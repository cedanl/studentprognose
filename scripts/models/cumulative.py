# cumulative.py

# --- Standard library ---
import os
import sys
import time
import json
import logging
import warnings
from pathlib import Path


# --- Third-party libraries ---
import numpy as np
from numpy import linalg as LA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import pandas as pd
import statsmodels.api as sm
import yaml
from dotenv import load_dotenv

# --- Project modules ---
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from scripts.utils.load_data import (
    load_cumulative,
    load_student_numbers_first_years,
    load_latest,
)
from scripts.prediction_methods.bayesian_models import BayesianRatioRegressor, BayesianClusterRegressor
from scripts.utils.helper import get_all_weeks_valid, get_pred_len, get_weeks_list, get_prediction_weeks_list, is_current_week
from scripts.standalone.evaluation import evaluate_predictions
from cli import parse_args


# --- Warnings and logging setup ---
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.filterwarnings("ignore", message="Too few observations*")
logger = logging.getLogger(__name__)

# --- Environment setup ---
load_dotenv()
ROOT_PATH = os.getenv("ROOT_PATH")

# --- Constant variable names ---
GROUP_COLS = [
    "Collegejaar", "Croho groepeernaam", "Faculteit",
    "Examentype", "Herkomst"
]

NUMERIC_COLS = [
    "Ongewogen vooraanmelders", "Gewogen vooraanmelders",
    "Aantal aanmelders met 1 aanmelding", "Inschrijvingen"
]

WEEK_COL = ["Weeknummer"]

TARGET_COL = ['Aantal_studenten']    

RENAME_MAP = {
    "Type hoger onderwijs": "Examentype",
    "Groepeernaam Croho": "Croho groepeernaam",
}

# --- Main cumulative class ---

class Cumulative():
    def __init__(self, data_cumulative, data_studentcount, data_latest, configuration):
        self.data_cumulative = data_cumulative
        self.data_studentcount = data_studentcount
        self.data_latest = data_latest
        self.configuration = configuration
        self.pred_len = None

        # Cached xgboost models
        self.xgboost_models = {}

        # Backup data
        self.data_cumulative_backup = self.data_cumulative.copy()

         # Store processing variables
        self.preprocessed = False

    # --------------------------------------------------
    # -- General helper functions --
    # --------------------------------------------------
    
    def _get_transformed_data(self, data: pd.DataFrame, column: str = "ts") -> pd.DataFrame:
        """
        Drops duplicates, filters data from start_year onwards, and transforms from long to wide format.
        """
        # Drop duplicates and filter years
        df = data.drop_duplicates()
        df = df[df["Collegejaar"] >= self.configuration["start_year"]]

        # Keep relevant columns
        df = df.loc[:, GROUP_COLS + TARGET_COL + [column, "Weeknummer"]].drop_duplicates()

        # Temporary filler for missing target col
        df[TARGET_COL] = df[TARGET_COL].fillna(99999)

        # Pivot to wide format
        df_wide = df.pivot_table(
            index=GROUP_COLS + TARGET_COL,
            columns="Weeknummer",
            values=column,
            aggfunc="sum",
            fill_value=0
        ).reset_index()

        # Flatten column names and reorder based on valid weeks
        df_wide.columns = df_wide.columns.map(str)
        valid_weeks = get_all_weeks_valid(df_wide.columns)
        df_wide = df_wide[GROUP_COLS + TARGET_COL + valid_weeks]

        # Remove temporary filler
        df_wide[TARGET_COL] = df_wide[TARGET_COL].replace(99999, np.nan)

        # Make sure week 52 is is similar to week 51
        df_wide['52'] = df_wide['51']

        return df_wide
    

    # --------------------------------------------------
    # -- Preprocessing --
    # --------------------------------------------------    

    def preprocess(self) -> pd.DataFrame:
        """
        Cleans, filters, aggregates, and merges cumulative pre-application data.
        """
        
        # --- Data copy ---
        df = self.data_cumulative.copy()

        # --- Rename columns ---
        df = df.rename(columns=RENAME_MAP)

        # --- Convert numeric columns to float64 ---
        for col in NUMERIC_COLS:
            if pd.api.types.is_string_dtype(df[col]):
                df[col] = pd.to_numeric(
                    df[col], 
                    decimal=',', 
                    thousands='.', 
                    errors='coerce'
                )

        df[NUMERIC_COLS] = df[NUMERIC_COLS].astype('float64')

        # --- Filter for first-year and pre-master students ---
        mask = (df["Hogerejaars"] == "Nee") | (df["Examentype"] == "Pre-master")
        df = df[mask]

        # --- Group and aggregate data ---
        processed_df = df.groupby(GROUP_COLS + WEEK_COL, as_index=False)[NUMERIC_COLS].sum()

        # --- Merge with student count data (if it exists) ---
        if self.data_studentcount is not None:
            processed_df = processed_df.merge(
                self.data_studentcount,
                on=[col for col in GROUP_COLS if col != "Faculteit"],
                how="left",
            )
        
        # --- Create the 'ts' (time series) target column by adding 'Gewogen vooraanmelders' and 'Inschrijvingen' ---
        processed_df["ts"] = (
            processed_df["Gewogen vooraanmelders"] + processed_df["Inschrijvingen"]
        )

        # --- Standardize faculty codes ---
        faculty_transformation = self.configuration["faculty"]
        processed_df["Faculteit"] = processed_df["Faculteit"].replace(faculty_transformation)

        # --- Set week 39 and week 40 to 0 ---
        processed_df.loc[processed_df["Weeknummer"] == 39, "ts"] = 0
        processed_df.loc[processed_df["Weeknummer"] == 40, "ts"] = 0

        # --- Final sorting, ordering, and duplicate removal ---
        
        # Determine the final column order, keeping original columns first
        final_cols_order = GROUP_COLS + WEEK_COL + NUMERIC_COLS
        existing_cols = set(final_cols_order)
        # Add any new columns from the merge and the 'ts' column
        new_cols = [col for col in processed_df.columns if col not in existing_cols]
        final_cols_order.extend(new_cols)

        processed_df = (
            processed_df.sort_values(by=GROUP_COLS + WEEK_COL, ignore_index=True)
            .drop_duplicates()
            [final_cols_order]  # Enforce consistent column order
        )

        # --- Update Instance Attributes ---
        self.data_cumulative_backup = self.data_cumulative.copy()
        self.data_cumulative = processed_df

        self.preprocessed = True

        return self.data_cumulative
    
    # --------------------------------------------------
    # -- Prediction pre-applicants with SARIMA --
    # --------------------------------------------------
    def _create_time_series(self, train, target, predict_week):

        # --- First create one large time series from the train data ---
        train = train.loc[:, '39':'38'] # Only the weekly data
        train_ts_data = train.values.flatten() # Flatten into a numpy array

        # --- Similar process to the target data
        valid_weeks = [str(x) for x in get_weeks_list(predict_week)] 
        target = target[valid_weeks]
        target_ts_data = target.values.flatten() # Flatten into a numpy array

        # --- Combine them togehter --- 
        ts_data = np.concatenate([train_ts_data, target_ts_data])

        return ts_data

    def _fit_sarima(self, ts_data: np.ndarray, model_name: str, predict_year: int):
        model_path = os.path.join(self.configuration["other_paths"]["cumulative_sarima_models"].replace("${root_path}", ROOT_PATH), f"{model_name}.json")

        sarimax_args = dict(
            order=(1, 0, 1),
            seasonal_order=(1, 1, 1, 52),
            #trend="c" if len(ts_data) < 52 else None,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        model = sm.tsa.SARIMAX(ts_data, **sarimax_args)

        if os.path.exists(model_path):
            try:
                with open(model_path, "r") as f:
                    model_data = json.load(f)
                loaded_params = model_data["model_params"]
                trained_year = model_data["trained_year"]

                if predict_year > trained_year:
                    fitted_model = model.fit(disp=False)
                else:
                    param_array = [loaded_params[name] for name in model.param_names]
                    fitted_model = model.fit(start_params=param_array, disp=False)
                    return fitted_model
                
            except KeyError:
                fitted_model = model.fit(disp=False)
        else:
            fitted_model = model.fit(disp=False)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "w") as f:
            json.dump(
                {"trained_year": predict_year, "model_params": dict(zip(fitted_model.param_names, fitted_model.params))},
                f, indent=4
            )

        return fitted_model

    def run_sarima(self, train, target, combo, predict_week, predict_year) -> list[float]:
        """
        Predict pre-registrations with SARIMA
        """

        # --- Pprediction length ---
        pred_len = get_pred_len(predict_week)

        # --- Prediction ---
        try:
            ts_data_weighted = self._create_time_series(train, target, predict_week)
            model_name = f"{combo.iloc[0]}{combo.iloc[1]}{combo.iloc[2]}"
            results = self._fit_sarima(ts_data_weighted, model_name, predict_year)
            predictions = results.forecast(steps=pred_len).tolist()
            return predictions

        except (LA.LinAlgError, IndexError, ValueError):
            return []

    def predict_preapplicants(self, df: pd.DataFrame, predict_year: int, predict_week: int):
        
        df_preapplicants = self._get_transformed_data(self.data_cumulative, column = 'Gewogen vooraanmelders')

        # Prepare a results dataframe
        df_results = df.copy()
        df_results["Voorspelde gewogen vooraanmelders"] = np.nan

        # Get unique combinations
        combos = df_results[["Croho groepeernaam", "Herkomst", "Examentype", 'Faculteit']].drop_duplicates()

        for _, combo in combos.iterrows():
            programme, herkomst, examentype, faculteit = combo

            # --- Filter the data and split --- 
            train_prog_preapps, _, target_preapps = self._filter_data(df_preapplicants, programme, herkomst, examentype, faculteit, predict_year)
            if len(train_prog_preapps) > 2:
                preapps_pred = self.run_sarima(train_prog_preapps, target_preapps, combo, predict_week, predict_year)
            else:
                preapps_pred = []

            # Assign to all matching rows in df_results
            mask = (
                (df_results["Croho groepeernaam"] == programme) &
                (df_results["Herkomst"] == herkomst) &
                (df_results["Examentype"] == examentype)
            )
            try:
                df_results.loc[mask, "Voorspelde gewogen vooraanmelders"] = preapps_pred
            except ValueError:
                pass

        return df_results


    # --------------------------------------------------
    # -- Prediction actual student inflow --
    # --------------------------------------------------

    # --- Helpers ---
    def _filter_data(self, data: pd.DataFrame, programme: str, herkomst: str, examentype: str, faculteit: str, predict_year: int) -> pd.DataFrame:

        covid_year = self.configuration["covid_year"]
        numerus_fixus = self.configuration.get("numerus_fixus", {})
        used_to_be_nf = self.configuration.get("used_to_be_numerus_fixus", {})

        # --- Remove COVID year ---
        data = data[data.Collegejaar != covid_year]

        # --- Split into train and target ---
        train = data[data.Collegejaar < predict_year]
        target = data[
            (data.Collegejaar == predict_year)
            & (data["Croho groepeernaam"] == programme)
            & (data.Herkomst == herkomst)
            & (data.Examentype == examentype) 
        ]

        # --- If programme is numerus fixus, restrict training data accordingly ---
        if programme in numerus_fixus:
            train_filtered = train[
                (train["Croho groepeernaam"] == programme)
                & (train.Herkomst == herkomst)
                & (train.Examentype == examentype)
            ]
            # For NF: train_prog == train_total
            return train_filtered, train_filtered, target

        # --- Exclude years when programmes were NF ---
        exclude_pairs = {
            (prog, year)
            for prog, years in used_to_be_nf.items()
            for year in years
        }

        train_pairs = train[["Croho groepeernaam", "Collegejaar"]].apply(tuple, axis=1)

        if not programme in used_to_be_nf.keys():
            train = train[~train_pairs.isin(exclude_pairs)].copy()

        # --- Programme-specific and total training subsets ---
        train_prog = train[
            (train["Croho groepeernaam"] == programme)
            & (train.Herkomst == herkomst)
            & (train.Examentype == examentype)
        ]

        train_total = train[
            (train.Herkomst == herkomst)
            & (train.Examentype == examentype)
            #& (train.Faculteit == faculteit)
            & (~train["Croho groepeernaam"].isin(self.configuration.get("numerus_fixus", [])))
        ]

        return train_prog, train_total, target

    ### --- Main logic --- ###
    def predict_student_inflow(self, df: pd.DataFrame, predict_year: int, predict_week: int, print_output=False, verbose = False):
        """
        Vectorized prediction using clustering per programme/herkomst/examentype.
        """
        df_wide = self._get_transformed_data(self.data_cumulative)

        # Prepare a results dataframe
        df_results = df.copy()
        df_results["Cumulative_ratio"] = np.nan
        df_results["Cumulative_mean"] = np.nan

        # --- Create the prediction models ---
        ratio_predictor = BayesianRatioRegressor(predict_week, verbose=verbose)
        cluster_predictor = BayesianClusterRegressor(predict_week, verbose=verbose)

        # Get unique combinations
        combos = df_results[["Croho groepeernaam", "Herkomst", "Examentype", 'Faculteit']].drop_duplicates()

        for _, combo in combos.iterrows():
            programme, herkomst, examentype, faculteit = combo

            # --- Filter the data and split --- 
            train_prog, train_total, target = self._filter_data(df_wide, programme, herkomst, examentype, faculteit, predict_year)

            # -- Ratio prediction --
            if not len(train_prog) == 0:
                try:
                    ratio_predictor.fit(train_prog.drop(columns='Aantal_studenten'), train_prog['Aantal_studenten'].copy())
                    ratio_pred = ratio_predictor.predict(target)[0]
                except (ValueError, IndexError, TypeError, UnboundLocalError):
                    ratio_pred = 0 
            else:
                ratio_pred = np.nan

            # -- Mean prediction --
            try:
                cluster_predictor.fit(train_prog, train_total, target)
                mean_pred = cluster_predictor.predict()[0]
            except (ValueError, IndexError, TypeError, UnboundLocalError):
                mean_pred = 0

            # Assign to all matching rows in df_results
            mask = (
                (df_results["Croho groepeernaam"] == programme) &
                (df_results["Herkomst"] == herkomst) &
                (df_results["Examentype"] == examentype)
            )
            df_results.loc[mask, "Cumulative_ratio"] = ratio_pred
            df_results.loc[mask, "Cumulative_mean"] = mean_pred

            if print_output:
                print(
                    f"Cumulative (ratio) prediction for {programme}, {herkomst}, {examentype}, year: {predict_year}, week: {predict_week}: {ratio_pred}"
                )

                print(
                    f"Cumulative (mean) prediction for {programme}, {herkomst}, {examentype}, year: {predict_year}, week: {predict_week}: {mean_pred}"
                )

        return df_results


    # --------------------------------------------------
    # -- Full prediction loop --
    # --------------------------------------------------
    
    ### --- Main logic --- ###
    def run_full_prediction_loop(self, predict_year: int, predict_week: int, write_file: bool, print_output: bool, args = None):
        """
        Run the full prediction loop for all years and weeks.
        """
        logger.info("Running cumulative prediction loop")

         # --- Preprocess data (if not already done) ---
        if not self.preprocessed:
            self.preprocess()

        # --- Apply filtering from configuration ---
        filtering = self.configuration["filtering"]

        # --- Filter data ---
        base_mask = np.ones(len(self.data_latest), dtype=bool) 

        # --- Apply conditional filters from configuration ---
        if filtering["programme"]:
            base_mask &= self.data_latest["Croho groepeernaam"].isin(filtering["programme"])
        if filtering["herkomst"]:
            base_mask &= self.data_latest["Herkomst"].isin(filtering["herkomst"])
        if filtering["examentype"]:
            base_mask &= self.data_latest["Examentype"].isin(filtering["examentype"])
        
        # --- Apply year and week filters ---
        base_mask &= self.data_latest["Collegejaar"] == predict_year

        prediction_mask = base_mask & (self.data_latest["Weeknummer"] == predict_week)

        # --- Apply mask ---
        prediction_df = self.data_latest.loc[prediction_mask, GROUP_COLS + WEEK_COL].copy()

        # --- Prediction ---
        prediction_df = self.predict_student_inflow(prediction_df, predict_year, predict_week, print_output=print_output, verbose=args.verbose)

        # --- Assign predictions back to main dataframe ---
        self.data_latest.loc[prediction_df.index, "Cumulative_ratio"] = prediction_df["Cumulative_ratio"]
        self.data_latest.loc[prediction_df.index, "Cumulative_mean"] = prediction_df["Cumulative_mean"]

        # --- Add the predicited pre-applicants if required ---
        if is_current_week(predict_year, predict_week):
            logger.info("The predict week equals current week, also predicting preapplicants..")
            preapps_mask = base_mask & self.data_latest["Weeknummer"].isin(get_prediction_weeks_list(predict_week))
            prediction_df_preapps = self.data_latest.loc[preapps_mask, GROUP_COLS + WEEK_COL].copy()
            prediction_df_preapps = self.predict_preapplicants(prediction_df_preapps, predict_year, predict_week)
            self.data_latest.loc[prediction_df_preapps.index, "Voorspelde gewogen vooraanmelders"] = prediction_df_preapps["Voorspelde gewogen vooraanmelders"]
            logger.info("Pre-applicants prediction finished.")


        # --- Write the file ---
        if write_file:
            output_path = self.configuration["paths"]["output"]["path_output"].replace("${time}", time.strftime("%Y%m%d_%H%M%S"))
            self.data_latest.to_excel(output_path, index=False, engine="xlsxwriter")

        # --- Evaluate predictions (if required) ---
        if args.evaluate:
            evaluate_predictions(self.data_latest, 'Aantal_studenten', ['Cumulative_ratio', 'Cumulative_mean'], self.configuration, args, 'Prognose_ratio')

        logger.info("Cumulative prediction done")

# --- Main function ---
def main():
    # --- Parse arguments ---
    args = parse_args()

    # --- Load configuration ---
    CONFIG_FILE = Path("configuration.yaml")
    with open(CONFIG_FILE, "r") as f:
        configuration = yaml.safe_load(f)  

    # --- Load data ---
    cumulative_data = load_cumulative()
    student_counts = load_student_numbers_first_years()
    latest_data = load_latest()

    # --- Initialize model ---
    cumulative_model = Cumulative(cumulative_data, student_counts, latest_data, configuration)

    # --- Run prediction loop ---
    for year in args.years:
        for week in args.weeks:
            cumulative_model.run_full_prediction_loop(
                predict_year=year,
                predict_week=week,
                write_file=args.write_file,
                print_output=args.print,
                args=args
            )


if __name__ == "__main__":
    main()

