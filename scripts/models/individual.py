# individual.py

# --- Standard library ---
import os
import sys
import math
import json
import time
import logging
import warnings
from datetime import date
from pathlib import Path

# --- Third-party libraries ---
import numpy as np
import pandas as pd
import yaml
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from dotenv import load_dotenv

# --- Project modules ---
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from scripts.prediction_methods.bayesian_models import BayesianRatioRegressor, BayesianClusterRegressor, BayesianTSRegressor
from scripts.utils.load_data import (
    load_individual,
    load_distances,
    load_latest,
    load_cumulative
)
from scripts.utils.helper import get_all_weeks_valid, get_weeks_list
from cli import parse_args
from scripts.standalone.evaluation import evaluate_predictions
from scripts.utils.clustering import cluster_programme

# --- Warnings and logging setup ---
warnings.simplefilter("ignore", ConvergenceWarning)
logger = logging.getLogger(__name__)

# --- Environment setup ---
load_dotenv()
ROOT_PATH = os.getenv("ROOT_PATH")

# --- Constant variable names ---
GROUP_COLS = [
    "Collegejaar", "Croho groepeernaam", "Faculteit",
    "Examentype", "Herkomst"
]

CATEGORICAL_COLS = [
    "Opleiding",
    "Type vooropleiding",
    "Nationaliteit",
    "EER",
    "Geslacht",
    "Geverifieerd adres land",
    "School eerste vooropleiding",
    "Land code eerste vooropleiding"
]

NUMERIC_COLS = [
    "Sleutel_count",
    "is_numerus_fixus",
    "Afstand",
    "Deadlineweek"
]

WEEK_COL = ["Weeknummer"]

TARGET_COL = ['Inschrijfstatus']    


# --- Main individual class ---

class Individual():
    def __init__(self, data_individual, data_distances, data_latest, configuration, data_cumulative = None):
        self.data_individual = data_individual
        self.data_cumulative = data_cumulative
        self.data_distances = data_distances
        self.data_latest = data_latest
        self.configuration = configuration
        self.pred_len = None

        # Backup data
        self.data_individual_backup = self.data_individual.copy()

    # --------------------------------------------------
    # -- General helper functions --
    # --------------------------------------------------
    
    def _get_transformed_data(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Filters, cleans, and reshapes data from long to wide format.
        """
        start_year = self.configuration["individual_start_year"]

        # Filter and keep only necessary columns (avoid chained indexing)
        df = (
            df.loc[df["Collegejaar"] >= start_year, GROUP_COLS + ["Weeknummer", column]]
            .assign(**{column: pd.to_numeric(df[column], errors="coerce")})
        )

        # Pivot once to wide format
        pivot = (
            df.pivot_table(
                index=GROUP_COLS,
                columns="Weeknummer",
                values=column,
                aggfunc="sum",
                fill_value=0,
            )
            .sort_index(axis=1)
            .reset_index()
        )

        # Determine valid weeks (should be strings)
        pivot.columns = pivot.columns.map(str)

        valid_weeks = get_all_weeks_valid(pivot.columns)
        pivot = pivot[GROUP_COLS + valid_weeks]

        # Compute cumulative sum across weeks (vectorized)
        pivot[valid_weeks] = pivot[valid_weeks].cumsum(axis=1)

        merged = pivot.merge(
            self.data_latest[GROUP_COLS + ['Aantal_studenten']],
            on=GROUP_COLS,
            how='left'
        )

        merged = merged.drop_duplicates()

        return merged
    

    # --------------------------------------------------
    # -- Preprocessing --
    # --------------------------------------------------

    ### --- Helpers --- ###
    def to_weeknummer(self, datum):
        try:
            day, month, year = map(int, datum.split("-"))
            return date(year, month, day).isocalendar()[1]
        except (AttributeError, ValueError):
            return np.nan

    def get_herkomst(self, nat, eer):
        if nat == "Nederlandse":
            return "NL"
        elif eer == "J":
            return "EER"
        return "Niet-EER"

    def get_deadlineweek(self, row):
        return row["Weeknummer"] == 17 and (
            row["Croho groepeernaam"] not in list(self.configuration["numerus_fixus"].keys())
            or row["Examentype"] != "Bachelor"
        )

    # --- Main logic ---
    def preprocess(self, df) -> pd.DataFrame:
        """
        Preprocess the input data for further analysis.
        """

        # --- Load and clean base dataset ---
        df = df.drop(columns=["Aantal studenten"])

        # --- Filter out specific English programme in 2021 ---
        mask = (
            (df["Croho groepeernaam"] == "B English Language and Culture")
            & (df["Collegejaar"] == 2021)
            & (df["Examentype"] != "Propedeuse Bachelor")
        )
        df = df[~mask]

        # --- Add count of entries per key ---
        df["Sleutel_count"] = df.groupby(["Collegejaar", "Sleutel"])["Sleutel"].transform(
            "count"
        )

        # --- Convert dates to week numbers ---
        df["Datum intrekking vooraanmelding"] = df["Datum intrekking vooraanmelding"].apply(
            self.to_weeknummer
        )
        df["Weeknummer"] = df["Datum Verzoek Inschr"].apply(self.to_weeknummer)

        # --- Derive origin from nationality and EER flag ---
        df["Herkomst"] = df.apply(lambda x: self.get_herkomst(x["Nationaliteit"], x["EER"]), axis=1)

        # --- Keep only entries with September or October intake ---
        df = df[
            df["Ingangsdatum"].str.contains("01-09-")
            | df["Ingangsdatum"].str.contains("01-10-")
        ]

        # --- Update RU faculty name ---
        df["Faculteit"] = df["Faculteit"].replace(self.configuration["faculty"])

        # --- Add numerus fixus flag ---
        df["is_numerus_fixus"] = (
            df["Croho groepeernaam"].isin(list(self.configuration["numerus_fixus"].keys()))
            & (df["Examentype"] == "Bachelor")
        ).astype(int)

        # --- Normalize exam type names ---
        df["Examentype"] = df["Examentype"].replace("Propedeuse Bachelor", "Bachelor")

        # --- Filter on valid enrollment status and exam types ---
        df = df[df["Inschrijfstatus"].notna()]
        df = df[df["Examentype"].isin(["Bachelor", "Master", "Pre-master"])]

        # --- Collapse rare nationalities into 'Overig' ---
        counts = df["Nationaliteit"].value_counts()
        rare_values = counts[counts < 100].index
        df["Nationaliteit"] = df["Nationaliteit"].replace(rare_values, "Overig")

        # --- Add distances if available ---
        if self.data_distances is not None:
            afstand_lookup = self.data_distances.set_index("Geverifieerd adres plaats")["Afstand"]
            df["Afstand"] = df["Geverifieerd adres plaats"].map(afstand_lookup)
        else:
            df["Afstand"] = np.nan

        # --- Determine deadline week flag ---
        df["Deadlineweek"] = df.apply(self.get_deadlineweek, axis=1)

        # --- Drop unneeded columns ---
        df = df.drop(columns=["Sleutel"])

        # --- Special handling for pre-master entries ---
        premaster_mask = df["Examentype"] == "Pre-master"
        df.loc[
            premaster_mask, ["Is eerstejaars croho opleiding", "Is hogerejaars", "BBC ontvangen"]
        ] = [1, 0, 0]

        # --- Final filtering on enrollment status ---
        df = df[
            (df["Is eerstejaars croho opleiding"] == 1)
            & (df["Is hogerejaars"] == 0)
            & (df["BBC ontvangen"] == 0) 
            & (df['Hoofdopleiding'] == 'ja')]

        # --- Final cleanup ---
        df = df[GROUP_COLS + CATEGORICAL_COLS + NUMERIC_COLS + WEEK_COL + TARGET_COL + ["Datum intrekking vooraanmelding"]]

        return df

    # --------------------------------------------------
    # -- Prediction of pre-applicant probabilities (chance that someone will enroll) --
    # --------------------------------------------------
    
    ### --- Main logic --- ###
    def predict_preapplicant_probabilities(self, df: pd.DataFrame, predict_year: int, predict_week: int) -> pd.DataFrame:
        """
        Predict the probability that a pre-applicant will enroll for each individual.
        Uses per-group models based on 'herkomst' and 'examentype'.
        Numerus fixus programmes are kept in the dataframe but are handled separately.
        """

        weeks_to_predict = get_weeks_list(predict_week)
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        numerus_fixus_list = list(self.configuration["numerus_fixus"].keys())

        # --- Train/Test Split ---
        train_mask = (df["Collegejaar"] < predict_year) & (df["Collegejaar"] >= self.configuration["individual_start_year"])

        # --- Test mask ---
        filtering = self.configuration["filtering"]

        # --- Filter data ---
        test_mask = np.ones(len(df), dtype=bool) 

        # --- Apply conditional filters from configuration ---
        if filtering["programme"]:
            test_mask &= df["Croho groepeernaam"].isin(filtering["programme"])
        if filtering["herkomst"]:
            test_mask &= df["Herkomst"].isin(filtering["herkomst"])
        if filtering["examentype"]:
            test_mask &= df["Examentype"].isin(filtering["examentype"])
        
        # --- Apply year and week filters ---
        test_mask &= df["Collegejaar"] == predict_year
        
        # --- Apply mask ---
        train = df[train_mask].copy()
        test = df[test_mask].copy()

        # --- Remove numerus fixus programmes from training and test for non-numerus predictions ---
        train = train[~train["Opleiding"].isin(numerus_fixus_list)]
        test_non_nf = test[~test["Opleiding"].isin(numerus_fixus_list)]

        # --- Filter out cancelled registrations ---
        if predict_week <= 38:
            cancellation_filter = train["Datum intrekking vooraanmelding"].isna() | (
                (train["Datum intrekking vooraanmelding"] >= predict_week) & (train["Datum intrekking vooraanmelding"] < 39)
            )
        else:
            cancellation_filter = (
                train["Datum intrekking vooraanmelding"].isna()
                | (train["Datum intrekking vooraanmelding"] > predict_week)
                | (train["Datum intrekking vooraanmelding"] < 39)
            )
        train = train[cancellation_filter]

        # --- Target Mapping ---
        status_map = {
            "Ingeschreven": 1,
            "Uitgeschreven": 1,
            "Geannuleerd": 0,
            "Verzoek tot inschrijving": 0,
            "Studie gestaakt": 0,
            "Aanmelding vervolgen": 0,
        }
        original_statuses = test_non_nf[TARGET_COL[0]].copy()
        train[TARGET_COL[0]] = train[TARGET_COL[0]].map(status_map)
        test_non_nf[TARGET_COL[0]] = test_non_nf[TARGET_COL[0]].map(status_map)

        # --- Prepare predictions container ---
        final_predictions = pd.Series(index=test_non_nf.index, dtype=float)

        # --- Iterate over groups ---
        group_cols = ["Herkomst", "Examentype", "Croho groepeernaam"]
        for group_values, test_group in test_non_nf.groupby(group_cols):
            train_group = train.copy()
            for col, val in zip(group_cols, group_values):
                train_group = train_group[train_group[col] == val]
            if train_group.empty or train_group[TARGET_COL[0]].nunique() < 2:
                train_group = train.copy()

            # --- Features and Labels ---
            X_train_group = train_group.drop(columns=[TARGET_COL[0]])
            y_train_group = train_group[TARGET_COL[0]]
            X_test_group = test_group.drop(columns=[TARGET_COL[0]])

            # --- Preprocessing + Model Pipeline ---
            preprocessor = ColumnTransformer(
                transformers=[
                    ("numeric", "passthrough", NUMERIC_COLS),
                    ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                    CATEGORICAL_COLS + GROUP_COLS + WEEK_COL),
                ],
                remainder="drop",
            )

            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", XGBClassifier(objective="binary:logistic", learning_rate=0.001, eval_metric="auc", random_state=0))
            ])

            # --- Fit and predict ---
            pipeline.fit(X_train_group, y_train_group)
            probas = pipeline.predict_proba(X_test_group)[:, 1]

            # --- Post-processing for cancellations ---
            cancelled_flag = (original_statuses.loc[test_group.index] == "Geannuleerd")
            week_mask = np.isin(test_group["Datum intrekking vooraanmelding"].to_numpy(), weeks_to_predict)
            mask = cancelled_flag.to_numpy() & week_mask
            probas[mask] = 0

            # Store predictions
            final_predictions.loc[test_group.index] = probas

        # --- Assign predictions back to main dataframe ---
        df.loc[:, TARGET_COL[0]] = df[TARGET_COL[0]].map(status_map)
        df.loc[final_predictions.index, TARGET_COL[0]] = final_predictions

        return df



    # --------------------------------------------------
    # -- Prediction of inflow  --
    # --------------------------------------------------
    
    # --- Helpers ---
    def _filter_data(self, data: pd.DataFrame, programme: str, herkomst: str, examentype: str, predict_year: int) -> pd.DataFrame:

        covid_year = self.configuration["covid_year"]
        numerus_fixus = self.configuration.get("numerus_fixus", {})
        used_to_be_nf = self.configuration.get("used_to_be_numerus_fixus", {})

        # --- Remove COVID year ---
        data = data[data.Collegejaar != covid_year]

        # --- Also remove 2021 (this year has faulty data) ---
        data = data[data.Collegejaar != 2021]

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

        # Remove pairs that used to NF
        if not programme in used_to_be_nf.keys():
            train_pairs = train[["Croho groepeernaam", "Collegejaar"]].apply(tuple, axis=1)
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
            & (~train["Croho groepeernaam"].isin(self.configuration.get("numerus_fixus", [])))
        ]

        return train_prog, train_total, target


    ### --- Main logic --- ###
    def predict_student_inflow(self, df: pd.DataFrame, predict_year: int, predict_week: int, print_output=False, verbose=False):
        """
        Vectorized prediction using clustering per programme/herkomst/examentype.
        """

        # -- Transform the data from long to wide --
        df_wide = self._get_transformed_data(self.data_individual_copy, column = TARGET_COL[0])

        # Prepare a results dataframe
        df_results = df.copy()
        df_results["Individual_ratio"] = np.nan
        df_results["Individual_mean"] = np.nan

        # --- Create the prediction models ---
        ratio_predictor = BayesianRatioRegressor(predict_week, verbose=verbose)
        cluster_predictor = BayesianClusterRegressor(predict_week, verbose=verbose)

        # Get unique combinations
        combos = df_results[["Croho groepeernaam", "Herkomst", "Examentype"]].drop_duplicates()

        for _, combo in combos.iterrows():
            programme, herkomst, examentype = combo

            # --- Filter the data and split --- 
            train_prog, train_total, target = self._filter_data(df_wide, programme, herkomst, examentype, predict_year)

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
            except (ValueError, IndexError, TypeError, UnboundLocalError, AttributeError):
                mean_pred = 0

            # Assign to all matching rows in df_results
            mask = (
                (df_results["Croho groepeernaam"] == programme) &
                (df_results["Herkomst"] == herkomst) &
                (df_results["Examentype"] == examentype)
            )
            df_results.loc[mask, "Individual_ratio"] = ratio_pred
            df_results.loc[mask, "Individual_mean"] = mean_pred

            if print_output:
                print(
                    f"Individual (ratio) prediction for {programme}, {herkomst}, {examentype}, year: {predict_year}, week: {predict_week}: {ratio_pred}"
                )

                print(
                    f"Individual (mean) prediction for {programme}, {herkomst}, {examentype}, year: {predict_year}, week: {predict_week}: {mean_pred}"
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
        logger.info("Running individual prediction loop")

        # --- Get copy of the full dataset ---
        self.data_individual_copy = self.data_individual.copy()

        # --- Preprocess if not done ---
        self.data_individual_copy = self.preprocess(self.data_individual_copy)

        # --- Predict preapplicant probabilities ---
        self.data_individual_copy = self.predict_preapplicant_probabilities(self.data_individual_copy, predict_year, predict_week)

        # --- Apply filtering from configuration ---
        filtering = self.configuration["filtering"]

        # --- Filter data ---
        mask = np.ones(len(self.data_latest), dtype=bool) 

        # --- Apply conditional filters from configuration ---
        if filtering["programme"]:
            mask &= self.data_latest["Croho groepeernaam"].isin(filtering["programme"])
        if filtering["herkomst"]:
            mask &= self.data_latest["Herkomst"].isin(filtering["herkomst"])
        if filtering["examentype"]:
            mask &= self.data_latest["Examentype"].isin(filtering["examentype"])
        
        # --- Apply year and week filters ---
        mask &= self.data_latest["Collegejaar"] == predict_year
        prediction_df = self.data_latest.loc[mask, GROUP_COLS + WEEK_COL].copy()

        mask &= self.data_latest["Weeknummer"] == predict_week
        
        # --- Apply mask ---
        prediction_df = self.data_latest.loc[mask, GROUP_COLS + WEEK_COL].copy()

        # --- Prediction ---
        prediction_df = self.predict_student_inflow(prediction_df, predict_year, predict_week, print_output=print_output, verbose = args.verbose)

        # --- Assign predictions back to main dataframe ---
        self.data_latest.loc[prediction_df.index, "Individual_ratio"] = prediction_df["Individual_ratio"]
        self.data_latest.loc[prediction_df.index, "Individual_mean"] = prediction_df["Individual_mean"]

        # --- Write the file ---
        if write_file:
            output_path = self.configuration["paths"]["output"]["path_output"].replace("${time}", time.strftime("%Y%m%d_%H%M%S"))
            self.data_latest.to_excel(output_path, index=False, engine="xlsxwriter")


        # --- Evaluate predictions (if required) ---
        if args.evaluate:
            evaluate_predictions(self.data_latest, 'Aantal_studenten', ['Individual_ratio', 'Individual_mean'], self.configuration, args, 'Prognose_ratio')

        logger.info("Individual prediction done")



# --- Main function ---
def main():
    # --- Parse arguments ---
    args = parse_args()

    # --- Load configuration ---
    CONFIG_FILE = Path("configuration.yaml")
    with open(CONFIG_FILE, "r") as f:
        configuration = yaml.safe_load(f)  

    # --- Load data ---
    individual_data = load_individual()
    distances = load_distances()
    latest_data = load_latest()
    data_cumulative = load_cumulative()

    # --- Initialize model ---
    individual_model = Individual(individual_data, distances, latest_data, configuration, data_cumulative)

    # --- Run prediction loop ---
    for year in args.years:
        for week in args.weeks:
            individual_model.run_full_prediction_loop(
                predict_year=year,
                predict_week=week,
                write_file=args.write_file,
                print_output=args.print,
                args=args
            )



if __name__ == "__main__":
    main()