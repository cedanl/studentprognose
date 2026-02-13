# ensemble.py

# --- Standard library ---
import os
import sys
import logging
import time
from pathlib import Path

# --- Third-party libraries ---
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from dotenv import load_dotenv

# --- Project modules ---
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from scripts.utils.load_data import (
    load_latest,
)
from cli import parse_args
from scripts.standalone.evaluation import evaluate_predictions

# --- Warnings and logging setup ---
logger = logging.getLogger(__name__)

# --- Environment setup ---
load_dotenv()
ROOT_PATH = os.getenv("ROOT_PATH")

# --- Constant variable names ---
GROUP_COLS = [
    "Collegejaar", "Croho groepeernaam", "Faculteit",
    "Examentype", "Herkomst"
]

PREDICTION_COLS = ['Cumulative_ratio', 'Cumulative_mean', 'Individual_ratio']#, 'Individual_mean']

WEEK_COL = ["Weeknummer"]

TARGET_COL = ['Aantal_studenten']  

# --- Main ratio class ---

class Ensemble():
    def __init__(self, data_latest, configuration):
        self.data_latest = data_latest
        self.configuration = configuration

    # --------------------------------------------------
    # -- Predicting using the ensemble method --
    # --------------------------------------------------

    ### --- Helpers --- ###
    def _filter_data(self, df: pd.DataFrame, programme: str, herkomst: str, examentype: str, faculteit: str, predict_year: int, predict_week: int) -> pd.DataFrame:
        df = df[df["Collegejaar"] >= self.configuration["ensemble_start_year"]]

        population_mask = (
            (df['Croho groepeernaam'] == programme) &
            (df['Collegejaar'] <= predict_year) &
            (df['Herkomst'] == herkomst) &
            (df['Weeknummer'] == predict_week) &
            (df['Examentype'] == examentype) &
            (df["Collegejaar"] != 2020) &
            (df["Collegejaar"] >= (predict_year - 2)) # Only last two years
        )
        
        filtered = df[population_mask].copy()
        
        return filtered

    def _clean_data(self, df: pd.DataFrame, predict_year: int) -> pd.DataFrame:
        # Remove rows where all columns are 0
        df = df.drop(df.index[df[PREDICTION_COLS].sum(axis=1) == 0])

        # Remove rows where target is NaN, but only if it's NOT the predict_year
        df = df[~(df['Collegejaar'] != predict_year) | df[TARGET_COL[0]].notna()]

        # Replace Inf/-Inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaNs with group-wise mean
        #df[PREDICTION_COLS] = df.groupby(["Croho groepeernaam", "Faculteit", "Examentype", "Herkomst"])[PREDICTION_COLS].transform(lambda x: x.fillna(x.mean()))

        # Replace NANs with 0
        df[PREDICTION_COLS] = df[PREDICTION_COLS].replace(np.nan, 0)

        return df

    def _calculate_weights(self, X_train, y_train):
        """
        Calculate optimal weights for a linear ensemble:
            y_pred = w1*m1 + w2*m2 + ... + wn*mn
        """
        X = np.array(X_train)
        y = np.array(y_train)

        n_models = X.shape[1]

        # Constrained optimization
        def objective(w):
            return np.sum((y - X.dot(w))**2)

        # Initial guess: equal weights
        w0 = np.ones(n_models) / n_models

        # Constraints
        constraints = []
        bounds = None

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        bounds = [(0, None)] * n_models

        result = minimize(objective, w0, bounds=bounds, constraints=constraints)

        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)
        
        weights = result.x

        return weights

    
    ### --- Main logic --- ###
    def make_ensemble_prediction(self, df, predict_year: int, programme) -> int:
        """
        Predict the inflow of students based on the ratio of pre-applicants.
        """

        # --- Clean to make sure for no corrupt values or empty values ---
        try:
            df = self._clean_data(df, predict_year)
        except ValueError:
            return 0

        # --- Make a train and test split ---
        train = df[df["Collegejaar"] < predict_year]
        test = df[df["Collegejaar"] == predict_year]

        x_train = train[PREDICTION_COLS]
        y_train = train[TARGET_COL]
        
        x_test = test[PREDICTION_COLS]

        # --- Filter Columns First ---
        # Determine valid columns based on x_test logic
        valid_cols = (x_test.mean() > 1) & (x_test.ne(0).any())
        selected_cols = x_test.columns[valid_cols]

        # Filter x_train and x_test to use ONLY the selected columns
        x_train_selected = x_train[selected_cols]
        x_test_selected = x_test[selected_cols]

        # --- Calculate Weights on the subset ---
        if x_train_selected is not None and not x_train_selected.empty:
            try:
                # pass the filtered training data so weights match the shape of selected_cols
                weights = self._calculate_weights(x_train_selected, y_train)
                weights = np.array(weights)
            except RuntimeError:
                count = len(selected_cols)
                weights = np.full(count, 1 / count) if count > 0 else np.array([])
        else:
            count = len(selected_cols)
            weights = np.full(count, 1 / count) if count > 0 else np.array([])

        # --- Compute Prediction ---
        # Now weights and x_test_selected have matching dimensions
        if len(weights) == 0:
            prediction = 0
        else:
            prediction = round(np.sum(weights * x_test_selected.values))

        return prediction
    
    def predict_using_ensemble_method(self, df: pd.DataFrame, predict_year: int, predict_week: int, print_output=False):
        """
        Vectorized prediction using clustering per programme/herkomst/examentype.
        """
        # Prepare a results dataframe
        df_results = df.copy()
        df_results["Ensemble_prediction"] = np.nan

        # Get unique combinations
        combos = df_results[["Croho groepeernaam", "Herkomst", "Examentype", 'Faculteit']].drop_duplicates()

        for _, combo in combos.iterrows():
            programme, herkomst, examentype, faculteit = combo

            # --- Filter the data and split --- 
            df_combo = self._filter_data(self.data_latest, programme, herkomst, examentype, faculteit, predict_year, predict_week)

            ensemble_pred = self.make_ensemble_prediction(df_combo, predict_year, programme)

            # Assign to all matching rows in df_results
            mask = (
                (df_results["Croho groepeernaam"] == programme) &
                (df_results["Herkomst"] == herkomst) &
                (df_results["Examentype"] == examentype)
            )
            df_results.loc[mask, "Ensemble_prediction"] = ensemble_pred

            if print_output:
                print(
                    f"Ensemble prediction for {programme}, {herkomst}, {examentype}, year: {predict_year}, week: {predict_week}: {ensemble_pred}"
                )

        return df_results


    # --------------------------------------------------
    # -- Full prediction loop --
    # --------------------------------------------------

    def run_full_prediction_loop(self, predict_year: int, predict_week: int, write_file: bool, print_output: bool, args = None):

        """
        Run the full prediction loop for all years and weeks.
        """
        logger.info('Running Ensemble prediction loop')

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
        mask &= self.data_latest["Weeknummer"] == predict_week

        # --- Apply mask ---
        prediction_df = self.data_latest.loc[mask, GROUP_COLS + WEEK_COL + PREDICTION_COLS + TARGET_COL].copy()

        # --- Prediction ---
        prediction_df = self.predict_using_ensemble_method(prediction_df, predict_year, predict_week, print_output=print_output)

        # --- Assign predictions back to main dataframe ---
        self.data_latest.loc[prediction_df.index, "Ensemble_prediction"] = prediction_df["Ensemble_prediction"]

        # --- Write the file ---
        if write_file:
            output_path = self.configuration["paths"]["output"]["path_output"].replace("${time}", time.strftime("%Y%m%d_%H%M%S"))
            self.data_latest.to_excel(output_path, index=False, engine="xlsxwriter")

        # --- Evaluate predictions (if required) ---
        if args.evaluate:
            evaluate_predictions(self.data_latest, 'Aantal_studenten', ['Ensemble_prediction'], self.configuration, args, 'Prognose_ratio')

        logger.info('Ensemble prediction done')

# --- Main function ---
def main():
    # --- Parse arguments ---
    args = parse_args()

    # --- Load configuration ---
    CONFIG_FILE = Path("configuration.yaml")
    with open(CONFIG_FILE, "r") as f:
        configuration = yaml.safe_load(f)  

    # --- Load data ---
    latest_data = load_latest()

    # --- Initialize model ---
    ensemble_model = Ensemble(latest_data, configuration)

    # --- Run prediction loop ---
    for year in args.years:
        for week in args.weeks:
            ensemble_model.run_full_prediction_loop(
                predict_year=year,
                predict_week=week,
                write_file=args.write_file,
                print_output=args.print,
                args=args
            )


if __name__ == "__main__":
    main()
    

    