# higher-years.py

# --- Standard library ---
import os
import sys
import logging
import warnings
from datetime import date
import time
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

from scripts.utils.load_data import (
    load_latest,
    load_oktober_file
)
from cli import parse_args


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
    "Studievorm code",
    "Geslacht",
    "Type vooropleiding eerste vooropleiding"
]

NUMERIC_COLS = [
    "hoeveelste_jaar",
    "Aantal eerstejaarsinstelling",
    "Aantal eerstejaars croho",
    "Aantal Hoofdinschrijvingen",
    "Aantal neveninschrijvingen",
    "andere_opleiding"
]

WEEK_COL = ["Weeknummer"]

ID_COL = ["ID"]

TARGET_COL = ['next_year_registered']  

RENAME_MAP = {
    "Examentype code": "Examentype",
    "Groepeernaam Croho": "Croho groepeernaam",
    "Naam faculteit Nederlands": "Faculteit",
    "EER-NL-nietEER": "Herkomst"
}

# --- Main higher-years class ---

class HigherYearsPredictor:
    """
    Predicts student retention (registration in the next academic year)
    based on historical data using an XGBoost model.

    The prediction is performed separately for each combination of
    'Groepeernaam Croho' (program group) and 'EER-NL-nietEER' (origin).
    """

    def __init__(self, data_latest: pd.DataFrame, configuration):
        self.data_latest = data_latest.copy()
        self.configuration = configuration

        # Keep backups
        self.data_latest_copy = data_latest.copy()

        # Store processing variables
        self.preprocessed = False
    
    # --------------------------------------------------
    # -- First check if predictions for that year were already done --
    # --------------------------------------------------
    def check_predictions_requirement(self, predict_year):
        """
        Checks if predictions for the given year have already been made.
        """
        # --- Check if predictions for the given year have already been made ---
        if "Prediction_higheryears" not in self.data_latest.columns:
            return False

        # Filter for the given year
        mask_year = self.data_latest["Collegejaar"] == predict_year

        # Check if any non-NaN predictions exist
        return self.data_latest.loc[mask_year, "Prediction_higheryears"].notna().any()

        
    # --------------------------------------------------
    # -- Preprocessing --
    # --------------------------------------------------
    def preprocess(self) -> pd.DataFrame:
        """
        Performs initial preprocessing steps common to all predictions.
        """
        # --- Create a copy of the october data ---
        df = self.data_october.copy()

        # --- Rename columns ---
        df = df.rename(columns=RENAME_MAP)

        # --- Sort values by ID, Collegejaar, and Croho groepeernaam ---
        df = df.sort_values(by=["ID", "Collegejaar", "Croho groepeernaam"]).copy()

        # --- Create a column to check if the student is also registered in the next year ---
        df['next_year'] = df.groupby(['ID', 'Croho groepeernaam'])['Collegejaar'].shift(-1)
        df['next_Aantal_Hoofdinschrijvingen'] = df.groupby(['ID', 'Croho groepeernaam'])['Aantal Hoofdinschrijvingen'].shift(-1)

        # Only 1 if next year exists AND next Aantal Hoofdinschrijvingen is 1
        df['next_year_registered'] = ((df['next_year'] == df['Collegejaar'] + 1) &
                                    (df['next_Aantal_Hoofdinschrijvingen'] == 1)).astype(int)

        # Optional: drop helper columns
        df.drop(columns=['next_year', 'next_Aantal_Hoofdinschrijvingen'], inplace=True)
        df["next_year_registered"] = (
            df["next_year_registered"].fillna(False).astype(int)
        )  # Handle NaNs and convert

        # --- Calculate year in program and if student has other registrations ---
        df["hoeveelste_jaar"] = df.groupby(["ID", "Croho groepeernaam"]).cumcount() + 1
        df["andere_opleiding"] = df.duplicated(subset=["ID", "Collegejaar"], keep=False).astype(
            int
        )

        # --- Filter based on general criteria ---
        valid_exam_types = ["Bachelor eerstejaars", "Bachelor hogerejaars", "Master"]
        df = df[df["Examentype"].isin(valid_exam_types)]
        df = df[
            df["Aantal Hoofdinschrijvingen"] == 1
        ]  

        # --- Rename 'bachelor hogerejaars' to bachelor
        df.loc[df['Examentype'] == 'Bachelor hogerejaars', 'Examentype'] = 'Bachelor'
        df.loc[df['Examentype'] == 'Bachelor eerstejaars', 'Examentype'] = 'Bachelor'

        # --- Select relevant columns (features + ID + target) ---
        columns_to_keep = ID_COL + GROUP_COLS + CATEGORICAL_COLS + NUMERIC_COLS + TARGET_COL

        # --- Ensure all needed columns exist before selecting ---
        missing_cols = [col for col in columns_to_keep if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in input data: {missing_cols}")

        df = df[columns_to_keep]

        # --- Only keep the valid years ---
        df = df[df["Collegejaar"] >= self.configuration["start_year"]]

        # --- Store the df ---
        self.data_october = df

        # --- Set preprocessed to True ---
        self.preprocessed = True
        
        return df

    # --------------------------------------------------
    # -- Prediction of higher years --
    # --------------------------------------------------

    ### --- Helpers --- ###
    def _split_and_encode(
        self, group_data: pd.DataFrame, predict_year: int):
        """
        Splits data into train/test sets based on year and performs one-hot encoding.
        """
        # --- Split train/test based on Collegejaar ---
        train = group_data[group_data["Collegejaar"] <= predict_year - 2].copy()
        test = group_data[group_data["Collegejaar"] == predict_year - 1].copy()

        # --- Select features and target ---
        X_train = train[GROUP_COLS + CATEGORICAL_COLS + NUMERIC_COLS]
        y_train = train[TARGET_COL]
        X_test = test[GROUP_COLS + CATEGORICAL_COLS + NUMERIC_COLS]
        y_test = test[TARGET_COL]

        return X_train, y_train, X_test, y_test

    def _build_model(self, X_train, y_train):
        """ Build the xgboost model """
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", NUMERIC_COLS),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS + GROUP_COLS),
            ]
        )

        # Define pipeline with preprocessing + model
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", XGBClassifier(learning_rate= 0.01, objective= "binary:logistic", eval_metric= "logloss", n_estimators=200)),
            ]
        )

        # Fit and store the model
        model = model.fit(X_train, y_train)

        return model

    def _add_predictions_during_year(
        self,
        opleiding: str,
        examentype: str,
        herkomst: str,
        predict_year: int
    ):
        """
        Adds extra students that are not in the october data but likely still enroll on different moments
        """

        # Adjust for examentype bachelor

        differences = []
        
        for year in range(predict_year - 3, predict_year):
            # expected students (fall back to 0 if no matches)
            expected_students = self.data_october[
                (self.data_october['Croho groepeernaam'] == opleiding) &
                (self.data_october['Herkomst'] == herkomst) &
                (self.data_october['Collegejaar'] == year - 1) &
                (self.data_october['Examentype'] == examentype)
            ]['next_year_registered'].sum()

            # actual observed students
            students_next_year = self.data_latest[
                (self.data_latest['Croho groepeernaam'] == opleiding) &
                (self.data_latest['Herkomst'] == herkomst) &
                (self.data_latest['Collegejaar'] == year) &
                (self.data_latest['Examentype'] == examentype)
            ]['Aantal_studenten_higher_years'].mean()


            difference = students_next_year - expected_students
            differences.append(difference)
        
        return float(np.mean(differences)) if differences else np.nan

    ### --- Main prediction function --- ### 
    def predict_higher_years(
        self,
        programme: str,
        examentype: str,
        herkomst: str,
        predict_year: int,
        verbose: bool
    ) -> float:
        """
        Trains an XGBoost model and predicts higher years for a specific group.
        """
        # Filter out pre-masters first
        if examentype == 'Pre-master':
            return np.nan

        # Filter data for the specific group
        try:
            df = self.data_october[
                (self.data_october["Croho groepeernaam"] == programme)
                & (self.data_october["Examentype"] == examentype)
                & (self.data_october["Herkomst"] == herkomst)
            ]
        except KeyError:
            prediction_sum = 0
            if verbose:
                print(
                    f"Higher-years prediction for {programme}, {herkomst}, {examentype}, year: {predict_year}: {prediction_sum}"
                )
            return prediction_sum

        # Split and encode
        X_train, y_train, X_test, y_test = self._split_and_encode(
            df, predict_year
        )

        try:
            # Train the XGBoost model
            model = self._build_model(X_train, y_train)

            # Make predictions (probabilities)
            y_pred_proba = model.predict_proba(X_test)

            # Return the sum of predicted probabilities for this group
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            prediction_sum = y_pred_proba.sum()

            # Add the applicants that apply during the year
            prediction_sum += self._add_predictions_during_year(
                programme, examentype, herkomst, predict_year
            )

            prediction_sum = round(prediction_sum)
        except ValueError:
            prediction_sum = 0

        if verbose:
            print(
                f"Higher-years prediction for {programme}, {herkomst}, {examentype}, year: {predict_year}: {prediction_sum}"
            )

        return prediction_sum


    # --------------------------------------------------
    # -- Full prediction loop --
    # --------------------------------------------------

    def run_full_prediction_loop(self, predict_year: int, write_file: bool, print_output: bool, args, refit: bool = True):

        """
        Run the full prediction loop for all years and weeks.
        """
        # --- First check if predictions not already exist ---
        if self.check_predictions_requirement(predict_year) and not refit:
            logger.info(f"Higher-years predictions for year {predict_year} already exist.")
            return 

        logger.info('Running higher-years prediction loop')

        self.data_october = load_oktober_file()

        # --- Preprocess data (if not done yet) ---
        if not self.preprocessed:
            self.preprocess()

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
        # --- Apply mask ---
        prediction_df = self.data_latest.loc[mask, GROUP_COLS].copy()

        # --- Keep the unique values for group cols ---
        prediction_df = prediction_df.drop_duplicates(subset=GROUP_COLS)


        # --- Prediction ---
        prediction_df["Prediction_higheryears"] = prediction_df.apply(
            lambda row: self.predict_higher_years(
                programme=row["Croho groepeernaam"],
                herkomst=row["Herkomst"],
                examentype=row["Examentype"],
                predict_year=predict_year,
                verbose=print_output
            ),
            axis=1,
        )

        # --- Map ratio predictions back into latest data ---
        higher_years_map = prediction_df.set_index(GROUP_COLS)["Prediction_higheryears"].to_dict()
        self.data_latest["Prediction_higheryears"] = [
            higher_years_map.get(tuple(row[col] for col in GROUP_COLS), row["Prediction_higheryears"] )
            for _, row in self.data_latest.iterrows()
        ]

        # --- Write the file (if required) ---
        if write_file:
            output_path = self.configuration["paths"]["output"]["path_output"].replace("${time}", time.strftime("%Y%m%d_%H%M%S"))
            self.data_latest.to_excel(output_path, index=False, engine="xlsxwriter")


        logger.info('Higher years prediction done')


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
    higheryears_model = HigherYearsPredictor(latest_data, configuration)

    # --- Main prediction loop ---
    for year in args.years:
        higheryears_model.run_full_prediction_loop(
            predict_year=year,
            write_file=args.write_file,
            print_output=args.print,
            args = args
        )
    
if __name__ == "__main__":
    main()