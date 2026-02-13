# baseline.py

# --- Standard library ---
import os
import sys
import logging
import time
import warnings
from pathlib import Path

# --- Third-party libraries ---
import numpy as np
import pandas as pd
import yaml
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from dotenv import load_dotenv

# --- Project modules ---
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from scripts.utils.load_data import (
    load_cumulative,
    load_student_numbers_first_years,
    load_latest,
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

# --- Main ratio class ---

class Baseline():
    def __init__(self, data_cumulative, data_studentcount, data_latest, configuration):
        self.data_cumulative = data_cumulative
        self.data_studentcount = data_studentcount
        self.data_latest = data_latest
        self.configuration = configuration
        self.skip_years = 0
        self.pred_len = None

        # Backup data
        self.data_cumulative_backup = self.data_cumulative.copy()

        # Store processing variables
        self.preprocessed = False

    # --------------------------------------------------
    # -- Preprocessing --
    # --------------------------------------------------
    def preprocess(self) -> pd.DataFrame:
        """
        Cleans, filters, aggregates, and merges cumulative pre-application data.
        This is the same function as in cumulative.py
        """
        
        # --- Data copy ---
        df = self.data_cumulative.copy()

        # 1. Rename columns
        df = df.rename(columns=RENAME_MAP)

        # 2. Convert numeric columns to float64
        for col in NUMERIC_COLS:
            if pd.api.types.is_string_dtype(df[col]):
                df[col] = pd.to_numeric(
                    df[col], 
                    decimal=',', 
                    thousands='.', 
                    errors='coerce'
                )

        df[NUMERIC_COLS] = df[NUMERIC_COLS].astype('float64')

        # 3. Filter for first-year and pre-master students
        mask = (df["Hogerejaars"] == "Nee") | (df["Examentype"] == "Pre-master")
        df = df[mask]

        # 4. Group and aggregate data
        processed_df = df.groupby(GROUP_COLS + WEEK_COL, as_index=False)[NUMERIC_COLS].sum()

        # 5. Merge with student count data (if it exists)
        if self.data_studentcount is not None:
            processed_df = processed_df.merge(
                self.data_studentcount,
                on=[col for col in GROUP_COLS if col != "Faculteit"],
                how="left",
            )
        
        # 6. Create the 'ts' (time series) target column by adding 'Gewogen vooraanmelders' and 'Inschrijvingen'
        processed_df["ts"] = (
            processed_df["Gewogen vooraanmelders"] + processed_df["Inschrijvingen"]
        )

        # 7. Standardize faculty codes
        faculty_transformation = self.configuration["faculty"]
        processed_df["Faculteit"] = processed_df["Faculteit"].replace(faculty_transformation)

        # 8. Set week 39 and week 40 to 0
        processed_df.loc[processed_df["Weeknummer"] == 39, "ts"] = 0
        processed_df.loc[processed_df["Weeknummer"] == 40, "ts"] = 0
        
        # 9. Final sorting, ordering, and duplicate removal
        
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
    # -- Predicting with the ratio method --
    # --------------------------------------------------

    ### --- Helpers --- ###
    def _filter_data(self, df: pd.DataFrame, predict_year: int, predict_week: int, programme: str, examentype: str, herkomst: str) -> pd.DataFrame:
        df = df[df["Collegejaar"] >= self.configuration["start_year"]]
        
        filtered = df[
            (df["Herkomst"] == herkomst)
            & (df["Collegejaar"] <= predict_year)
            & (df['Weeknummer'] == predict_week)
            & (df["Croho groepeernaam"] == programme)
            & (df["Examentype"] == examentype)
        ]
        return filtered
    
    def _get_ratio(self, df: pd.DataFrame, predict_year: int, training_years: int = 1) -> float:
        for years in range(training_years, 0, -1):  
            subset = df[
                (df["Collegejaar"] >= predict_year - years) & 
                (df["Collegejaar"] < predict_year)
            ].copy()
            ts_sum = subset["ts"].fillna(0).sum()
            student_sum = subset["Aantal_studenten"].fillna(0).sum()
        
        if student_sum > 0:  
            return ts_sum / student_sum 
        
        return 0.5 # some default value
    
    ### --- Main logic --- ###
    def predict_baseline(self,
        programme: str,
        herkomst: str,
        examentype: str,
        predict_year: int,
        predict_week: int,
        print_output: bool = False
    ) -> int:
        """
        Predict the inflow of students based on the ratio (1 year) of pre-applicants.
        """
        
        # --- Data copy ---
        df = self.data_cumulative.copy()

        # --- Preprocess data (if not done yet) ---
        if not self.preprocessed:
            self.preprocess()

        # --- Filter data ---
        df = self._filter_data(df, predict_year, predict_week, programme, examentype, herkomst)

        # --- Get ratio ---
        ratio = self._get_ratio(df, predict_year)

        # --- Prediction logic ---
        prediction = df.loc[df["Collegejaar"] == predict_year, "ts"] / ratio 

        # --- Return prediction ---
        try:
            prediction = round(prediction.squeeze())
        except (ValueError, AttributeError, OverflowError):
            prediction = 0
        
        if isinstance(prediction, pd.Series):
            prediction = prediction.iloc[0] if not prediction.empty else 0
        else:
            prediction = prediction if prediction is not None else 0

        if print_output:
            print(
                f"Baseline prediction for {programme}, {herkomst}, {examentype}, year: {predict_year}, week: {predict_week}: {prediction}"
            )

        return prediction

    # --------------------------------------------------
    # -- Full prediction loop --
    # --------------------------------------------------

    def run_full_prediction_loop(self, predict_year: int, predict_week: int, write_file: bool, print_output: bool = False, args = None):

        """
        Run the full prediction loop for all years and weeks.
        """
        logger.info('Running baseline prediction loop')

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
        mask &= self.data_latest["Weeknummer"] == predict_week

        # --- Apply mask ---
        prediction_df = self.data_latest.loc[mask, GROUP_COLS + WEEK_COL].copy()

        # --- Prediction ---
        prediction_df["Baseline"] = prediction_df.apply(
            lambda row: self.predict_baseline(
                programme=row["Croho groepeernaam"],
                herkomst=row["Herkomst"],
                examentype=row["Examentype"],
                predict_year=predict_year,
                predict_week=predict_week,
                print_output=print_output
            ),
            axis=1,
        )

        # --- Map ratio predictions back into latest data ---
        ratio_map = prediction_df.set_index(GROUP_COLS + WEEK_COL)["Baseline"].to_dict()
        self.data_latest["Baseline"] = [
            ratio_map.get(tuple(row[col] for col in GROUP_COLS + WEEK_COL), row["Baseline"] )
            for _, row in self.data_latest.iterrows()
        ]

        # --- Write the file ---
        if write_file:
            output_path = self.configuration["paths"]["output"]["path_output"].replace("${time}", time.strftime("%Y%m%d_%H%M%S"))
            self.data_latest.to_excel(output_path, index=False, engine="xlsxwriter")


        logger.info('Baseline prediction done')

def main():
    # Parse arguments
    args = parse_args()

    # Load configuration
    CONFIG_FILE = Path("configuration.yaml")
    with open(CONFIG_FILE, "r") as f:
        configuration = yaml.safe_load(f)  

    # Load data
    cumulative_data = load_cumulative()
    student_counts = load_student_numbers_first_years()
    latest_data = load_latest()

    # Initialize model
    baseline_model = Baseline(cumulative_data, student_counts, latest_data, configuration)

    for year in args.years:
        for week in args.weeks:
            baseline_model.run_full_prediction_loop(
                predict_year=year,
                predict_week=week,
                write_file=args.write_file,
                print_output = args.print,
                args=args
            )


if __name__ == "__main__":
    main()
    

    