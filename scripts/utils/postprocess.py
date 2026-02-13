# postprocess.py

# --- Standard library ---
import os
import sys
import time
import logging
from pathlib import Path

# --- Third-party libraries ---
import yaml
from dotenv import load_dotenv

# --- Project modules ---
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from scripts.utils.load_data import (
    load_cumulative,
    load_latest,
    load_individual
)
from scripts.models.individual import Individual
from cli import parse_args

# --- Warnings and logging setup ---
logger = logging.getLogger(__name__)

# --- Environment setup ---
load_dotenv()
ROOT_PATH = os.getenv("ROOT_PATH")


def add_applicant_data(data_cumulative, data_latest, predict_year, predict_week):
    """
    Add applicant data to the latest data.
    """
    # Define matching key columns
    key_cols = ["Weeknummer", "Examentype", "Croho groepeernaam", "Herkomst", "Collegejaar"]

    # Columns to update
    update_cols = [
        "Gewogen vooraanmelders",
        "Ongewogen vooraanmelders",
        "Aantal aanmelders met 1 aanmelding",
        "Inschrijvingen"
    ]

    # Rename columns in cumulative_data to match latest_data
    RENAME_MAP = {
        "Type hoger onderwijs": "Examentype",
        "Groepeernaam Croho": "Croho groepeernaam",
    }
    data_cumulative = data_cumulative.rename(columns=RENAME_MAP)

    # Filter cumulative_data to the target week/year
    cumulative_filtered = data_cumulative[
        (data_cumulative["Collegejaar"] == predict_year)
        & (data_cumulative["Weeknummer"] == predict_week)
    ]

    # Drop duplicates for unique mapping
    cumulative_unique = (
        cumulative_filtered
        .sort_values(key_cols)
        .drop_duplicates(subset=key_cols, keep="last")
    )

    # Build dictionary mapping keys -> cumulative values
    cumulative_map = cumulative_unique.set_index(key_cols)[update_cols].to_dict(orient="index")

    # Filter latest_data to only rows that match the target week/year
    mask = (data_latest["Collegejaar"] == predict_year) & (data_latest["Weeknummer"] == predict_week)
    subset = data_latest.loc[mask].copy()

    # Overwrite values using the mapping
    for col in update_cols:
        subset[col] = [
            cumulative_map.get(tuple(row[k] for k in key_cols), {}).get(col, row[col])
            for _, row in subset.iterrows()
        ]

    # Write back updated rows
    data_latest.loc[mask, update_cols] = subset[update_cols].values

    return data_latest


def clean_individual_data(data_individual, latest_data, configuration):
    individual_model = Individual(data_individual, None, latest_data, configuration)

    df = individual_model.preprocess()

    df = df[['Collegejaar', 'Croho groepeernaam', 'Faculteit', 'Examentype', 'Herkomst', 'Type vooropleiding', 'Weeknummer', 'Inschrijfstatus', 'Datum intrekking vooraanmelding']]

    return df

    


def postprocess(data_cumulative, data_latest, predict_year, predict_week):

    logger.info(f"Postprocessing...")

    # --- Add applicant data ---
    data_latest = add_applicant_data(data_cumulative, data_latest, predict_year, predict_week)

    logger.info(f"Postprocessing done")

    return data_latest




def main():
    # --- Parse arguments ---
    args = parse_args()

    # --- Load configuration ---
    CONFIG_FILE = Path("configuration.yaml")
    with open(CONFIG_FILE, "r") as f:
        configuration = yaml.safe_load(f)  

    # --- Load data ---
    cumulative_data = load_cumulative()
    individual_data = load_individual()
    latest_data = load_latest()

    # --- Run prediction loop ---
    for year in args.years:
        for week in args.weeks:
            latest_data = postprocess(cumulative_data, latest_data, year, week)

    #individual_cleaned = clean_individual_data(individual_data, latest_data, configuration)

    # --- Write the files ---
    output_path = configuration["paths"]["output"]["path_output"].replace("${time}", time.strftime("%Y%m%d_%H%M%S"))
    latest_data.to_excel(output_path, index=False, engine="xlsxwriter")

    #individual_cleaned_path = configuration["paths"]["output"]["path_individual_cleaned"].replace("${root_path}", ROOT_PATH)

    #individual_cleaned.to_csv(individual_cleaned_path, index=False, sep=';')

    logger.info(f"Output written to: {output_path}")