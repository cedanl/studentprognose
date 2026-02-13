# update_total_file.py

"""
With this script you can update the total file in terms of studentnumbers or prediction years. The script has two different functionalities:
- Adds the newly generated studentnumbers into the total file for last year
- Adds new prediction rows for the following year
"""

# --- imports ---
import sys
from pathlib import Path
from itertools import product
import time
import yaml
import pandas as pd
import logging

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

logger = logging.getLogger(__name__)
from scripts.utils.load_data import load_oktober_file, load_latest

# --- Constant variable names ---
GROUP_COLS = [
    "Collegejaar", "Croho groepeernaam", 
    "Examentype", "Herkomst"
]

RENAME_MAP = {
    "Examentype code": "Examentype",
    "Groepeernaam Croho": "Croho groepeernaam",
    "EER-NL-nietEER": "Herkomst"
}


# --- Adds the newly generated studentnumbers into the total file for last year ---
def calculate_first_year_student_counts(oktober_data, current_year):
    """
    Calculates the first year studentnumbers from the oktobercijfers.
    """
    # --- rename columns ---
    oktober_data = oktober_data.rename(columns=RENAME_MAP)

    # --- clean and filter oktobercijfers data ---
    oktober_data["Examentype"] = oktober_data["Examentype"].replace(
        {"Bachelor eerstejaars": "Bachelor"}
    )

    oktober_data = oktober_data[
        (oktober_data["Aantal Hoofdinschrijvingen"] == 1)
        & (
            (oktober_data["Examentype"] == "Pre-master")
            | (oktober_data["Aantal eerstejaars croho"] == 1)
        )
        & (~oktober_data["Examentype"].isin(["Bachelor hogerejaars", "Master post-initieel"]))
        & (oktober_data["Collegejaar"] <= 2019)
    ]

    # --- group and count ---
    total_first_year_students = (
        oktober_data
        .groupby(GROUP_COLS)
        .size()                # count rows per group
        .reset_index(name="Aantal_studenten")
    )

    return total_first_year_students

def calculate_higher_years_student_counts(oktober_data, current_year):
    """
    Calculates the higher year studentnumbers from the oktobercijfers.
    """
    # --- rename columns ---
    oktober_data = oktober_data.rename(columns=RENAME_MAP)

    # --- clean and filter oktobercijfers data ---
    oktober_data["Examentype"] = oktober_data["Examentype"].replace(
        {"Bachelor hogerejaars": "Bachelor"}
    )

    oktober_data = oktober_data[
        (oktober_data["Aantal Hoofdinschrijvingen"] == 1)
        & ~oktober_data["Examentype"].isin(["Master post-initieel", "Pre-master", "Bachelor eerstejaars"])
        & (
            (oktober_data["Examentype"] == "Bachelor")
            | ((oktober_data["Examentype"] == "Master") & (oktober_data["Aantal eerstejaars croho"] == 0))
        )
        & (oktober_data["Collegejaar"] <= 2019)
    ]

    # --- group and count ---
    total_higher_year_students = (
        oktober_data
        .groupby(GROUP_COLS)
        .size()                # count rows per group
        .reset_index(name="Aantal_studenten_higher_years")
    )

    return total_higher_year_students

def add_student_counts_to_total_file(total_file, oktober_file, current_year):
    """
    Adds the newly generated studentnumbers into the total file for last year.
    """

    # --- calculate student counts ---
    total_first_year_students = calculate_first_year_student_counts(oktober_file, current_year)
    total_higher_year_students = calculate_higher_years_student_counts(oktober_file, current_year)

    # --- update student counts ---
    total_file = total_file.set_index(GROUP_COLS)
    total_higher_year_students = total_higher_year_students.set_index(GROUP_COLS)
    total_first_year_students = total_first_year_students.set_index(GROUP_COLS)

    # update higher-years
    mask = total_higher_year_students.index.intersection(total_file.index)
    total_file.loc[mask, "Aantal_studenten_higher_years"] = total_higher_year_students.loc[mask, "Aantal_studenten_higher_years"]

    # update first-years
    mask = total_first_year_students.index.intersection(total_file.index)
    total_file.loc[mask, "Aantal_studenten"] = total_first_year_students.loc[mask, "Aantal_studenten"]

    # --- reset index ---
    total_file = total_file.reset_index()

    return total_file


# --- Adds new prediction rows for the following year ---
def add_new_year(total_file, current_year, configuration):
    """
    Adds new prediction rows for the following year.
    Safeguard: does nothing if the next year already exists.
    """
    
    df = total_file.copy()
    next_year = current_year + 1

    # --- Safeguard ---
    if next_year in df['Collegejaar'].values:
        print(f"Year {next_year} already exists. No rows added.")
        return df

    new_programmes = list(configuration['new_programmes'].keys())

    # --- Prepare unique values ---
    croho_unique = list(
    df.loc[df["Collegejaar"] == current_year, "Croho groepeernaam"]
        .dropna()
        .unique()
    )
    if current_year == df['Collegejaar'].max():
        croho_unique = list(set(croho_unique + new_programmes))
    
    herkomst_unique = [x for x in df["Herkomst"].dropna().unique() if x != "onbekend"]
    weeknummer_unique = df["Weeknummer"].dropna().unique()

    # Map Croho to Faculteit
    faculteit_map = df.dropna(subset=["Faculteit"]) \
                      .drop_duplicates("Croho groepeernaam") \
                      .set_index("Croho groepeernaam")["Faculteit"].to_dict()

    # --- Generate combinations ---
    all_combinations = []
    for croho in croho_unique:
        if croho.startswith("B"):
            ex_types = ["Bachelor", "Pre-master"]
        elif croho.startswith("M"):
            ex_types = ["Master"]
        else:
            ex_types = df["Examentype"].dropna().unique()
        
        combos = product([croho], ex_types, herkomst_unique, weeknummer_unique)
        all_combinations.extend(combos)

    combinations_df = pd.DataFrame(all_combinations, columns=["Croho groepeernaam", "Examentype", "Herkomst", "Weeknummer"])

    # --- Assign Faculteit ---
    combinations_df["Faculteit"] = combinations_df["Croho groepeernaam"].map(faculteit_map)

    # --- Override Faculteit for new programmes from configuration ---
    for new_prog in new_programmes:
        combinations_df.loc[
            combinations_df["Croho groepeernaam"] == new_prog, "Faculteit"
        ] = configuration['new_programmes'][new_prog]

    # --- Add missing columns ---
    missing_cols = [c for c in df.columns if c not in combinations_df.columns and c != "Collegejaar"]
    for col in missing_cols:
        combinations_df[col] = None

    # --- Set new year ---
    combinations_df['Collegejaar'] = next_year

    # --- Reorder columns and append ---
    combinations_df = combinations_df[df.columns]
    combined_df = pd.concat([df, combinations_df], ignore_index=True)

    return combined_df



# --- main ---
def main():
    """
    Here define the current year and which functions you want to use.
    """
    logger.info("Starting update_total_file script")

    # --- load configuration ---
    CONFIG_FILE = Path("configuration.yaml")
    with open(CONFIG_FILE, "r") as f:
        configuration = yaml.safe_load(f)   

    # --- define current year (not the upcoming prediction year) ---
    current_year = 2025

    # --- load data ---
    data_oktober = load_oktober_file()
    data_total = load_latest()

    # --- add student counts (uncomment if you want to add student counts) ---
    #data_total = add_student_counts_to_total_file(data_total, data_oktober, current_year)

    # --- add new year (uncomment if you want to add new year) ---
    data_total = add_new_year(data_total, current_year, configuration)

    # --- save total file (uncomment if you want to save total file) ---
    output_path = configuration["paths"]["output"]["path_output"].replace("${time}", time.strftime("%Y%m%d_%H%M%S"))
    data_total.to_excel(output_path, index=False, engine="xlsxwriter")

    logger.info(f"Total file saved to: {output_path}")


if __name__ == "__main__":
    main()
