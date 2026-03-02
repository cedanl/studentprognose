"""
Generate a chronological subset of DEMO data files for faster testing.

Keeps only the last N years of data (default: 2022-2024) to reduce SARIMA/XGBoost
training time while preserving enough seasonal history. Data order is preserved
(no shuffling) since SARIMA requires chronological input.

Usage:
    uv run scripts/create_test_subset.py
"""

import csv
import os

import pandas as pd

CUTOFF_YEAR = 2022

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "data", "input")

# Source files
SRC_CUMULATIVE = os.path.join(INPUT_DIR, "vooraanmeldingen_cumulatief_DEMO.csv")
SRC_INDIVIDUAL = os.path.join(INPUT_DIR, "vooraanmeldingen_individueel_DEMO.csv")
SRC_STUDENT_COUNT = os.path.join(
    INPUT_DIR, "student_count_first-years_inclusief_pre-master_DEMO.xlsx"
)

# Output files (overwrites DEMO files)
OUT_CUMULATIVE = os.path.join(INPUT_DIR, "vooraanmeldingen_cumulatief_DEMO.csv")
OUT_INDIVIDUAL = os.path.join(INPUT_DIR, "vooraanmeldingen_individueel_DEMO.csv")
OUT_STUDENT_COUNT = os.path.join(
    INPUT_DIR, "student_count_first-years_inclusief_pre-master_DEMO.xlsx"
)


def subset_cumulative() -> tuple[int, int]:
    """Subset cumulative CSV by keeping rows with Collegejaar >= CUTOFF_YEAR.

    Returns (total_rows, rows_written).
    """
    year_col_index = 1  # "Collegejaar" is column index 1

    total = 0
    written = 0
    with (
        open(SRC_CUMULATIVE, "r", newline="", encoding="utf-8-sig") as infile,
        open(OUT_CUMULATIVE, "w", newline="", encoding="utf-8-sig") as outfile,
    ):
        reader = csv.reader(infile, delimiter=";")
        writer = csv.writer(outfile, delimiter=";")

        header = next(reader)
        writer.writerow(header)

        for row in reader:
            total += 1
            if int(row[year_col_index]) >= CUTOFF_YEAR:
                writer.writerow(row)
                written += 1

    return total, written


def subset_individual() -> tuple[int, int]:
    """Subset individual CSV by keeping rows with Collegejaar >= CUTOFF_YEAR.

    Preserves the description row (row 2) that load_data.py skips via skiprows=[1].
    Returns (total_rows, rows_written).
    """
    year_col_index = 3  # "Collegejaar" is column index 3

    total = 0
    written = 0
    with (
        open(SRC_INDIVIDUAL, "r", newline="", encoding="utf-8-sig") as infile,
        open(OUT_INDIVIDUAL, "w", newline="", encoding="utf-8-sig") as outfile,
    ):
        reader = csv.reader(infile, delimiter=";")
        writer = csv.writer(outfile, delimiter=";")

        # Row 0: header
        header = next(reader)
        writer.writerow(header)

        # Row 1: description row (skipped by load_data.py via skiprows=[1])
        description = next(reader)
        writer.writerow(description)

        # Data rows
        for row in reader:
            total += 1
            if int(row[year_col_index]) >= CUTOFF_YEAR:
                writer.writerow(row)
                written += 1

    return total, written


def subset_student_count() -> tuple[int, int]:
    """Subset student count Excel by keeping rows with Collegejaar >= CUTOFF_YEAR.

    Returns (total_rows, rows_written).
    """
    df = pd.read_excel(SRC_STUDENT_COUNT)
    total = len(df)

    df_subset = df[df["Collegejaar"] >= CUTOFF_YEAR]
    df_subset.to_excel(OUT_STUDENT_COUNT, index=False)

    return total, len(df_subset)


def main():
    for path in [SRC_CUMULATIVE, SRC_INDIVIDUAL, SRC_STUDENT_COUNT]:
        if not os.path.exists(path):
            print(f"ERROR: Source file not found: {path}")
            return

    print(f"Creating subset with Collegejaar >= {CUTOFF_YEAR}\n")

    cum_total, cum_written = subset_cumulative()
    print(f"Cumulative: {cum_total:,} -> {cum_written:,} rows ({cum_written/cum_total:.1%})")

    ind_total, ind_written = subset_individual()
    print(f"Individual: {ind_total:,} -> {ind_written:,} rows ({ind_written/ind_total:.1%})")

    sc_total, sc_written = subset_student_count()
    print(f"Student count: {sc_total:,} -> {sc_written:,} rows ({sc_written/sc_total:.1%})")

    print(f"\nSubset files written to {INPUT_DIR}/")
    print(f"  - {os.path.basename(OUT_CUMULATIVE)}")
    print(f"  - {os.path.basename(OUT_INDIVIDUAL)}")
    print(f"  - {os.path.basename(OUT_STUDENT_COUNT)}")


if __name__ == "__main__":
    main()
