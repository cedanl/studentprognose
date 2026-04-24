from studentprognose.utils.weeks import convert_nan_to_zero, get_all_weeks_valid

import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()


TRANSFORM_GROUP_COLS = [
    "Collegejaar",
    "Faculteit",
    "Herkomst",
    "Examentype",
    "Croho groepeernaam",
]


def transform_data(data_input: pd.DataFrame, targ_col: str) -> pd.DataFrame:
    """Makes a certain pivot_wider where it transforms the data from long to wide."""
    group_cols = TRANSFORM_GROUP_COLS

    data = data_input[group_cols + [targ_col, "Weeknummer"]]

    data = data.drop_duplicates()

    data = data.pivot(index=group_cols, columns="Weeknummer", values=targ_col).reset_index()

    data.columns = map(str, data.columns)

    colnames = group_cols + get_all_weeks_valid(data.columns)

    data = data[colnames]

    data = data.fillna(0)

    return data


def replace_latest_data(data_latest: pd.DataFrame, data: pd.DataFrame, predict_year, predict_week):
    merge_cols = ["Examentype", "Croho groepeernaam", "Herkomst", "Collegejaar", "Weeknummer"]

    merged = data_latest.merge(data, on=merge_cols, how="outer", suffixes=("", "_B"))

    remaining_cols = list(set(data.columns) - set(merge_cols))

    for col in remaining_cols:
        merged[col] = merged[f"{col}_B"].where(merged[f"{col}_B"].notna(), merged[col])

    data_latest = merged.drop(columns=[f"{col}_B" for col in remaining_cols])

    data_latest = data_latest.reset_index(drop=True)

    return data_latest


