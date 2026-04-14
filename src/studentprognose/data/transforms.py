from studentprognose.utils.weeks import *
from studentprognose.models.xgboost_classifier import DEFAULT_STATUS_MAP

import pandas as pd
import numpy as np
from tqdm import tqdm
import collections

tqdm.pandas()

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

pd.options.mode.chained_assignment = None  # default='warn'


def transform(
    data: pd.DataFrame, target_year: int, last_week: int, old_method=True
) -> pd.DataFrame:
    """
    Transforms the dataframe into a workable dataframe suitable for prediction. It groups the data and creates the
    cumulative sum of the pre-applications.
    """

    data = data[data.Collegejaar <= target_year]

    group_cols = [
        "Collegejaar",
        "Faculteit",
        "Herkomst",
        "Examentype",
        "Croho groepeernaam",
    ]

    all_weeks = []
    all_weeks = all_weeks + [str(i) for i in range(39, 53)]
    all_weeks = all_weeks + [str(i) for i in range(1, 39)]

    target_year_weeknummers = []
    if int(last_week) > 38:
        target_year_weeknummers = target_year_weeknummers + [
            str(i) for i in range(39, int(last_week) + 1)
        ]
    elif int(last_week) < 39:
        target_year_weeknummers = target_year_weeknummers + [str(i) for i in range(39, 53)]
        target_year_weeknummers = target_year_weeknummers + [
            str(i) for i in range(1, int(last_week) + 1)
        ]

    if old_method:
        data = data[group_cols + ["Inschrijvingen_predictie", "Inschrijfstatus", "Weeknummer"]]
    else:
        data = data[group_cols + ["Weeknummer"]]

    data["Weeknummer"] = data["Weeknummer"].astype(str)

    if old_method:
        data["Inschrijfstatus"] = data["Inschrijfstatus"].map(DEFAULT_STATUS_MAP)

    data = data.groupby(group_cols + ["Weeknummer"]).sum(numeric_only=False).reset_index()

    def _transform_data_inner(input_data, target_col, weeknummers):
        data2 = input_data.reset_index().drop(["index", target_col], axis=1)
        input_data = input_data.pivot(
            index=group_cols, columns="Weeknummer", values=target_col
        ).reset_index()

        input_data.columns = map(str, input_data.columns)
        colnames = group_cols + weeknummers
        input_data = input_data.reindex(columns=colnames)

        input_data = input_data.fillna(0)
        input_data = input_data.melt(
            ignore_index=False, id_vars=group_cols, value_vars=weeknummers
        )

        input_data = input_data.rename(columns={"variable": "Weeknummer", "value": target_col})

        input_data = input_data.merge(data2, on=group_cols + ["Weeknummer"], how="left")

        input_data = input_data.fillna(0)

        input_data["Cumulative_sum_within_year"] = input_data.groupby(group_cols)[
            target_col
        ].transform(pd.Series.cumsum)

        return input_data

    data_real = data[data.Collegejaar != target_year]
    data_real = _transform_data_inner(data_real, "Inschrijfstatus", all_weeks)

    data_predict = data[data.Collegejaar == target_year]
    data_predict = _transform_data_inner(
        data_predict, "Inschrijvingen_predictie", target_year_weeknummers
    )

    data = pd.concat([data_real, data_predict])

    return data


def transform_data(data_input: pd.DataFrame, targ_col: str) -> pd.DataFrame:
    """Makes a certain pivot_wider where it transforms the data from long to wide."""
    group_cols = [
        "Collegejaar",
        "Faculteit",
        "Herkomst",
        "Examentype",
        "Croho groepeernaam",
    ]

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


def create_total_file(
    data: pd.DataFrame,
    vooraanmeldingen: pd.DataFrame,
    data_student_numbers: pd.DataFrame,
) -> pd.DataFrame:
    studentenaantallen = data_student_numbers

    data = data[
        [
            "Croho groepeernaam",
            "Collegejaar",
            "Herkomst",
            "Weeknummer",
            "SARIMA_cumulative",
            "SARIMA_individual",
            "Voorspelde vooraanmelders",
        ]
    ]

    total = data.merge(
        studentenaantallen,
        on=["Croho groepeernaam", "Collegejaar", "Herkomst"],
        how="left",
    )

    total = total.merge(
        vooraanmeldingen,
        on=["Croho groepeernaam", "Collegejaar", "Herkomst", "Weeknummer"],
        how="left",
    )

    return total


def calculate_volume_predicted_data(
    data_first_years, data_second_years, predict_year, predict_week
):
    data = data_first_years.merge(
        data_second_years,
        on=[
            "Collegejaar",
            "Faculteit",
            "Examentype",
            "Herkomst",
            "Croho groepeernaam",
            "Weeknummer",
        ],
        how="left",
    )

    data["SARIMA_cumulative"] = np.nan
    data["SARIMA_individual"] = np.nan
    data["Voorspelde vooraanmelders"] = np.nan

    predict_mask = (data["Collegejaar"] == predict_year) & (data["Weeknummer"] == predict_week)
    data.loc[predict_mask, "SARIMA_cumulative"] = (
        data.loc[predict_mask, ["SARIMA_cumulative_x", "SARIMA_cumulative_y"]]
        .applymap(convert_nan_to_zero)
        .sum(axis=1)
    )
    data.loc[predict_mask, "SARIMA_individual"] = (
        data.loc[predict_mask, ["SARIMA_individual_x", "SARIMA_individual_y"]]
        .applymap(convert_nan_to_zero)
        .sum(axis=1)
    )

    data.loc[~predict_mask, "Voorspelde vooraanmelders"] = (
        data.loc[
            ~predict_mask,
            ["Voorspelde vooraanmelders_x", "Voorspelde vooraanmelders_y"],
        ]
        .applymap(convert_nan_to_zero)
        .sum(axis=1)
    )

    return data[
        [
            "Collegejaar",
            "Faculteit",
            "Examentype",
            "Herkomst",
            "Croho groepeernaam",
            "Weeknummer",
            "SARIMA_cumulative",
            "SARIMA_individual",
            "Voorspelde vooraanmelders",
        ]
    ]


def sum_volume_data_cumulative(data_first_years, data_second_years):
    data = data_first_years.merge(
        data_second_years,
        on=[
            "Weeknummer",
            "Collegejaar",
            "Faculteit",
            "Examentype",
            "Herkomst",
            "Croho groepeernaam",
        ],
        how="left",
    )

    data["Gewogen vooraanmelders"] = (
        data[["Gewogen vooraanmelders_x", "Gewogen vooraanmelders_y"]]
        .applymap(convert_nan_to_zero)
        .sum(axis=1)
    )
    data["Ongewogen vooraanmelders"] = (
        data[["Ongewogen vooraanmelders_x", "Ongewogen vooraanmelders_y"]]
        .applymap(convert_nan_to_zero)
        .sum(axis=1)
    )
    data["Aantal aanmelders met 1 aanmelding"] = (
        data[
            [
                "Aantal aanmelders met 1 aanmelding_x",
                "Aantal aanmelders met 1 aanmelding_y",
            ]
        ]
        .applymap(convert_nan_to_zero)
        .sum(axis=1)
    )
    data["Inschrijvingen"] = (
        data[["Inschrijvingen_x", "Inschrijvingen_y"]].applymap(convert_nan_to_zero).sum(axis=1)
    )

    return data[
        [
            "Weeknummer",
            "Collegejaar",
            "Faculteit",
            "Examentype",
            "Herkomst",
            "Croho groepeernaam",
            "Gewogen vooraanmelders",
            "Ongewogen vooraanmelders",
            "Aantal aanmelders met 1 aanmelding",
            "Inschrijvingen",
        ]
    ]
