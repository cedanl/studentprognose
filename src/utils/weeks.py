import pandas as pd
import collections
from enum import Enum


class DataOption(Enum):
    INDIVIDUAL = 1
    CUMULATIVE = 2
    BOTH_DATASETS = 3

    @property
    def filename_suffix(self):
        return {
            DataOption.INDIVIDUAL: "individueel",
            DataOption.CUMULATIVE: "cumulatief",
            DataOption.BOTH_DATASETS: "beide",
        }[self]


class StudentYearPrediction(Enum):
    FIRST_YEARS = 1
    HIGHER_YEARS = 2
    VOLUME = 3


HIGHER_YEARS_COLUMNS = [
    "Croho groepeernaam",
    "Collegejaar",
    "Herkomst",
    "Weeknummer",
    "SARIMA_cumulative",
    "SARIMA_individual",
    "Voorspelde vooraanmelders",
    "Aantal_studenten",
    "Faculteit",
    "Examentype",
    "Gewogen vooraanmelders",
    "Ongewogen vooraanmelders",
    "Aantal aanmelders met 1 aanmelding",
    "Inschrijvingen",
    "Weighted_ensemble_prediction",
]


def get_max_week_from_weeks(weeks: pd.Series) -> int:
    if len(weeks.unique()) == 52:
        max_week = 38
    else:
        max_week = 38 + len(weeks.unique())
        if max_week > 52:
            max_week = max_week - 52

    return max_week


def get_weeks_list(weeknummer: int) -> list:
    weeks = []
    if int(weeknummer) > 38:
        weeks = weeks + [i for i in range(39, int(weeknummer) + 1)]
    elif int(weeknummer) < 39:
        weeks = weeks + [i for i in range(39, 53)]
        weeks = weeks + [i for i in range(1, int(weeknummer) + 1)]

    return weeks


def get_max_week(predict_year, max_year, data, key):
    if predict_year == max_year:
        max_week = get_max_week_from_weeks(data[data[key] == predict_year]["Weeknummer"])
    else:
        if predict_year == 2021:
            max_week = 38
        else:
            max_week = get_max_week_from_weeks(data[data[key] == predict_year]["Weeknummer"])

    return max_week


def increment_week(week):
    if week == 52:
        return 1
    else:
        return week + 1


def decrement_week(week):
    if week == 1:
        return 52
    else:
        return week - 1


def convert_nan_to_zero(number):
    if pd.isnull(number):
        return 0
    else:
        return number


def get_all_weeks_valid(columns):
    return list(
        (collections.Counter(get_all_weeks_ordered()) & collections.Counter(columns)).elements()
    )


def get_all_weeks_ordered():
    return [
        "39",
        "40",
        "41",
        "42",
        "43",
        "44",
        "45",
        "46",
        "47",
        "48",
        "49",
        "50",
        "51",
        "52",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
    ]
