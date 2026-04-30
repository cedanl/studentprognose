import pandas as pd
from enum import Enum
from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK, WEEKS_PER_YEAR


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


def detect_last_available(data: pd.DataFrame) -> tuple[int, int | None]:
    """Retourneer het hoogste Collegejaar en het bijbehorende hoogste Weeknummer.

    Args:
        data: DataFrame met minimaal een ``Collegejaar``-kolom en optioneel een
            ``Weeknummer``-kolom.

    Returns:
        Een tuple ``(jaar, week)`` waarbij ``week`` ``None`` is als de
        ``Weeknummer``-kolom niet aanwezig is in ``data``.
    """
    max_year = int(data["Collegejaar"].dropna().astype(int).max())

    if "Weeknummer" not in data.columns:
        return max_year, None

    year_data = data[data["Collegejaar"].astype(int) == max_year]
    max_week = int(year_data["Weeknummer"].dropna().astype(int).max())
    return max_year, max_week


def get_max_week_from_weeks(weeks: pd.Series) -> int:
    if len(weeks.unique()) == WEEKS_PER_YEAR:
        max_week = FINAL_ACADEMIC_WEEK
    else:
        max_week = FINAL_ACADEMIC_WEEK + len(weeks.unique())
        if max_week > WEEKS_PER_YEAR:
            max_week = max_week - WEEKS_PER_YEAR

    return max_week


def get_weeks_list(weeknummer: int) -> list:
    weeks = []
    if int(weeknummer) > FINAL_ACADEMIC_WEEK:
        weeks = weeks + [i for i in range(FINAL_ACADEMIC_WEEK + 1, int(weeknummer) + 1)]
    elif int(weeknummer) < FINAL_ACADEMIC_WEEK + 1:
        weeks = weeks + [i for i in range(FINAL_ACADEMIC_WEEK + 1, WEEKS_PER_YEAR + 1)]
        weeks = weeks + [i for i in range(1, int(weeknummer) + 1)]

    return weeks


def get_max_week(predict_year, max_year, data, key):
    if predict_year == max_year:
        max_week = get_max_week_from_weeks(
            data[data[key] == predict_year]["Weeknummer"]
        )
    else:
        if predict_year == 2021:
            # COVID-jaar: het aanmeldpatroon week 38 week was afwijkend waardoor
            # get_max_week_from_weeks() een onjuiste waarde teruggeeft. Zie #84.
            max_week = FINAL_ACADEMIC_WEEK
        else:
            max_week = get_max_week_from_weeks(
                data[data[key] == predict_year]["Weeknummer"]
            )

    return max_week


def increment_week(week):
    if week == WEEKS_PER_YEAR:
        return 1
    else:
        return week + 1


def decrement_week(week):
    if week == 1:
        return WEEKS_PER_YEAR
    else:
        return week - 1


def convert_nan_to_zero(number):
    if pd.isnull(number):
        return 0
    else:
        return number


def week_sort_key(w: int) -> int:
    """Sort key so the academic year runs 39 → 52, 1 → 38."""
    return w - 39 if w >= 39 else w + 13


def compute_pred_len(predict_week: int) -> int:
    if predict_week > FINAL_ACADEMIC_WEEK:
        return FINAL_ACADEMIC_WEEK + WEEKS_PER_YEAR - predict_week
    return FINAL_ACADEMIC_WEEK - predict_week


def get_all_weeks_valid(columns):
    col_set = set(columns)
    return [w for w in get_all_weeks_ordered() if w in col_set]


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
