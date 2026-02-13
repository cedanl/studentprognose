# helper.py

import pandas as pd
import collections
import datetime

# --- Helper Functions in use ---

def get_all_weeks_ordered():
    """Return a list of all valid week numbers in order"""
    return [
        "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "1", "2", "3", "4", "5", "6",
        "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26",
        "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38"
    ]

def get_all_weeks_valid(columns):
    """Return a sorted list of valid week numbers from DataFrame columns"""
    return list((collections.Counter(get_all_weeks_ordered()) & collections.Counter(columns)).elements())

def get_weeks_list(weeknummer: int) -> list:
    """Return a list of weeks up to max_week (taking into account that weeks start at 39 instead of 1)"""
    weeknummer = int(weeknummer)
    if weeknummer >= 39:
        return list(range(39, weeknummer + 1))
    else:
        return list(range(39, 53)) + list(range(1, weeknummer + 1))

def get_pred_len(weeknummer: int) -> int:
    """Return the prediction length based on the given week number."""
    return (38 + 52 - weeknummer) if weeknummer > 38 else (38 - weeknummer)

def get_current_len(weeknummer: int) -> int:
    if weeknummer <= 38:
        return weeknummer + 14
    else:
        return weeknummer - 38

def is_current_week(predict_year: int, predict_week: int) -> bool:
    """Check if the given year and week correspond to the current calendar year and week."""
    current_year, current_week, _ = datetime.date.today().isocalendar()
    if current_week > 39:
        current_year += 1
    return predict_year == current_year and predict_week == current_week


def get_prediction_weeks_list(weeknummer: int) -> list[int]:
    """Return a list of prediction weeks (taking into account that weeks start at 39 instead of 1)"""
    weeknummer = int(weeknummer) + 1
    if weeknummer >= 39:
        return list(range(weeknummer, 53)) + list(range(1, 39))
    return list(range(weeknummer, 39))



# --- Currently Unused Helper Functions ---

def get_max_week_from_weeks(weeks: pd.Series) -> int:
    """
    Determine the maximum week number in a dataset for a given year (usually 38 for a finished year)

    Args:
        weeks (pd.Series): A series containing unique week numbers.

    Returns:
        int: The maximum week number calculated from the input series.
    """
    if len(weeks.unique()) == 52:
        max_week = 38
    else:
        max_week = 38 + len(weeks.unique())
        if max_week > 52:
            max_week = max_week - 52

    return max_week


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


def academic_week_number(year, week):
    """
    Convert (year, week) to a linear number where week 39 is treated as week 0
    and the year runs from week 39..52 + 1..38.
    """
    # Shift week numbers: 39→0, 40→1, …, 52→13, 1→14, …, 38→51
    if week >= 39:
        return (year, week - 39)
    else:
        return (year + 1, week + 13)  # push weeks 1–38 into the next academic year


