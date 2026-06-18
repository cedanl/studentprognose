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


def detect_last_available(
    data: pd.DataFrame, final_week: int = FINAL_ACADEMIC_WEEK
) -> tuple[int, int | None]:
    """Retourneer het hoogste Collegejaar en het bijbehorende hoogste Weeknummer.

    Args:
        data: DataFrame met minimaal een ``Collegejaar``-kolom en optioneel een
            ``Weeknummer``-kolom.
        final_week: Laatste week van het academisch jaar (default 38; UvA 36).
            Bepaalt de seizoensvolgorde waarmee de "hoogste" week wordt gekozen.

    Returns:
        Een tuple ``(jaar, week)`` waarbij ``week`` ``None`` is als de
        ``Weeknummer``-kolom niet aanwezig is in ``data``.
    """
    max_year = int(data["Collegejaar"].dropna().astype(int).max())

    if "Weeknummer" not in data.columns:
        return max_year, None

    year_data = data[data["Collegejaar"].astype(int) == max_year]
    weeks = year_data["Weeknummer"].dropna().astype(int).tolist()
    max_week = max(weeks, key=lambda w: week_sort_key(w, final_week))
    return max_year, max_week


def get_max_week_from_weeks(weeks: pd.Series, final_week: int = FINAL_ACADEMIC_WEEK) -> int:
    if len(weeks.unique()) == WEEKS_PER_YEAR:
        max_week = final_week
    else:
        max_week = final_week + len(weeks.unique())
        if max_week > WEEKS_PER_YEAR:
            max_week = max_week - WEEKS_PER_YEAR

    return max_week


def get_weeks_list(weeknummer: int, final_week: int = FINAL_ACADEMIC_WEEK) -> list:
    weeks = []
    if int(weeknummer) > final_week:
        weeks = weeks + [i for i in range(final_week + 1, int(weeknummer) + 1)]
    elif int(weeknummer) < final_week + 1:
        weeks = weeks + [i for i in range(final_week + 1, WEEKS_PER_YEAR + 1)]
        weeks = weeks + [i for i in range(1, int(weeknummer) + 1)]

    return weeks


def get_max_week(predict_year, max_year, data, key, final_week: int = FINAL_ACADEMIC_WEEK):
    if predict_year == max_year:
        max_week = get_max_week_from_weeks(
            data[data[key] == predict_year]["Weeknummer"], final_week
        )
    else:
        if predict_year == 2021:
            # COVID-jaar: het aanmeldpatroon week 38 week was afwijkend waardoor
            # get_max_week_from_weeks() een onjuiste waarde teruggeeft. Zie #84.
            max_week = final_week
        else:
            max_week = get_max_week_from_weeks(
                data[data[key] == predict_year]["Weeknummer"], final_week
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


def academic_start_week(final_week: int = FINAL_ACADEMIC_WEEK) -> int:
    """Eerste week van het academisch jaar (de reset-week net na ``final_week``).

    Bij de default ``final_week=38`` is dit week 39; voor de UvA-aanmeldfase
    (``final_week=36``) is dit week 37.
    """
    return final_week + 1 if final_week < WEEKS_PER_YEAR else 1


def week_sort_key(w: int, final_week: int = FINAL_ACADEMIC_WEEK) -> int:
    """Sort key zodat het academisch jaar loopt van ``start`` → 52 → ``final_week``.

    De ``+ 1`` in de lente-offset houdt ruimte voor ISO-week 53 (lange ISO-jaren):
    die valt als najaarsweek tussen 52 en week 1 in, zonder sleutel-botsing met
    week 1 (de oude formule gaf ``week_sort_key(53) == week_sort_key(1)``).
    """
    start = academic_start_week(final_week)
    return w - start if w >= start else w + (WEEKS_PER_YEAR - start + 1)


def compute_pred_len(predict_week: int, final_week: int = FINAL_ACADEMIC_WEEK) -> int:
    if predict_week > final_week:
        return final_week + WEEKS_PER_YEAR - predict_week
    return final_week - predict_week


def get_all_weeks_valid(columns, final_week: int = FINAL_ACADEMIC_WEEK):
    col_set = set(columns)
    return [w for w in get_all_weeks_ordered(final_week) if w in col_set]


def get_all_weeks_ordered(final_week: int = FINAL_ACADEMIC_WEEK):
    """De weken van één academisch jaar in seizoensvolgorde, als strings.

    Loopt van de reset-week (``final_week`` + 1) via week 52 door naar
    ``final_week`` van het jaar erop. Voor de default ``final_week=38`` is dit
    ``["39", ..., "52", "1", ..., "38"]``; voor UvA (``36``) eindigt de reeks op
    ``"36"`` en start hij op ``"37"``.
    """
    start = academic_start_week(final_week)
    weeks = list(range(start, WEEKS_PER_YEAR + 1)) + list(range(1, final_week + 1))
    return [str(w) for w in weeks]
