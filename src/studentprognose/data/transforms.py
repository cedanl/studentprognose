from studentprognose.utils.weeks import academic_start_week, get_all_weeks_valid, week_sort_key
from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK, WEEKS_PER_YEAR

import pandas as pd


TRANSFORM_GROUP_COLS = [
    "Collegejaar",
    "Faculteit",
    "Herkomst",
    "Examentype",
    "Croho groepeernaam",
]


def transform_data(
    data_input: pd.DataFrame,
    targ_col: str,
    final_week: int = FINAL_ACADEMIC_WEEK,
    interpolate_interior_gaps: bool = False,
) -> pd.DataFrame:
    """Makes a certain pivot_wider where it transforms the data from long to wide.

    ``final_week`` bepaalt de laatste week van het academisch jaar en daarmee de
    seizoensvolgorde van de weekkolommen (default 38; UvA-aanmeldfase 36).

    Ontbrekende weken worden 0. Met ``interpolate_interior_gaps`` worden weken
    die binnen een rij tússen twee waarnemingen ontbreken eerst lineair
    geïnterpoleerd; alleen start- en eindweken zonder waarneming worden dan 0.
    Bedoeld voor cumulatieve reeksen, niet voor indicatoren zoals ``Deadline``.
    """
    group_cols = TRANSFORM_GROUP_COLS

    data = data_input[group_cols + [targ_col, "Weeknummer"]]

    data = data.drop_duplicates()

    data = data.pivot(index=group_cols, columns="Weeknummer", values=targ_col).reset_index()

    data.columns = map(str, data.columns)

    colnames = group_cols + get_all_weeks_valid(data.columns, final_week)

    data = data[colnames]

    if interpolate_interior_gaps:
        week_cols = colnames[len(group_cols):]
        data[week_cols] = (
            data[week_cols].astype("float64").interpolate(axis=1, limit_area="inside")
        )

    data = data.fillna(0)

    return data


def drop_weeks_after_predict_week(
    data: pd.DataFrame,
    from_year: int,
    predict_week: int,
    final_week: int = FINAL_ACADEMIC_WEEK,
) -> pd.DataFrame:
    """Verwijder waarnemingen die op het voorspelmoment nog niet bestonden.

    Een run voor een afgerond collegejaar bevat ook de weken ná de voorspelweek.
    Die lagen op het voorspelmoment in de toekomst. Zonder afkappen kunnen ze via
    interpolatie, features of een mislukte SARIMA-forecast in de voorspelling
    terechtkomen, en lijkt een terugblik beter dan een live run (#318). Na deze
    stap ziet het model exact wat een live run in die week zou zien.

    Args:
        data: Cumulatieve data in lang formaat (``Collegejaar``, ``Weeknummer``).
        from_year: Eerste collegejaar dat wordt afgekapt; dit en latere jaren
            houden alleen de weken t/m ``predict_week`` (academische volgorde).
            Eerdere jaren (de trainingsjaren) blijven volledig.
        predict_week: Laatste bekende week.
        final_week: Laatste week van het academisch jaar.

    Returns:
        ``data`` zonder de rijen van ``from_year`` en later ná ``predict_week``.
    """
    week_key = {w: week_sort_key(w, final_week) for w in range(1, WEEKS_PER_YEAR + 2)}
    limit = week_sort_key(int(predict_week), final_week)
    later = (data["Collegejaar"].astype(int) >= int(from_year)) & (
        data["Weeknummer"].astype(int).map(week_key) > limit
    )
    return data.loc[~later]


def ensure_start_week(data: pd.DataFrame, final_week: int = FINAL_ACADEMIC_WEEK) -> pd.DataFrame:
    """Zorg dat de startweek van het academisch jaar als kolom bestaat.

    De startweek (``final_week + 1``) markeert in de aaneengeplakte reeks de
    overgang naar een nieuw jaar. Ontbreekt de kolom, dan wordt hij met 0
    toegevoegd (er waren nog geen aanmelders). Bestaat hij al, dan blijven de
    echte waarden staan; voorheen werden die altijd met 0 overschreven (#303).
    """
    column = str(academic_start_week(final_week))
    if column not in data.columns:
        data[column] = 0
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


