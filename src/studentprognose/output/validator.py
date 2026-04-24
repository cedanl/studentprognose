"""Post-prediction outputkwaliteitschecks.

Twee checks worden uitgevoerd nadat de modellen hun voorspellingen hebben
opgeleverd:

  trend-realisme   – waarschuwt als Ensemble_prediction sterk afwijkt van
                     vorig jaar of vorige week. Nooit een hard stop.
  NF-cap controle  – waarschuwt als de gesommeerde voorspelling per NF-opleiding
                     het geconfigureerde plafond overschrijdt (exclusief Pre-master).

Beide checks zijn puur informatief. Ze stoppen de pipeline niet — de gebruiker
beslist of actie nodig is.
"""

import warnings

import pandas as pd

from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK


def run_post_prediction_checks(
    data: pd.DataFrame,
    data_cumulative: "pd.DataFrame | None",
    predict_year: int,
    predict_week: int,
    numerus_fixus_list: dict,
) -> None:
    """Run all post-prediction output quality checks. Prints warnings only."""
    if data is None or "Ensemble_prediction" not in data.columns:
        return

    predictions = data[
        (data["Collegejaar"] == predict_year)
        & (data["Weeknummer"] == predict_week)
        & data["Ensemble_prediction"].notna()
    ]
    if predictions.empty:
        return

    _check_trend_realism(predictions, data, data_cumulative, predict_year, predict_week)
    _check_numerus_fixus_caps(predictions, numerus_fixus_list, predict_year, predict_week)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_trend_realism(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    data_cumulative: "pd.DataFrame | None",
    predict_year: int,
    predict_week: int,
) -> None:
    group_cols = ["Herkomst", "Croho groepeernaam", "Examentype"]
    if any(c not in predictions.columns for c in group_cols):
        return

    # Year-over-year: compare against same week last year from cumulative data
    last_year_data = None
    if data_cumulative is not None and "Gewogen vooraanmelders" in data_cumulative.columns:
        last_year_data = data_cumulative[
            (data_cumulative["Collegejaar"] == predict_year - 1)
            & (data_cumulative["Weeknummer"] == predict_week)
        ]

    # Week-over-week: compare against predictions from last week in same run.
    # Exclude week 1 (no prior week) and FINAL_ACADEMIC_WEEK+1 (prior week is
    # end of previous cycle, not previous prediction in this run).
    last_week_data = data[
        (data["Collegejaar"] == predict_year)
        & (data["Weeknummer"] == predict_week - 1)
        & data["Ensemble_prediction"].notna()
    ] if predict_week > 1 and predict_week != FINAL_ACADEMIC_WEEK + 1 else pd.DataFrame()

    curr_preds = predictions.groupby(group_cols)["Ensemble_prediction"].sum().reset_index()

    if last_year_data is not None and not last_year_data.empty:
        ly_agg = (
            last_year_data.groupby(group_cols)["Gewogen vooraanmelders"]
            .sum()
            .reset_index()
            .rename(columns={"Gewogen vooraanmelders": "_ly"})
        )
        merged = pd.merge(curr_preds, ly_agg, on=group_cols, how="inner")
        for _, row in merged.iterrows():
            pred = row["Ensemble_prediction"]
            act_ly = row["_ly"]
            if act_ly <= 0:
                continue
            abs_diff = abs(pred - act_ly)
            rel_diff = abs_diff / act_ly
            if rel_diff > 0.50 and abs_diff > 20:
                label = f"{row['Herkomst']} | {row['Croho groepeernaam']} | {row['Examentype']}"
                warnings.warn(
                    f"Trend-realisme jaar {predict_year} week {predict_week}: "
                    f"ensemble wijkt sterk af van vorig jaar voor {label}. "
                    f"Voorspelling: {pred:.0f}, vorig jaar: {act_ly:.0f} "
                    f"({rel_diff:.0%} afwijking).",
                    UserWarning,
                    stacklevel=2,
                )

    if not last_week_data.empty:
        lw_agg = (
            last_week_data.groupby(group_cols)["Ensemble_prediction"]
            .sum()
            .reset_index()
            .rename(columns={"Ensemble_prediction": "_lw"})
        )
        merged = pd.merge(curr_preds, lw_agg, on=group_cols, how="inner")
        for _, row in merged.iterrows():
            pred = row["Ensemble_prediction"]
            act_lw = row["_lw"]
            if act_lw <= 0:
                continue
            abs_diff = abs(pred - act_lw)
            rel_diff = abs_diff / act_lw
            if rel_diff > 0.30 and abs_diff > 15:
                label = f"{row['Herkomst']} | {row['Croho groepeernaam']} | {row['Examentype']}"
                warnings.warn(
                    f"Trend-realisme jaar {predict_year} week {predict_week}: "
                    f"ensemble wijkt sterk af van vorige week voor {label}. "
                    f"Voorspelling: {pred:.0f}, vorige week: {act_lw:.0f} "
                    f"({rel_diff:.0%} week-op-week afwijking).",
                    UserWarning,
                    stacklevel=2,
                )


def _check_numerus_fixus_caps(
    predictions: pd.DataFrame,
    numerus_fixus_list: dict,
    predict_year: int,
    predict_week: int,
) -> None:
    if not numerus_fixus_list:
        return

    if "Examentype" not in predictions.columns:
        return

    # Pre-master studenten vallen buiten het NF-instroomplafond: ze zijn al
    # ingeschreven in een overgangstraject en tellen niet mee als nieuwe
    # eerstejaars. Inclusie zou NF-programma's structureel boven plafond doen
    # lijken terwijl de instroom feitelijk correct is.
    non_premaster = predictions[predictions["Examentype"] != "Pre-master"]
    totals = (
        non_premaster.groupby("Croho groepeernaam")["Ensemble_prediction"]
        .sum()
        .reset_index()
    )

    for _, row in totals.iterrows():
        programme = row["Croho groepeernaam"]
        total = row["Ensemble_prediction"]
        if programme in numerus_fixus_list:
            cap = numerus_fixus_list[programme]
            if cap > 0 and total > cap:
                warnings.warn(
                    f"NF-cap overschrijding jaar {predict_year} week {predict_week}: "
                    f"{programme} — voorspeld {total:.0f}, plafond {cap}.",
                    UserWarning,
                    stacklevel=2,
                )
