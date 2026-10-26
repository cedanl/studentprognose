"""Post-prediction outputkwaliteitschecks.

Twee checks worden uitgevoerd nadat de modellen hun voorspellingen hebben
opgeleverd:

  trend-realisme   – waarschuwt als Ensemble_prediction sterk afwijkt van
                     vorig jaar of vorige week. Nooit een hard stop.
  NF-cap controle  – waarschuwt als de gesommeerde voorspelling per NF-opleiding
                     het geconfigureerde plafond overschrijdt (exclusief Pre-master).

Voor het CWF-model (``-d cwf``) draaien dezelfde checks op ``Prognose_cwf``,
aangevuld met:

  interval-consistentie – waarschuwt als ondergrens ≤ voorspelling ≤ bovengrens
                          niet geldt of de grenzen geen hele getallen zijn.
  betrouwbaarheidslabel – waarschuwt bij een label buiten Groen/Oranje/Rood.
  fysiek plafond        – waarschuwt als de bovengrens boven ongewogen
                          aanmelders + inschrijvingen ligt terwijl de vulgraad
                          het plafond activeert.

Alle checks zijn puur informatief. Ze stoppen de pipeline niet — de gebruiker
beslist of actie nodig is.
"""

import warnings

import numpy as np
import pandas as pd

from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK

CWF_PREDICTION_COL = "Prognose_cwf"
CWF_LOWER_COL = "cwf_ondergrens"
CWF_UPPER_COL = "cwf_bovengrens"
CWF_RELIABILITY_COL = "cwf_betrouwbaarheid"
CWF_LABELS = ("Groen", "Oranje", "Rood")


def run_post_prediction_checks(
    data: pd.DataFrame,
    data_cumulative: "pd.DataFrame | None",
    predict_year: int,
    predict_week: int,
    numerus_fixus_list: dict,
    final_week: int = FINAL_ACADEMIC_WEEK,
    cwf_params: "dict | None" = None,
) -> None:
    """Run all post-prediction output quality checks. Prints warnings only.

    ``cwf_params`` (``model_config.cwf_params``) bepaalt voor ``-d cwf`` vanaf
    welke vulgraad en voor welke examentypes het fysieke plafond geldt.
    """
    if data is None:
        return
    if "Ensemble_prediction" in data.columns:
        prediction_col = "Ensemble_prediction"
    elif CWF_PREDICTION_COL in data.columns:
        prediction_col = CWF_PREDICTION_COL
    else:
        return

    predictions = data[
        (data["Collegejaar"] == predict_year)
        & (data["Weeknummer"] == predict_week)
        & data[prediction_col].notna()
    ]
    if predictions.empty:
        return

    _check_trend_realism(
        predictions, data, data_cumulative, predict_year, predict_week, final_week,
        prediction_col=prediction_col,
    )
    _check_numerus_fixus_caps(
        predictions, numerus_fixus_list, predict_year, predict_week, prediction_col=prediction_col,
    )
    if prediction_col == CWF_PREDICTION_COL:
        _check_cwf_interval(predictions, predict_year, predict_week)
        _check_cwf_labels(predictions, predict_year, predict_week)
        _check_cwf_ceiling(predictions, predict_year, predict_week, cwf_params)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_trend_realism(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    data_cumulative: "pd.DataFrame | None",
    predict_year: int,
    predict_week: int,
    final_week: int = FINAL_ACADEMIC_WEEK,
    prediction_col: str = "Ensemble_prediction",
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
    # Exclude week 1 (no prior week) and final_week+1 (prior week is
    # end of previous cycle, not previous prediction in this run).
    last_week_data = data[
        (data["Collegejaar"] == predict_year)
        & (data["Weeknummer"] == predict_week - 1)
        & data[prediction_col].notna()
    ] if predict_week > 1 and predict_week != final_week + 1 else pd.DataFrame()

    curr_preds = (
        predictions.groupby(group_cols)[prediction_col].sum().reset_index()
        .rename(columns={prediction_col: "_pred"})
    )

    if last_year_data is not None and not last_year_data.empty:
        ly_agg = (
            last_year_data.groupby(group_cols)["Gewogen vooraanmelders"]
            .sum()
            .reset_index()
            .rename(columns={"Gewogen vooraanmelders": "_ly"})
        )
        merged = pd.merge(curr_preds, ly_agg, on=group_cols, how="inner")
        for _, row in merged.iterrows():
            pred = row["_pred"]
            act_ly = row["_ly"]
            if act_ly <= 0:
                continue
            abs_diff = abs(pred - act_ly)
            rel_diff = abs_diff / act_ly
            # Ensemble_prediction is de voorspelde eindinstroom; act_ly is de
            # gewogen vooraanmelding vorig jaar op dezelfde week — geen directe
            # appels-met-appels vergelijking. De check werkt als signaal omdat
            # de conversieverhouding historisch stabiel is: extreme YoY-afwijking
            # wijst op een modelfout of uitzonderlijk jaar, niet op een normaal
            # seizoenspatroon. AND-logica (beide drempels) filtert ruis bij
            # kleine programma's zonder de absolute vloer van de pre-check.
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
            last_week_data.groupby(group_cols)[prediction_col]
            .sum()
            .reset_index()
            .rename(columns={prediction_col: "_lw"})
        )
        merged = pd.merge(curr_preds, lw_agg, on=group_cols, how="inner")
        for _, row in merged.iterrows():
            pred = row["_pred"]
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
    prediction_col: str = "Ensemble_prediction",
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
        non_premaster.groupby("Croho groepeernaam")[prediction_col]
        .sum()
        .reset_index()
    )

    for _, row in totals.iterrows():
        programme = row["Croho groepeernaam"]
        total = row[prediction_col]
        if programme in numerus_fixus_list:
            cap = numerus_fixus_list[programme]
            if cap > 0 and total > cap:
                warnings.warn(
                    f"NF-cap overschrijding jaar {predict_year} week {predict_week}: "
                    f"{programme} — voorspeld {total:.0f}, plafond {cap}.",
                    UserWarning,
                    stacklevel=2,
                )


def _check_cwf_interval(predictions: pd.DataFrame, predict_year: int, predict_week: int) -> None:
    """Interval moet ``ondergrens ≤ voorspelling ≤ bovengrens`` zijn, in hele getallen."""
    if CWF_LOWER_COL not in predictions.columns or CWF_UPPER_COL not in predictions.columns:
        return
    rows = predictions.dropna(subset=[CWF_LOWER_COL, CWF_UPPER_COL])
    pred, low, up = rows[CWF_PREDICTION_COL], rows[CWF_LOWER_COL], rows[CWF_UPPER_COL]
    inconsistent = rows[(low > pred) | (pred > up) | (low < 0)]
    if not inconsistent.empty:
        warnings.warn(
            f"CWF-interval jaar {predict_year} week {predict_week}: {len(inconsistent)} rij(en) "
            "waarbij niet geldt 0 ≤ ondergrens ≤ voorspelling ≤ bovengrens.",
            UserWarning,
            stacklevel=2,
        )
    fractional = rows[(pred % 1 != 0) | (low % 1 != 0) | (up % 1 != 0)]
    if not fractional.empty:
        warnings.warn(
            f"CWF-interval jaar {predict_year} week {predict_week}: {len(fractional)} rij(en) "
            "met een voorspelling of grens die geen heel getal is.",
            UserWarning,
            stacklevel=2,
        )


def _check_cwf_labels(predictions: pd.DataFrame, predict_year: int, predict_week: int) -> None:
    """Elk CWF-label moet Groen, Oranje of Rood zijn."""
    if CWF_RELIABILITY_COL not in predictions.columns:
        return
    invalid = predictions[~predictions[CWF_RELIABILITY_COL].isin(CWF_LABELS)]
    if not invalid.empty:
        warnings.warn(
            f"CWF-betrouwbaarheid jaar {predict_year} week {predict_week}: {len(invalid)} rij(en) "
            f"met een onbekend label (verwacht {', '.join(CWF_LABELS)}).",
            UserWarning,
            stacklevel=2,
        )


def _check_cwf_ceiling(
    predictions: pd.DataFrame,
    predict_year: int,
    predict_week: int,
    cwf_params: "dict | None",
) -> None:
    """Bovengrens ≤ ongewogen aanmelders + inschrijvingen zodra het plafond actief is."""
    needed = (CWF_UPPER_COL, "cwf_vulgraad", "Ongewogen vooraanmelders", "Inschrijvingen")
    if any(c not in predictions.columns for c in needed):
        return
    params = cwf_params or {}
    threshold = params.get("upper_bound_cap_fill_rate", 0.98)
    exam_types = params.get("upper_bound_cap_exam_types", ["Bachelor", "Master"])
    rows = predictions[
        predictions["Examentype"].isin(exam_types)
        & (predictions["cwf_vulgraad"] >= threshold)
        & predictions[CWF_UPPER_COL].notna()
    ]
    ceiling = rows["Ongewogen vooraanmelders"] + rows["Inschrijvingen"]
    # De bovengrens mag de voorspelling zelf nooit onderschrijden; alleen
    # overschrijdingen bóven max(voorspelling, plafond) zijn een fout.
    over = rows[rows[CWF_UPPER_COL] > np.maximum(ceiling, rows[CWF_PREDICTION_COL]) + 0.5]
    if not over.empty:
        warnings.warn(
            f"CWF-plafond jaar {predict_year} week {predict_week}: {len(over)} rij(en) met een "
            "bovengrens boven ongewogen aanmelders + inschrijvingen terwijl de cyclus "
            "historisch gezien al voltooid is.",
            UserWarning,
            stacklevel=2,
        )
