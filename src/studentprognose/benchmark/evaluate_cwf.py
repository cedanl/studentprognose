"""Backtest van het CWF-model voor ``studentprognose benchmark -d cwf``.

Voorspelt voor elk afgerond collegejaar de instroom vanaf dezelfde voorspelweek,
alsof het dat jaar live draaide: het model ziet alleen eerdere jaren (de vulgraad
heeft geen look-ahead en de historie is begrensd op ``Collegejaar < testjaar``).
Vergelijkt met de naïeve voorspelling "evenveel als vorig jaar" en meet hoe vaak
de werkelijke instroom binnen het predictie-interval viel.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from studentprognose.benchmark.metrics import mae, mape
from studentprognose.strategies.cwf import (
    CWF_LOWER_COL,
    CWF_PREDICTION_COL,
    CWF_RELIABILITY_COL,
    CWF_UPPER_COL,
    KEY_COLS,
    CWFStrategy,
)
from studentprognose.utils.weeks import DataOption

ACTUAL = "Aantal_studenten"
NAIVE = "Naief_vorig_jaar"


def backtest_cwf(
    data_cumulative: pd.DataFrame,
    data_studentcount: pd.DataFrame,
    configuration: dict,
    predict_week: int,
    min_history_years: int = 2,
) -> pd.DataFrame:
    """Draai het CWF-model per testjaar en koppel realisatie en naïeve voorspelling.

    Args:
        data_cumulative: Ruwe cumulatieve data (zoals ``load_data`` hem levert).
        data_studentcount: Eerstejaars-studentaantallen.
        configuration: Geladen configuratie.
        predict_week: Voorspelweek.
        min_history_years: Minimaal aantal eerdere jaren met realisatie voordat
            een jaar als testjaar telt.

    Returns:
        Eén rij per testjaar × opleiding × examentype × herkomst met prognose,
        interval, label, realisatie en naïeve voorspelling.
    """
    strategy = CWFStrategy(
        data_cumulative,
        data_studentcount,
        configuration,
        None,
        None,
        os.getcwd(),
        DataOption.CWF,
        None,
    )
    strategy.preprocess()

    count_years = sorted(
        int(y) for y in data_studentcount["Collegejaar"].dropna().unique()
    )
    data_years = {int(y) for y in strategy.data_cumulative["Collegejaar"].unique()}
    test_years = [y for y in count_years[min_history_years:] if y in data_years]

    counts = strategy._studentcount.groupby(["Collegejaar"] + KEY_COLS, as_index=False)[
        ACTUAL
    ].sum()
    frames = []
    for year in test_years:
        week_rows = strategy.data_cumulative[
            (strategy.data_cumulative["Collegejaar"] == year)
            & (strategy.data_cumulative["Weeknummer"] == predict_week)
        ]
        if week_rows.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = strategy.predict_nr_of_students(year, predict_week)
        if result is None or result.empty:
            continue
        result = result.merge(counts, on=["Collegejaar"] + KEY_COLS, how="left")
        previous = counts[counts["Collegejaar"] == year - 1].drop(columns="Collegejaar")
        result = result.merge(
            previous.rename(columns={ACTUAL: NAIVE}), on=KEY_COLS, how="left"
        )
        result["test_year"] = year
        frames.append(result)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _metrics(frame: pd.DataFrame) -> dict:
    known = frame.dropna(subset=[ACTUAL, CWF_PREDICTION_COL])
    actual = known[ACTUAL].to_numpy(dtype=float)
    pred = known[CWF_PREDICTION_COL].to_numpy(dtype=float)
    lo = known[CWF_LOWER_COL].to_numpy(dtype=float)
    hi = known[CWF_UPPER_COL].to_numpy(dtype=float)
    has_interval = np.isfinite(lo) & np.isfinite(hi)
    inside = (actual >= lo) & (actual <= hi)
    width = (hi - lo) / np.where(pred > 0, pred, np.nan)

    naive = (
        known[NAIVE].to_numpy(dtype=float)
        if NAIVE in known
        else np.full(len(known), np.nan)
    )
    both = np.isfinite(naive)
    total = np.sum(np.abs(actual))
    return {
        "n": len(known),
        "mae": mae(actual, pred),
        "mape": mape(actual, pred),
        "wape": float(np.sum(np.abs(actual - pred)) / total) if total else np.nan,
        "coverage": float(inside[has_interval].mean())
        if has_interval.any()
        else np.nan,
        "median_rel_width": float(np.nanmedian(width[has_interval]))
        if has_interval.any()
        else np.nan,
        "naive_mae": mae(actual[both], naive[both]) if both.any() else np.nan,
        "naive_wape": (
            float(
                np.sum(np.abs(actual[both] - naive[both]))
                / np.sum(np.abs(actual[both]))
            )
            if both.any() and np.sum(np.abs(actual[both]))
            else np.nan
        ),
    }


def summarise_cwf(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Samenvatting per testjaar (plus totaal) en per betrouwbaarheidslabel.

    Numerus-fixusopleidingen zijn al door de aanroeper uitgesloten waar gewenst.
    """
    by_year = [{"test_year": "alle", **_metrics(results)}]
    for year, frame in results.groupby("test_year"):
        by_year.append({"test_year": int(year), **_metrics(frame)})
    by_label = [
        {"label": label, **_metrics(frame)}
        for label, frame in results.groupby(CWF_RELIABILITY_COL)
    ]
    order = {"Groen": 0, "Oranje": 1, "Rood": 2}
    by_label_df = pd.DataFrame(by_label)
    if not by_label_df.empty:
        by_label_df = by_label_df.sort_values("label", key=lambda s: s.map(order))
    return pd.DataFrame(by_year), by_label_df
