"""Aggregeer pipeline-output tot scalaire evaluatiemetrieken.

Vergelijkt de voorspelkolommen in de pipeline-output (zoals teruggegeven door
:func:`studentprognose.run_pipeline_from_dataframes`) met de gerealiseerde
``Aantal_studenten`` en vat dat samen tot scalaire metrieken per voorspelmethode.
Bedoeld voor backtesting en monitoring: log de uitkomst naar MLflow
(MS Fabric / Databricks) of een eigen monitoring-tabel.

Belangrijk: evalueren kan alleen voor een (jaar, week) waarvoor de realisatie al
bekend is. Voor het lopende, nog niet afgeronde collegejaar bestaat er geen
``Aantal_studenten`` en is evaluatie per definitie niet mogelijk — kies dan een
eerder collegejaar (backtest) en geef de bijbehorende realisatie mee als
``data_student_numbers``.

De metrieken zelf worden berekend met de canonieke primitieven uit
:mod:`studentprognose.benchmark.metrics`, zodat ze identiek zijn aan wat de
benchmark-CLI en de per-rij ``MAE_*``/``MAPE_*``-kolommen in de output gebruiken.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from studentprognose.benchmark.metrics import mae as _mae, mape as _mape, rmse as _rmse

# Kandidaat-voorspelkolommen, van "samengesteld" naar "enkelvoudig model".
# Exact dezelfde set die PostProcessor._create_error_columns van foutkolommen
# voorziet, zodat evaluate_predictions dezelfde kolommen meet als de pipeline.
# 'Baseline' (duplicaat van Prognose_ratio) en 'Voorspelde vooraanmelders'
# (geen studentaantal-voorspelling) horen hier bewust niet bij.
_CANDIDATE_PREDICTIONS = (
    "Weighted_ensemble_prediction",
    "Average_ensemble_prediction",
    "Ensemble_prediction",
    "Prognose_ratio",
    "SARIMA_cumulative",
    "SARIMA_individual",
)

# Volgorde van de metriekkolommen in het resultaat-DataFrame.
_METRIC_COLS = ["n", "mae", "mape", "wape", "rmse", "bias", "r2"]

# MLflow staat alleen [A-Za-z0-9_\-./ ] toe in metric-namen.
_MLFLOW_KEY_INVALID = re.compile(r"[^A-Za-z0-9_\-./ ]")


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error: sum|fout| / sum|werkelijk|.

    Robuuster dan MAPE bij kleine opleidingen: weegt elke voorspelling naar
    omvang in plaats van elke (opleiding × herkomst) gelijk. Retourneert een
    fractie (0.08 == 8%).
    """
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Gemiddelde signed fout (voorspeld - werkelijk). Positief = overschatting."""
    if len(y_true) == 0:
        return np.nan
    return float(np.mean(y_pred - y_true))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Determinatiecoëfficiënt R². NaN bij < 2 punten of nul-variantie."""
    if len(y_true) < 2:
        return np.nan
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return np.nan
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - ss_res / ss_tot)


def _valid_mask(
    frame: pd.DataFrame,
    actual_column: str,
    pred: str,
    nf_set: set,
    programme_column: str,
) -> pd.Series:
    """Bepaal welke rijen meetellen voor voorspelkolom ``pred``.

    Gebruikt bij voorkeur de door de pipeline berekende ``MAE_<pred>``-kolom: die
    codeert al zowel ``actual``- en ``pred``-notna als de numerus-fixus-uitsluiting,
    zodat de aggregatie exact dezelfde populatie meet als de per-rij foutkolommen.
    Valt terug op een eigen notna-masker (plus optionele NF-uitsluiting) wanneer die
    foutkolom ontbreekt, bijvoorbeeld bij een zelf toegevoegde voorspelkolom.
    """
    mae_col = f"MAE_{pred}"
    if mae_col in frame.columns:
        return frame[mae_col].notna()

    valid = frame[actual_column].notna() & frame[pred].notna()
    if nf_set and programme_column in frame.columns:
        valid &= ~frame[programme_column].isin(nf_set)
    return valid


def _metrics_for(
    frame: pd.DataFrame,
    actual_column: str,
    pred: str,
    nf_set: set,
    programme_column: str,
) -> dict | None:
    """Bereken het scalaire metriekrijtje voor één voorspelkolom op één (sub)frame."""
    valid = _valid_mask(frame, actual_column, pred, nf_set, programme_column)
    sub = frame[valid]
    if sub.empty:
        return None

    y_true = sub[actual_column].to_numpy(dtype="float64")
    y_pred = sub[pred].to_numpy(dtype="float64")

    return {
        "n": int(len(sub)),
        "mae": _mae(y_true, y_pred),
        "mape": _mape(y_true, y_pred),
        "wape": _wape(y_true, y_pred),
        "rmse": _rmse(y_true, y_pred),
        "bias": _bias(y_true, y_pred),
        "r2": _r2(y_true, y_pred),
    }


def evaluate_predictions(
    data: pd.DataFrame,
    *,
    actual_column: str = "Aantal_studenten",
    prediction_columns: list[str] | None = None,
    week: int | None = None,
    group_by: str | list[str] | None = None,
    numerus_fixus: list[str] | dict | None = None,
    programme_column: str = "Croho groepeernaam",
) -> pd.DataFrame:
    """Vat de voorspel-output samen tot scalaire evaluatiemetrieken per model.

    Vergelijkt elke voorspelkolom met ``actual_column`` (de gerealiseerde
    studentaantallen) en geeft per voorspelmethode MAE, MAPE, WAPE, RMSE, bias en R²
    terug. Rijen zonder realisatie of zonder voorspelling worden per voorspelkolom
    overgeslagen; numerus-fixusopleidingen tellen niet mee (consistent met de
    ``MAE_*``-kolommen die de pipeline al berekent).

    Args:
        data: Pipeline-output, zoals teruggegeven door
            :func:`run_pipeline_from_dataframes`.
        actual_column: Kolom met de gerealiseerde aantallen. Standaard
            ``"Aantal_studenten"``.
        prediction_columns: Welke voorspelkolommen te evalueren. Bij ``None`` worden
            de standaard modelkolommen gebruikt die in ``data`` aanwezig zijn.
        week: Beperk de evaluatie tot één peilweek (``Weeknummer``). Sterk aanbevolen
            bij cumulatieve output: alleen de peilweekrij draagt de
            ``SARIMA_cumulative``-voorspelling, terwijl latere weken enkel de
            vooraanmeldcurve bevatten. Geef de gebruikte ``WEEK`` mee voor een
            zuivere vergelijking tussen modellen.
        group_by: Optionele kolom(men) om per segment te rapporteren (bijv.
            ``"Examentype"`` of ``["Examentype", "Herkomst"]``).
        numerus_fixus: NF-opleidingen om uit te sluiten in het terugvalpad (wanneer er
            geen ``MAE_<pred>``-kolom is). Accepteert een lijst of een dict (sleutels).
        programme_column: Kolom met de opleidingsnaam, gebruikt voor NF-uitsluiting.

    Returns:
        Een long-format DataFrame met één rij per (voorspelkolom × groep) en de
        kolommen ``prediction``, eventuele ``group_by``-kolommen, en
        ``n, mae, mape, wape, rmse, bias, r2``. MAPE en WAPE zijn fracties
        (0.08 == 8%). Leeg DataFrame als niets evalueerbaar is.

    Raises:
        ValueError: Als ``data`` ``None`` is, of ``actual_column`` ontbreekt — dat
            laatste betekent meestal dat je een nog niet afgerond collegejaar
            probeert te evalueren (geef realisatiedata mee of kies een eerder jaar).

    Example::

        from studentprognose import run_pipeline_from_dataframes, evaluate_predictions, DataOption

        result = run_pipeline_from_dataframes(
            year=2023, week=12,
            data_cumulative=df_cum, data_student_numbers=df_sc,
            dataset=DataOption.CUMULATIVE, save_output=False,
        )
        metrics = evaluate_predictions(result, week=12)
        print(metrics)
    """
    if data is None:
        raise ValueError("evaluate_predictions kreeg geen DataFrame (data is None).")

    if actual_column not in data.columns:
        raise ValueError(
            f"Kolom {actual_column!r} ontbreekt in de output, dus er valt niets te "
            "evalueren. Dit gebeurt als je een nog niet afgerond collegejaar voorspelt: "
            "geef realisatiedata mee via data_student_numbers en evalueer een eerder "
            "(afgerond) collegejaar."
        )

    group_cols: list[str] = (
        [group_by] if isinstance(group_by, str) else list(group_by) if group_by else []
    )

    frame = data
    if week is not None:
        if "Weeknummer" not in frame.columns:
            raise ValueError("week-filter gevraagd maar kolom 'Weeknummer' ontbreekt.")
        frame = frame[frame["Weeknummer"] == week]
        if frame.empty:
            warnings.warn(
                f"Geen rijen voor week {week}; resultaat is leeg.",
                UserWarning,
                stacklevel=2,
            )

    if prediction_columns is None:
        prediction_columns = [c for c in _CANDIDATE_PREDICTIONS if c in frame.columns]
    else:
        missing = [c for c in prediction_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"Voorspelkolommen ontbreken in data: {missing}")

    if not prediction_columns:
        warnings.warn(
            "Geen evalueerbare voorspelkolommen gevonden in de output.",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=["prediction"] + group_cols + _METRIC_COLS)

    if isinstance(numerus_fixus, dict):
        nf_set = set(numerus_fixus.keys())
    elif numerus_fixus:
        nf_set = set(numerus_fixus)
    else:
        nf_set = set()

    rows: list[dict] = []
    for pred in prediction_columns:
        if group_cols:
            for group_vals, sub in frame.groupby(group_cols, dropna=False):
                metrics = _metrics_for(sub, actual_column, pred, nf_set, programme_column)
                if metrics is None:
                    continue
                if not isinstance(group_vals, tuple):
                    group_vals = (group_vals,)
                row = {"prediction": pred}
                row.update(dict(zip(group_cols, group_vals)))
                row.update(metrics)
                rows.append(row)
        else:
            metrics = _metrics_for(frame, actual_column, pred, nf_set, programme_column)
            if metrics is None:
                continue
            row = {"prediction": pred}
            row.update(metrics)
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["prediction"] + group_cols + _METRIC_COLS)

    result = pd.DataFrame(rows, columns=["prediction"] + group_cols + _METRIC_COLS)
    return result.sort_values(["prediction"] + group_cols, ignore_index=True)


def pivot_metrics(
    metrics: pd.DataFrame,
    *,
    value: str = "wape",
    over: str = "Collegejaar",
    summary: bool = True,
) -> pd.DataFrame:
    """Vat gegroepeerde metrieken samen tot een model × ``over``-matrix.

    Bedoeld om een backtest over meerdere afgeronde jaren (of peilweken) in één oogopslag
    te lezen: één rij per voorspelmodel, één kolom per groepswaarde van ``over`` (bijv. per
    collegejaar), gevuld met de gekozen metriek ``value`` (standaard WAPE — robuust tegen
    kleine opleidingen). Met ``summary=True`` komen er samenvattende kolommen bij — het
    gemiddelde over de groepen plus de spreiding (``min``/``max``) — zodat je ziet of de
    jaar-op-jaar variatie de verschillen tussen modellen overschaduwt.

    Werkt op de output van :func:`evaluate_predictions`; voer eerst de evaluatie per groep
    uit met ``group_by``. Pivoteer je op een grovere as dan waarop is gegroepeerd, dan worden
    de fijnere groepen per cel gemiddeld (ongewogen).

    Args:
        metrics: Long-format resultaat van :func:`evaluate_predictions`, dat naast
            ``prediction`` ook de groepskolom ``over`` bevat.
        value: Welke metriek in de cellen komt; één van ``n, mae, mape, wape, rmse, bias, r2``.
            Standaard ``"wape"`` (robuust tegen kleine opleidingen).
        over: De groepskolom die de matrixkolommen vormt (bijv. ``"Collegejaar"`` of
            ``"Weeknummer"``). Standaard ``"Collegejaar"``.
        summary: Voeg ``mean``/``min``/``max`` over de groepen toe, plus ``n_groups``
            (aantal groepen met een waarde). Standaard ``True``.

    Returns:
        Een wide-format DataFrame, geïndexeerd op ``prediction``, met één kolom per
        groepswaarde en eventueel de samenvattingskolommen. Leeg DataFrame als ``metrics``
        leeg is.

    Raises:
        ValueError: Als ``metrics`` geen ``prediction``-kolom heeft, of als ``value``/``over``
            ontbreken — een ontbrekende ``over``-kolom betekent meestal dat je ``group_by``
            in :func:`evaluate_predictions` bent vergeten.

    Example::

        # Backtest: voorspel meerdere afgeronde jaren op dezelfde peilweek en score ze samen.
        frames = [
            run_pipeline_from_dataframes(
                year=jaar, week=WEEK, data_cumulative=df_cum,
                data_student_numbers=df_sc, dataset=DataOption.CUMULATIVE, save_output=False,
            )
            for jaar in (2021, 2022, 2023)
        ]
        metrics = evaluate_predictions(pd.concat(frames, ignore_index=True),
                                       week=WEEK, group_by="Collegejaar")
        print(pivot_metrics(metrics, value="wape", over="Collegejaar").round(3))
    """
    if "prediction" not in metrics.columns:
        raise ValueError(
            "metrics mist de kolom 'prediction'; geef het resultaat van evaluate_predictions door."
        )
    if value not in metrics.columns:
        raise ValueError(f"Onbekende metriek {value!r}; kies uit {_METRIC_COLS}.")
    if over not in metrics.columns:
        raise ValueError(
            f"Groepskolom {over!r} ontbreekt in metrics. Gebruik group_by={over!r} in "
            "evaluate_predictions zodat de metriek per groep wordt berekend."
        )

    if metrics.empty:
        return pd.DataFrame(index=pd.Index([], name="prediction"))

    table = metrics.pivot_table(index="prediction", columns=over, values=value, aggfunc="mean")
    table = table.sort_index(axis=1)
    table.columns.name = None
    # Toon hele jaren/weken als int (2023) i.p.v. float (2023.0), net als dashboard.py.
    if pd.api.types.is_float_dtype(table.columns):
        table.columns = table.columns.astype("int64")

    if summary:
        period_cols = list(table.columns)
        periods = table[period_cols]
        table["mean"] = periods.mean(axis=1)
        table["min"] = periods.min(axis=1)
        table["max"] = periods.max(axis=1)
        table["n_groups"] = periods.notna().sum(axis=1).astype(int)

    return table


def to_mlflow_metrics(
    metrics: pd.DataFrame,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Vlak een :func:`evaluate_predictions`-resultaat af tot ``{naam: waarde}``.

    Klaar voor ``mlflow.log_metrics(...)`` in een MS Fabric- of Databricks-notebook.
    Sleutels zijn ``f"{prefix}{METRIC}_{prediction}"``; bij een gegroepeerd resultaat
    worden de groepswaarden achteraan de sleutel toegevoegd. ``NaN``-waarden worden
    overgeslagen (MLflow accepteert geen ``NaN``) en tekens die MLflow niet toestaat
    in metric-namen worden vervangen door ``_``.

    Args:
        metrics: Resultaat van :func:`evaluate_predictions`.
        prefix: Optioneel voorvoegsel voor elke sleutel (bijv. ``"cumulatief/"``).

    Returns:
        Platte dict van metricnaam naar float.

    Example::

        import mlflow
        metrics = evaluate_predictions(result, week=12)
        mlflow.log_metrics(to_mlflow_metrics(metrics))
    """
    group_cols = [c for c in metrics.columns if c not in (["prediction"] + _METRIC_COLS)]

    out: dict[str, float] = {}
    for _, row in metrics.iterrows():
        suffix_parts = [str(row[g]) for g in group_cols]
        for metric in _METRIC_COLS:
            val = row[metric]
            if pd.isna(val):
                continue
            key = prefix + "_".join([metric.upper(), str(row["prediction"])] + suffix_parts)
            key = _MLFLOW_KEY_INVALID.sub("_", key)
            out[key] = float(val)
    return out
