"""Uitlezen en samenvatten van de voorspellingsoutput voor het resultatenoverzicht.

Pure logica (geen NiceGUI). De voorspelling is verankerd op de predict-week: de
kolom met de voorspelling (bijv. ``SARIMA_cumulative`` of ``Ensemble_prediction``)
is alleen op die week gevuld. :func:`prediction_rows` reduceert de output daarom
tot één rij per opleiding × herkomst.
"""

from __future__ import annotations

import glob
import os

import pandas as pd

#: Voorkeursvolgorde voor de primaire voorspelkolom (eerste die bestaat wint).
_PREDICTION_COLUMNS = [
    "Weighted_ensemble_prediction",
    "Ensemble_prediction",
    "SARIMA_cumulative",
    "SARIMA_individual",
]

#: Kolomprefixen van foutmaten die als "model" in de vergelijking meetellen.
_MAE_PREFIX = "MAE_"
_MAPE_PREFIX = "MAPE_"

PROGRAMME_COL = "Croho groepeernaam"


def find_output_files(output_dir: str) -> list[tuple[str, str]]:
    """Zoek de definitieve outputbestanden (geen prelim/totaal).

    Returns:
        Lijst van ``(label, pad)``, nieuwste eerst.
    """
    results: list[tuple[str, str]] = []
    for path in glob.glob(os.path.join(output_dir, "output_*.xlsx")):
        name = os.path.basename(path)
        if name.startswith("output_prelim") or "_ci_test" in name:
            continue
        results.append((name, path))
    results.sort(key=lambda t: os.path.getmtime(t[1]), reverse=True)
    return results


def load_output(path: str) -> pd.DataFrame:
    """Laad een outputbestand (expliciete engine voor .xlsx)."""
    return pd.read_excel(path, engine="openpyxl")


def primary_prediction_column(df: pd.DataFrame) -> str | None:
    """Bepaal de primaire voorspelkolom die in ``df`` aanwezig is."""
    for col in _PREDICTION_COLUMNS:
        if col in df.columns:
            return col
    return None


def prediction_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Reduceer tot de verankerrijen (de predict-week) — één per opleiding×herkomst.

    De predict-week is de week waarop de primaire voorspelkolom gevuld is. Valt
    terug op de kleinste weeknummer als er geen voorspelkolom is.
    """
    pred_col = primary_prediction_column(df)
    if pred_col is not None and df[pred_col].notna().any():
        anchored = df[df[pred_col].notna()]
        if not anchored.empty:
            return anchored.copy()
    if "Weeknummer" in df.columns and df["Weeknummer"].notna().any():
        first_week = int(df["Weeknummer"].min())
        return df[df["Weeknummer"] == first_week].copy()
    return df.copy()


def _clean_metric(series: pd.Series) -> pd.Series:
    """Vervang inf door NaN zodat gemiddelden robuust zijn."""
    return pd.to_numeric(series, errors="coerce").replace(
        [float("inf"), float("-inf")], float("nan")
    )


def model_comparison(rows: pd.DataFrame) -> dict[str, float]:
    """Gemiddelde MAE per model (uit de ``MAE_``-kolommen).

    Returns:
        ``{modelnaam: gemiddelde_mae}`` (lege dict als er geen MAE-kolommen zijn).
    """
    result: dict[str, float] = {}
    for col in rows.columns:
        if col.startswith(_MAE_PREFIX):
            mean = _clean_metric(rows[col]).mean()
            if pd.notna(mean):
                result[col[len(_MAE_PREFIX) :]] = float(mean)
    return result


def best_model_by_mape(rows: pd.DataFrame) -> tuple[str | None, float | None]:
    """Bepaal het model met de laagste gemiddelde MAPE.

    Returns:
        ``(modelnaam, gemiddelde_mape)`` of ``(None, None)``.
    """
    best_name, best_value = None, None
    for col in rows.columns:
        if col.startswith(_MAPE_PREFIX):
            mean = _clean_metric(rows[col]).mean()
            if pd.notna(mean) and (best_value is None or mean < best_value):
                best_name, best_value = col[len(_MAPE_PREFIX) :], float(mean)
    return best_name, best_value


def compute_kpis(df: pd.DataFrame) -> dict:
    """Bereken de KPI's voor het overzicht.

    Returns:
        Dict met ``n_programmes``, ``year``, ``week``, ``best_model``,
        ``best_mape`` (fractie, of ``None``).
    """
    rows = prediction_rows(df)
    n_programmes = (
        int(rows[PROGRAMME_COL].nunique()) if PROGRAMME_COL in rows.columns else 0
    )
    year = (
        int(rows["Collegejaar"].max())
        if "Collegejaar" in rows.columns and rows["Collegejaar"].notna().any()
        else None
    )
    week = (
        int(rows["Weeknummer"].min())
        if "Weeknummer" in rows.columns and rows["Weeknummer"].notna().any()
        else None
    )
    best_model, best_mape = best_model_by_mape(rows)
    return {
        "n_programmes": n_programmes,
        "year": year,
        "week": week,
        "best_model": best_model,
        "best_mape": best_mape,
    }


def mape_bucket(mape: float | None) -> str:
    """Kleurbucket voor een MAPE-waarde (fractie): groen/oranje/rood.

    Returns:
        ``"positive"`` (<0.10), ``"warning"`` (<0.25) of ``"negative"`` (≥0.25).
        ``"grey"`` bij ontbrekende waarde.
    """
    if mape is None or (isinstance(mape, float) and pd.isna(mape)):
        return "grey"
    if mape < 0.10:
        return "positive"
    if mape < 0.25:
        return "warning"
    return "negative"


def programme_rows(df: pd.DataFrame) -> list[dict]:
    """Bouw de tabelrijen voor het overzicht, gesorteerd op absolute afwijking.

    Elke rij bevat opleiding, herkomst, examentype, voorspelling, werkelijk,
    afwijking, mape en een kleurbucket.
    """
    rows = prediction_rows(df)
    pred_col = primary_prediction_column(rows)
    mape_col = _pick_mape_for(pred_col)
    records: list[dict] = []

    for _, r in rows.iterrows():
        prediction = _safe_float(r.get(pred_col)) if pred_col else None
        actual = _safe_float(r.get("Aantal_studenten"))
        deviation = (
            abs(prediction - actual)
            if prediction is not None and actual is not None
            else None
        )
        mape = None
        if mape_col and mape_col in rows.columns:
            mape = _safe_float(r.get(mape_col))
        records.append(
            {
                "opleiding": str(r.get(PROGRAMME_COL, "")),
                "herkomst": str(r.get("Herkomst", "")),
                "examentype": str(r.get("Examentype", "")),
                "voorspelling": round(prediction, 1)
                if prediction is not None
                else None,
                "werkelijk": round(actual, 1) if actual is not None else None,
                "afwijking": round(deviation, 1) if deviation is not None else None,
                "mape": round(mape, 3) if mape is not None else None,
                "kleur": mape_bucket(mape),
            }
        )

    records.sort(key=lambda d: (d["afwijking"] is None, -(d["afwijking"] or 0)))
    return records


def _pick_mape_for(pred_col: str | None) -> str | None:
    """Kies de MAPE-kolom die bij de primaire voorspelkolom hoort."""
    if pred_col is None:
        return None
    candidate = f"{_MAPE_PREFIX}{pred_col}"
    return candidate


def audit_diff(totaal: pd.DataFrame, *, shift_threshold: float = 0.10) -> dict:
    """Vergelijk de twee laatste runs in het ``_totaal``-auditbestand.

    Args:
        totaal: Het audittrail-DataFrame (bevat een ``Run_date``-kolom).
        shift_threshold: Relatieve verschuiving vanaf waar een opleiding als
            "grote verschuiving" telt (default 10%).

    Returns:
        Dict met ``available`` (bool), en bij True: ``new_programmes`` (lijst) en
        ``shifts`` (lijst van dicts met opleiding, oud, nieuw, verschil-fractie).
    """
    if "Run_date" not in totaal.columns:
        return {"available": False}
    dates = sorted(totaal["Run_date"].dropna().unique())
    if len(dates) < 2:
        return {"available": False}

    prev, curr = dates[-2], dates[-1]
    pred_col = primary_prediction_column(totaal)
    if pred_col is None:
        return {"available": False}

    key_cols = [PROGRAMME_COL, "Herkomst", "Examentype"]
    key_cols = [c for c in key_cols if c in totaal.columns]

    def _anchor(date):
        sub = totaal[(totaal["Run_date"] == date) & (totaal[pred_col].notna())]
        return sub.set_index(key_cols)[pred_col]

    prev_s, curr_s = _anchor(prev), _anchor(curr)

    new_keys = set(map(_key_str, curr_s.index)) - set(map(_key_str, prev_s.index))
    shifts = []
    for idx, new_val in curr_s.items():
        if idx in prev_s.index:
            old_val = prev_s.loc[idx]
            old_scalar = _safe_float(
                old_val.iloc[0] if hasattr(old_val, "iloc") else old_val
            )
            new_scalar = _safe_float(
                new_val.iloc[0] if hasattr(new_val, "iloc") else new_val
            )
            if old_scalar and new_scalar and old_scalar != 0:
                rel = (new_scalar - old_scalar) / abs(old_scalar)
                if abs(rel) >= shift_threshold:
                    shifts.append(
                        {
                            "opleiding": _key_str(idx),
                            "oud": round(old_scalar, 1),
                            "nieuw": round(new_scalar, 1),
                            "verschil": round(rel, 3),
                        }
                    )
    shifts.sort(key=lambda d: -abs(d["verschil"]))
    return {
        "available": True,
        "new_programmes": sorted(new_keys),
        "shifts": shifts,
    }


def _key_str(idx) -> str:
    if isinstance(idx, tuple):
        return " · ".join(str(x) for x in idx)
    return str(idx)


def _safe_float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f
