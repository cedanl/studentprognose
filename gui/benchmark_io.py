"""Samenvatten van benchmarkresultaten en parsen van tune-output.

Pure logica (geen NiceGUI). De benchmark-CSV's bevatten één rij per
(opleiding × herkomst × …) per model; :func:`summarize` aggregeert naar één rij
per model. :func:`extract_config_snippets` haalt de JSON-config-snippets uit de
console-uitvoer van het ``tune``-commando.
"""

from __future__ import annotations

import json

import pandas as pd

#: Per benchmarkbestand: (metriekkolom, hoger_is_beter, label).
BENCHMARK_METRICS = {
    "benchmark_timeseries.csv": ("mape", False, "MAPE"),
    "benchmark_regressor.csv": ("mape", False, "MAPE"),
    "benchmark_classifier.csv": ("auc_roc", True, "AUC-ROC"),
}


def summarize(
    df: pd.DataFrame, metric: str, *, higher_is_better: bool = False
) -> list[dict]:
    """Aggregeer de benchmark naar één rij per model, gesorteerd op de metriek.

    Args:
        df: Benchmark-DataFrame met een ``model``-kolom en de metriekkolom.
        metric: Naam van de metriekkolom (bijv. ``mape``).
        higher_is_better: True als een hogere waarde beter is (bijv. AUC).

    Returns:
        Lijst met dicts ``{model, metric, n, winner}`` — beste model eerst,
        ``winner=True`` op de beste rij.
    """
    if "model" not in df.columns or metric not in df.columns:
        return []

    clean = df.copy()
    clean[metric] = pd.to_numeric(clean[metric], errors="coerce").replace(
        [float("inf"), float("-inf")], float("nan")
    )

    grouped = (
        clean.groupby("model")[metric].mean().reset_index().dropna(subset=[metric])
    )
    grouped = grouped.sort_values(metric, ascending=not higher_is_better)

    records = []
    for i, (_, row) in enumerate(grouped.iterrows()):
        records.append(
            {
                "model": str(row["model"]),
                "metric": round(float(row[metric]), 4),
                "winner": i == 0,
            }
        )
    return records


def extract_config_snippets(text: str) -> list[dict]:
    """Haal de JSON-config-snippets met ``model_config`` uit tune-console-tekst.

    Scant naar gebalanceerde ``{...}``-blokken en behoudt de blokken die geldige
    JSON zijn én een ``model_config``-sleutel bevatten.

    Args:
        text: De (gecombineerde) console-uitvoer van het ``tune``-commando.

    Returns:
        Lijst van geparste snippet-dicts (kan leeg zijn).
    """
    snippets: list[dict] = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    block = text[start : i + 1]
                    try:
                        parsed = json.loads(block)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict) and "model_config" in parsed:
                        snippets.append(parsed)
    return snippets


def deep_merge(base: dict, overlay: dict) -> dict:
    """Diepe merge van ``overlay`` in ``base`` (nieuwe dict, muteert niets).

    Nested dicts worden recursief samengevoegd; niet-dict-waarden uit ``overlay``
    overschrijven die uit ``base``.
    """
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
