"""Gedeelde helpers voor alle studentprognose-notebooks.

Houd de notebooks zelf vrij van setup-boilerplate. Importeer hier vanuit:

    from _helpers import load_cumulative, with_realisatie, suppress_stdout

Alle functies werken op de meegeleverde demodata in ``data/input/``.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

import pandas as pd


# ----------------------------------------------------------------------
# Paadbeheer
# ----------------------------------------------------------------------

def project_root() -> Path:
    """Wissel naar de projectroot ongeacht waar de notebook draait."""
    p = Path.cwd()
    if p.name == "notebooks":
        p = p.parent
    os.chdir(p)
    return p


# ----------------------------------------------------------------------
# Datalading (cumulatief + studentaantallen + optioneel individueel)
# ----------------------------------------------------------------------

def preprocess_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    """Renames en typecasts zoals ``CumulativeStrategy.preprocess()`` doet.

    Effect:
    - ``Type hoger onderwijs`` → ``Examentype``
    - ``Groepeernaam Croho`` → ``Croho groepeernaam``
    - Hogerejaars uitsluiten (de pipeline voorspelt alleen eerstejaars)
    - Cast numerieke kolommen, voeg ``ts`` toe
    """
    df = df.copy()
    for col in (
        "Gewogen vooraanmelders",
        "Ongewogen vooraanmelders",
        "Aantal aanmelders met 1 aanmelding",
        "Inschrijvingen",
    ):
        if pd.api.types.is_string_dtype(df[col].dtype):
            df[col] = df[col].str.replace(".", "").str.replace(",", ".")
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    df = df.rename(columns={
        "Type hoger onderwijs": "Examentype",
        "Groepeernaam Croho": "Croho groepeernaam",
    })
    df = df[df["Hogerejaars"] == "Nee"].copy()
    df["ts"] = df["Gewogen vooraanmelders"].fillna(0) + df["Inschrijvingen"].fillna(0)
    df = df.astype({"Weeknummer": "int32", "Collegejaar": "int32"})
    return df


def load_cumulative(preprocessed: bool = True):
    """Laad de demodata en lever (data_cumulative, data_studentcount, data_cumulative_raw).

    Args:
        preprocessed: of de cumulatieve data al gepreprocessed moet zijn met
            ``preprocess_cumulative``. Zet op ``False`` om de RAW vorm te krijgen
            (nodig voor ``run_pipeline_from_dataframes`` dat zelf preprocesst).
    """
    from studentprognose.config import load_defaults
    from studentprognose.data.loader import load_data
    from studentprognose.utils.weeks import DataOption

    config = load_defaults()
    _, raw, sc, *_ = load_data(config, DataOption.CUMULATIVE)
    data_cum = preprocess_cumulative(raw) if preprocessed else raw
    return data_cum, sc, raw


def load_individueel(n_sample: int | None = 5000, seed: int = 42):
    """Laad een sample van de individuele aanmelddata, hernoem kolommen kanoniek.

    Args:
        n_sample: aantal rijen om te samplen (None = volledig bestand).
        seed: random_state voor reproduceerbaarheid.

    Returns:
        DataFrame met kanonieke kolomnamen, of ``None`` als het bestand ontbreekt.
    """
    path = Path("data/input/vooraanmeldingen_individueel.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=";", skiprows=[1], low_memory=False)
    if n_sample is not None and len(df) > n_sample:
        df = df.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# Realisatie & vergelijking
# ----------------------------------------------------------------------

def with_realisatie(
    predictions: pd.DataFrame,
    data_studentcount: pd.DataFrame,
    pred_col: str = "Prognose",
) -> pd.DataFrame:
    """Voeg werkelijke instroom (realisatie) toe en bereken absolute + procentuele fout.

    Args:
        predictions: DataFrame met minimaal de identifier-kolommen + ``pred_col``.
        data_studentcount: ``Aantal_studenten`` per (Collegejaar, Croho groepeernaam, Herkomst, Examentype).
        pred_col: naam van de voorspelkolom in ``predictions``.

    Returns:
        Kopie van ``predictions`` met extra kolommen ``Realisatie``, ``Fout`` (absoluut)
        en ``Fout_pct`` (afgerond %).
    """
    merge_keys = ["Collegejaar", "Croho groepeernaam", "Herkomst", "Examentype"]
    out = predictions.merge(
        data_studentcount[merge_keys + ["Aantal_studenten"]].rename(
            columns={"Aantal_studenten": "Realisatie"}
        ),
        on=merge_keys,
        how="left",
    )
    out["Fout"] = (out[pred_col] - out["Realisatie"]).round(1)
    out["Fout_pct"] = (
        (out[pred_col] - out["Realisatie"]) / out["Realisatie"] * 100
    ).round(1)
    return out


# ----------------------------------------------------------------------
# Pipeline-spam onderdrukken
# ----------------------------------------------------------------------

@contextmanager
def suppress_stdout():
    """Onderdruk print()-output tijdelijk — handig voor pipeline-runs."""
    saved = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = saved


# ----------------------------------------------------------------------
# Plot-defaults
# ----------------------------------------------------------------------

def setup_matplotlib():
    """Standaard matplotlib-instellingen voor alle notebooks."""
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
