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
    """Bereid de cumulatieve data voor zoals de pipeline dat doet.

    Gebruikt dezelfde functie als het cumulatieve spoor
    (``prepare_cumulative_data``): kolommen hernoemen, alleen eerstejaars, de
    programmasleutel als Isatcode (getal) en optellen over instellingen tot één
    rij per opleiding × herkomst × examentype × jaar × week. Daarna komt de
    kolom ``ts`` erbij (gewogen vooraanmelders + inschrijvingen), de reeks die
    SARIMA voorspelt.
    """
    from studentprognose.strategies.cumulative import prepare_cumulative_data

    df = prepare_cumulative_data(df.copy())
    df["ts"] = df["Gewogen vooraanmelders"].fillna(0) + df["Inschrijvingen"].fillna(0)
    return df.astype({"Weeknummer": "int32", "Collegejaar": "int32"})


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


INDIVIDUEEL_PADEN = (
    Path("data/input/vooraanmeldingen_individueel.csv"),
    Path("data/input_raw/individuele_aanmelddata.csv"),
)


def load_individueel(n_sample: int | None = 5000, seed: int = 42):
    """Laad (een sample van) de individuele aanmelddata.

    Zoekt eerst de ETL-output (``data/input/vooraanmeldingen_individueel.csv``)
    en anders de ruwe demodata die ``studentprognose init`` downloadt
    (``data/input_raw/individuele_aanmelddata.csv``).

    Args:
        n_sample: aantal rijen om te samplen (None = volledig bestand).
        seed: random_state voor reproduceerbaarheid.

    Returns:
        DataFrame met de individuele aanmeldingen, of ``None`` als geen van
        beide bestanden bestaat.
    """
    path = next((p for p in INDIVIDUEEL_PADEN if p.exists()), None)
    if path is None:
        return None
    df = pd.read_csv(path, sep=";", low_memory=False)
    if n_sample is not None and len(df) > n_sample:
        df = df.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    return df


def opleidingsnamen() -> dict[int, str]:
    """Opzoektabel Isatcode → officiële opleidingsnaam (meegeleverde DUO-tabel)."""
    from studentprognose.data.programme_names import load_programme_names

    return load_programme_names()


def label(code, namen: dict | None = None, max_len: int = 40) -> str:
    """Leesbaar label voor een opleiding: ``"56604 B Psychologie"``."""
    namen = opleidingsnamen() if namen is None else namen
    naam = namen.get(code, "") if not isinstance(code, str) else ""
    return f"{code} {naam}".strip()[:max_len]


def grootste_opleidingen(
    data_studentcount: pd.DataFrame, n: int = 15, examentype: str | None = None
) -> list:
    """De ``n`` opleidingen met de grootste instroom in het laatste jaar met realisatie.

    Handig om zware runs (het cumulatieve spoor) in een notebook te beperken.
    """
    sc = data_studentcount
    if examentype is not None:
        sc = sc[sc["Examentype"] == examentype]
    laatste = sc[sc["Collegejaar"] == sc["Collegejaar"].max()]
    totaal = laatste.groupby("Croho groepeernaam")["Aantal_studenten"].sum()
    return [int(c) for c in totaal.nlargest(n).index]


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
