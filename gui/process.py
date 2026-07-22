"""Lokaliseren en samenstellen van CLI-aanroepen.

De GUI voert de bestaande ``studentprognose``-console-script uit via een
subprocess. Deze module is pure logica (geen NiceGUI) zodat ze getest kan worden;
het live streamen naar de UI gebeurt in :mod:`gui.components.log_stream`.
"""

from __future__ import annotations

import os
import shutil
import sys


def locate_cli() -> str:
    """Vind het pad naar de ``studentprognose``-console-script.

    Zoekt eerst naast de draaiende Python-interpreter (de venv-``bin``/``Scripts``),
    zodat de GUI en de CLI gegarandeerd dezelfde omgeving delen; valt terug op de
    ``PATH``.

    Returns:
        Absoluut pad naar het uitvoerbare bestand.

    Raises:
        FileNotFoundError: Als de console-script nergens gevonden wordt.
    """
    bindir = os.path.dirname(sys.executable)
    for name in ("studentprognose", "studentprognose.exe"):
        candidate = os.path.join(bindir, name)
        if os.path.isfile(candidate):
            return candidate

    found = shutil.which("studentprognose")
    if found:
        return found

    raise FileNotFoundError(
        "De 'studentprognose'-CLI is niet gevonden. Installeer het pakket in "
        "dezelfde omgeving als de GUI (bijv. `uv sync --extra gui`)."
    )


#: GUI-labels → CLI-codes voor dataset en cohort.
DATASET_CODES = {"Individueel": "i", "Cumulatief": "c", "Beide": "b"}
COHORT_CODES = {"Eerstejaars": "f", "Hogerejaars": "h", "Volume": "v"}


def build_run_args(
    *,
    dataset: str | None = None,
    cohort: str | None = None,
    years: str = "",
    weeks: str = "",
    institutions: list[str] | None = None,
    skip_years: int = 0,
    noetl: bool = False,
    dashboard: bool = False,
    no_warnings: bool = False,
    yes: bool = True,
) -> list[str]:
    """Bouw de CLI-argumenten voor een voorspellingsrun.

    Jaar- en weekvelden worden op witruimte gesplitst en als losse tokens
    doorgegeven; de CLI zelf verwerkt bereiksyntaxis zoals ``10:20`` of ``10 : 20``.

    Args:
        dataset: GUI-label (Individueel/Cumulatief/Beide) of ``None`` (default b).
        cohort: GUI-label (Eerstejaars/Hogerejaars/Volume) of ``None``.
        years: Vrije tekst met jaartallen/bereiken (bijv. ``"2023 2024"``).
        weeks: Vrije tekst met weeknummers/bereiken (bijv. ``"1:38"``).
        institutions: Brincodes/instellingsnamen (leeg = alle).
        skip_years: Aantal jaren overslaan (backtesting); 0 = geen.
        noetl: Sla ETL + validatie over.
        dashboard: Genereer dashboards.
        no_warnings: Onderdruk UserWarnings.
        yes: Sla de interactieve validatieprompt over (default True voor de GUI,
            omdat een subprocess geen interactieve invoer heeft).

    Returns:
        De argumentenlijst ná ``studentprognose``.
    """
    args: list[str] = []

    if dataset and dataset in DATASET_CODES:
        args += ["-d", DATASET_CODES[dataset]]
    if cohort and cohort in COHORT_CODES:
        args += ["-sy", COHORT_CODES[cohort]]
    if years.strip():
        args += ["-y", *years.split()]
    if weeks.strip():
        args += ["-w", *weeks.split()]
    if institutions:
        args += ["--institution", *institutions]
    if skip_years and skip_years > 0:
        args += ["-sk", str(skip_years)]
    if noetl:
        args.append("--noetl")
    if dashboard:
        args.append("--dashboard")
    if no_warnings:
        args.append("--no-warnings")
    if yes:
        args.append("--yes")

    return args


def preview_command(args: list[str]) -> str:
    """Bouw een leesbare weergave van het commando voor de UI-preview.

    Args:
        args: Argumenten ná ``studentprognose`` (bijv. ``["-d", "c", "-w", "6"]``).

    Returns:
        Een string zoals ``studentprognose -d c -w 6``.
    """
    parts = ["studentprognose", *args]
    return " ".join(parts)
