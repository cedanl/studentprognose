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
DATASET_CODES = {"Individueel": "i", "Cumulatief": "c", "Beide": "b", "CWF": "cwf"}
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
    dashboard: bool = True,
    no_warnings: bool = False,
    yes: bool = True,
) -> list[str]:
    """Bouw de CLI-argumenten voor een voorspellingsrun.

    Jaar- en weekvelden worden op witruimte gesplitst en als losse tokens
    doorgegeven; de CLI zelf verwerkt bereiksyntaxis zoals ``10:20`` of ``10 : 20``.

    Args:
        dataset: GUI-label (Individueel/Cumulatief/Beide/CWF) of ``None`` (default b).
        cohort: GUI-label (Eerstejaars/Hogerejaars/Volume) of ``None``.
        years: Vrije tekst met jaartallen/bereiken (bijv. ``"2023 2024"``).
        weeks: Vrije tekst met weeknummers/bereiken (bijv. ``"1:38"``).
        institutions: Brincodes/instellingsnamen (leeg = alle).
        skip_years: Aantal jaren overslaan (backtesting); 0 = geen.
        noetl: Sla ETL + validatie over.
        dashboard: Genereer dashboards (default True voor de GUI, zodat het
            dashboard op Resultaten altijd bij de laatste run hoort).
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


#: Config-sleutel per dataset met de eerdere-runsdata die ``-sy h`` nodig heeft.
#: ``Beide`` heeft geen eigen bron en kan daarom geen hogerejaars voorspellen.
_HIGHER_YEARS_SOURCE_KEYS = {
    "Cumulatief": "path_latest_cumulative",
    "Individueel": "path_latest_individual",
    "CWF": "path_latest_cwf",
}


def higher_years_source(dataset: str | None, config: dict | None) -> str | None:
    """Geef het (relatieve) pad dat een hogerejaarsrun voor ``dataset`` inleest.

    Args:
        dataset: GUI-label van de dataset.
        config: Geladen projectconfiguratie (mag ``None`` zijn).

    Returns:
        Het pad uit ``paths.path_latest_*``, of ``None`` als de dataset geen
        hogerejaarsbron heeft (``Beide``) of het pad niet is ingesteld.
    """
    key = _HIGHER_YEARS_SOURCE_KEYS.get(dataset or "")
    if key is None:
        return None
    path = ((config or {}).get("paths") or {}).get(key) or None
    return path


def run_conflicts(
    *,
    dataset: str | None,
    cohort: str | None,
    skip_years: int,
    project_dir: str | None,
    config: dict | None,
) -> list[str]:
    """Vind parametercombinaties die de CLI gegarandeerd zou weigeren.

    Zo krijgt de gebruiker de uitleg vóór de run, in plaats van een mislukte
    run na minuten wachten (#322).

    Args:
        dataset: GUI-label van de dataset (Cumulatief/Individueel/Beide/CWF).
        cohort: GUI-label van het cohort (Eerstejaars/Hogerejaars/Volume).
        skip_years: Aantal jaren overslaan (backtesting).
        project_dir: Projectmap (voor het controleren van invoerbestanden).
        config: Geladen projectconfiguratie.

    Returns:
        Leesbare foutmeldingen; leeg als de combinatie uitvoerbaar is.
    """
    errors: list[str] = []
    if dataset == "CWF" and skip_years > 0:
        errors.append(
            "Het CWF-model ondersteunt geen backtest: zet 'Jaren overslaan' op 0."
        )
    if cohort == "Hogerejaars":
        source = higher_years_source(dataset, config)
        if source is None:
            errors.append(
                f"Hogerejaars voorspellen kan niet met dataset '{dataset}'. "
                "Kies Cumulatief, Individueel of CWF."
            )
        else:
            full = (
                source
                if os.path.isabs(source)
                else os.path.join(project_dir or "", source)
            )
            if not os.path.isfile(full):
                errors.append(
                    "Hogerejaars voorspellen kan in dit project nog niet: daarvoor "
                    "zijn eerdere prognoses mét de gerealiseerde aantallen nodig in "
                    f"'{source}', en dat bestand ontbreekt. Een gewone run maakt "
                    "het niet aan. Zie in de documentatie 'Draaien & CLI' → "
                    "Hogerejaars hoe je het maakt; tot die tijd voorspel je "
                    "Eerstejaars of Volume."
                )
    return errors


#: Markeringen in de CLI-uitvoer van problemen die zonder ``--yes`` de run
#: hadden gestopt (hard stop) of een vraag hadden gesteld (soft-validatie).
#: De hard stop komt als Python-warning binnen ("...: UserWarning: [--yes] ...");
#: op die volledige vorm matchen voorkomt dat de bronregel die Python eronder
#: print (``warnings.warn(f"[--yes] {msg}", ...)``) als melding meetelt.
_OVERRIDE_MARKERS = ("UserWarning: [--yes] ", "[PROBLEEM] ")


def overridden_problems(lines: list[str]) -> list[str]:
    """Vind validatieproblemen die de GUI via ``--yes`` heeft overgeslagen.

    De GUI draait altijd met ``--yes`` (er is geen interactieve invoer). Een
    hard stop of validatievraag wordt dan een regel in de log. Deze functie
    haalt die regels eruit, zodat de GUI ze na een geslaagde run zichtbaar kan
    maken in plaats van stil "Geslaagd" te tonen (#322).

    Args:
        lines: Uitvoerregels van de CLI (stdout + stderr).

    Returns:
        De meldingen zonder markering, uniek en in volgorde van voorkomen.
    """
    found: list[str] = []
    for line in lines:
        for marker in _OVERRIDE_MARKERS:
            idx = line.find(marker)
            if idx != -1:
                msg = line[idx + len(marker):].strip()
                if msg and msg not in found:
                    found.append(msg)
                break
    return found
