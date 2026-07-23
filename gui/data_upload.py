"""Data-upload helpers voor de wizard: opslaan en valideren van inputbestanden.

De validatielogica weerspiegelt ``studentprognose/data/validation.py`` maar roept
nooit ``sys.exit()`` aan — in de GUI worden fouten teruggegeven als datastructuur.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from studentprognose.utils.telbestand_filenames import (
    compile_patterns,
    match_telbestand,
    week_from_match,
)


class FileStatus(Enum):
    CHECKING = "checking"
    VALID = "valid"
    WARNINGS = "warnings"
    ERRORS = "errors"


@dataclass
class FileCheckResult:
    """Resultaat van de kwaliteitscontrole van één bestand."""

    filename: str
    status: FileStatus
    hard_errors: list[str] = field(default_factory=list)
    soft_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int | None = None


# Validatie-defaults gespiegeld van validation.py — geen import om
# circulaire afhankelijkheden en sys.exit()-aanroepen te vermijden.
_CFG: dict = {
    "collegejaar_min_offset": 15,
    "collegejaar_max_offset": 2,
    "weeknummer_min": 1,
    "weeknummer_max": 53,
    "nan_warning_threshold": 0.05,
    "nan_error_threshold": 0.30,
    "telbestand": {
        "required_columns": [
            "Studiejaar", "Isatcode", "Groepeernaam", "Aantal", "meercode_V",
            "Status", "Herinschrijving", "Hogerejaars", "Herkomst",
        ],
        "herkomst_allowed": ["N", "E", "R"],
        "herinschrijving_allowed": ["J", "N"],
        "hogerejaars_allowed": ["J", "N"],
        "separator": ";",
    },
    "individueel": {
        "critical_columns": [
            "Collegejaar", "Croho", "Inschrijfstatus", "Datum Verzoek Inschr",
        ],
    },
    "oktober": {
        "critical_columns": [
            "Collegejaar", "Isatcode", "Aantal eerstejaars croho",
            "EER-NL-nietEER", "Examentype code", "Aantal Hoofdinschrijvingen",
        ],
    },
}


@dataclass
class TelCoverage:
    """Tijdsdekkingsstatistieken afgeleid uit geüploade telbestanden."""

    years: list[int]
    present: dict[int, set[int]]   # year → set of week numbers
    gaps: list[tuple[int, int]]    # (year, week) within expected range but missing
    total: int                     # aantal geldige bestanden


def compute_tel_coverage(results: dict[str, FileCheckResult]) -> TelCoverage | None:
    """Leid week/jaar-dekking af uit de verzameling telbestand-resultaten.

    Gaten worden bepaald als weken die ontbreken binnen het aaneengesloten
    bereik [eerste week van eerste jaar … laatste week van laatste jaar].
    Geeft None terug als er geen geldige bestanden zijn.
    """
    patterns = compile_patterns(None)
    present: dict[int, set[int]] = {}

    for filename, result in results.items():
        if result.status not in (FileStatus.VALID, FileStatus.WARNINGS):
            continue
        match = match_telbestand(filename, patterns)
        if match is None:
            continue
        try:
            week = week_from_match(match)
            year = int(match.group("year"))
        except (ValueError, IndexError):
            continue
        present.setdefault(year, set()).add(week)

    if not present:
        return None

    years = sorted(present.keys())
    min_year, max_year = years[0], years[-1]
    gaps: list[tuple[int, int]] = []

    for year in range(min_year, max_year + 1):
        year_weeks = present.get(year, set())
        if year == min_year == max_year:
            w_start = min(year_weeks) if year_weeks else 1
            w_end = max(year_weeks) if year_weeks else 1
        elif year == min_year:
            w_start = min(present[min_year]) if present.get(min_year) else 1
            w_end = 52
        elif year == max_year:
            w_start = 1
            w_end = max(present[max_year]) if present.get(max_year) else 52
        else:
            w_start, w_end = 1, 52
        for w in range(w_start, w_end + 1):
            if w not in year_weeks:
                gaps.append((year, w))

    return TelCoverage(
        years=years,
        present=present,
        gaps=gaps,
        total=sum(len(ws) for ws in present.values()),
    )


def _to_status(hard: list, soft: list, warnings: list) -> FileStatus:
    if hard or soft:
        return FileStatus.ERRORS
    if warnings:
        return FileStatus.WARNINGS
    return FileStatus.VALID


# ---------------------------------------------------------------------------
# Public API — alle functies nemen (project_dir, original_filename, content)
# zodat _UploadZone een uniforme aanroepconventie kan gebruiken.
# ---------------------------------------------------------------------------

def save_and_validate_telbestand(
    project_dir: str, filename: str, content: bytes
) -> FileCheckResult:
    """Sla een telbestand op in data/input_raw/telbestanden/ en valideer het."""
    dest_dir = os.path.join(project_dir, "data", "input_raw", "telbestanden")
    os.makedirs(dest_dir, exist_ok=True)
    filepath = os.path.join(dest_dir, filename)
    with open(filepath, "wb") as f:
        f.write(content)
    return _check_telbestand(filepath, filename)


def save_and_validate_individueel(
    project_dir: str, _original_filename: str, content: bytes
) -> FileCheckResult:
    """Sla individuele aanmelddata op (vaste naam) en valideer het."""
    dest = os.path.join(project_dir, "data", "input_raw", "individuele_aanmelddata.csv")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(content)
    return _check_individueel(dest)


def save_and_validate_oktober(
    project_dir: str, _original_filename: str, content: bytes
) -> FileCheckResult:
    """Sla het oktober-bestand op (vaste naam) en valideer het."""
    dest = os.path.join(project_dir, "data", "input_raw", "oktober_bestand.xlsx")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(content)
    return _check_oktober(dest)


def scan_existing_files(project_dir: str) -> dict:
    """Controleer en valideer bestanden die al aanwezig zijn in het project.

    Returns:
        Dict met sleutels ``"telbestanden"`` (dict filename → result),
        ``"individueel"`` (result of None), ``"oktober"`` (result of None).
    """
    out: dict = {"telbestanden": {}, "individueel": None, "oktober": None}

    tel_dir = os.path.join(project_dir, "data", "input_raw", "telbestanden")
    if os.path.isdir(tel_dir):
        patterns = compile_patterns(None)
        for fname in sorted(os.listdir(tel_dir)):
            if match_telbestand(fname, patterns):
                out["telbestanden"][fname] = _check_telbestand(
                    os.path.join(tel_dir, fname), fname
                )

    ind = os.path.join(project_dir, "data", "input_raw", "individuele_aanmelddata.csv")
    if os.path.isfile(ind):
        out["individueel"] = _check_individueel(ind)

    okt = os.path.join(project_dir, "data", "input_raw", "oktober_bestand.xlsx")
    if os.path.isfile(okt):
        out["oktober"] = _check_oktober(okt)

    return out


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _check_telbestand(filepath: str, filename: str) -> FileCheckResult:
    hard: list[str] = []
    soft: list[str] = []
    warn: list[str] = []

    patterns = compile_patterns(None)
    fname = os.path.basename(filepath)
    match = match_telbestand(fname, patterns)
    if not match:
        hard.append(
            "Bestandsnaam past niet op een herkend Studielink-patroon. "
            "Verwacht bijv. 'telbestandY2024W10.csv' of "
            "'telbestand_sl_20241007_v01_2024.csv'. "
            "Pas het patroon aan via 'telbestand_filename_patterns' in configuration.json "
            "als jouw instelling een andere naamgeving gebruikt."
        )
        return FileCheckResult(filename=filename, status=FileStatus.ERRORS, hard_errors=hard)

    week_nr = week_from_match(match)
    w_min, w_max = _CFG["weeknummer_min"], _CFG["weeknummer_max"]
    if not (w_min <= week_nr <= w_max):
        hard.append(
            f"Weeknummer {week_nr} valt buiten het verwachte bereik [{w_min}–{w_max}]."
        )

    tel = _CFG["telbestand"]
    try:
        df = pd.read_csv(filepath, sep=tel["separator"], low_memory=False)
    except Exception as exc:
        hard.append(
            f"Bestand kan niet worden gelezen: {exc}. "
            "Controleer of het een geldige CSV is met puntkomma (;) als scheidingsteken."
        )
        return FileCheckResult(filename=filename, status=FileStatus.ERRORS, hard_errors=hard)

    missing = [c for c in tel["required_columns"] if c not in df.columns]
    if missing:
        hard.append(
            f"Vereiste kolommen ontbreken: {', '.join(missing)}. "
            "Controleer het scheidingsteken (verwacht: ';') en of dit een Studielink-export is."
        )
        return FileCheckResult(filename=filename, status=FileStatus.ERRORS, hard_errors=hard)

    current_year = datetime.date.today().year
    y_min = current_year - _CFG["collegejaar_min_offset"]
    y_max = current_year + _CFG["collegejaar_max_offset"]
    studiejaar = pd.to_numeric(df["Studiejaar"], errors="coerce")
    invalid_yrs = sorted(
        int(y) for y in studiejaar[~studiejaar.between(y_min, y_max)].dropna().unique()
    )
    if invalid_yrs:
        soft.append(
            f"Studiejaar bevat waarden buiten verwacht bereik {y_min}–{y_max}: {invalid_yrs}. "
            "Controleer of dit de juiste bestanden zijn."
        )

    for col, allowed in [
        ("Herinschrijving", tel["herinschrijving_allowed"]),
        ("Hogerejaars", tel["hogerejaars_allowed"]),
        ("Herkomst", tel["herkomst_allowed"]),
    ]:
        series = df[col].astype(str).str.strip()
        invalid_vals = sorted(
            v for v in series[
                ~series.isin(allowed) & series.notna() & (series != "nan") & (series != "")
            ].unique()
        )
        if invalid_vals:
            soft.append(
                f"Kolom '{col}' bevat ongeldige waarden: {invalid_vals} "
                f"(toegestaan: {allowed}). "
                "De pipeline probeert ze automatisch te normaliseren."
            )

    neg_count = int((pd.to_numeric(df["Aantal"], errors="coerce") < 0).sum())
    if neg_count:
        soft.append(
            f"Kolom 'Aantal' bevat {neg_count} rij(en) met negatieve waarden. "
            "Dit kan duiden op gecorrigeerde inschrijvingen."
        )

    for col in ["Aantal", "meercode_V"]:
        _append_nan_messages(df, col, filename, warn, soft)

    return FileCheckResult(
        filename=filename,
        status=_to_status(hard, soft, warn),
        hard_errors=hard,
        soft_errors=soft,
        warnings=warn,
        row_count=len(df),
    )


def _check_individueel(filepath: str) -> FileCheckResult:
    hard: list[str] = []
    soft: list[str] = []
    warn: list[str] = []
    filename = "individuele_aanmelddata.csv"

    try:
        df = pd.read_csv(filepath, sep=";", low_memory=False)
    except Exception as exc:
        hard.append(
            f"Bestand kan niet worden gelezen: {exc}. "
            "Controleer of het een geldige CSV is met puntkomma (;) als scheidingsteken."
        )
        return FileCheckResult(filename=filename, status=FileStatus.ERRORS, hard_errors=hard)

    critical = _CFG["individueel"]["critical_columns"]
    missing = [c for c in critical if c not in df.columns]
    if missing:
        hard.append(
            f"Vereiste kolommen ontbreken: {', '.join(missing)}. "
            "Pas de kolomnamen aan via 'columns.individual' in configuration.json "
            "als jouw instelling andere namen gebruikt."
        )
        return FileCheckResult(filename=filename, status=FileStatus.ERRORS, hard_errors=hard)

    for col in critical:
        _append_nan_messages(df, col, filename, warn, soft)

    return FileCheckResult(
        filename=filename,
        status=_to_status(hard, soft, warn),
        hard_errors=hard,
        soft_errors=soft,
        warnings=warn,
        row_count=len(df),
    )


def _check_oktober(filepath: str) -> FileCheckResult:
    hard: list[str] = []
    soft: list[str] = []
    warn: list[str] = []
    filename = "oktober_bestand.xlsx"

    try:
        df = pd.read_excel(filepath)
    except Exception as exc:
        hard.append(
            f"Bestand kan niet worden gelezen: {exc}. "
            "Controleer of het een geldig Excel-bestand (.xlsx) is."
        )
        return FileCheckResult(filename=filename, status=FileStatus.ERRORS, hard_errors=hard)

    critical = _CFG["oktober"]["critical_columns"]
    missing = [c for c in critical if c not in df.columns]
    if missing:
        hard.append(
            f"Vereiste kolommen ontbreken: {', '.join(missing)}. "
            "Pas de kolomnamen aan via 'columns.oktober' in configuration.json "
            "als jouw instelling andere namen gebruikt."
        )
        return FileCheckResult(filename=filename, status=FileStatus.ERRORS, hard_errors=hard)

    current_year = datetime.date.today().year
    y_min = current_year - _CFG["collegejaar_min_offset"]
    y_max = current_year + _CFG["collegejaar_max_offset"]
    collegejaar = pd.to_numeric(df["Collegejaar"], errors="coerce")
    invalid_yrs = sorted(
        int(y) for y in collegejaar[~collegejaar.between(y_min, y_max)].dropna().unique()
    )
    if invalid_yrs:
        soft.append(
            f"Collegejaar bevat waarden buiten verwacht bereik {y_min}–{y_max}: {invalid_yrs}."
        )

    for col in critical:
        if col in df.columns:
            _append_nan_messages(df, col, filename, warn, soft)

    return FileCheckResult(
        filename=filename,
        status=_to_status(hard, soft, warn),
        hard_errors=hard,
        soft_errors=soft,
        warnings=warn,
        row_count=len(df),
    )


def _append_nan_messages(
    df: pd.DataFrame,
    col: str,
    filename: str,
    warn: list[str],
    soft: list[str],
) -> None:
    if col not in df.columns or len(df) == 0:
        return
    rate = df[col].isna().mean()
    pct = f"{rate:.0%}"
    if rate >= _CFG["nan_error_threshold"]:
        soft.append(
            f"Kolom '{col}' heeft {pct} ontbrekende waarden "
            f"(drempel: {_CFG['nan_error_threshold']:.0%}). "
            "Prognoses voor betrokken opleidingen kunnen onbetrouwbaar zijn."
        )
    elif rate >= _CFG["nan_warning_threshold"]:
        warn.append(f"Kolom '{col}' heeft {pct} ontbrekende waarden.")
