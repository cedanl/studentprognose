"""Data-upload helpers voor de wizard: opslaan en valideren van inputbestanden.

De validatielogica weerspiegelt ``studentprognose/data/validation.py`` maar roept
nooit ``sys.exit()`` aan — in de GUI worden fouten teruggegeven als datastructuur.
"""

from __future__ import annotations

import copy
import datetime
import json
import os
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from studentprognose.data.validation import _DEFAULT_VALIDATION_CFG
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
    years: list[int] | None = None  # unieke jaren in het bestand (bijv. Collegejaar)
    actual_columns: list[str] = field(default_factory=list)   # kolommen in het bestand vóór mapping
    missing_required: list[str] = field(default_factory=list) # verwachte kolommen die ontbreken


def _build_gui_validation_cfg() -> dict:
    """Bouw de validatie-config van de upload-wizard.

    De **enige bron van waarheid** voor drempels en kolomlijsten is
    ``studentprognose.data.validation._DEFAULT_VALIDATION_CFG``; door daaruit af
    te leiden kunnen de GUI-drempels (NaN-drempels, jaar-offsets, weekbereik)
    niet stil uiteendrijven met de pipeline. Daarop passen we twee **bewuste**
    versoepelingen toe, omdat de wizard *ruwe* bestanden vóór de ETL valideert:

    * ``Groepeernaam`` is niet vereist — het UvA SQL (SL) formaat levert die niet
      en de ETL genereert de kolom zelf uit ``Isatcode`` (#232).
    * ``Herkomst`` mag ook ``"O"`` (onbekend) bevatten naast ``N``/``E``/``R``;
      de pipeline normaliseert dat verderop.

    Returns:
        Een diepe kopie van de canonieke config met de GUI-versoepelingen.
    """
    cfg = copy.deepcopy(_DEFAULT_VALIDATION_CFG)
    tel = cfg["telbestand"]
    tel["required_columns"] = [
        c for c in tel["required_columns"] if c != "Groepeernaam"
    ]
    if "O" not in tel["herkomst_allowed"]:
        tel["herkomst_allowed"] = [*tel["herkomst_allowed"], "O"]
    return cfg


#: Validatie-config afgeleid uit de canonieke pipeline-bron (zie
#: :func:`_build_gui_validation_cfg`).
_CFG: dict = _build_gui_validation_cfg()


@dataclass
class TelCoverage:
    """Tijdsdekkingsstatistieken afgeleid uit geüploade telbestanden."""

    years: list[int]
    present: dict[int, set[int]]   # year → set of week numbers
    gaps: list[tuple[int, int]]    # (year, week) within expected range but missing
    total: int                     # aantal geldige bestanden


@dataclass
class OverlapInfo:
    """Overlap tussen telbestand-jaren en oktober-jaren."""

    tel_years: list[int]
    okt_years: list[int]
    intersection: list[int]
    year_range: list[int]  # aaneengesloten reeks min..max over beide datasets


def compute_overlap(
    cov: TelCoverage | None,
    okt_result: FileCheckResult | None,
) -> OverlapInfo | None:
    """Bereken de jaar-overlap tussen telbestanden en het oktober-bestand.

    Geeft None terug als één van beide ontbreekt of ongeldig is.
    """
    if cov is None or okt_result is None:
        return None
    if okt_result.status not in (FileStatus.VALID, FileStatus.WARNINGS):
        return None
    if not okt_result.years:
        return None

    tel_set = set(cov.years)
    okt_set = set(okt_result.years)
    all_years = tel_set | okt_set
    if not all_years:
        return None

    return OverlapInfo(
        tel_years=sorted(tel_set),
        okt_years=sorted(okt_set),
        intersection=sorted(tel_set & okt_set),
        year_range=list(range(min(all_years), max(all_years) + 1)),
    )


@dataclass
class DataYearBounds:
    """Effectief traindata-bereik afgeleid uit de aanwezige projectbestanden.

    Attributes:
        train_start: Vroegste trainingsjaar — ``max(config-ondergrens,
            eerste overlap-jaar)``. Dit is het eerste jaar waarvoor zowel
            aanmeld- (telbestand) als realisatiedata (oktober) bestaat.
        train_end: Laatste jaar met realisatiedata (laatste overlap-jaar).
            Training kan niet verder reiken dan dit jaar.
        tel_years: Jaren waarvoor telbestanden aanwezig zijn.
        okt_years: Jaren in het oktober-bestand.
        overlap: Doorsnede van ``tel_years`` en ``okt_years`` (gesorteerd).
    """

    train_start: int
    train_end: int
    tel_years: list[int]
    okt_years: list[int]
    overlap: list[int]


def _config_min_training_year(project_dir: str) -> int | None:
    """Lees ``model_config.min_training_year`` uit de projectconfiguratie."""
    cfg_path = os.path.join(project_dir, "configuration", "configuration.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        val = cfg.get("model_config", {}).get("min_training_year")
        return int(val) if val is not None else None
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return None


def _telbestand_years(project_dir: str) -> set[int]:
    """Leid telbestand-jaren af uit bestandsnamen (leest geen CSV-inhoud)."""
    tel_dir = os.path.join(project_dir, "data", "input_raw", "telbestanden")
    if not os.path.isdir(tel_dir):
        return set()
    patterns = compile_patterns(None)
    years: set[int] = set()
    for fname in os.listdir(tel_dir):
        match = match_telbestand(fname, patterns)
        if match is None:
            continue
        try:
            years.add(int(match.group("year")))
        except (ValueError, IndexError):
            continue
    return years


def _oktober_years(project_dir: str) -> set[int]:
    """Leid de collegejaren uit het oktober-bestand af.

    Leest alleen de ``Collegejaar``-kolom (met kolomnaam-mapping toegepast) en
    filtert op een plausibel bereik. Bewust losgekoppeld van de volledige
    bestandsvalidatie: het jaarbereik is ook bruikbaar wanneer andere vereiste
    kolommen (nog) ontbreken of anders heten.
    """
    okt_path = os.path.join(project_dir, "data", "input_raw", "oktober_bestand.xlsx")
    if not os.path.isfile(okt_path):
        return set()
    try:
        df = pd.read_excel(okt_path)
    except (OSError, ValueError, KeyError):
        return set()

    column_map = load_project_col_map(project_dir, "oktober")
    if column_map:
        rename_map = {inst: canon for canon, inst in column_map.items() if inst != canon}
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

    if "Collegejaar" not in df.columns:
        return set()

    current_year = datetime.date.today().year
    y_min = current_year - _CFG["collegejaar_min_offset"]
    y_max = current_year + _CFG["collegejaar_max_offset"]
    collegejaar = pd.to_numeric(df["Collegejaar"], errors="coerce")
    return {
        int(y) for y in collegejaar[collegejaar.between(y_min, y_max)].dropna().unique()
    }


def scan_data_year_bounds(project_dir: str) -> DataYearBounds | None:
    """Bepaal het effectieve traindata-bereik uit de aanwezige projectbestanden.

    Telbestand-jaren komen (goedkoop) uit de bestandsnamen; oktober-jaren uit
    het gevalideerde Excel-bestand. Het traindata-bereik is de doorsnede van
    beide, met de config-ondergrens (``min_training_year``) als vloer voor het
    startjaar.

    Returns:
        ``DataYearBounds`` als beide datasets aanwezig zijn en overlappen,
        anders ``None`` (bijv. wanneer de data nog niet is geüpload).
    """
    tel_years = _telbestand_years(project_dir)
    if not tel_years:
        return None

    okt_years = _oktober_years(project_dir)
    if not okt_years:
        return None

    overlap = sorted(tel_years & okt_years)
    if not overlap:
        return None

    floor = _config_min_training_year(project_dir)
    train_start = max(overlap[0], floor) if floor is not None else overlap[0]
    train_end = overlap[-1]
    if train_start > train_end:
        # Config-vloer ligt voorbij alle overlap-jaren — geen bruikbaar bereik.
        return None

    return DataYearBounds(
        train_start=train_start,
        train_end=train_end,
        tel_years=sorted(tel_years),
        okt_years=sorted(okt_years),
        overlap=overlap,
    )


def selectable_exclusion_years(bounds: DataYearBounds | None) -> list[int]:
    """Jaren die als uitsluitingsregel gekozen mogen worden.

    Alleen jaren in de overlap tussen telbestanden en het oktober-bestand
    komen in aanmerking, begrensd door de config-ondergrens
    (``train_start``): dat zijn de enige jaren die daadwerkelijk in de
    trainingsdata terechtkomen en dus zinvol zijn om uit te sluiten. Een jaar
    zonder overlap (of onder de ondergrens) zit sowieso niet in de training —
    het als uitsluiting aanbieden zou misleidend zijn.

    Args:
        bounds: Het gescande traindata-bereik, of ``None`` wanneer de data nog
            niet (volledig) is geüpload.

    Returns:
        Oplopend gesorteerde lijst van kiesbare jaren. Leeg wanneer er geen
        bruikbaar traindata-bereik bekend is.
    """
    if bounds is None:
        return []
    return [y for y in bounds.overlap if bounds.train_start <= y <= bounds.train_end]


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


def _sniff_separator(filepath: str) -> str:
    """Kies het scheidingsteken dat de meeste kolommen oplevert.

    csv.Sniffer faalt als data-velden het kandidaat-scheidingsteken bevatten
    (bijv. puntkomma's in tekst terwijl de echte separator een tab is). We
    proberen alle drie kandidaten en kiezen degene die de breedste tabel geeft.
    """
    required = set(_CFG["telbestand"]["required_columns"])
    best_sep = ";"
    best_score: tuple[int, int] = (-1, -1)
    for sep in (";", ",", "\t"):
        try:
            with open(filepath, encoding="utf-8", errors="replace") as fh:
                header = fh.readline()
        except OSError:
            continue
        cols = {c.strip().strip('"').strip("'") for c in header.split(sep)}
        # Primair: meeste vereiste kolommen; secundair: meeste kolommen totaal
        score = (len(cols & required), len(cols))
        if score > best_score:
            best_score = score
            best_sep = sep
    return best_sep


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

def safe_telbestand_name(filename: str) -> str:
    """Normaliseer een geüploade telbestand-naam tot een veilige basisnaam.

    Verwijdert padcomponenten (``os.path.basename``) zodat een naam als
    ``../../etc/passwd`` nooit buiten de ``telbestanden``-map kan schrijven of
    verwijderen (padtraversal). Backslashes worden ook als scheidingsteken
    behandeld zodat Windows-paden op een POSIX-server niet doorlekken. De naam
    wordt daarna lowercase gemaakt — de conventie waarmee bestanden worden
    opgeslagen.

    Args:
        filename: De door de client aangeleverde bestandsnaam.

    Returns:
        Een pad-loze, lowercase bestandsnaam.

    Raises:
        ValueError: Als er na normalisatie geen geldige naam overblijft (leeg,
            of ``.``/``..``).
    """
    base = os.path.basename(filename.replace("\\", "/")).lower().strip()
    if not base or base in {".", ".."}:
        raise ValueError(f"Ongeldige bestandsnaam: {filename!r}")
    return base


def load_project_col_map(project_dir: str, key: str) -> dict[str, str]:
    """Lees de kolomnaam-mapping voor *key* uit de projectconfiguratie.

    Geeft een lege dict terug als het bestand niet bestaat of niet leesbaar is.
    """
    cfg_path = os.path.join(project_dir, "configuration", "configuration.json")
    if not os.path.isfile(cfg_path):
        return {}
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        return dict(cfg.get("columns", {}).get(key, {}))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def save_project_col_map(project_dir: str, key: str, mapping: dict[str, str]) -> None:
    """Sla de kolomnaam-mapping voor *key* op in configuration.json van het project.

    Overschrijft alleen het relevante sub-sleutel; andere config blijft intact.
    """
    cfg_path = os.path.join(project_dir, "configuration", "configuration.json")
    if not os.path.isfile(cfg_path):
        return
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.setdefault("columns", {})[key] = mapping
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)
            f.write("\n")
    except (OSError, json.JSONDecodeError):
        pass


def save_and_validate_telbestand(
    project_dir: str, filename: str, content: bytes
) -> FileCheckResult:
    """Sla een telbestand op in data/input_raw/telbestanden/ en valideer het."""
    dest_dir = os.path.join(project_dir, "data", "input_raw", "telbestanden")
    os.makedirs(dest_dir, exist_ok=True)
    filename = safe_telbestand_name(filename)
    filepath = os.path.join(dest_dir, filename)
    with open(filepath, "wb") as f:
        f.write(content)
    column_map = load_project_col_map(project_dir, "telbestand")
    return _check_telbestand(filepath, filename, column_map)


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
    column_map = load_project_col_map(project_dir, "oktober")
    return _check_oktober(dest, column_map)


def revalidate_telbestand(project_dir: str, filename: str) -> FileCheckResult:
    """Hervalideer een bestaand telbestand op schijf met de huidige kolomnaam-mapping."""
    saved_filename = safe_telbestand_name(filename)  # zoals opgeslagen (zie save_and_validate_telbestand)
    filepath = os.path.join(project_dir, "data", "input_raw", "telbestanden", saved_filename)
    column_map = load_project_col_map(project_dir, "telbestand")
    return _check_telbestand(filepath, filename, column_map)


def revalidate_oktober(project_dir: str) -> FileCheckResult:
    """Hervalideer het bestaande oktober-bestand op schijf met de huidige kolomnaam-mapping."""
    filepath = os.path.join(project_dir, "data", "input_raw", "oktober_bestand.xlsx")
    column_map = load_project_col_map(project_dir, "oktober")
    return _check_oktober(filepath, column_map)


def delete_telbestand(project_dir: str, filename: str) -> None:
    """Verwijder een telbestand van schijf."""
    try:
        saved_filename = safe_telbestand_name(filename)
    except ValueError:
        return
    path = os.path.join(project_dir, "data", "input_raw", "telbestanden", saved_filename)
    if os.path.isfile(path):
        os.remove(path)


def delete_individueel(project_dir: str, _filename: str) -> None:
    """Verwijder het individuele aanmeldbestand van schijf."""
    path = os.path.join(project_dir, "data", "input_raw", "individuele_aanmelddata.csv")
    if os.path.isfile(path):
        os.remove(path)


def delete_oktober(project_dir: str, _filename: str) -> None:
    """Verwijder het oktober-bestand van schijf."""
    path = os.path.join(project_dir, "data", "input_raw", "oktober_bestand.xlsx")
    if os.path.isfile(path):
        os.remove(path)


def scan_existing_files(project_dir: str) -> dict:
    """Controleer en valideer bestanden die al aanwezig zijn in het project.

    Returns:
        Dict met sleutels ``"telbestanden"`` (dict filename → result),
        ``"individueel"`` (result of None), ``"oktober"`` (result of None).
    """
    out: dict = {"telbestanden": {}, "individueel": None, "oktober": None}

    tel_col_map = load_project_col_map(project_dir, "telbestand")
    okt_col_map = load_project_col_map(project_dir, "oktober")

    tel_dir = os.path.join(project_dir, "data", "input_raw", "telbestanden")
    if os.path.isdir(tel_dir):
        patterns = compile_patterns(None)
        for fname in sorted(os.listdir(tel_dir)):
            if match_telbestand(fname, patterns):
                out["telbestanden"][fname] = _check_telbestand(
                    os.path.join(tel_dir, fname), fname, tel_col_map
                )

    ind = os.path.join(project_dir, "data", "input_raw", "individuele_aanmelddata.csv")
    if os.path.isfile(ind):
        out["individueel"] = _check_individueel(ind)

    okt = os.path.join(project_dir, "data", "input_raw", "oktober_bestand.xlsx")
    if os.path.isfile(okt):
        out["oktober"] = _check_oktober(okt, okt_col_map)

    return out


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _check_telbestand(
    filepath: str,
    filename: str,
    column_map: dict[str, str] | None = None,
) -> FileCheckResult:
    hard: list[str] = []
    soft: list[str] = []
    warn: list[str] = []

    patterns = compile_patterns(None)
    fname = os.path.basename(filepath)
    match = match_telbestand(fname, patterns)
    if not match:
        hard.append(
            "Bestandsnaam past niet op een herkend Studielink-patroon. "
            "Verwacht bijv. 'telbestandY2024W10.csv', "
            "'telbestand_sl_20241007_v01_2024.csv' of "
            "'Telbestand_SL_2020_V96_20210802.csv'. "
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
    sep = _sniff_separator(filepath)
    try:
        df = pd.read_csv(filepath, sep=sep, low_memory=False)
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        hard.append(
            f"Bestand kan niet worden gelezen: {exc}. "
            "Controleer of het een geldige CSV is (puntkomma- of kommagescheiden)."
        )
        return FileCheckResult(filename=filename, status=FileStatus.ERRORS, hard_errors=hard)

    actual_columns = list(df.columns)

    # Pas kolomnaam-mapping toe: institutienaam → canonieke naam
    if column_map:
        rename_map = {inst: canon for canon, inst in column_map.items() if inst != canon}
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

    missing = [c for c in tel["required_columns"] if c not in df.columns]
    if missing:
        hard.append(
            f"Vereiste kolommen ontbreken: {', '.join(missing)}. "
            f"Herkend scheidingsteken: '{sep}'. "
            "Controleer of dit een Studielink-export is of koppel de kolomnamen hieronder."
        )
        return FileCheckResult(
            filename=filename,
            status=FileStatus.ERRORS,
            hard_errors=hard,
            actual_columns=actual_columns,
            missing_required=missing,
        )

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
        actual_columns=actual_columns,
    )


def _check_individueel(filepath: str) -> FileCheckResult:
    hard: list[str] = []
    soft: list[str] = []
    warn: list[str] = []
    filename = "individuele_aanmelddata.csv"

    try:
        df = pd.read_csv(filepath, sep=";", low_memory=False)
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
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


def _check_oktober(
    filepath: str,
    column_map: dict[str, str] | None = None,
) -> FileCheckResult:
    hard: list[str] = []
    soft: list[str] = []
    warn: list[str] = []
    filename = "oktober_bestand.xlsx"

    try:
        df = pd.read_excel(filepath)
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        hard.append(
            f"Bestand kan niet worden gelezen: {exc}. "
            "Controleer of het een geldig Excel-bestand (.xlsx) is."
        )
        return FileCheckResult(filename=filename, status=FileStatus.ERRORS, hard_errors=hard)

    actual_columns = list(df.columns)

    # Pas kolomnaam-mapping toe: institutienaam → canonieke naam
    if column_map:
        rename_map = {inst: canon for canon, inst in column_map.items() if inst != canon}
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

    critical = _CFG["oktober"]["critical_columns"]
    missing = [c for c in critical if c not in df.columns]
    if missing:
        hard.append(
            f"Vereiste kolommen ontbreken: {', '.join(missing)}. "
            "Koppel de kolomnamen hieronder of pas 'columns.oktober' aan in configuration.json."
        )
        return FileCheckResult(
            filename=filename,
            status=FileStatus.ERRORS,
            hard_errors=hard,
            actual_columns=actual_columns,
            missing_required=missing,
        )

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

    valid_years = sorted(
        int(y) for y in collegejaar[collegejaar.between(y_min, y_max)].dropna().unique()
    )

    return FileCheckResult(
        filename=filename,
        status=_to_status(hard, soft, warn),
        hard_errors=hard,
        soft_errors=soft,
        warnings=warn,
        row_count=len(df),
        years=valid_years,
        actual_columns=actual_columns,
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
