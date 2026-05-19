"""Pre-ETL datakwaliteitscontrole.

Dit module valideert alle ruwe inputbestanden voordat de ETL-pipeline start.
Ongeldige data wordt nooit stilzwijgend omgezet — de gebruiker wordt altijd
gewaarschuwd of de pipeline stopt.

Gedrag per type bevinding
--------------------------
hard error  → pipeline stopt direct, los op en probeer opnieuw
soft error  → pre-flight prompt; gebruik ``--yes`` om door te gaan (CI/CD)
warning     → gelogd, pipeline loopt door

Tolerantie voor institutie-specifieke data
------------------------------------------
Elke instelling kan kolomnamen en lichte type-afwijkingen hebben.
Dit module probeert data te normaliseren voordat het valideert:

- **Kolomnamen** voor ``individuele_aanmelddata`` en het telbestand studenten
  (``oktober_bestand.xlsx``) worden opgezocht via
  ``configuration["columns"]["individual"]`` en
  ``configuration["columns"]["oktober"]``. Pas die mapping aan in
  ``configuration.json`` als jouw instelling andere namen gebruikt.
  Telbestanden komen van Studielink en zijn gestandaardiseerd; die
  kolomnamen zijn vast.

- **Witruimte** in categorische waarden (``"J "`` → ``"J"``) wordt
  automatisch gestript. Een waarschuwing verschijnt als normalisatie
  nodig was.

- **Numerieke strings** (``"2024"`` of ``"2.024"``) worden gecoerceerd
  naar getallen voor bereikcontroles. Een waarschuwing verschijnt als
  coercering nodig was. Dit is consistent met hoe de ETL
  (``_convert_to_float``) dezelfde variaties al afhandelt.

Configuratie
------------
``_DEFAULT_VALIDATION_CFG`` is de **enige bron van waarheid** voor
validatiedefaults. De waarden staan hier in de code, niet in
``configuration.json``, om dubbele definities te vermijden.

Wil je een instelling overschrijven? Voeg een ``"validation"``-blok toe
aan je ``configuration.json``. Je hoeft alleen de afwijkende waarden op
te nemen:

.. code-block:: json

    {
        "validation": {
            "nan_error_threshold": 0.20,
            "telbestand": {
                "herkomst_allowed": ["N", "E", "R", "ONBEKEND"]
            }
        }
    }

Een nieuw validatiecheck toevoegen
-----------------------------------
1. Voeg de configuratiesleutel toe aan ``_DEFAULT_VALIDATION_CFG``
   (en eventueel als override-voorbeeld in de README).
2. Schrijf een helperfunctie ``_check_*`` die een ``ValidationResult``
   als parameter accepteert en fouten/waarschuwingen appended.
3. Roep de functie aan vanuit de betreffende ``_validate_*``-functie.
4. Schrijf een test in ``tests/studentprognose/test_validation.py``.
"""

import datetime
import os
import re
import sys
from dataclasses import dataclass, field

import pandas as pd


# ---------------------------------------------------------------------------
# ANSI color helpers (disabled when stdout is not a terminal)
# ---------------------------------------------------------------------------

def _supports_color():
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _green(text):
    if _supports_color():
        return f"\033[32m{text}\033[0m"
    return text


def _red(text):
    if _supports_color():
        return f"\033[31m{text}\033[0m"
    return text


def _bold(text):
    if _supports_color():
        return f"\033[1m{text}\033[0m"
    return text


def _dim(text):
    if _supports_color():
        return f"\033[2m{text}\033[0m"
    return text


# Enige bron van waarheid voor validatiedefaults.
# Waarden hier worden gebruikt tenzij configuration.json ze overschrijft.
# Zie de module-docstring hierboven voor uitleg en het override-patroon.
_DEFAULT_VALIDATION_CFG = {
    "collegejaar_min_offset": 15,
    "collegejaar_max_offset": 2,
    "weeknummer_min": 1,
    "weeknummer_max": 53,  # ISO 8601: sommige jaren hebben week 53 (bijv. 2026)
    "nan_warning_threshold": 0.05,
    "nan_error_threshold": 0.30,
    "telbestand": {
        "required_columns": [
            "Studiejaar", "Isatcode", "Groepeernaam", "Aantal", "meercode_V",
            "Herinschrijving", "Hogerejaars", "Herkomst",
        ],
        "herkomst_allowed": ["N", "E", "R"],
        "herinschrijving_allowed": ["J", "N"],
        "hogerejaars_allowed": ["J", "N"],
    },
    # critical_columns zijn kanonieke namen; de werkelijke kolomnamen worden
    # opgezocht via configuration["columns"]["individual"] / ["oktober"].
    "individueel": {
        "critical_columns": ["Collegejaar", "Croho", "Inschrijfstatus", "Datum Verzoek Inschr"],
    },
    "oktober": {
        "critical_columns": [
            "Collegejaar", "Groepeernaam Croho", "Aantal eerstejaars croho",
            "EER-NL-nietEER", "Examentype code", "Aantal Hoofdinschrijvingen",
        ],
    },
}


@dataclass
class ValidationResult:
    hard_errors: list = field(default_factory=list)
    soft_errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def validate_raw_data(configuration, yes=False, data_option=None):
    """Validate all raw input files before ETL. Blocks on hard errors, prompts on soft errors.

    Skipped automatically when --noetl is used, on the assumption that if ETL has
    run before, the data was validated at that point.

    ``data_option`` controls *which* files are required. If omitted, both
    telbestanden and individueel zijn vereist (overeenkomend met ``-d both``).
    Wanneer de gebruiker expliciet ``-d individual`` of ``-d cumulative`` koos,
    wordt het ongebruikte bestand niet als hard-error gerapporteerd.
    """
    print("==== Valideren van ruwe inputdata ====")

    paths = configuration["paths"]
    cwd = os.getcwd()

    _print_file_overview(cwd, paths)

    # Top-level merge: voegt flat overrides toe (bijv. nan_error_threshold).
    # Nested sub-dicts (telbestand, individueel, oktober) worden hier shallow
    # vervangen als de gebruiker ze overschrijft. Elke _validate_*-functie
    # doet daarom een tweede sub-merge met _DEFAULT_VALIDATION_CFG als basis,
    # zodat een partial override nooit sleutels verliest. Voeg je een nieuwe
    # sectie toe, volg dan hetzelfde patroon: zie _validate_telbestanden.
    validation_cfg = {**_DEFAULT_VALIDATION_CFG, **configuration.get("validation", {})}
    columns_cfg = configuration.get("columns", {})
    result = ValidationResult()

    needs_telbestanden, needs_individueel = _required_files_for(data_option)

    _validate_telbestanden(cwd, paths, validation_cfg, result, required=needs_telbestanden)
    _validate_individueel(cwd, paths, validation_cfg, columns_cfg, result, required=needs_individueel)
    _validate_oktober(cwd, paths, validation_cfg, columns_cfg, result)

    _handle_result(result, yes)
    print("==== Validatie voltooid ====\n")


def _required_files_for(data_option):
    """Return ``(needs_telbestanden, needs_individueel)`` voor de gekozen ``-d``-modus.

    Bij ``None`` of ``BOTH_DATASETS`` zijn beide vereist. De file-overview-tabel
    laat de gebruiker zien welke alternatieve modi nog werken als er één ontbreekt.
    """
    from studentprognose.utils.weeks import DataOption

    if data_option == DataOption.INDIVIDUAL:
        return False, True
    if data_option == DataOption.CUMULATIVE:
        return True, False
    return True, True


# ---------------------------------------------------------------------------
# File overview table
# ---------------------------------------------------------------------------

def _check_file_presence(cwd, paths):
    """Check which input files exist and return a list of (label, relative_path, found, required_for)."""
    telbestanden_rel = paths.get("path_raw_telbestanden", "data/input_raw/telbestanden")
    telbestanden_dir = os.path.join(cwd, telbestanden_rel)
    telbestanden_ok = (
        os.path.isdir(telbestanden_dir)
        and any(re.search(r"telbestandY\d{4}W\d+", f) for f in os.listdir(telbestanden_dir))
    )

    individueel_rel = paths.get("path_raw_individueel", "data/input_raw/individuele_aanmelddata.csv")
    individueel_ok = os.path.exists(os.path.join(cwd, individueel_rel))

    oktober_rel = paths.get("path_raw_october", "data/input_raw/oktober_bestand.xlsx")
    oktober_ok = os.path.exists(os.path.join(cwd, oktober_rel)) if oktober_rel else False

    return [
        (telbestanden_rel, telbestanden_ok, "-d cumulative, -d both"),
        (individueel_rel, individueel_ok, "-d individual, -d both"),
        (oktober_rel, oktober_ok, "studentaantallen (optioneel, telbestand studenten)"),
    ]


def _print_file_overview(cwd, paths):
    """Print a table showing which input files are present and which run modes are available."""
    files = _check_file_presence(cwd, paths)

    col_file = max(len(f[0]) for f in files)
    col_status = 8
    header_file = "Bestand".ljust(col_file)
    header_status = "Status".ljust(col_status)
    header_needed = "Nodig voor"
    separator = "─" * (col_file + col_status + len(header_needed) + 6)

    print(f"\n  {_bold(header_file)}  {_bold(header_status)}  {_bold(header_needed)}")
    print(f"  {separator}")

    for rel_path, found, needed_for in files:
        status = _green("✓") if found else _red("✗")
        label = rel_path.ljust(col_file)
        print(f"  {label}  {status}         {needed_for}")

    telbestanden_ok = files[0][1]
    individueel_ok = files[1][1]

    modes = [
        ("-d cumulative", telbestanden_ok, [files[0][0]]),
        ("-d individual", individueel_ok, [files[1][0]]),
        ("-d both", telbestanden_ok and individueel_ok, [f[0] for f in files[:2] if not f[1]]),
    ]

    any_unavailable = any(not available for _, available, _ in modes)

    if any_unavailable:
        print(f"\n  {_bold('Beschikbare modi:')}")
        for mode, available, missing in modes:
            if available:
                print(f"    {mode:<18} {_green('✓')}")
            else:
                verb = "ontbreken" if len(missing) > 1 else "ontbreekt"
                missing_str = ", ".join(missing) + f" {verb}"
                print(f"    {mode:<18} {_red('✗')}  {missing_str}")
    else:
        print(f"\n  {_green('Alle bestanden aanwezig')} — alle modi beschikbaar.")

    print()


# ---------------------------------------------------------------------------
# Per-file validators
# ---------------------------------------------------------------------------

def _validate_telbestanden(cwd, paths, validation_cfg, result, required=True):
    telbestanden_dir = os.path.join(
        cwd, paths.get("path_raw_telbestanden", "data/input_raw/telbestanden")
    )
    individueel_path = os.path.join(
        cwd, paths.get("path_raw_individueel", "data/input_raw/individuele_aanmelddata.csv")
    )
    hint = (
        " — Tip: draai met `-d individual` om alleen de individuele aanmelddata te gebruiken."
        if os.path.exists(individueel_path) else ""
    )

    if not os.path.isdir(telbestanden_dir):
        if required:
            result.hard_errors.append(
                f"Telbestanden-map niet gevonden: {telbestanden_dir}{hint}"
            )
        return

    files = sorted([
        f for f in os.listdir(telbestanden_dir)
        if re.search(r"telbestandY\d{4}W\d+", f)
    ])

    if not files:
        if required:
            result.hard_errors.append(
                f"Geen telbestanden gevonden in {telbestanden_dir} "
                f"(verwacht patroon: telbestandY{{jaar}}W{{week}}.csv){hint}"
            )
        return

    _check_telbestand_completeness(files, validation_cfg, result)

    current_year = datetime.date.today().year
    year_min = current_year - validation_cfg["collegejaar_min_offset"]
    year_max = current_year + validation_cfg["collegejaar_max_offset"]
    weeknummer_min = validation_cfg["weeknummer_min"]
    weeknummer_max = validation_cfg["weeknummer_max"]
    nan_warn = validation_cfg["nan_warning_threshold"]
    nan_error = validation_cfg["nan_error_threshold"]

    telbestand_cfg = {**_DEFAULT_VALIDATION_CFG["telbestand"], **validation_cfg.get("telbestand", {})}
    required_columns = telbestand_cfg["required_columns"]
    herkomst_allowed = telbestand_cfg["herkomst_allowed"]
    herinschrijving_allowed = telbestand_cfg["herinschrijving_allowed"]
    hogerejaars_allowed = telbestand_cfg["hogerejaars_allowed"]

    for filename in files:
        filepath = os.path.join(telbestanden_dir, filename)
        try:
            df = pd.read_csv(filepath, sep=";", low_memory=False)
        except Exception as e:
            result.hard_errors.append(f"{filename}: kan bestand niet lezen ({e})")
            continue

        missing_cols = [c for c in required_columns if c not in df.columns]
        if missing_cols:
            result.hard_errors.append(
                f"{filename}: ontbrekende kolommen: {', '.join(missing_cols)}"
            )
            continue

        match = re.search(r"telbestandY(\d{4})W(\d+)", filename)
        if match:
            week_nr = int(match.group(2))
            if not (weeknummer_min <= week_nr <= weeknummer_max):
                result.hard_errors.append(
                    f"{filename}: weeknummer {week_nr} valt buiten verwacht bereik "
                    f"[{weeknummer_min}, {weeknummer_max}]"
                )

        studiejaar, was_coerced = _coerce_to_numeric(df["Studiejaar"])
        if was_coerced:
            result.warnings.append(
                f"{filename}: kolom 'Studiejaar' is geen numeriek type — "
                f"waarden worden automatisch geconverteerd"
            )
        invalid_years = studiejaar[~studiejaar.between(year_min, year_max)].dropna().unique()
        if len(invalid_years) > 0:
            result.soft_errors.append(
                f"{filename}: Studiejaar bevat onverwachte waarden: "
                f"{sorted(int(y) for y in invalid_years)} (verwacht: {year_min}–{year_max})"
            )

        for col, allowed in [
            ("Herinschrijving", herinschrijving_allowed),
            ("Hogerejaars", hogerejaars_allowed),
            ("Herkomst", herkomst_allowed),
        ]:
            normalized, was_normalized = _normalize_series(df[col])
            if was_normalized:
                result.warnings.append(
                    f"{filename}: kolom '{col}' bevat witruimte — "
                    f"waarden worden automatisch gestript"
                )
            _check_categoricals_per_programme(
                df.assign(**{col: normalized}),
                col, allowed, "Groepeernaam", filename, result,
            )

        meercode_v, _ = _coerce_to_numeric(df["meercode_V"])
        zero_meercode = df[meercode_v == 0]
        if not zero_meercode.empty:
            programmes = zero_meercode["Groepeernaam"].dropna().unique()
            result.soft_errors.append(
                f"{filename}: meercode_V = 0 (deling door nul in ETL) voor "
                f"{len(programmes)} opleiding(en): "
                + _format_programmes(programmes)
            )

        neg_aantal = df[df["Aantal"] < 0]
        if not neg_aantal.empty:
            programmes = neg_aantal["Groepeernaam"].dropna().unique()
            result.soft_errors.append(
                f"{filename}: Aantal < 0 voor "
                f"{len(programmes)} opleiding(en): "
                + _format_programmes(programmes)
            )

        for col in ["Aantal", "meercode_V"]:
            _check_nan_rate(df, col, filename, nan_warn, nan_error, result)


def _validate_individueel(cwd, paths, validation_cfg, columns_cfg, result, required=True):
    path = os.path.join(
        cwd, paths.get("path_raw_individueel", "data/input_raw/individuele_aanmelddata.csv")
    )

    if not os.path.exists(path):
        if required:
            telbestanden_dir = os.path.join(
                cwd, paths.get("path_raw_telbestanden", "data/input_raw/telbestanden")
            )
            has_telbestanden = (
                os.path.isdir(telbestanden_dir)
                and any(re.search(r"telbestandY\d{4}W\d+", f) for f in os.listdir(telbestanden_dir))
            )
            hint = (
                " — Tip: draai met `-d cumulative` om alleen de telbestanden te gebruiken."
                if has_telbestanden else ""
            )
            result.hard_errors.append(
                f"Individuele aanmelddata niet gevonden: {path}{hint}"
            )
        return

    try:
        df = pd.read_csv(path, sep=";", low_memory=False)
    except Exception as e:
        result.hard_errors.append(
            f"individuele_aanmelddata.csv: kan bestand niet lezen ({e})"
        )
        return

    col_map = columns_cfg.get("individual", {})
    individueel_cfg = {**_DEFAULT_VALIDATION_CFG["individueel"], **validation_cfg.get("individueel", {})}
    critical_columns = individueel_cfg["critical_columns"]

    missing_cols = [
        f"{_resolve_column(c, col_map)} (verwacht als '{c}')"
        for c in critical_columns
        if _resolve_column(c, col_map) not in df.columns
    ]
    if missing_cols:
        result.hard_errors.append(
            f"individuele_aanmelddata.csv: ontbrekende kolommen: {', '.join(missing_cols)}"
        )
        return

    nan_warn = validation_cfg["nan_warning_threshold"]
    nan_error = validation_cfg["nan_error_threshold"]
    for canonical in critical_columns:
        actual = _resolve_column(canonical, col_map)
        _check_nan_rate(df, actual, "individuele_aanmelddata.csv", nan_warn, nan_error, result)


def _validate_oktober(cwd, paths, validation_cfg, columns_cfg, result):
    path = os.path.join(
        cwd, paths.get("path_raw_october", "data/input_raw/oktober_bestand.xlsx")
    )

    if not os.path.exists(path):
        return

    try:
        df = pd.read_excel(path)
    except Exception as e:
        result.hard_errors.append(
            f"oktober_bestand.xlsx: kan bestand niet lezen ({e})"
        )
        return

    col_map = columns_cfg.get("oktober", {})
    oktober_cfg = {**_DEFAULT_VALIDATION_CFG["oktober"], **validation_cfg.get("oktober", {})}
    critical_columns = oktober_cfg["critical_columns"]

    missing_cols = [
        f"{_resolve_column(c, col_map)} (verwacht als '{c}')"
        for c in critical_columns
        if _resolve_column(c, col_map) not in df.columns
    ]
    if missing_cols:
        result.hard_errors.append(
            f"oktober_bestand.xlsx: ontbrekende kolommen: {', '.join(missing_cols)}"
        )
        return

    collegejaar_col = _resolve_column("Collegejaar", col_map)
    collegejaar, was_coerced = _coerce_to_numeric(df[collegejaar_col])
    if was_coerced:
        result.warnings.append(
            f"oktober_bestand.xlsx: kolom '{collegejaar_col}' is geen numeriek type — "
            f"waarden worden automatisch geconverteerd"
        )

    current_year = datetime.date.today().year
    year_min = current_year - validation_cfg["collegejaar_min_offset"]
    year_max = current_year + validation_cfg["collegejaar_max_offset"]
    invalid_years = collegejaar[~collegejaar.between(year_min, year_max)].dropna().unique()
    if len(invalid_years) > 0:
        result.soft_errors.append(
            f"oktober_bestand.xlsx: {collegejaar_col} bevat onverwachte waarden: "
            f"{sorted(int(y) for y in invalid_years)} (verwacht: {year_min}–{year_max})"
        )

    nan_warn = validation_cfg["nan_warning_threshold"]
    nan_error = validation_cfg["nan_error_threshold"]
    for canonical in critical_columns:
        actual = _resolve_column(canonical, col_map)
        _check_nan_rate(df, actual, "oktober_bestand.xlsx", nan_warn, nan_error, result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_column(canonical, col_map):
    """Resolve a canonical column name to the institution-specific name via the columns config."""
    return col_map.get(canonical, canonical)


def _normalize_series(series):
    """Strip leading/trailing whitespace from a string series.

    Returns (normalized_series, was_changed).
    """
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return series, False
    normalized = series.str.strip()
    was_changed = bool((normalized.fillna("") != series.fillna("")).any())
    return normalized, was_changed


def _coerce_to_numeric(series):
    """Coerce a series to numeric, handling Dutch decimal notation (comma separator).

    Returns (coerced_series, was_coerced). If already numeric, returns as-is.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series, False
    coerced = pd.to_numeric(
        series.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    return coerced, True


def _check_telbestand_completeness(files, validation_cfg, result):
    """Warn when there are unexpected gaps (> 2 weeks) between telbestanden within a year."""
    weeks_per_year = {}
    for filename in files:
        match = re.search(r"telbestandY(\d{4})W(\d+)", filename)
        if match:
            year, week = int(match.group(1)), int(match.group(2))
            weeks_per_year.setdefault(year, []).append(week)

    for year, weeks in sorted(weeks_per_year.items()):
        weeks_sorted = sorted(weeks)
        gaps = [
            (weeks_sorted[i], weeks_sorted[i + 1])
            for i in range(len(weeks_sorted) - 1)
            if weeks_sorted[i + 1] - weeks_sorted[i] > 2
        ]
        if gaps:
            gap_str = ", ".join(f"week {a} → week {b}" for a, b in gaps)
            result.warnings.append(
                f"Telbestanden jaar {year}: gaten gevonden tussen weken: {gap_str}"
            )


def _check_categoricals_per_programme(df, col, allowed, programme_col, filename, result):
    """Report categorical violations grouped by programme."""
    invalid_mask = ~df[col].isin(allowed) & df[col].notna()
    if not invalid_mask.any():
        return

    invalid_df = df[invalid_mask]
    invalid_values = list(invalid_df[col].unique())

    if programme_col in df.columns:
        programmes = invalid_df[programme_col].dropna().unique()
        result.soft_errors.append(
            f"{filename}: kolom '{col}' bevat ongeldige waarden {invalid_values} "
            f"(toegestaan: {allowed}) voor {len(programmes)} opleiding(en): "
            + _format_programmes(programmes)
        )
    else:
        result.soft_errors.append(
            f"{filename}: kolom '{col}' bevat ongeldige waarden {invalid_values} "
            f"(toegestaan: {allowed})"
        )


def _check_nan_rate(df, col, filename, warn_threshold, error_threshold, result):
    if col not in df.columns or len(df) == 0:
        return
    rate = df[col].isna().mean()
    pct = f"{rate:.0%}"
    if rate >= error_threshold:
        result.soft_errors.append(
            f"{filename}: kolom '{col}' heeft {pct} ontbrekende waarden "
            f"(foutdrempel: {error_threshold:.0%})"
        )
    elif rate >= warn_threshold:
        result.warnings.append(
            f"{filename}: kolom '{col}' heeft {pct} ontbrekende waarden "
            f"(waarschuwingsdrempel: {warn_threshold:.0%})"
        )


def _format_programmes(programmes, limit=5):
    names = [str(p) for p in programmes[:limit]]
    suffix = " ..." if len(programmes) > limit else ""
    return ", ".join(names) + suffix


# ---------------------------------------------------------------------------
# Result handler
# ---------------------------------------------------------------------------

def _handle_result(result, yes):
    for w in result.warnings:
        print(f"  [WAARSCHUWING] {w}")

    if result.hard_errors:
        print("\nValidatie mislukt — de pipeline kan niet worden gestart:")
        for e in result.hard_errors:
            print(f"  [FOUT] {e}")
        print(
            "\nLos de bovenstaande fouten op in je ruwe inputdata of configuratie "
            "en probeer opnieuw."
        )
        sys.exit(1)

    if result.soft_errors:
        print("\nValidatie gevonden problemen die prognoses kunnen beïnvloeden:")
        for e in result.soft_errors:
            print(f"  [PROBLEEM] {e}")

        if yes:
            print(
                "\n--yes opgegeven: pipeline wordt gestart ondanks bovenstaande problemen. "
                "Prognoses voor de betreffende opleidingen kunnen onbetrouwbaar zijn "
                "of worden overgeslagen."
            )
            return

        print(
            "\nAls je doorgaat, kunnen sommige opleidingen worden overgeslagen of "
            "onbetrouwbare prognoses opleveren. Je zou dan minder opleidingen "
            "terugkrijgen dan verwacht."
        )
        print(
            "Tip: gebruik --yes om deze prompt te omzeilen in geautomatiseerde runs."
        )

        try:
            answer = input("\nWil je doorgaan met de pipeline? [j/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nPipeline gestopt.")
            sys.exit(0)

        if answer != "j":
            print(
                "\nPipeline gestopt. Los de problemen op in je ruwe inputdata "
                "en probeer opnieuw."
            )
            sys.exit(0)

        print("Doorgaan met pipeline...\n")
