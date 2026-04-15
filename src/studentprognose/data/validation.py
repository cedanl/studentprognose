"""Pre-ETL datakwaliteitscontrole.

Dit module valideert alle ruwe inputbestanden voordat de ETL-pipeline start.
Ongeldige data wordt nooit stilzwijgend omgezet — de gebruiker wordt altijd
gewaarschuwd of de pipeline stopt.

Gedrag per type bevinding
--------------------------
hard error  → pipeline stopt direct, los op en probeer opnieuw
soft error  → pre-flight prompt; gebruik ``--yes`` om door te gaan (CI/CD)
warning     → gelogd, pipeline loopt door

Configuratie
------------
``_DEFAULT_VALIDATION_CFG`` is de **enige bron van waarheid** voor
validatiedefaults. De waarden staan hier in de code, niet in
``configuration.json``, om dubbele definities te vermijden.

Wil je een instelling overschrijven voor jouw installatie? Voeg een
``"validation"``-blok toe aan je ``configuration.json``. Je hoeft alleen de
waarden op te nemen die afwijken van de defaults:

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


# Enige bron van waarheid voor validatiedefaults.
# Waarden hier worden gebruikt tenzij configuration.json ze overschrijft.
# Zie de module-docstring hierboven voor uitleg en het override-patroon.
_DEFAULT_VALIDATION_CFG = {
    "collegejaar_min_offset": 15,
    "collegejaar_max_offset": 2,
    "weeknummer_min": 1,
    "weeknummer_max": 52,
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
    "individueel": {
        "required_columns": [
            "Collegejaar", "Croho", "Inschrijfstatus", "Datum Verzoek Inschr",
        ],
    },
    "oktober": {
        "required_columns": [
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


def validate_raw_data(configuration, yes=False):
    """Validate all raw input files before ETL. Blocks on hard errors, prompts on soft errors.

    Skipped automatically when --noetl is used, on the assumption that if ETL has
    run before, the data was validated at that point.
    """
    print("==== Valideren van ruwe inputdata ====")

    paths = configuration["paths"]
    validation_cfg = {**_DEFAULT_VALIDATION_CFG, **configuration.get("validation", {})}
    cwd = os.getcwd()
    result = ValidationResult()

    _validate_telbestanden(cwd, paths, validation_cfg, result)
    _validate_individueel(cwd, paths, validation_cfg, result)
    _validate_oktober(cwd, paths, validation_cfg, result)

    _handle_result(result, yes)
    print("==== Validatie voltooid ====\n")


# ---------------------------------------------------------------------------
# Per-file validators
# ---------------------------------------------------------------------------

def _validate_telbestanden(cwd, paths, validation_cfg, result):
    telbestanden_dir = os.path.join(
        cwd, paths.get("path_raw_telbestanden", "data/input_raw/telbestanden")
    )

    if not os.path.isdir(telbestanden_dir):
        result.hard_errors.append(
            f"Telbestanden-map niet gevonden: {telbestanden_dir}"
        )
        return

    files = sorted([
        f for f in os.listdir(telbestanden_dir)
        if re.search(r"telbestandY\d{4}W\d+", f)
    ])

    if not files:
        result.hard_errors.append(
            f"Geen telbestanden gevonden in {telbestanden_dir} "
            f"(verwacht patroon: telbestandY{{jaar}}W{{week}}.csv)"
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

        invalid_years = (
            df[~df["Studiejaar"].between(year_min, year_max)]["Studiejaar"]
            .dropna()
            .unique()
        )
        if len(invalid_years) > 0:
            result.soft_errors.append(
                f"{filename}: Studiejaar bevat onverwachte waarden: "
                f"{sorted(invalid_years)} (verwacht: {year_min}–{year_max})"
            )

        _check_categoricals_per_programme(
            df, "Herinschrijving", herinschrijving_allowed, "Groepeernaam", filename, result
        )
        _check_categoricals_per_programme(
            df, "Hogerejaars", hogerejaars_allowed, "Groepeernaam", filename, result
        )
        _check_categoricals_per_programme(
            df, "Herkomst", herkomst_allowed, "Groepeernaam", filename, result
        )

        zero_meercode = df[df["meercode_V"] == 0]
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


def _validate_individueel(cwd, paths, validation_cfg, result):
    path = os.path.join(
        cwd, paths.get("path_raw_individueel", "data/input_raw/individuele_aanmelddata.csv")
    )

    if not os.path.exists(path):
        result.warnings.append(
            f"Individuele aanmelddata niet gevonden: {path} — wordt overgeslagen."
        )
        return

    try:
        df = pd.read_csv(path, sep=";", low_memory=False)
    except Exception as e:
        result.hard_errors.append(
            f"individuele_aanmelddata.csv: kan bestand niet lezen ({e})"
        )
        return

    individueel_cfg = {**_DEFAULT_VALIDATION_CFG["individueel"], **validation_cfg.get("individueel", {})}
    required_columns = individueel_cfg["required_columns"]

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        result.hard_errors.append(
            f"individuele_aanmelddata.csv: ontbrekende kolommen: {', '.join(missing_cols)}"
        )
        return

    nan_warn = validation_cfg["nan_warning_threshold"]
    nan_error = validation_cfg["nan_error_threshold"]
    for col in ["Collegejaar", "Croho", "Inschrijfstatus"]:
        _check_nan_rate(df, col, "individuele_aanmelddata.csv", nan_warn, nan_error, result)


def _validate_oktober(cwd, paths, validation_cfg, result):
    path = os.path.join(
        cwd, paths.get("path_raw_october", "data/input_raw/oktober_bestand.xlsx")
    )

    if not os.path.exists(path):
        result.warnings.append(
            f"Oktober-bestand niet gevonden: {path} — studentaantallen worden niet berekend."
        )
        return

    try:
        df = pd.read_excel(path)
    except Exception as e:
        result.hard_errors.append(
            f"oktober_bestand.xlsx: kan bestand niet lezen ({e})"
        )
        return

    oktober_cfg = {**_DEFAULT_VALIDATION_CFG["oktober"], **validation_cfg.get("oktober", {})}
    required_columns = oktober_cfg["required_columns"]

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        result.hard_errors.append(
            f"oktober_bestand.xlsx: ontbrekende kolommen: {', '.join(missing_cols)}"
        )
        return

    current_year = datetime.date.today().year
    year_min = current_year - validation_cfg["collegejaar_min_offset"]
    year_max = current_year + validation_cfg["collegejaar_max_offset"]

    invalid_years = (
        df[~df["Collegejaar"].between(year_min, year_max)]["Collegejaar"]
        .dropna()
        .unique()
    )
    if len(invalid_years) > 0:
        result.soft_errors.append(
            f"oktober_bestand.xlsx: Collegejaar bevat onverwachte waarden: "
            f"{sorted(invalid_years)} (verwacht: {year_min}–{year_max})"
        )

    nan_warn = validation_cfg["nan_warning_threshold"]
    nan_error = validation_cfg["nan_error_threshold"]
    for col in ["Collegejaar", "Aantal eerstejaars croho", "Aantal Hoofdinschrijvingen"]:
        _check_nan_rate(df, col, "oktober_bestand.xlsx", nan_warn, nan_error, result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
