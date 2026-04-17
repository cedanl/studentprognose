import json
import sys

# Enkelvoudige bron voor defaults — geen extern JSON-bestand nodig.
# Aanpassen hier werkt direct voor zowel dev als geïnstalleerd package.
_DEFAULTS = {
    "paths": {
        "path_cumulative": "data/input/vooraanmeldingen_cumulatief.csv",
        "path_latest_individual": "data/input/totaal_individueel.xlsx",
        "path_latest_cumulative": "data/input/totaal_cumulatief.xlsx",
        "path_cumulative_new": "",
        "path_ensemble_weights": "data/input/ensemble_weights.xlsx",
        "path_raw_october": "data/input_raw/oktober_bestand.xlsx",
        "path_student_count_first-years": "data/input/student_count_first-years.xlsx",
        "path_student_count_higher-years": "data/input/student_count_higher-years.xlsx",
        "path_student_volume": "data/input/student_volume.xlsx",
        "path_ratios": "data/input/ratiobestand.xlsx",
        "path_individual": "data/input/vooraanmeldingen_individueel.csv",
        "path_raw_telbestanden": "data/input_raw/telbestanden",
        "path_raw_individueel": "data/input_raw/individuele_aanmelddata.csv",
    },
    "model_config": {
        "status_mapping": {
            "Ingeschreven": 1,
            "Geannuleerd": 0,
            "Uitgeschreven": 1,
            "Verzoek tot inschrijving": 0,
            "Studie gestaakt": 0,
            "Aanmelding vervolgen": 0,
        },
        "min_training_year": 2016,
    },
    "numerus_fixus": {},
    "ensemble_override_cumulative": [],
    "exclude_from_combined": [],
    "ensemble_weights": {
        "master_week_17_23": {"individual": 0.2, "cumulative": 0.8},
        "week_30_34": {"individual": 0.6, "cumulative": 0.4},
        "week_35_37": {"individual": 0.7, "cumulative": 0.3},
        "default": {"individual": 0.5, "cumulative": 0.5},
    },
    "columns": {
        "individual": {
            "Sleutel": "Sleutel",
            "Datum Verzoek Inschr": "Datum Verzoek Inschr",
            "Ingangsdatum": "Ingangsdatum",
            "Collegejaar": "Collegejaar",
            "Datum intrekking vooraanmelding": "Datum intrekking vooraanmelding",
            "Inschrijfstatus": "Inschrijfstatus",
            "Faculteit": "Faculteit",
            "Examentype": "Examentype",
            "Croho": "Croho",
            "Croho groepeernaam": "Croho groepeernaam",
            "Opleiding": "Opleiding",
            "Hoofdopleiding": "Hoofdopleiding",
            "Eerstejaars croho jaar": "Eerstejaars croho jaar",
            "Is eerstejaars croho opleiding": "Is eerstejaars croho opleiding",
            "Is hogerejaars": "Is hogerejaars",
            "BBC ontvangen": "BBC ontvangen",
            "Type vooropleiding": "Type vooropleiding",
            "Nationaliteit": "Nationaliteit",
            "EER": "EER",
            "Geslacht": "Geslacht",
            "Geverifieerd adres postcode": "Geverifieerd adres postcode",
            "Geverifieerd adres plaats": "Geverifieerd adres plaats",
            "Geverifieerd adres land": "Geverifieerd adres land",
            "Studieadres postcode": "Studieadres postcode",
            "Studieadres land": "Studieadres land",
            "School code eerste vooropleiding": "School code eerste vooropleiding",
            "School eerste vooropleiding": "School eerste vooropleiding",
            "Plaats code eerste vooropleiding": "Plaats code eerste vooropleiding",
            "Land code eerste vooropleiding": "Land code eerste vooropleiding",
            "Aantal studenten": "Aantal studenten",
        },
        "oktober": {
            "Collegejaar": "Collegejaar",
            "Groepeernaam Croho": "Groepeernaam Croho",
            "Aantal eerstejaars croho": "Aantal eerstejaars croho",
            "EER-NL-nietEER": "EER-NL-nietEER",
            "Examentype code": "Examentype code",
            "Aantal Hoofdinschrijvingen": "Aantal Hoofdinschrijvingen",
        },
        "cumulative": {
            "Korte naam instelling": "Korte naam instelling",
            "Collegejaar": "Collegejaar",
            "Weeknummer rapportage": "Weeknummer rapportage",
            "Weeknummer": "Weeknummer",
            "Faculteit": "Faculteit",
            "Type hoger onderwijs": "Type hoger onderwijs",
            "Groepeernaam Croho": "Groepeernaam Croho",
            "Naam Croho opleiding Nederlands": "Naam Croho opleiding Nederlands",
            "Croho": "Croho",
            "Herinschrijving": "Herinschrijving",
            "Hogerejaars": "Hogerejaars",
            "Herkomst": "Herkomst",
            "Gewogen vooraanmelders": "Gewogen vooraanmelders",
            "Ongewogen vooraanmelders": "Ongewogen vooraanmelders",
            "Aantal aanmelders met 1 aanmelding": "Aantal aanmelders met 1 aanmelding",
            "Inschrijvingen": "Inschrijvingen",
        },
    },
}

_FILTERING_DEFAULTS = {
    "filtering": {
        "programme": [],
        "herkomst": [],
        "examentype": [],
    }
}

_VALID_RULE_KEYS = {"year", "year_before", "year_after", "herkomst", "examentype", "opleiding"}


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base. override wins on conflicts."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_filtering(file_path: str) -> dict:
    """Load a filtering config. Falls back to empty filter if file_path doesn't exist."""
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return _FILTERING_DEFAULTS.copy()


def load_configuration(file_path: str) -> dict:
    """Load configuration, deep-merged on top of package defaults.

    If file_path does not exist, the bundled defaults are returned as-is.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            user_config = json.load(f)
        cfg = _deep_merge(_DEFAULTS, user_config)
    except FileNotFoundError:
        print(f"Waarschuwing: configuratiebestand niet gevonden: {file_path!r} — package defaults worden gebruikt.")
        cfg = _DEFAULTS.copy()

    if "excluded_data_points" in cfg:
        _validate_excluded_data_points(cfg["excluded_data_points"], file_path)
    return cfg


def _validate_excluded_data_points(rules, file_path):
    if not isinstance(rules, list):
        print(
            f"Configuratiefout in {file_path}: "
            f"'excluded_data_points' moet een lijst zijn, niet {type(rules).__name__}."
        )
        sys.exit(1)

    for i, rule in enumerate(rules):
        if not isinstance(rule, dict):
            print(
                f"Configuratiefout in {file_path}: "
                f"'excluded_data_points[{i}]' moet een object zijn."
            )
            sys.exit(1)

        unknown = set(rule) - _VALID_RULE_KEYS
        if unknown:
            print(
                f"Configuratiefout in {file_path}: "
                f"'excluded_data_points[{i}]' bevat onbekende sleutels: {sorted(unknown)}. "
                f"Toegestaan: {sorted(_VALID_RULE_KEYS)}."
            )
            sys.exit(1)

        if not rule:
            print(
                f"Configuratiefout in {file_path}: "
                f"'excluded_data_points[{i}]' is leeg — voeg ten minste één filtercriterium toe."
            )
            sys.exit(1)

        for int_key in ("year", "year_before", "year_after"):
            if int_key in rule and not isinstance(rule[int_key], int):
                print(
                    f"Configuratiefout in {file_path}: "
                    f"'excluded_data_points[{i}].{int_key}' moet een geheel getal zijn."
                )
                sys.exit(1)

        if "year_before" in rule and "year_after" in rule:
            if rule["year_before"] <= rule["year_after"]:
                print(
                    f"Configuratiefout in {file_path}: "
                    f"'excluded_data_points[{i}]' heeft een onmogelijk jaarbereik: "
                    f"year_before ({rule['year_before']}) moet groter zijn dan "
                    f"year_after ({rule['year_after']}). "
                    f"De combinatie sluit nooit rijen uit."
                )
                sys.exit(1)
