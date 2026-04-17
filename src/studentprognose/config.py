import json
import sys
from importlib.resources import files

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


def load_defaults() -> dict:
    """Load the bundled default configuration from package data."""
    data = files("studentprognose.configuration").joinpath("configuration.json").read_text(encoding="utf-8")
    return json.loads(data)


def load_filtering(file_path: str) -> dict:
    """Load a filtering config. Falls back to bundled base.json if file_path doesn't exist."""
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        data = files("studentprognose.configuration.filtering").joinpath("base.json").read_text(encoding="utf-8")
        return json.loads(data)


def load_configuration(file_path: str) -> dict:
    """Load configuration, deep-merged on top of package defaults.

    If file_path does not exist, the bundled defaults are returned as-is.
    """
    defaults = load_defaults()

    try:
        with open(file_path, encoding="utf-8") as f:
            user_config = json.load(f)
        cfg = _deep_merge(defaults, user_config)
    except FileNotFoundError:
        cfg = defaults

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
