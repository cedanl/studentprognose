import json
import sys

_VALID_RULE_KEYS = {"year", "year_before", "year_after", "herkomst", "examentype", "opleiding"}


def load_configuration(file_path):
    with open(file_path) as f:
        cfg = json.load(f)
    # load_configuration wordt ook aangeroepen voor filtering.json, dat deze
    # sleutel nooit mag bevatten. Een .get() met lege lijst-default zou dat
    # stil doorlaten; de expliciete presence-check maakt de contractbreuk zichtbaar.
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
