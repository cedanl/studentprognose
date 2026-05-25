import json
import os
import sys
from importlib.resources import files
from types import SimpleNamespace

from studentprognose.utils.telbestand_filenames import _placeholder_to_regex

_VALID_RULE_KEYS = {"year", "year_before", "year_after", "herkomst", "examentype", "opleiding"}

_VALID_TIMESERIES_MODELS = {"sarima", "ets", "theta", "auto_arima"}
_VALID_REGRESSOR_MODELS = {"xgboost", "ridge", "random_forest", "gradient_boosting", "extra_trees"}
_VALID_CLASSIFIER_MODELS = {"xgboost", "random_forest", "logistic_regression", "gradient_boosting", "extra_trees"}


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
    """Load the bundled generic default configuration."""
    data = files("studentprognose.configuration").joinpath("configuration.json").read_text(encoding="utf-8")
    return json.loads(data)


def load_defaults_filtering() -> dict:
    """Load the bundled default filtering config (no filters applied)."""
    data = files("studentprognose.configuration.filtering").joinpath("base.json").read_text(encoding="utf-8")
    return json.loads(data)


def load_filtering(file_path: str) -> dict:
    """Load a filtering config. Falls back to bundled base.json if file_path doesn't exist."""
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return load_defaults_filtering()


def load_configuration(file_path: str) -> dict:
    """Load configuration, deep-merged on top of package defaults.

    Falls back to bundled defaults when file_path doesn't exist, so the
    package works after pip install without a local configuration file.
    """
    defaults = load_defaults()

    try:
        with open(file_path, encoding="utf-8") as f:
            user_config = json.load(f)
        cfg = _deep_merge(defaults, user_config)
    except FileNotFoundError:
        print(f"Waarschuwing: configuratiebestand niet gevonden: {file_path!r} — package defaults worden gebruikt.")
        cfg = defaults

    if "excluded_data_points" in cfg:
        _validate_excluded_data_points(cfg["excluded_data_points"], file_path)
    _validate_model_config(cfg, file_path)
    _validate_runtime(cfg, file_path)
    _validate_telbestand_filename_patterns(cfg, file_path)
    return cfg


def get_columns(config: dict) -> SimpleNamespace:
    """Get column names as a namespace from configuration.

    Each attribute maps a semantic role to the actual column name used by
    this institution.  Example: ``cols.programme`` resolves to
    ``"Croho groepeernaam"`` with the default configuration.
    """
    return SimpleNamespace(**config.get("column_roles", {}))


def get_model_features(config: dict) -> dict:
    """Get model feature lists (numeric / categorical) from configuration."""
    return config.get("model_features", {})


def get_cpu_count(config: dict) -> int:
    """Resolve the number of CPU cores to use for parallel work.

    Pure lookup with no side-effects. The value has already been validated
    and capped (with a one-time warning) during :func:`_validate_runtime`
    when the configuration was loaded.

    - ``runtime.cpu_count`` is ``None`` (or missing): returns ``os.cpu_count()``,
      or ``1`` when the OS does not report a count (e.g. some
      cgroup-constrained containers).
    - Otherwise: returns the configured integer as-is.
    """
    requested = config.get("runtime", {}).get("cpu_count")
    if requested is None:
        return os.cpu_count() or 1
    return requested


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


def _validate_model_config(cfg, file_path):
    model_config = cfg.get("model_config", {})

    ts_model = model_config.get("cumulative_timeseries")
    if ts_model is not None and ts_model not in _VALID_TIMESERIES_MODELS:
        print(
            f"Configuratiefout in {file_path}: "
            f"'model_config.cumulative_timeseries' is '{ts_model}'. "
            f"Geldige opties: {sorted(_VALID_TIMESERIES_MODELS)}."
        )
        sys.exit(1)

    reg_model = model_config.get("cumulative_regressor")
    if reg_model is not None and reg_model not in _VALID_REGRESSOR_MODELS:
        print(
            f"Configuratiefout in {file_path}: "
            f"'model_config.cumulative_regressor' is '{reg_model}'. "
            f"Geldige opties: {sorted(_VALID_REGRESSOR_MODELS)}."
        )
        sys.exit(1)

    clf_model = model_config.get("individual_classifier")
    if clf_model is not None and clf_model not in _VALID_CLASSIFIER_MODELS:
        print(
            f"Configuratiefout in {file_path}: "
            f"'model_config.individual_classifier' is '{clf_model}'. "
            f"Geldige opties: {sorted(_VALID_CLASSIFIER_MODELS)}."
        )
        sys.exit(1)


def _validate_telbestand_filename_patterns(cfg, file_path):
    """Fail fast op ongeldige telbestand-bestandsnaampatronen.

    Een patroon moet zowel ``{year}`` als ``{week}`` bevatten — anders kan de
    pipeline geen jaar/week uit de bestandsnaam destilleren. Ontbrekende
    sleutel, lege lijst, lege string of lijst van geldige strings worden alle
    geaccepteerd; ``compile_patterns`` valt terug op de default waar nodig.
    """
    raw = cfg.get("telbestand_filename_patterns")
    if raw is None or raw == [] or raw == "":
        return

    if isinstance(raw, str):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = raw
    else:
        print(
            f"Configuratiefout in {file_path}: "
            f"'telbestand_filename_patterns' moet een string of lijst van strings zijn, "
            f"niet {type(raw).__name__}."
        )
        sys.exit(1)

    for i, pattern in enumerate(candidates):
        if not isinstance(pattern, str):
            print(
                f"Configuratiefout in {file_path}: "
                f"'telbestand_filename_patterns[{i}]' moet een string zijn, "
                f"niet {type(pattern).__name__}."
            )
            sys.exit(1)
        try:
            _placeholder_to_regex(pattern)
        except ValueError as exc:
            print(
                f"Configuratiefout in {file_path}: {exc} "
                f"Voorbeeld: \"telbestandY{{year}}W{{week}}\"."
            )
            sys.exit(1)


def _validate_runtime(cfg, file_path):
    runtime = cfg.get("runtime", {})
    if not isinstance(runtime, dict):
        print(
            f"Configuratiefout in {file_path}: "
            f"'runtime' moet een object zijn, niet {type(runtime).__name__}."
        )
        sys.exit(1)

    cpu_count = runtime.get("cpu_count")
    if cpu_count is None:
        return

    if isinstance(cpu_count, bool) or not isinstance(cpu_count, int):
        print(
            f"Configuratiefout in {file_path}: "
            f"'runtime.cpu_count' moet een geheel getal zijn of null, "
            f"niet {type(cpu_count).__name__}."
        )
        sys.exit(1)

    if cpu_count < 1:
        print(
            f"Configuratiefout in {file_path}: "
            f"'runtime.cpu_count' is {cpu_count} — moet >= 1 zijn, of null voor automatische detectie."
        )
        sys.exit(1)

    # Cap aan beschikbare cores. Eenmalig hier — geen herhaalde waarschuwing
    # tijdens predict-loops over (jaar × week)-combinaties.
    available = os.cpu_count() or 1
    if cpu_count > available:
        print(
            f"Waarschuwing in {file_path}: "
            f"'runtime.cpu_count' ({cpu_count}) is hoger dan het aantal beschikbare "
            f"cores ({available}). Verlaagd naar {available}."
        )
        cfg["runtime"]["cpu_count"] = available
