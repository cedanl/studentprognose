import json
import os
import sys
from importlib.resources import files
from types import SimpleNamespace

from studentprognose.utils.telbestand_filenames import _placeholder_to_regex
from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK
from studentprognose.utils.programme_key import normalize_programme_keys

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
    _validate_cwf_params(cfg, file_path)
    _validate_runtime(cfg, file_path)
    _validate_telbestand_filename_patterns(cfg, file_path)
    _validate_institution_filter(cfg, file_path)

    # Normaliseer de numerus_fixus-keys (programmesleutels) naar hetzelfde dtype
    # als de datakolom: een numerieke Isatcode-key wordt int en matcht zo de
    # Int64-kolom in .isin/dict-lookups; legacy naam-keys blijven string.
    if isinstance(cfg.get("numerus_fixus"), dict):
        cfg["numerus_fixus"] = normalize_programme_keys(cfg["numerus_fixus"])
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


def get_final_academic_week(config: dict) -> int:
    """Laatste week van het academisch jaar uit de configuratie.

    Default :data:`FINAL_ACADEMIC_WEEK` (38, het Radboud/Studielink-instellings-
    formaat). Het UvA SQL-telbestand levert de aanmeldfase tot ~week 36 (geen
    leveringen week 37–39), dus ``model_config.final_academic_week`` staat daar
    op 36. Bepaalt de seizoensvensters in weeks/sarima/cumulative/postprocessor.
    """
    return config.get("model_config", {}).get("final_academic_week", FINAL_ACADEMIC_WEEK)


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
    # De geldige modelnamen komen uit de model-registries (single source of
    # truth): dezelfde dicts die de pipeline gebruikt om modellen te
    # instantiëren. Function-local import omdat dit de enige plek in config.py
    # is die de registries nodig heeft, en het models-pakket config.py terug
    # importeert (models -> xgboost_regressor -> config); lokaal importeren
    # houdt config.py's module-surface los van die cyclus.
    from studentprognose.models import (
        CLASSIFIER_REGISTRY,
        FORECASTER_REGISTRY,
        REGRESSOR_REGISTRY,
    )

    valid_timeseries = set(FORECASTER_REGISTRY)
    valid_regressor = set(REGRESSOR_REGISTRY)
    valid_classifier = set(CLASSIFIER_REGISTRY)

    model_config = cfg.get("model_config", {})

    ts_model = model_config.get("cumulative_timeseries")
    if ts_model is not None and ts_model not in valid_timeseries:
        print(
            f"Configuratiefout in {file_path}: "
            f"'model_config.cumulative_timeseries' is '{ts_model}'. "
            f"Geldige opties: {sorted(valid_timeseries)}."
        )
        sys.exit(1)

    reg_model = model_config.get("cumulative_regressor")
    if reg_model is not None and reg_model not in valid_regressor:
        print(
            f"Configuratiefout in {file_path}: "
            f"'model_config.cumulative_regressor' is '{reg_model}'. "
            f"Geldige opties: {sorted(valid_regressor)}."
        )
        sys.exit(1)

    clf_model = model_config.get("individual_classifier")
    if clf_model is not None and clf_model not in valid_classifier:
        print(
            f"Configuratiefout in {file_path}: "
            f"'model_config.individual_classifier' is '{clf_model}'. "
            f"Geldige opties: {sorted(valid_classifier)}."
        )
        sys.exit(1)

    reg_params = model_config.get("regressor_params")
    if reg_params is not None:
        if not isinstance(reg_params, dict):
            print(
                f"Configuratiefout in {file_path}: "
                f"'model_config.regressor_params' moet een object zijn "
                f"(regressornaam → parameters), niet {type(reg_params).__name__}."
            )
            sys.exit(1)
        for reg_name, params in reg_params.items():
            if reg_name not in valid_regressor:
                print(
                    f"Configuratiefout in {file_path}: "
                    f"'model_config.regressor_params' bevat onbekende regressor '{reg_name}'. "
                    f"Geldige opties: {sorted(valid_regressor)}."
                )
                sys.exit(1)
            if not isinstance(params, dict):
                print(
                    f"Configuratiefout in {file_path}: "
                    f"'model_config.regressor_params.{reg_name}' moet een object zijn "
                    f"(parameternaam → waarde), niet {type(params).__name__}."
                )
                sys.exit(1)

    tuning_grid = model_config.get("tuning_grid")
    if tuning_grid is not None and not isinstance(tuning_grid, dict):
        print(
            f"Configuratiefout in {file_path}: "
            f"'model_config.tuning_grid' moet een object zijn "
            f"(parameternaam → lijst van waarden), niet {type(tuning_grid).__name__}."
        )
        sys.exit(1)

    fc_params = model_config.get("forecaster_params")
    if fc_params is not None:
        if not isinstance(fc_params, dict):
            print(
                f"Configuratiefout in {file_path}: "
                f"'model_config.forecaster_params' moet een object zijn "
                f"(forecasternaam → parameters), niet {type(fc_params).__name__}."
            )
            sys.exit(1)
        for fc_name, params in fc_params.items():
            if fc_name not in valid_timeseries:
                print(
                    f"Configuratiefout in {file_path}: "
                    f"'model_config.forecaster_params' bevat onbekende forecaster '{fc_name}'. "
                    f"Geldige opties: {sorted(valid_timeseries)}."
                )
                sys.exit(1)
            if not isinstance(params, dict):
                print(
                    f"Configuratiefout in {file_path}: "
                    f"'model_config.forecaster_params.{fc_name}' moet een object zijn "
                    f"(parameternaam → waarde), niet {type(params).__name__}."
                )
                sys.exit(1)

    sarima_tuning_grid = model_config.get("sarima_tuning_grid")
    if sarima_tuning_grid is not None and not isinstance(sarima_tuning_grid, dict):
        print(
            f"Configuratiefout in {file_path}: "
            f"'model_config.sarima_tuning_grid' moet een object zijn "
            f"(ordenaam → lijst van waarden), niet {type(sarima_tuning_grid).__name__}."
        )
        sys.exit(1)


# Bereikeisen per numerieke CWF-parameter: (ondergrens, bovengrens, ondergrens
# exclusief). Parameters die hier ontbreken moeten alleen ≥ 0 zijn.
_CWF_RANGES = {
    "cv_lambda": (0, None, True),
    "max_history_years": (1, None, False),
    "ratio_tolerance": (0, None, True),
    "size_factor": (1, None, True),
    "max_group_programmes": (1, None, False),
    "trend_r2_threshold": (0, 1, False),
    "ratio_floor": (0, None, True),
    "confidence_level": (0, 1, True),
    "pool_outlier_quantile": (0, 1, True),
    "yoy_absolute_weight": (0, 1, False),
    "fill_rate_recency_alpha": (0, None, False),
    "upper_bound_cap_fill_rate": (0, 1, False),
    "reliability_orange_min_fill": (0, 1, False),
    "reliability_green_min_fill": (0, 1, False),
}


def cwf_params_errors(raw) -> list[str]:
    """Foutmeldingen voor een ``model_config.cwf_params``-object (leeg = geldig).

    Controleert onbekende sleutels, types en bereiken. Gedeeld door de CLI
    (:func:`_validate_cwf_params`, stopt bij de eerste fout) en de GUI-editor
    (toont alle fouten).

    Args:
        raw: De waarde van ``model_config.cwf_params``.

    Returns:
        Lijst met leesbare foutmeldingen.
    """
    from studentprognose.models.cwf import CWFParams

    prefix = "'model_config.cwf_params"
    if not isinstance(raw, dict):
        return [f"{prefix}' moet een object zijn, niet {type(raw).__name__}."]

    defaults = CWFParams.defaults_as_dict()
    errors: list[str] = []
    unknown = sorted(set(raw) - set(defaults))
    if unknown:
        errors.append(
            f"{prefix}' bevat onbekende sleutel(s) {unknown}. Geldige sleutels: {sorted(defaults)}."
        )

    def _is_number(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    for key, value in raw.items():
        if key not in defaults:
            continue
        default = defaults[key]
        if key == "training_start_year":
            if value is not None and not (isinstance(value, int) and not isinstance(value, bool)):
                errors.append(f"{prefix}.{key}' moet een jaartal of null zijn, niet {value!r}.")
            continue
        if isinstance(default, bool):
            if not isinstance(value, bool):
                errors.append(f"{prefix}.{key}' moet true of false zijn, niet {value!r}.")
            continue
        if isinstance(default, list):
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                errors.append(f"{prefix}.{key}' moet een lijst van examentypes zijn, niet {value!r}.")
            continue
        if isinstance(default, dict):
            if not isinstance(value, dict):
                errors.append(f"{prefix}.{key}' moet een object zijn, niet {type(value).__name__}.")
            continue
        if isinstance(default, int) and not (isinstance(value, int) and not isinstance(value, bool)):
            errors.append(f"{prefix}.{key}' moet een geheel getal zijn, niet {value!r}.")
            continue
        if isinstance(default, float) and not _is_number(value):
            errors.append(f"{prefix}.{key}' moet een getal zijn, niet {value!r}.")
            continue

        low, high, exclusive = _CWF_RANGES.get(key, (0, None, False))
        too_low = value <= low if exclusive else value < low
        if too_low or (high is not None and value > high):
            bound = f"> {low}" if exclusive else f">= {low}"
            if high is not None:
                bound += f" en <= {high}"
            errors.append(f"{prefix}.{key}' moet {bound} zijn, niet {value!r}.")

    overrides = raw.get("year_weight_overrides", {})
    if isinstance(overrides, dict):
        for year, weight in overrides.items():
            if not str(year).isdigit() or not _is_number(weight) or weight < 0:
                errors.append(
                    f"{prefix}.year_weight_overrides' verwacht jaartal → gewicht >= 0, "
                    f"niet {year!r}: {weight!r}."
                )

    by_exam = raw.get("reliability_green_max_own_error_by_exam_type", {})
    if isinstance(by_exam, dict):
        for exam_type, value in by_exam.items():
            if not _is_number(value) or value < 0:
                errors.append(
                    f"{prefix}.reliability_green_max_own_error_by_exam_type.{exam_type}' moet "
                    f"een getal >= 0 zijn, niet {value!r}."
                )

    priors = raw.get("predefined_priors", {})
    if isinstance(priors, dict):
        for programme, per_exam in priors.items():
            if not isinstance(per_exam, dict) or not all(
                isinstance(per_origin, dict)
                and all(_is_number(v) and v >= 0 for v in per_origin.values())
                for per_origin in per_exam.values()
            ):
                errors.append(
                    f"{prefix}.predefined_priors.{programme}' moet de vorm "
                    "{examentype: {herkomst: aantal}} hebben met aantallen >= 0."
                )
    return errors


def _validate_cwf_params(cfg, file_path):
    """Valideer ``model_config.cwf_params`` (CWF-model).

    Onbekende sleutels, verkeerde types en waarden buiten het geldige bereik
    stoppen de pipeline met exitcode 1, zodat een tikfout niet stil op de
    default terugvalt.
    """
    raw = cfg.get("model_config", {}).get("cwf_params")
    if raw is None:
        return
    errors = cwf_params_errors(raw)
    if errors:
        print(f"Configuratiefout in {file_path}: {errors[0]}")
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


def _validate_institution_filter(cfg, file_path):
    """Valideer ``institution_filter`: een lijst van instellingscodes/-namen.

    Leeg of afwezig = geen filter (alle instellingen). Elke waarde moet een string
    of geheel getal zijn (Brincodes zijn tekstcodes zoals ``"00IC"``, maar een
    numerieke code als int wordt geaccepteerd en later als string vergeleken).
    """
    institutions = cfg.get("institution_filter")
    if institutions is None:
        return

    if not isinstance(institutions, list):
        print(
            f"Configuratiefout in {file_path}: "
            f"'institution_filter' moet een lijst van instellingscodes zijn, "
            f"niet {type(institutions).__name__}."
        )
        sys.exit(1)

    for i, value in enumerate(institutions):
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            print(
                f"Configuratiefout in {file_path}: "
                f"'institution_filter[{i}]' moet een string of geheel getal zijn, "
                f"niet {type(value).__name__}."
            )
            sys.exit(1)
