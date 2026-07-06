"""CLI-orchestratie voor ``studentprognose tune``.

Laadt en preprocesst de cumulatieve data (hergebruikt de benchmark-loaders) en
tunet één of beide trappen van het cumulatieve spoor:

* ``regressor`` (stap 2) — :func:`tune_regressor` op de feature-matrix.
* ``sarima`` (stap 1) — :func:`tune_sarima` op de tijdreeks-curve.

Per trap wordt een resultatenoverzicht geprint (met ✓ op de winnaar) plus een
kant-en-klaar config-snippet om de beste waarden vast te leggen.
"""

from studentprognose.config import load_configuration, get_final_academic_week
from studentprognose.models.tuning import (
    DEFAULT_PARAM_GRIDS,
    DEFAULT_SARIMA_GRID,
    format_tuning_results,
    tune_regressor,
    tune_sarima,
)
from studentprognose.utils.weeks import DataOption, academic_start_week


def run_tuning(
    configuration_path: str = "configuration/configuration.json",
    predict_week: int = 12,
    data_option: DataOption = DataOption.CUMULATIVE,
    target: str = "regressor",
) -> dict | None:
    """Draai tuning voor het cumulatieve spoor.

    Args:
        configuration_path: Pad naar het configuratiebestand.
        predict_week: Week waarvandaan voorspeld wordt (bepaalt lag-features en
            de tijdreeks-horizon, net als in productie).
        data_option: Alleen ``CUMULATIVE`` wordt ondersteund — beide trappen
            horen bij het cumulatieve spoor.
        target: ``"regressor"`` (stap 2, default), ``"sarima"`` (stap 1) of
            ``"both"`` (beide, elk apart gelogd met eigen ✓-tabel).

    Returns:
        Een dict ``{target: result}`` met de :func:`tune_regressor` /
        :func:`tune_sarima` resultaten, of ``None`` als er geen data was.
    """
    if data_option != DataOption.CUMULATIVE:
        print(
            "Tuning ondersteunt momenteel alleen het cumulatieve spoor. "
            "Gebruik: studentprognose tune -d c"
        )
        return None

    # Lazy import: benchmark.runner trekt een brede importboom mee die we alleen
    # nodig hebben wanneer het tune-commando daadwerkelijk draait.
    from studentprognose.benchmark.runner import _load_cumulative_benchmark_data
    from studentprognose.models.sarima import _get_transformed_data
    from studentprognose.strategies.cumulative import _add_engineered_features

    config = load_configuration(configuration_path)
    model_config = config.get("model_config", {})
    min_year = model_config.get("min_training_year", 2016)

    print("Tuning: cumulatief spoor\n")
    print(f"Target: {target}, predict week: {predict_week}, min trainingsjaar: {min_year}\n")

    data_cumulative, data_studentcount = _load_cumulative_benchmark_data(config)
    if data_cumulative is None:
        print("Geen cumulatieve data gevonden. Tuning afgebroken.")
        return None

    do_regressor = target in ("regressor", "both")
    do_sarima = target in ("sarima", "both")
    results: dict[str, dict] = {}

    if do_sarima:
        sarima_grid = model_config.get("sarima_tuning_grid")
        grid_used = sarima_grid if sarima_grid is not None else DEFAULT_SARIMA_GRID
        print("── SARIMA (stap 1: tijdreeks-ordes) ──")
        print(f"Zoekruimte: {grid_used}\n")
        sarima_result = tune_sarima(
            data_cumulative,
            predict_week,
            config,
            grid=sarima_grid,
            min_training_year=min_year,
        )
        print(format_tuning_results(sarima_result))
        print()
        results["sarima"] = sarima_result

    if do_regressor:
        if data_studentcount is None:
            print("Geen studentaantallen gevonden — de regressor kan niet worden geëvalueerd.")
        else:
            name = model_config.get("cumulative_regressor", "xgboost")
            nf_list = list(config.get("numerus_fixus", {}).keys())
            grid = model_config.get("tuning_grid")

            final_week = get_final_academic_week(config)
            full_data = _get_transformed_data(data_cumulative.copy(), min_year, final_week)
            full_data[str(academic_start_week(final_week))] = 0
            full_data = _add_engineered_features(full_data, data_cumulative, predict_week)

            grid_used = grid if grid is not None else DEFAULT_PARAM_GRIDS.get(name, {})
            print(f"── Regressor (stap 2: {name}) ──")
            print(f"Zoekruimte: {grid_used or '(geen — alleen modeldefaults)'}\n")
            reg_result = tune_regressor(
                full_data,
                data_studentcount,
                config,
                regressor_name=name,
                grid=grid,
                min_training_year=min_year,
                numerus_fixus_list=nf_list,
            )
            print(format_tuning_results(reg_result))
            results["regressor"] = reg_result

    return results or None
