"""CLI-orchestratie voor ``studentprognose tune``.

Laadt en preprocesst de cumulatieve data (hergebruikt de benchmark-loaders),
bouwt dezelfde feature-matrix als productie, draait :func:`tune_regressor` en
print een resultatenoverzicht plus een kant-en-klaar configuratie-snippet om de
beste parameters in ``model_config.regressor_params`` vast te leggen.
"""

from studentprognose.config import load_configuration, get_final_academic_week
from studentprognose.models.tuning import (
    DEFAULT_PARAM_GRIDS,
    format_tuning_results,
    tune_regressor,
)
from studentprognose.utils.weeks import DataOption, academic_start_week


def run_tuning(
    configuration_path: str = "configuration/configuration.json",
    predict_week: int = 12,
    data_option: DataOption = DataOption.CUMULATIVE,
) -> dict | None:
    """Draai hyperparameter tuning voor de cumulatieve regressor.

    Args:
        configuration_path: Pad naar het configuratiebestand.
        predict_week: Week waarvandaan de feature-matrix wordt opgebouwd
            (bepaalt de lag-features, net als in productie).
        data_option: Alleen ``CUMULATIVE`` wordt ondersteund — de regressor
            hoort bij het cumulatieve spoor.

    Returns:
        Het resultaat van :func:`tune_regressor`, of ``None`` als er geen data
        of onvoldoende historie was.
    """
    if data_option != DataOption.CUMULATIVE:
        print(
            "Hyperparameter tuning ondersteunt momenteel alleen het cumulatieve "
            "spoor. Gebruik: studentprognose tune -d c"
        )
        return None

    # Lazy import: benchmark.runner trekt een brede importboom mee die we alleen
    # nodig hebben wanneer het tune-commando daadwerkelijk draait.
    from studentprognose.benchmark.runner import _load_cumulative_benchmark_data
    from studentprognose.models.sarima import _get_transformed_data
    from studentprognose.strategies.cumulative import _add_engineered_features

    config = load_configuration(configuration_path)
    name = config.get("model_config", {}).get("cumulative_regressor", "xgboost")
    min_year = config.get("model_config", {}).get("min_training_year", 2016)
    nf_list = list(config.get("numerus_fixus", {}).keys())
    grid = config.get("model_config", {}).get("tuning_grid")

    print("Hyperparameter tuning: cumulatieve regressor\n")
    print(f"Regressor: {name}, predict week: {predict_week}, min trainingsjaar: {min_year}")

    data_cumulative, data_studentcount = _load_cumulative_benchmark_data(config)
    if data_cumulative is None:
        print("Geen cumulatieve data gevonden. Tuning afgebroken.")
        return None
    if data_studentcount is None:
        print("Geen studentaantallen gevonden — de regressor kan niet worden geëvalueerd.")
        return None

    final_week = get_final_academic_week(config)
    full_data = _get_transformed_data(data_cumulative.copy(), min_year, final_week)
    full_data[str(academic_start_week(final_week))] = 0
    full_data = _add_engineered_features(full_data, data_cumulative, predict_week)

    grid_used = grid if grid is not None else DEFAULT_PARAM_GRIDS.get(name, {})
    print(f"Zoekruimte: {grid_used or '(geen — alleen modeldefaults)'}\n")

    result = tune_regressor(
        full_data,
        data_studentcount,
        config,
        regressor_name=name,
        grid=grid,
        min_training_year=min_year,
        numerus_fixus_list=nf_list,
    )

    _print_results(result)
    return result


def _print_results(result: dict) -> None:
    """Print het kandidatenoverzicht en een config-snippet voor de winnaar."""
    print(format_tuning_results(result))
