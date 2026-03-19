from src.utils.weeks import DataOption
from src.strategies.individual import IndividualStrategy
from src.strategies.cumulative import CumulativeStrategy
from src.strategies.combined import CombinedStrategy


def create_strategy(cfg, datasets, configuration, cwd):
    """Select and instantiate the appropriate prediction strategy.

    datasets is the 6-tuple returned by load_data():
        (data_individual, data_cumulative, data_student_numbers,
         data_latest, data_distances, ensemble_weights)
    """
    (
        data_individual,
        data_cumulative,
        data_student_numbers,
        data_latest,
        data_distances,
        ensemble_weights,
    ) = datasets

    if cfg.skip_years > 0 or cfg.data_option == DataOption.CUMULATIVE:
        if data_cumulative is None:
            raise Exception("Cumulative dataset not found")
        return CumulativeStrategy(
            data_cumulative, data_student_numbers, configuration,
            data_latest, ensemble_weights, cwd, cfg.data_option, cfg.ci_test_n,
        )

    if cfg.data_option == DataOption.BOTH_DATASETS:
        if data_individual is None:
            raise Exception("Individual dataset not found")
        if data_cumulative is None:
            raise Exception("Cumulative dataset not found")
        try:
            return CombinedStrategy(
                data_individual, data_cumulative, data_distances,
                data_student_numbers, configuration, data_latest,
                ensemble_weights, cwd, cfg.data_option, cfg.ci_test_n,
                cfg.years,
            )
        except ValueError as e:
            print(e)
            return CumulativeStrategy(
                data_cumulative, data_student_numbers, configuration,
                data_latest, ensemble_weights, cwd, cfg.data_option, cfg.ci_test_n,
            )

    if cfg.data_option == DataOption.INDIVIDUAL:
        if data_individual is None:
            raise Exception("Individual dataset not found")
        return IndividualStrategy(
            data_individual, data_distances, configuration,
            data_latest, ensemble_weights, data_student_numbers,
            cwd, cfg.data_option, cfg.ci_test_n,
        )

    raise Exception(f"Unknown data option: {cfg.data_option}")
