import os
import sys

from src.cli import parse_args
from src.config import load_configuration
from src.data.s02_loader import load_data
from src.data.s04_ci_subset import apply_ci_test_subset
from src.output.s10_postprocessor import PostProcessor
from src.strategies import create_strategy
from src.utils.weeks import DataOption, StudentYearPrediction, HIGHER_YEARS_COLUMNS


def main(argv):
    cfg = parse_args(argv)

    # Step 0: ETL (default — raw data → input data, skip with --noetl)
    if not cfg.noetl:
        from src.data.s01_etl import run_etl
        run_etl(load_configuration(cfg.configuration_path))

    print("Predicting for years: ", cfg.years, " and weeks: ", cfg.weeks)

    # Step 1: Load configuration and data
    print("Loading configuration...")
    configuration = load_configuration(cfg.configuration_path)
    filtering = load_configuration(cfg.filtering_path)

    print("Loading data...")
    datasets = load_data(configuration, cfg.data_option)
    if cfg.ci_test_n is not None:
        datasets = apply_ci_test_subset(cfg.ci_test_n, *datasets)

    _check_data_range(datasets, cfg)

    # Step 2: Initialize strategy (Individual / Cumulative / Combined)
    cwd = os.path.dirname(os.path.abspath(__file__))
    strategy = create_strategy(cfg, datasets, configuration, cwd)

    PostProcessor.check_output_writable(
        cfg.data_option, cfg.student_year_prediction, cfg.ci_test_n, cfg.filtering_path,
    )

    # Step 3: Preprocess (feature engineering)
    data_cumulative = _preprocess(strategy, cfg.student_year_prediction)

    # Step 4: Apply filtering
    strategy.set_filtering(
        filtering["filtering"]["programme"],
        filtering["filtering"]["herkomst"],
        filtering["filtering"]["examentype"],
    )

    # Step 5: Predict + evaluate (per year × week)
    for year in cfg.years:
        for week in cfg.weeks:
            _predict_and_postprocess(strategy, cfg, data_cumulative, year, week)

    # Step 6: Save output
    _save_results(strategy, cfg)


def _check_data_range(datasets, cfg):
    """Check if requested years/weeks exist in the loaded data and warn if not."""
    data_individual, data_cumulative, *_ = datasets

    # Pick whichever dataset is loaded based on the chosen mode
    data = data_cumulative if data_cumulative is not None else data_individual
    if data is None:
        return

    available_years = sorted(int(y) for y in data["Collegejaar"].dropna().unique())
    available_weeks = sorted(int(w) for w in data["Weeknummer"].dropna().unique())

    if not available_years:
        return

    missing_years = [y for y in cfg.years if y not in available_years]
    missing_weeks = [w for w in cfg.weeks if w not in available_weeks]

    if missing_years or missing_weeks:
        year_range = f"{available_years[0]}-{available_years[-1]}" if len(available_years) > 1 else str(available_years[0])
        week_range = f"{available_weeks[0]}-{available_weeks[-1]}" if len(available_weeks) > 1 else str(available_weeks[0])
        print(f"\nWaarschuwing: de gevraagde combinatie is niet (volledig) beschikbaar in de data.")
        print(f"  Beschikbare data: jaren {year_range}, weken {week_range}.")
        print(f"  Pas je flags aan tussen -y {year_range} en -w {week_range},")
        print(f"  of voeg nieuwe trainingsdata toe in data/input_raw/ om je gewenste tijdstip te voorspellen.")
        sys.exit(1)


def _preprocess(strategy, student_year_prediction):
    if student_year_prediction in (StudentYearPrediction.FIRST_YEARS, StudentYearPrediction.VOLUME):
        print("Preprocessing...")
        return strategy.preprocess()

    if student_year_prediction == StudentYearPrediction.HIGHER_YEARS:
        strategy.postprocessor.data = strategy.postprocessor.data_latest[HIGHER_YEARS_COLUMNS]
        return None


def _predict_and_postprocess(strategy, cfg, data_cumulative, year, week):
    if cfg.student_year_prediction in (StudentYearPrediction.FIRST_YEARS, StudentYearPrediction.VOLUME):
        print(f"Predicting first-years: {year}-{week}...")
        data = strategy.predict_nr_of_students(year, week, cfg.skip_years)
        if data is not None:
            strategy.postprocessor.prepare_data_for_output_prelim(
                data, year, week, data_cumulative, cfg.skip_years,
            )
            if cfg.data_option in (DataOption.CUMULATIVE, DataOption.BOTH_DATASETS):
                strategy.postprocessor.predict_with_ratio(data_cumulative, year)
            print("Postprocessing...")
            strategy.postprocessor.postprocess(year, week)

    strategy.postprocessor.ready_new_data()


def _save_results(strategy, cfg):
    if "test" not in cfg.filtering_path:
        if strategy.postprocessor.data is not None:
            print("Saving output...")
            strategy.postprocessor.save_output(cfg.student_year_prediction)
        else:
            print("No data to save. Saving output skipped.")


if __name__ == "__main__":
    main(sys.argv)
