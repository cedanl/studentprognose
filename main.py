import os
import sys

from src.cli import parse_args
from src.config import load_configuration
from src.data.loader import load_data
from src.utils.ci_subset import apply_ci_test_subset
from src.output.postprocessor import PostProcessor
from src.output.dashboard import DashboardBuilder
from src.strategies import create_strategy
from src.utils.weeks import DataOption, StudentYearPrediction, HIGHER_YEARS_COLUMNS


def main(argv):
    cfg = parse_args(argv)

    # Step 0: ETL (default — raw data → input data, skip with --noetl)
    if not cfg.noetl:
        from src.data.etl import run_etl
        run_etl(load_configuration(cfg.configuration_path))

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

    # Step 5: Validate prediction horizon
    invalid_weeks = [w for w in cfg.weeks if w == 38]
    if invalid_weeks:
        print(f"\nFout: week 38 is de laatste week van het academisch jaar.")
        print(f"  Er valt niets meer te voorspellen vanaf week 38 (pred_len = 0).")
        print(f"  Kies een eerdere week, bijvoorbeeld -w 37 of -w 1:37.")
        sys.exit(1)

    # Step 6: Print summary + predict (per year × week)
    _print_summary(datasets, cfg, strategy)
    for year in cfg.years:
        for week in cfg.weeks:
            _predict_and_postprocess(strategy, cfg, data_cumulative, year, week)

    # Step 7: Save output
    _save_results(strategy, cfg)


def _check_data_range(datasets, cfg):
    """Check if requested years/weeks exist in the loaded data and warn if not."""
    data_individual, data_cumulative, *_ = datasets

    # Pick whichever dataset is loaded based on the chosen mode
    data = data_cumulative if data_cumulative is not None else data_individual
    if data is None:
        return

    available_years = sorted(int(y) for y in data["Collegejaar"].dropna().unique())

    if not available_years:
        return

    missing_years = [y for y in cfg.years if y not in available_years]

    # Individual dataset has no Weeknummer column before preprocessing
    if "Weeknummer" in data.columns:
        available_weeks = sorted(int(w) for w in data["Weeknummer"].dropna().unique())
        missing_weeks = [w for w in cfg.weeks if w not in available_weeks]
    else:
        available_weeks = []
        missing_weeks = []

    if missing_years or missing_weeks:
        year_range = f"{available_years[0]}-{available_years[-1]}" if len(available_years) > 1 else str(available_years[0])
        week_range = f"{available_weeks[0]}-{available_weeks[-1]}" if len(available_weeks) > 1 else str(available_weeks[0])
        print(f"\nWaarschuwing: de gevraagde combinatie is niet (volledig) beschikbaar in de data.")
        print(f"  Beschikbare data: jaren {year_range}, weken {week_range}.")
        print(f"  Pas je flags aan tussen -y {year_range} en -w {week_range},")
        print(f"  of voeg nieuwe trainingsdata toe in data/input_raw/ om je gewenste tijdstip te voorspellen.")
        sys.exit(1)


def _print_summary(datasets, cfg, strategy):
    """Print a summary of what will be trained on and predicted."""
    data_individual, data_cumulative, *_ = datasets
    data = data_cumulative if data_cumulative is not None else data_individual

    # Training data range
    train_years = sorted(int(y) for y in data["Collegejaar"].dropna().unique()) if data is not None else []
    train_year_str = f"{train_years[0]}-{train_years[-1]}" if len(train_years) > 1 else str(train_years[0]) if train_years else "?"

    # Programmes
    programmes = strategy.programme_filtering
    if not programmes and data is not None:
        for col in ("Croho groepeernaam", "Groepeernaam Croho"):
            if col in data.columns:
                programmes = sorted(data[col].dropna().unique())
                break

    # Prediction details per week
    pred_details = []
    for week in cfg.weeks:
        pred_len = (38 + 52 - week) if week > 38 else (38 - week)
        end_week = 38
        start_week = week + 1 if week < 38 else (week + 1 if week < 52 else 1)
        pred_details.append(f"week {start_week} t/m {end_week} ({pred_len} weken vooruit)")

    print(f"\n{'=' * 40}")
    print(f"  Dataset:       {cfg.data_option.filename_suffix}")
    print(f"  Trainingsdata: jaren {train_year_str}")
    print(f"  Voorspelling:  jaar {cfg.years}, vanaf week {cfg.weeks}")
    for i, detail in enumerate(pred_details):
        prefix = "                 " if i > 0 else "  Voorspelt:     "
        print(f"{prefix}{detail}")
    print(f"  Opleidingen:   {', '.join(str(p) for p in programmes)}")
    print(f"{'=' * 40}\n")


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

            data_cumulative = getattr(strategy, "data_cumulative", None)
            if data_cumulative is None:
                data_cumulative = getattr(
                    getattr(strategy, "cumulative", None), "data_cumulative", None,
                )
            dashboard = DashboardBuilder(
                data=strategy.postprocessor.data,
                data_option=cfg.data_option,
                numerus_fixus_list=strategy.postprocessor.numerus_fixus_list,
                student_year_prediction=cfg.student_year_prediction,
                ci_test_n=cfg.ci_test_n,
                cwd=os.getcwd(),
                predict_week=cfg.weeks[-1] if cfg.weeks else None,
                data_cumulative=data_cumulative,
                data_studentcount=strategy.postprocessor.data_studentcount,
            )
            dashboard.build_and_save()
        else:
            print("No data to save. Saving output skipped.")


if __name__ == "__main__":
    main(sys.argv)
