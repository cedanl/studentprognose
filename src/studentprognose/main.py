import os
import sys

from studentprognose.cli import parse_args
from studentprognose.config import load_configuration
from studentprognose.data.loader import load_data
from studentprognose.data.prediction_validator import run_pre_prediction_checks
from studentprognose.utils.ci_subset import apply_ci_test_subset
from studentprognose.output.postprocessor import PostProcessor
from studentprognose.output.dashboard import DashboardBuilder
from studentprognose.output.validator import run_post_prediction_checks
from studentprognose.strategies import create_strategy
from studentprognose.utils.weeks import DataOption, StudentYearPrediction, HIGHER_YEARS_COLUMNS
from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK, WEEKS_PER_YEAR


def main(argv):
    cfg = parse_args(argv)

    # Step 0: Validate raw input data, then run ETL (skip both with --noetl)
    print("Loading configuration...")
    configuration = load_configuration(cfg.configuration_path)

    if not cfg.noetl:
        from studentprognose.data.validation import validate_raw_data
        from studentprognose.data.etl import run_etl
        validate_raw_data(configuration, yes=cfg.yes)
        run_etl(configuration)

    # Step 1: Load data
    filtering = load_configuration(cfg.filtering_path)

    print("Loading data...")
    datasets = load_data(configuration, cfg.data_option)
    if cfg.ci_test_n is not None:
        datasets = apply_ci_test_subset(cfg.ci_test_n, *datasets)

    _check_data_range(datasets, cfg)

    # Step 2: Initialize strategy (Individual / Cumulative / Combined)
    cwd = os.getcwd()
    strategy = create_strategy(cfg, datasets, configuration, cwd)

    PostProcessor.check_output_writable(
        cfg.data_option, cfg.student_year_prediction, cfg.ci_test_n,
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
    invalid_weeks = [w for w in cfg.weeks if w == FINAL_ACADEMIC_WEEK]
    if invalid_weeks:
        print(f"\nFout: week {FINAL_ACADEMIC_WEEK} is de laatste week van het academisch jaar.")
        print(f"  Er valt niets meer te voorspellen vanaf week {FINAL_ACADEMIC_WEEK} (pred_len = 0).")
        print(f"  Kies een eerdere week, bijvoorbeeld -w {FINAL_ACADEMIC_WEEK - 1} of -w 1:{FINAL_ACADEMIC_WEEK - 1}.")
        sys.exit(1)

    # Step 6: Print summary + predict (per year × week)
    _print_summary(datasets, cfg, strategy)
    for year in cfg.years:
        for week in cfg.weeks:
            _predict_and_postprocess(strategy, cfg, data_cumulative, year, week)

    # Step 7: Save output
    _save_results(strategy, cfg)


def cli():
    main(sys.argv)


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
        print("\nWaarschuwing: de gevraagde combinatie is niet (volledig) beschikbaar in de data.")
        print(f"  Beschikbare data: jaren {year_range}, weken {week_range}.")
        print(f"  Pas je flags aan tussen -y {year_range} en -w {week_range},")
        print("  of voeg nieuwe trainingsdata toe in data/input_raw/ om je gewenste tijdstip te voorspellen.")
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
        pred_len = (FINAL_ACADEMIC_WEEK + WEEKS_PER_YEAR - week) if week > FINAL_ACADEMIC_WEEK else (FINAL_ACADEMIC_WEEK - week)
        end_week = FINAL_ACADEMIC_WEEK
        start_week = week + 1 if week < FINAL_ACADEMIC_WEEK else (week + 1 if week < WEEKS_PER_YEAR else 1)
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
        if data_cumulative is not None and cfg.ci_test_n is None:
            run_pre_prediction_checks(
                data_cumulative, year, week, strategy.numerus_fixus_list, yes=cfg.yes,
            )

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

            if cfg.ci_test_n is None:
                run_post_prediction_checks(
                    strategy.postprocessor.data,
                    data_cumulative,
                    year,
                    week,
                    strategy.numerus_fixus_list,
                )

    strategy.postprocessor.ready_new_data()


def _save_results(strategy, cfg):
    if cfg.ci_test_n is None:
        if strategy.postprocessor.data is not None:
            print("Saving output...")
            strategy.postprocessor.save_output(cfg.student_year_prediction)

            data_cumulative = getattr(strategy, "data_cumulative", None)
            if data_cumulative is None:
                data_cumulative = getattr(
                    getattr(strategy, "cumulative", None), "data_cumulative", None,
                )
            xgboost_curve = getattr(strategy, "xgboost_curve", None)
            if xgboost_curve is None:
                xgboost_curve = getattr(
                    getattr(strategy, "individual", None), "xgboost_curve", None,
                )

            xgb_classifier_importance = getattr(
                getattr(strategy, "individual", None), "xgboost_importance", None,
            )
            xgb_regressor_importance = getattr(
                getattr(strategy, "cumulative", None), "xgboost_importance", None,
            )
            if xgb_regressor_importance is None and not hasattr(strategy, "individual"):
                xgb_regressor_importance = getattr(strategy, "xgboost_importance", None)

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
                data_xgboost_curve=xgboost_curve,
                xgb_classifier_importance=xgb_classifier_importance,
                xgb_regressor_importance=xgb_regressor_importance,
            )
            dashboard.build_and_save()
        else:
            print("No data to save. Saving output skipped.")
