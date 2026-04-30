import os
import sys
from typing import Optional

import pandas as pd

from studentprognose.cli import PipelineConfig, parse_args
from studentprognose.config import load_configuration, load_defaults_filtering, load_filtering
from studentprognose.data.loader import load_data
from studentprognose.data.prediction_validator import run_pre_prediction_checks
from studentprognose.utils.ci_subset import apply_ci_test_subset
from studentprognose.output.postprocessor import PostProcessor
from studentprognose.output.dashboard import DashboardBuilder
from studentprognose.output.validator import run_post_prediction_checks
from studentprognose.strategies import create_strategy
from studentprognose.utils.weeks import (
    DataOption,
    StudentYearPrediction,
    HIGHER_YEARS_COLUMNS,
    detect_last_available,
)
from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK, WEEKS_PER_YEAR


def main(argv):
    cfg = parse_args(argv)

    if cfg.command == "init":
        from studentprognose.init import run_init

        run_init()
        return

    # Step 0: Validate raw input data, then run ETL (skip both with --noetl)
    print("Loading configuration...")
    configuration = load_configuration(cfg.configuration_path)

    if not cfg.noetl:
        from studentprognose.data.validation import validate_raw_data
        from studentprognose.data.etl import run_etl

        validate_raw_data(configuration, yes=cfg.yes)
        run_etl(configuration)

    # Step 1: Load data
    filtering = load_filtering(cfg.filtering_path)

    print("Loading data...")
    datasets = load_data(configuration, cfg.data_option)
    if cfg.ci_test_n is not None:
        datasets = apply_ci_test_subset(cfg.ci_test_n, *datasets)

    _apply_auto_defaults(datasets, cfg)
    _check_data_range(datasets, cfg)

    # Steps 2-7: gemeenschappelijke pipeline-kern
    cwd = os.getcwd()
    _run_pipeline_core(cfg, datasets, configuration, filtering, cwd)


def cli():
    main(sys.argv)


def run_pipeline_cli(argv):
    """Wrapper rond main() voor gebruik als geïmporteerde functie.

    Accepteert een argv-lijst zoals sys.argv, inclusief de programmanaam als
    eerste element. Gebruik run_pipeline_from_dataframes() als je data al
    in-memory hebt.

    Example::

        from studentprognose import run_pipeline_cli
        run_pipeline_cli(["studentprognose", "--noetl", "-d", "c", "-y", "2025", "-w", "10"])
    """
    main(argv)


def run_pipeline_from_dataframes(
    year: int,
    week: int,
    data_cumulative: Optional[pd.DataFrame] = None,
    data_individual: Optional[pd.DataFrame] = None,
    configuration: Optional[dict] = None,
    data_student_numbers: Optional[pd.DataFrame] = None,
    data_latest: Optional[pd.DataFrame] = None,
    data_weighted_ensemble: Optional[pd.DataFrame] = None,
    dataset: DataOption = DataOption.BOTH_DATASETS,
    student_year_prediction: StudentYearPrediction = StudentYearPrediction.FIRST_YEARS,
    skip_years: int = 0,
    filtering: Optional[dict] = None,
    cwd: Optional[str] = None,
    save_output: bool = True,
) -> Optional[pd.DataFrame]:
    """Run de voorspellingspipeline met data die al als DataFrames in-memory aanwezig is.

    Bedoeld voor cloud- en script-gebruikers die hun data niet als lokale bestanden
    hebben (bijv. geladen vanuit Azure Blob Storage of Amazon S3). ETL wordt overgeslagen.

    Args:
        year: Academisch jaar om te voorspellen (bijv. ``2025``).
        week: ISO-weeknummer waarvandaan voorspeld wordt (bijv. ``10``).
        data_cumulative: Cumulatief vooraanmeld-DataFrame. Vereist wanneer
            ``dataset`` ``CUMULATIVE`` of ``BOTH_DATASETS`` is.
        data_individual: Individueel aanmeld-DataFrame. Vereist wanneer
            ``dataset`` ``INDIVIDUAL`` of ``BOTH_DATASETS`` is.
        configuration: Configuratiedict (zelfde structuur als ``configuration.json``).
            Bij ``None`` worden de gebundelde package-defaults gebruikt.
        data_student_numbers: Studentaantallen eerstejaars DataFrame (optioneel).
        data_latest: Laatste output Excel DataFrame voor hogerejaarsvoorspellingen (optioneel).
        data_weighted_ensemble: Ensemble-gewichten DataFrame (optioneel).
        dataset: Welke dataset(s) te gebruiken. Standaard ``BOTH_DATASETS``.
        student_year_prediction: Welke studentcohort te voorspellen. Standaard ``FIRST_YEARS``.
        skip_years: Aantal meest recente jaren om uit training te sluiten (backtesting).
        filtering: Filteringdict (zelfde structuur als een filtering-JSON-bestand).
            Bij ``None`` wordt de gebundelde ``base.json`` gebruikt (geen filters).
        cwd: Werkmap voor uitvoerbestanden. Standaard ``os.getcwd()``.
        save_output: Sla uitvoer op naar schijf. Standaard ``True``. Zet op ``False``
            voor puur in-memory gebruik (bijv. bij cloud-pipelines die het resultaat
            zelf opslaan).

    Returns:
        Het gepostprocessde voorspellings-DataFrame, of ``None`` als geen rijen
        overeenkwamen met de filters of voorspellingscriteria.

    Example::

        import pandas as pd
        from studentprognose import run_pipeline_from_dataframes, DataOption

        df_cum = pd.read_csv("vooraanmeldingen_cumulatief.csv", sep=";", skiprows=[1])

        # Alleen het DataFrame teruggeven, niets naar schijf schrijven
        result = run_pipeline_from_dataframes(
            year=2025,
            week=10,
            data_cumulative=df_cum,
            dataset=DataOption.CUMULATIVE,
            save_output=False,
        )
    """
    from studentprognose.config import load_defaults

    if week == FINAL_ACADEMIC_WEEK:
        raise ValueError(
            f"week {FINAL_ACADEMIC_WEEK} is de laatste week van het academisch jaar. "
            f"Kies een eerdere week, bijvoorbeeld week {FINAL_ACADEMIC_WEEK - 1}."
        )

    if configuration is None:
        configuration = load_defaults()

    if filtering is None:
        filtering = load_defaults_filtering()

    if cwd is None:
        cwd = os.getcwd()

    cfg = PipelineConfig(
        years=[year],
        weeks=[week],
        weeks_specified=True,
        years_specified=True,
        data_option=dataset,
        student_year_prediction=student_year_prediction,
        skip_years=skip_years,
        noetl=True,
        yes=True,
    )

    datasets = (
        data_individual,
        data_cumulative,
        data_student_numbers,
        data_latest,
        data_weighted_ensemble,
    )

    return _run_pipeline_core(cfg, datasets, configuration, filtering, cwd, save_output=save_output)


def _run_pipeline_core(cfg, datasets, configuration, filtering, cwd, save_output: bool = True):
    """Gemeenschappelijke pipeline-kern: strategy, preprocessing, predict, opslaan.

    Zowel ``main()`` (CLI) als ``run_pipeline_from_dataframes()`` (Python API)
    delegeren hiernaartoe na hun eigen setup-stappen.
    """
    # Step 2: Initialize strategy (Individual / Cumulative / Combined)
    strategy = create_strategy(cfg, datasets, configuration, cwd)

    if save_output:
        PostProcessor.check_output_writable(
            cfg.data_option,
            cfg.student_year_prediction,
            cfg.ci_test_n,
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
        print(
            f"\nFout: week {FINAL_ACADEMIC_WEEK} is de laatste week van het academisch jaar."
        )
        print(
            f"  Er valt niets meer te voorspellen vanaf week {FINAL_ACADEMIC_WEEK} (pred_len = 0)."
        )
        print(
            f"  Kies een eerdere week, bijvoorbeeld -w {FINAL_ACADEMIC_WEEK - 1} of -w 1:{FINAL_ACADEMIC_WEEK - 1}."
        )
        sys.exit(1)

    # Step 6: Print summary + predict (per year × week)
    _print_summary(datasets, cfg, strategy)
    for year in cfg.years:
        for week in cfg.weeks:
            _predict_and_postprocess(strategy, cfg, data_cumulative, year, week)

    # Step 7: Save output
    if save_output:
        _save_results(strategy, cfg)

    return strategy.postprocessor.data


def _apply_auto_defaults(datasets, cfg):
    """Override week/jaar defaults met de laatste beschikbare waarden uit de data.

    Wordt alleen toegepast als de gebruiker ``-w`` en/of ``-y`` niet heeft opgegeven.
    """
    if cfg.weeks_specified and cfg.years_specified:
        return

    data_individual, data_cumulative, *_ = datasets
    data = data_cumulative if data_cumulative is not None else data_individual
    if data is None:
        return

    auto_year, auto_week = detect_last_available(data)

    changed = False

    if not cfg.years_specified:
        cfg.years = [auto_year]
        changed = True

    if not cfg.weeks_specified and auto_week is not None:
        cfg.weeks = [auto_week]
        changed = True

    if changed:
        year_str = str(cfg.years[0])
        week_str = str(cfg.weeks[0]) if auto_week is not None else "onbekend"
        print(
            f"Geen week/jaar opgegeven — automatisch gekozen op basis van beschikbare data: "
            f"jaar {year_str}, week {week_str}."
        )
        print("Geef -w en -y mee om een andere combinatie te gebruiken.")


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
        year_range = (
            f"{available_years[0]}-{available_years[-1]}"
            if len(available_years) > 1
            else str(available_years[0])
        )
        week_range = (
            f"{available_weeks[0]}-{available_weeks[-1]}"
            if len(available_weeks) > 1
            else str(available_weeks[0])
        )
        print(
            "\nWaarschuwing: de gevraagde combinatie is niet (volledig) beschikbaar in de data."
        )
        print(f"  Beschikbare data: jaren {year_range}, weken {week_range}.")
        print(f"  Pas je flags aan tussen -y {year_range} en -w {week_range},")
        print(
            "  of voeg nieuwe trainingsdata toe in data/input_raw/ om je gewenste tijdstip te voorspellen."
        )
        sys.exit(1)


def _print_summary(datasets, cfg, strategy):
    """Print a summary of what will be trained on and predicted."""
    data_individual, data_cumulative, *_ = datasets
    data = data_cumulative if data_cumulative is not None else data_individual

    # Training data range
    train_years = (
        sorted(int(y) for y in data["Collegejaar"].dropna().unique())
        if data is not None
        else []
    )
    train_year_str = (
        f"{train_years[0]}-{train_years[-1]}"
        if len(train_years) > 1
        else str(train_years[0])
        if train_years
        else "?"
    )

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
        pred_len = (
            (FINAL_ACADEMIC_WEEK + WEEKS_PER_YEAR - week)
            if week > FINAL_ACADEMIC_WEEK
            else (FINAL_ACADEMIC_WEEK - week)
        )
        end_week = FINAL_ACADEMIC_WEEK
        start_week = (
            week + 1
            if week < FINAL_ACADEMIC_WEEK
            else (week + 1 if week < WEEKS_PER_YEAR else 1)
        )
        pred_details.append(
            f"week {start_week} t/m {end_week} ({pred_len} weken vooruit)"
        )

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
    if student_year_prediction in (
        StudentYearPrediction.FIRST_YEARS,
        StudentYearPrediction.VOLUME,
    ):
        print("Preprocessing...")
        return strategy.preprocess()

    if student_year_prediction == StudentYearPrediction.HIGHER_YEARS:
        if strategy.postprocessor.data_latest is None:
            print("\nFout: geen 'totaal'-bestand gevonden voor hogerjaarsvoorspelling.")
            print("  Dit bestand wordt aangemaakt na een eerste first-years run.")
            print("  Draai eerst: studentprognose -d cumulative (of -d both)")
            sys.exit(1)
        strategy.postprocessor.data = strategy.postprocessor.data_latest[
            HIGHER_YEARS_COLUMNS
        ]
        return None


def _predict_and_postprocess(strategy, cfg, data_cumulative, year, week):
    if cfg.student_year_prediction in (
        StudentYearPrediction.FIRST_YEARS,
        StudentYearPrediction.VOLUME,
    ):
        if data_cumulative is not None and cfg.ci_test_n is None:
            run_pre_prediction_checks(
                data_cumulative,
                year,
                week,
                strategy.numerus_fixus_list,
                yes=cfg.yes,
            )

        print(f"Predicting first-years: {year}-{week}...")
        data = strategy.predict_nr_of_students(year, week, cfg.skip_years)
        if data is not None:
            strategy.postprocessor.prepare_data_for_output_prelim(
                data,
                year,
                week,
                data_cumulative,
                cfg.skip_years,
            )
            if cfg.data_option in (DataOption.CUMULATIVE, DataOption.BOTH_DATASETS):
                strategy.postprocessor.predict_with_ratio(data_cumulative, year)
                strategy.postprocessor.add_applicant_data(data_cumulative, year, week)
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

            if len(cfg.weeks) > 1:
                print(
                    f"Let op: het dashboard toont alleen de prognose voor week {cfg.weeks[-1]} (laatste week in de reeks)."
                )

            dd = strategy.get_dashboard_data()
            dashboard = DashboardBuilder(
                data=strategy.postprocessor.data,
                data_option=cfg.data_option,
                numerus_fixus_list=strategy.postprocessor.numerus_fixus_list,
                student_year_prediction=cfg.student_year_prediction,
                ci_test_n=cfg.ci_test_n,
                cwd=os.getcwd(),
                predict_week=cfg.weeks[-1] if cfg.weeks else None,
                data_cumulative=dd["data_cumulative"],
                data_studentcount=strategy.postprocessor.data_studentcount,
                data_xgboost_curve=dd["xgboost_curve"],
                xgb_classifier_importance=dd["xgb_classifier_importance"],
                xgb_regressor_importance=dd["xgb_regressor_importance"],
            )
            dashboard.build_and_save()
        else:
            print("No data to save. Saving output skipped.")
