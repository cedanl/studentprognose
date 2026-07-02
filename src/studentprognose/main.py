import os
import sys
import traceback
from typing import Optional

import pandas as pd

from studentprognose.cli import PipelineConfig, parse_args
from studentprognose.config import (
    load_configuration, load_defaults_filtering, load_filtering, get_final_academic_week,
)
from studentprognose.data.loader import load_data, _normalize_programme_code
from studentprognose.data.prediction_validator import run_pre_prediction_checks
from studentprognose.data.range_check import (
    detect_data_range_mismatch,
    format_api_range_error,
    format_cli_range_warning,
)
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
from studentprognose.utils.constants import WEEKS_PER_YEAR


_TUNE_TARGETS = ("regressor", "sarima")


def _resolve_tune_targets(tune) -> dict:
    """Vertaal de ``tune``-parameter naar een ``{target: grid|None}``-dict.

    Ondersteunde vormen:

    * ``False`` / ``None`` → ``{}`` (niet tunen).
    * ``True`` → ``{"regressor": None}`` (alleen stap 2; backwards compatibel).
    * ``"regressor"`` / ``"sarima"`` / ``"both"`` → die trap(pen), default-grid.
    * ``dict`` met sleutels uit ``{"regressor", "sarima"}`` → die trappen, elke
      waarde een eigen zoekruimte (of ``None``/niet-dict = default-grid).

    Een falsy waarde (``False``, ``None``, ``{}``) betekent: niet tunen.

    Returns:
        Dict van target → zoekruimte (``None`` = ingebouwde default-grid).

    Raises:
        ValueError: Bij een onbekende string, onbekende dict-sleutels, of een
            type dat geen bool/str/dict is.
    """
    if not tune:
        return {}
    if tune is True:
        return {"regressor": None}
    if isinstance(tune, str):
        key = tune.lower()
        if key == "both":
            return {t: None for t in _TUNE_TARGETS}
        if key in _TUNE_TARGETS:
            return {key: None}
        raise ValueError(
            f"Onbekende tune-waarde '{tune}'. Kies True, 'regressor', 'sarima', "
            "'both' of een dict zoals {'regressor': {...}, 'sarima': {...}}."
        )
    if isinstance(tune, dict):
        unknown = set(tune) - set(_TUNE_TARGETS)
        if unknown:
            raise ValueError(
                f"tune-dict bevat onbekende sleutels {sorted(unknown)}. "
                f"Geldig: {list(_TUNE_TARGETS)}. "
                "Bijv. tune={'regressor': {...}, 'sarima': {...}}."
            )
        # Een niet-dict zoekruimte (bijv. True) betekent: gebruik de default-grid.
        return {k: (v if isinstance(v, dict) else None) for k, v in tune.items()}
    raise ValueError(
        f"tune moet bool, str of dict zijn, niet {type(tune).__name__}."
    )


def main(argv):
    cfg = parse_args(argv)

    if cfg.command == "init":
        from studentprognose.init import run_init

        run_init()
        return

    if cfg.command == "benchmark":
        from studentprognose.benchmark.runner import main as run_benchmark

        if not cfg.dataset_specified:
            print("Geef -d c (cumulatief) of -d i (individueel) mee bij benchmark.")
            sys.exit(1)
        if cfg.data_option == DataOption.BOTH_DATASETS:
            print("Benchmark ondersteunt geen -d both. Kies -d c (cumulatief) of -d i (individueel).")
            sys.exit(1)

        predict_week = cfg.weeks[0] if cfg.weeks else 12
        run_benchmark(cfg.configuration_path, predict_week, cfg.data_option)
        return

    if cfg.command == "tune":
        from studentprognose.tuning_runner import run_tuning

        if not cfg.dataset_specified:
            print("Geef -d c (cumulatief) mee bij tune.")
            sys.exit(1)
        if cfg.data_option != DataOption.CUMULATIVE:
            print("Tune ondersteunt alleen -d c (cumulatief): het betreft de cumulatieve regressor.")
            sys.exit(1)

        predict_week = cfg.weeks[0] if cfg.weeks else 12
        run_tuning(cfg.configuration_path, predict_week, cfg.data_option, cfg.tune_target)
        return

    # Step 0: Validate raw input data, then run ETL (skip both with --noetl)
    print("Loading configuration...")
    configuration = load_configuration(cfg.configuration_path)

    if not cfg.noetl:
        from studentprognose.data.validation import validate_raw_data
        from studentprognose.data.etl import run_etl

        validate_raw_data(configuration, yes=cfg.yes, data_option=cfg.data_option)
        run_etl(configuration)

    # Step 1: Load data
    filtering = load_filtering(cfg.filtering_path)

    print("Loading data...")
    datasets = load_data(configuration, cfg.data_option)
    if cfg.ci_test_n is not None:
        datasets = apply_ci_test_subset(cfg.ci_test_n, *datasets)

    _apply_auto_defaults(datasets, cfg, configuration)
    _check_data_range(datasets, cfg)

    # Steps 2-7: gemeenschappelijke pipeline-kern
    cwd = os.getcwd()
    _run_pipeline_core(cfg, datasets, configuration, filtering, cwd)


def cli():
    # Forceer UTF-8 op stdout/stderr. Op Windows is de console default cp1252,
    # waarin de box-drawing en status-iconen (─, ✓, ✗) uit de validatie-overzichten
    # niet kunnen worden gecodeerd; resultaat is een UnicodeEncodeError voor er
    # ook maar iets gebeurt. errors="replace" voorkomt dat een eventueel
    # niet-reconfigureerbare stream alsnog crasht — slechtere weergave dan
    # crashen.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass
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
    tune: bool | str | dict = False,
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
        tune: Opt-in tuning voor het cumulatieve spoor. Standaard ``False``
            (gebruikt de waarden uit ``model_config.regressor_params`` /
            ``forecaster_params`` of de modeldefaults — snel en reproduceerbaar).
            Kies welke van de twee trappen getuned wordt:

            * ``True`` of ``"regressor"`` — alleen de regressor (stap 2,
              hyperparameters).
            * ``"sarima"`` — alleen SARIMA (stap 1, ARIMA-ordes).
            * ``"both"`` — beide trappen, elk apart gelogd met een eigen ✓-tabel.
            * ``{"regressor": {...}, "sarima": {...}}`` — beide/één trap met een
              eigen zoekruimte (een waarde ``None``/weggelaten = ingebouwde grid).

            Elke gekozen trap draait éénmalig een tijd-bewuste grid search op de
            meegegeven data, voorspelt met de beste waarden en koppelt ze terug in
            de configuratie zodat je ze kunt vastleggen. Het kandidatenoverzicht
            (met ✓ op de winnaar) gaat naar de console; de functie retourneert het
            voorspellings-DataFrame. Heeft alleen effect op het cumulatieve spoor
            (``CUMULATIVE``/``BOTH_DATASETS``).

    Returns:
        Het gepostprocessde voorspellings-DataFrame, of ``None`` als geen rijen
        overeenkwamen met de filters of voorspellingscriteria.

    Raises:
        ValueError: Als ``week`` gelijk is aan ``FINAL_ACADEMIC_WEEK`` (laatste week
            van het academisch jaar), of als ``year``/``week`` buiten de range van de
            meegegeven trainingsdata valt. De melding toont de wél beschikbare jaren
            en weken, zodat je year/week kunt bijstellen of trainingsdata kunt toevoegen.

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
    cfg, datasets, configuration, filtering, cwd = _prepare_in_memory_run(
        year,
        week,
        data_cumulative=data_cumulative,
        data_individual=data_individual,
        data_student_numbers=data_student_numbers,
        data_latest=data_latest,
        data_weighted_ensemble=data_weighted_ensemble,
        configuration=configuration,
        filtering=filtering,
        cwd=cwd,
        dataset=dataset,
        student_year_prediction=student_year_prediction,
        skip_years=skip_years,
        tune=tune,
    )

    return _run_pipeline_core(cfg, datasets, configuration, filtering, cwd, save_output=save_output)


def build_dashboard_from_dataframes(
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
) -> str:
    """Genereer de interactieve Plotly-dashboards vanuit in-memory DataFrames.

    Zelfstandige tegenhanger van :func:`run_pipeline_from_dataframes` voor cloud- en
    notebook-gebruikers (bijv. Microsoft Fabric) die hun data al als DataFrame in het
    geheugen hebben. Draait de voorspelpipeline (zonder iets naar schijf te schrijven)
    en schrijft daarna de HTML-dashboards naar ``<cwd>/data/output/visualisations/``.

    De data-parameters zijn identiek aan die van :func:`run_pipeline_from_dataframes`;
    roep deze functie ná (of in plaats van) die functie aan met dezelfde argumenten.

    Args:
        year: Academisch jaar om te voorspellen (bijv. ``2025``).
        week: ISO-weeknummer waarvandaan voorspeld wordt (bijv. ``10``).
        data_cumulative: Cumulatief vooraanmeld-DataFrame. Vereist wanneer ``dataset``
            ``CUMULATIVE`` of ``BOTH_DATASETS`` is.
        data_individual: Individueel aanmeld-DataFrame. Vereist wanneer ``dataset``
            ``INDIVIDUAL`` of ``BOTH_DATASETS`` is.
        configuration: Configuratiedict (zelfde structuur als ``configuration.json``).
            Bij ``None`` worden de gebundelde package-defaults gebruikt.
        data_student_numbers: Studentaantallen eerstejaars DataFrame (optioneel, maar
            nodig voor de realisatie-/conversiegrafieken in het dashboard).
        data_latest: Laatste output Excel DataFrame voor hogerejaarsvoorspellingen (optioneel).
        data_weighted_ensemble: Ensemble-gewichten DataFrame (optioneel).
        dataset: Welke dataset(s) te gebruiken. Standaard ``BOTH_DATASETS``.
        student_year_prediction: Welke studentcohort te voorspellen. Standaard ``FIRST_YEARS``.
        skip_years: Aantal meest recente jaren om uit training te sluiten (backtesting).
        filtering: Filteringdict (zelfde structuur als een filtering-JSON-bestand).
            Bij ``None`` wordt de gebundelde ``base.json`` gebruikt (geen filters).
        cwd: Werkmap voor de uitvoer. De dashboards komen in
            ``<cwd>/data/output/visualisations/``. Standaard ``os.getcwd()``.

    Returns:
        Het pad naar de map waarin de dashboards zijn geschreven
        (``<cwd>/data/output/visualisations/``).

    Raises:
        ValueError: Als ``week`` de laatste academische week is, als ``year``/``week``
            buiten de range van de meegegeven data valt, of als geen enkele rij
            overeenkwam met de filters/voorspellingscriteria (dan valt er niets te
            visualiseren).

    Example::

        from studentprognose import build_dashboard_from_dataframes, DataOption

        output_dir = build_dashboard_from_dataframes(
            year=2025,
            week=10,
            data_cumulative=df_cum,
            data_student_numbers=sc,
            dataset=DataOption.CUMULATIVE,
            configuration=config,
        )
        print(f"Dashboards geschreven naar {output_dir}")
    """
    cfg, datasets, configuration, filtering, cwd = _prepare_in_memory_run(
        year,
        week,
        data_cumulative=data_cumulative,
        data_individual=data_individual,
        data_student_numbers=data_student_numbers,
        data_latest=data_latest,
        data_weighted_ensemble=data_weighted_ensemble,
        configuration=configuration,
        filtering=filtering,
        cwd=cwd,
        dataset=dataset,
        student_year_prediction=student_year_prediction,
        skip_years=skip_years,
    )

    # Draai de pipeline zonder schijf-output; we willen alleen de uitgevoerde
    # strategy (met voorspellingen én dashboard-data) om er de dashboards uit te bouwen.
    strategy = _run_pipeline_strategy(
        cfg, datasets, configuration, filtering, cwd, save_output=False
    )

    # Zowel None (geen voorspelling) als een lege DataFrame (alle rijen weggefilterd)
    # afvangen: DashboardBuilder doet o.a. int(data["Collegejaar"].max()) en zou op
    # leeg met een cryptische int(NaN)-fout crashen i.p.v. deze heldere melding.
    data = strategy.postprocessor.data
    if data is None or data.empty:
        raise ValueError(
            "Geen data om een dashboard van te bouwen: geen enkele rij kwam overeen met "
            "de filters of voorspellingscriteria. Controleer year/week, dataset en filtering."
        )

    # Anders dan de CLI-route (_try_build_dashboard) laten we fouten hier wél
    # propageren: dit is een expliciete, op zichzelf staande dashboard-actie, dus
    # stil falen zou de gebruiker misleiden.
    builder = _make_dashboard_builder(strategy, cfg, cwd)
    builder.build_and_save()
    return builder.base_dir


def _prepare_in_memory_run(
    year: int,
    week: int,
    *,
    data_cumulative: Optional[pd.DataFrame],
    data_individual: Optional[pd.DataFrame],
    data_student_numbers: Optional[pd.DataFrame],
    data_latest: Optional[pd.DataFrame],
    data_weighted_ensemble: Optional[pd.DataFrame],
    configuration: Optional[dict],
    filtering: Optional[dict],
    cwd: Optional[str],
    dataset: DataOption,
    student_year_prediction: StudentYearPrediction,
    skip_years: int,
    tune: bool | str | dict = False,
) -> tuple:
    """Gedeelde setup voor de in-memory routes.

    Zowel :func:`run_pipeline_from_dataframes` als
    :func:`build_dashboard_from_dataframes` gebruiken dit: rangecontrole,
    config-/filtering-defaults, opt-in tuning (``tune``), runtime-validatie,
    programmesleutel-normalisatie en het opbouwen van de :class:`PipelineConfig`.
    De dashboard-route laat ``tune`` op de default ``False`` (geen tuning).

    Returns:
        Een tuple ``(cfg, datasets, configuration, filtering, cwd)`` klaar voor de
        pipeline-kern.
    """
    from studentprognose.config import load_defaults, _validate_runtime

    # Zelfde vangnet als de CLI: faal direct met een heldere melding wanneer year/week
    # buiten de range van de meegegeven data valt, in plaats van een stille None of een
    # kernel-crash verderop in de pipeline. Draait vóór config-werk (fail fast); de
    # 2-tuple wordt door de detector via `*_` afgehandeld.
    mismatch = detect_data_range_mismatch(
        (data_individual, data_cumulative), [year], [week]
    )
    if mismatch is not None:
        raise ValueError(format_api_range_error(year, week, mismatch))

    if configuration is None:
        configuration = load_defaults()

    # Opt-in tuning vertaalt naar model_config.tune_targets die de
    # CumulativeStrategy leest. Diepe kopie zodat we de configuration-dict van de
    # aanroeper niet muteren (de strategy schrijft de gevonden params terug in
    # deze kopie).
    tune_targets = _resolve_tune_targets(tune)
    if tune_targets:
        import copy

        configuration = copy.deepcopy(configuration)
        model_config = configuration.setdefault("model_config", {})
        model_config["tune_targets"] = tune_targets

    final_week = get_final_academic_week(configuration)
    if week == final_week:
        raise ValueError(
            f"week {final_week} is de laatste week van het academisch jaar. "
            f"Kies een eerdere week, bijvoorbeeld week {final_week - 1}."
        )

    # Valideer runtime ook in de in-memory pad: gebruikers die een eigen
    # configuration dict samenstellen (bv. via load_defaults() + handmatige
    # aanpassing) krijgen dan dezelfde capping + waarschuwing als CLI-gebruikers.
    _validate_runtime(configuration, "<in-memory configuration>")

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

    # De in-memory route slaat load_data over en mist daarmee de
    # programmesleutel-normalisatie. Pas die hier alsnog toe (zelfde sleutel-rollen
    # als load_data) op kopieën — anders lekt een object/string-sleutel uit de
    # meegegeven DataFrames in self.data en crasht een latere merge op str-vs-Int64.
    # Kopiëren voorkomt dat we de DataFrames van de aanroeper muteren.
    _roles = configuration.get("column_roles", {})
    if data_cumulative is not None:
        data_cumulative = data_cumulative.copy()
        _normalize_programme_code(data_cumulative, _roles.get("croho_source"))
    if data_student_numbers is not None:
        data_student_numbers = data_student_numbers.copy()
        _normalize_programme_code(data_student_numbers, _roles.get("programme"))
    if data_weighted_ensemble is not None:
        data_weighted_ensemble = data_weighted_ensemble.copy()
        _normalize_programme_code(data_weighted_ensemble, "Programme")

    datasets = (
        data_individual,
        data_cumulative,
        data_student_numbers,
        data_latest,
        data_weighted_ensemble,
    )

    return cfg, datasets, configuration, filtering, cwd


def _run_pipeline_core(cfg, datasets, configuration, filtering, cwd, save_output: bool = True):
    """Gemeenschappelijke pipeline-kern: strategy, preprocessing, predict, opslaan.

    Zowel ``main()`` (CLI) als ``run_pipeline_from_dataframes()`` (Python API)
    delegeren hiernaartoe na hun eigen setup-stappen. Geeft het gepostprocessde
    voorspellings-DataFrame terug (of ``None``).
    """
    strategy = _run_pipeline_strategy(cfg, datasets, configuration, filtering, cwd, save_output)
    return strategy.postprocessor.data


def _run_pipeline_strategy(cfg, datasets, configuration, filtering, cwd, save_output: bool = True):
    """Draai de pipeline-kern en geef de uitgevoerde strategy terug.

    Gescheiden van :func:`_run_pipeline_core` zodat aanroepers die het
    strategy-object zelf nodig hebben (bijv. :func:`build_dashboard_from_dataframes`
    voor de dashboard-data) er niet de omweg via ``postprocessor.data`` voor hoeven te
    maken.
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
    final_week = get_final_academic_week(configuration)
    invalid_weeks = [w for w in cfg.weeks if w == final_week]
    if invalid_weeks:
        print(
            f"\nFout: week {final_week} is de laatste week van het academisch jaar."
        )
        print(
            f"  Er valt niets meer te voorspellen vanaf week {final_week} (pred_len = 0)."
        )
        print(
            f"  Kies een eerdere week, bijvoorbeeld -w {final_week - 1} of -w 1:{final_week - 1}."
        )
        sys.exit(1)

    # Step 6: Print summary + predict (per year × week)
    _print_summary(datasets, cfg, strategy)
    for year in cfg.years:
        for week in cfg.weeks:
            _predict_and_postprocess(strategy, cfg, data_cumulative, year, week, save_output)

    # Step 7: Save output
    if save_output:
        _save_results(strategy, cfg)

    return strategy


def _apply_auto_defaults(datasets, cfg, configuration):
    """Override week/jaar defaults met de laatste beschikbare waarden uit de data.

    Wordt alleen toegepast als de gebruiker ``-w`` en/of ``-y`` niet heeft opgegeven.
    """
    if cfg.weeks_specified and cfg.years_specified:
        return

    data_individual, data_cumulative, *_ = datasets
    data = data_cumulative if data_cumulative is not None else data_individual
    if data is None:
        return

    auto_year, auto_week = detect_last_available(data, get_final_academic_week(configuration))

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
    """Check if requested years/weeks exist in the loaded data and exit (CLI-pad) if not."""
    mismatch = detect_data_range_mismatch(datasets, cfg.years, cfg.weeks)
    if mismatch is None:
        return

    print(format_cli_range_warning(mismatch))
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
    final_week = strategy.final_academic_week
    pred_details = []
    for week in cfg.weeks:
        pred_len = (
            (final_week + WEEKS_PER_YEAR - week)
            if week > final_week
            else (final_week - week)
        )
        end_week = final_week
        start_week = (
            week + 1
            if week < final_week
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


def _predict_and_postprocess(strategy, cfg, data_cumulative, year, week, save_output: bool = True):
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
            if save_output:
                strategy.postprocessor.save_output_prelim()
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
                    strategy.final_academic_week,
                )

    strategy.postprocessor.ready_new_data()


def _save_results(strategy, cfg):
    if cfg.ci_test_n is None:
        if strategy.postprocessor.data is not None:
            print("Saving output...")
            strategy.postprocessor.save_output(cfg.student_year_prediction)
            strategy.postprocessor.save_totaal_audit_trail(cfg.student_year_prediction)

            if cfg.dashboard:
                _try_build_dashboard(strategy, cfg)
        else:
            print("No data to save. Saving output skipped.")


def _make_dashboard_builder(strategy, cfg, cwd: str) -> DashboardBuilder:
    """Bouw een :class:`DashboardBuilder` uit een uitgevoerde strategy en config.

    Gedeeld door de CLI-route (:func:`_try_build_dashboard`) en de Python-API
    (:func:`build_dashboard_from_dataframes`) zodat de constructie op één plek staat.
    """
    dd = strategy.get_dashboard_data()
    return DashboardBuilder(
        data=strategy.postprocessor.data,
        data_option=cfg.data_option,
        numerus_fixus_list=strategy.postprocessor.numerus_fixus_list,
        student_year_prediction=cfg.student_year_prediction,
        ci_test_n=cfg.ci_test_n,
        cwd=cwd,
        predict_week=cfg.weeks[-1] if cfg.weeks else None,
        data_cumulative=dd["data_cumulative"],
        data_studentcount=strategy.postprocessor.data_studentcount,
        data_xgboost_curve=dd["xgboost_curve"],
        xgb_classifier_importance=dd["xgb_classifier_importance"],
        xgb_regressor_importance=dd["xgb_regressor_importance"],
        final_academic_week=strategy.final_academic_week,
    )


def _try_build_dashboard(strategy, cfg):
    """Build the Plotly dashboard. A failure here is logged but never crashes the pipeline."""
    if len(cfg.weeks) > 1:
        print(
            f"Let op: het dashboard toont alleen de prognose voor week {cfg.weeks[-1]} (laatste week in de reeks)."
        )

    try:
        _make_dashboard_builder(strategy, cfg, os.getcwd()).build_and_save()
    except Exception:
        log_path = os.path.join(os.getcwd(), "data", "output", "dashboard_error.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(
            f"Waarschuwing: dashboard niet gegenereerd. Zie {log_path} voor details."
        )
