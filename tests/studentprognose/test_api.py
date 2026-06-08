"""Tests for the public Python API exported from studentprognose.__init__."""

import importlib

import pandas as pd
import pytest


def test_load_configuration_importable():
    from studentprognose import load_configuration

    config = load_configuration("__nonexistent_path__.json")
    assert isinstance(config, dict)
    assert "paths" in config


def test_load_filtering_importable():
    from studentprognose import load_filtering

    filtering = load_filtering("__nonexistent_path__.json")
    assert isinstance(filtering, dict)
    assert "filtering" in filtering


def test_load_defaults_filtering_importable():
    from studentprognose import load_defaults_filtering

    filtering = load_defaults_filtering()
    assert isinstance(filtering, dict)
    assert "filtering" in filtering


def test_pipeline_config_importable():
    from studentprognose import PipelineConfig

    cfg = PipelineConfig()
    assert cfg.noetl is False


def test_data_option_importable():
    from studentprognose import DataOption

    assert DataOption.CUMULATIVE is not None
    assert DataOption.INDIVIDUAL is not None
    assert DataOption.BOTH_DATASETS is not None


def test_student_year_prediction_importable():
    from studentprognose import StudentYearPrediction

    assert StudentYearPrediction.FIRST_YEARS is not None
    assert StudentYearPrediction.HIGHER_YEARS is not None
    assert StudentYearPrediction.VOLUME is not None


def test_run_pipeline_cli_importable():
    from studentprognose import run_pipeline_cli

    assert callable(run_pipeline_cli)


def test_run_pipeline_from_dataframes_importable():
    from studentprognose import run_pipeline_from_dataframes

    assert callable(run_pipeline_from_dataframes)


def test_run_pipeline_from_dataframes_rejects_final_week():
    from studentprognose import run_pipeline_from_dataframes
    from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK

    with pytest.raises(ValueError, match=str(FINAL_ACADEMIC_WEEK)):
        run_pipeline_from_dataframes(year=2025, week=FINAL_ACADEMIC_WEEK)


def _make_prediction_frame():
    """Minimale voorspelling met exact de kolommen die de postprocessor selecteert."""
    return pd.DataFrame({
        "Croho groepeernaam": ["B Foo"],
        "Faculteit":          ["FdM"],
        "Examentype":         ["Bachelor"],
        "Collegejaar":        [2025],
        "Herkomst":           ["NL"],
        "Weeknummer":         [10],
        "SARIMA_cumulative":  [100.0],
        "SARIMA_individual":  [110.0],
        "Voorspelde vooraanmelders": [120.0],
    })


def _install_fake_strategy(monkeypatch, data_option):
    """Vervang create_strategy door een lichtgewicht fake.

    Doel: de save_output-gating in de orchestratielaag end-to-end via de publieke
    ``run_pipeline_from_dataframes`` testen, zónder de zware SARIMA/XGBoost-compute
    (conform de projectconventie dat unit tests de volledige pipeline niet draaien).
    De fake gebruikt een ECHTE PostProcessor, zodat de echte ``save_output_prelim``-
    schrijfpoging daadwerkelijk wordt gegate.
    """
    import studentprognose.main as main_mod
    from studentprognose.config import load_defaults
    from studentprognose.output.postprocessor import PostProcessor

    class _FakeStrategy:
        def __init__(self, cwd, opt):
            configuration = load_defaults()
            self.postprocessor = PostProcessor(
                configuration=configuration,
                data_latest=None,
                ensemble_weights=None,
                data_studentcount=None,
                cwd=cwd,
                data_option=opt,
                ci_test_n=None,
            )
            self.numerus_fixus_list = configuration["numerus_fixus"]
            self.programme_filtering = []

        def preprocess(self):
            return None

        def set_filtering(self, *args, **kwargs):
            pass

        def predict_nr_of_students(self, year, week, skip_years=0):
            return _make_prediction_frame()

    def _fake_create_strategy(cfg, datasets, configuration, cwd):
        return _FakeStrategy(cwd, cfg.data_option)

    monkeypatch.setattr(main_mod, "create_strategy", _fake_create_strategy)


@pytest.mark.parametrize("save_output, expect_files", [(False, False), (True, True)])
def test_run_pipeline_respects_save_output(tmp_path, monkeypatch, save_output, expect_files):
    """save_output=False schrijft niets naar data/output/; save_output=True wel (issue #219)."""
    from studentprognose import run_pipeline_from_dataframes, DataOption

    _install_fake_strategy(monkeypatch, DataOption.INDIVIDUAL)
    # tmp_path als cwd: zowel check_output_writable (os.getcwd()) als de
    # postprocessor (self.CWD) schrijven dan binnen tmp_path.
    monkeypatch.chdir(tmp_path)

    data_individual = pd.DataFrame({
        "Collegejaar": [2025],
        "Croho groepeernaam": ["B Foo"],
    })

    run_pipeline_from_dataframes(
        year=2025,
        week=10,
        data_individual=data_individual,
        dataset=DataOption.INDIVIDUAL,
        save_output=save_output,
    )

    output_dir = tmp_path / "data" / "output"
    written = {p.name for p in output_dir.glob("*")} if output_dir.exists() else set()

    if expect_files:
        # Criterium: bij save_output=True blijven álle outputtypen behouden —
        # het tussenresultaat, het eindresultaat én de audittrail.
        expected = {
            "output_prelim_individueel.xlsx",
            "output_first-years_individueel.xlsx",
            "_totaal_first-years_individueel.xlsx",
        }
        assert expected <= written, f"ontbrekende outputs: {expected - written}"
    else:
        assert written == set(), (
            f"save_output=False mag niets schrijven, maar vond {sorted(written)}"
        )


def test_all_exports_present():
    import studentprognose

    for name in studentprognose.__all__:
        obj = getattr(studentprognose, name, None)
        assert obj is not None, (
            f"'{name}' is listed in __all__ but cannot be imported from studentprognose"
        )
        assert hasattr(importlib.import_module("studentprognose"), name)
