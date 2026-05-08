"""Tests for the public Python API exported from studentprognose.__init__."""

import importlib
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


def test_all_exports_present():
    import studentprognose

    for name in studentprognose.__all__:
        obj = getattr(studentprognose, name, None)
        assert obj is not None, (
            f"'{name}' is listed in __all__ but cannot be imported from studentprognose"
        )
        assert hasattr(importlib.import_module("studentprognose"), name)
