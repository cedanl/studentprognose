"""Tests for the public Python API exported from studentprognose.__init__."""

import importlib


def test_load_configuration_importable():
    """load_configuration must be importable and fall back to bundled defaults."""
    from studentprognose import load_configuration

    # Pass a non-existent path so the function falls back to bundled defaults
    config = load_configuration("__nonexistent_path__.json")
    assert isinstance(config, dict)
    assert "paths" in config


def test_load_filtering_importable():
    """load_filtering must be importable and fall back to bundled base.json."""
    from studentprognose import load_filtering

    filtering = load_filtering("__nonexistent_path__.json")
    assert isinstance(filtering, dict)
    assert "filtering" in filtering


def test_pipeline_config_importable():
    """PipelineConfig must be importable and have sensible defaults."""
    from studentprognose import PipelineConfig

    cfg = PipelineConfig()
    assert cfg.noetl is False


def test_data_option_importable():
    """DataOption enum must be importable and have the expected members."""
    from studentprognose import DataOption

    assert DataOption.CUMULATIVE is not None
    assert DataOption.INDIVIDUAL is not None
    assert DataOption.BOTH_DATASETS is not None


def test_student_year_prediction_importable():
    """StudentYearPrediction enum must be importable and have the expected members."""
    from studentprognose import StudentYearPrediction

    assert StudentYearPrediction.FIRST_YEARS is not None
    assert StudentYearPrediction.HIGHER_YEARS is not None
    assert StudentYearPrediction.VOLUME is not None


def test_run_pipeline_from_dataframes_importable():
    """run_pipeline_from_dataframes must be importable from the top-level package."""
    from studentprognose import run_pipeline_from_dataframes

    assert callable(run_pipeline_from_dataframes)


def test_all_exports_present():
    """Every name listed in __all__ must be importable from the package."""
    import studentprognose

    for name in studentprognose.__all__:
        obj = getattr(studentprognose, name, None)
        assert obj is not None, (
            f"'{name}' is listed in __all__ but cannot be imported from studentprognose"
        )
        # Also verify it is importable via importlib
        importlib.import_module("studentprognose")
        assert hasattr(importlib.import_module("studentprognose"), name)
