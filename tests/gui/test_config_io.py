"""Tests voor gui.config_io — laden, valideren, opslaan van configuratie."""

import json

import pytest

from gui import config_io


def test_roundtrip_load_save(tmp_path):
    path = tmp_path / "configuration.json"
    original = {"model_config": {"min_training_year": 2016}, "numerus_fixus": {}}
    path.write_text(json.dumps(original), encoding="utf-8")

    loaded = config_io.load_config(str(path))
    assert loaded == original

    loaded["model_config"]["min_training_year"] = 2018
    config_io.save_config(str(path), loaded)

    reloaded = config_io.load_config(str(path))
    assert reloaded["model_config"]["min_training_year"] == 2018


def test_save_is_readable_json_utf8(tmp_path):
    path = tmp_path / "c.json"
    config_io.save_config(str(path), {"a": "café", "b": [1, 2]})
    text = path.read_text(encoding="utf-8")
    assert "café" in text  # geen \u-escapes
    assert text.endswith("\n")


def test_validate_ensemble_weights_ok():
    weights = {
        "default": {"individual": 0.5, "cumulative": 0.5},
        "week_30_34": {"individual": 0.6, "cumulative": 0.4},
    }
    assert config_io.validate_ensemble_weights(weights) == []


def test_validate_ensemble_weights_detects_bad_sum():
    weights = {"default": {"individual": 0.7, "cumulative": 0.5}}
    errors = config_io.validate_ensemble_weights(weights)
    assert len(errors) == 1
    assert "default" in errors[0]


def test_validate_ensemble_weights_missing_key():
    weights = {"default": {"individual": 0.5}}
    errors = config_io.validate_ensemble_weights(weights)
    assert len(errors) == 1
    assert "cumulative" in errors[0]


def test_parse_json_rejects_non_object():
    with pytest.raises(ValueError, match="JSON-object"):
        config_io.parse_json("[1, 2, 3]")


def test_parse_json_rejects_invalid():
    with pytest.raises(json.JSONDecodeError):
        config_io.parse_json("{niet: geldig}")


def test_validate_config_aggregates_ensemble_errors():
    config = {"ensemble_weights": {"default": {"individual": 0.9, "cumulative": 0.9}}}
    assert len(config_io.validate_config(config)) == 1
