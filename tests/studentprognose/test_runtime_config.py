"""Tests for runtime configuration (cpu_count)."""

import json
from unittest.mock import patch

import pytest

from studentprognose.config import get_cpu_count, load_configuration


# ---------------------------------------------------------------------------
# get_cpu_count — pure lookup (no side-effects)
# ---------------------------------------------------------------------------

class TestGetCpuCount:
    def test_none_falls_back_to_os_cpu_count(self):
        with patch("studentprognose.config.os.cpu_count", return_value=8):
            assert get_cpu_count({"runtime": {"cpu_count": None}}) == 8

    def test_missing_runtime_falls_back_to_os_cpu_count(self):
        with patch("studentprognose.config.os.cpu_count", return_value=4):
            assert get_cpu_count({}) == 4

    def test_os_returns_none_falls_back_to_one(self):
        with patch("studentprognose.config.os.cpu_count", return_value=None):
            assert get_cpu_count({}) == 1

    def test_explicit_value_used_when_below_available(self):
        # Note: bound checks happen in _validate_runtime — here we just verify
        # that whatever sits in cfg is returned as-is.
        with patch("studentprognose.config.os.cpu_count", return_value=8):
            assert get_cpu_count({"runtime": {"cpu_count": 2}}) == 2

    def test_value_equal_to_available_is_used(self):
        with patch("studentprognose.config.os.cpu_count", return_value=4):
            assert get_cpu_count({"runtime": {"cpu_count": 4}}) == 4

    def test_returns_cfg_value_verbatim_when_above_available(self):
        # get_cpu_count itself does NOT cap. Capping is _validate_runtime's job.
        # This test pins that contract so a future regression doesn't reintroduce
        # the per-call warning.
        with patch("studentprognose.config.os.cpu_count", return_value=4):
            assert get_cpu_count({"runtime": {"cpu_count": 99}}) == 99

    def test_no_side_effects_on_repeated_calls(self, capsys):
        # Repeated calls must NOT print anything — proves there's no per-call
        # warning regression even when cfg has a high value.
        cfg = {"runtime": {"cpu_count": 99}}
        with patch("studentprognose.config.os.cpu_count", return_value=4):
            for _ in range(20):
                get_cpu_count(cfg)
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# load_configuration — runtime validation + one-time cap
# ---------------------------------------------------------------------------

class TestLoadConfigurationRuntime:
    def test_default_runtime_loads(self, tmp_path):
        cfg = {"numerus_fixus": {}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        result = load_configuration(str(f))
        assert result["runtime"]["cpu_count"] is None

    def test_explicit_cpu_count_loads(self, tmp_path):
        cfg = {"runtime": {"cpu_count": 4}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with patch("studentprognose.config.os.cpu_count", return_value=8):
            result = load_configuration(str(f))
        assert result["runtime"]["cpu_count"] == 4

    def test_zero_cpu_count_exits(self, tmp_path):
        cfg = {"runtime": {"cpu_count": 0}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with pytest.raises(SystemExit) as exc:
            load_configuration(str(f))
        assert exc.value.code == 1

    def test_negative_cpu_count_exits(self, tmp_path):
        cfg = {"runtime": {"cpu_count": -1}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with pytest.raises(SystemExit) as exc:
            load_configuration(str(f))
        assert exc.value.code == 1

    def test_non_int_cpu_count_exits(self, tmp_path):
        cfg = {"runtime": {"cpu_count": "vier"}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with pytest.raises(SystemExit) as exc:
            load_configuration(str(f))
        assert exc.value.code == 1

    def test_bool_cpu_count_exits(self, tmp_path):
        # bool is a subclass of int in Python — must be rejected explicitly,
        # otherwise `True` would silently mean 1 core.
        cfg = {"runtime": {"cpu_count": True}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with pytest.raises(SystemExit) as exc:
            load_configuration(str(f))
        assert exc.value.code == 1

    def test_runtime_not_a_dict_exits(self, tmp_path):
        cfg = {"runtime": "not-a-dict"}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with pytest.raises(SystemExit) as exc:
            load_configuration(str(f))
        assert exc.value.code == 1

    def test_value_above_available_is_capped_with_one_time_warning(self, tmp_path, capsys):
        # Capping happens during load_configuration, not during get_cpu_count.
        # The warning is printed exactly once — directly tackling the bug
        # that triggered this refactor.
        cfg = {"runtime": {"cpu_count": 16}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with patch("studentprognose.config.os.cpu_count", return_value=4):
            result = load_configuration(str(f))
        assert result["runtime"]["cpu_count"] == 4
        out = capsys.readouterr().out
        assert out.count("Verlaagd naar") == 1, (
            f"Verwacht exact één cap-waarschuwing, kreeg {out.count('Verlaagd naar')}.\n"
            f"Output: {out!r}"
        )
        assert "16" in out
        assert "4" in out

    def test_capped_value_is_persisted_in_cfg(self, tmp_path, capsys):
        # Once capped, subsequent get_cpu_count() calls must see the new value
        # without re-warning — guarantees idempotency across the predict loop.
        cfg = {"runtime": {"cpu_count": 99}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with patch("studentprognose.config.os.cpu_count", return_value=2):
            result = load_configuration(str(f))
            capsys.readouterr()  # leeg na load
            for _ in range(10):
                assert get_cpu_count(result) == 2
        assert capsys.readouterr().out == "", "get_cpu_count mag na load niet meer waarschuwen"

    def test_value_below_available_keeps_value_and_no_warning(self, tmp_path, capsys):
        cfg = {"runtime": {"cpu_count": 2}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with patch("studentprognose.config.os.cpu_count", return_value=8):
            result = load_configuration(str(f))
        assert result["runtime"]["cpu_count"] == 2
        assert "Verlaagd naar" not in capsys.readouterr().out

    def test_value_equal_to_available_no_warning(self, tmp_path, capsys):
        cfg = {"runtime": {"cpu_count": 4}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with patch("studentprognose.config.os.cpu_count", return_value=4):
            result = load_configuration(str(f))
        assert result["runtime"]["cpu_count"] == 4
        assert "Verlaagd naar" not in capsys.readouterr().out

    def test_os_returns_none_caps_to_one(self, tmp_path, capsys):
        # Edge case: cgroup-constrained container where os.cpu_count() is None.
        # Any positive request must cap to 1.
        cfg = {"runtime": {"cpu_count": 8}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with patch("studentprognose.config.os.cpu_count", return_value=None):
            result = load_configuration(str(f))
        assert result["runtime"]["cpu_count"] == 1
        assert "Verlaagd naar 1" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# In-memory pad: run_pipeline_from_dataframes valideert ook
# ---------------------------------------------------------------------------

class TestInMemoryConfigValidation:
    """Verzeker dat het in-memory pad (run_pipeline_from_dataframes) dezelfde
    capping toepast als de CLI. Anders zou een gebruiker die load_defaults()
    gebruikt en handmatig cpu_count zet alsnog ongecapt door joblib gaan.

    We mocken hier alleen het stuk waar de validatie wordt aangeroepen — niet
    de hele pipeline — om snel en focused te blijven.
    """

    def test_validate_runtime_caps_in_memory_cfg(self, capsys):
        from studentprognose.config import _validate_runtime, get_cpu_count

        cfg = {"runtime": {"cpu_count": 99}}
        with patch("studentprognose.config.os.cpu_count", return_value=4):
            _validate_runtime(cfg, "<in-memory configuration>")
        assert cfg["runtime"]["cpu_count"] == 4
        out = capsys.readouterr().out
        assert "Verlaagd naar 4" in out
        assert "<in-memory configuration>" in out
        # Subsequent get_cpu_count returns capped value zonder side-effect
        with patch("studentprognose.config.os.cpu_count", return_value=4):
            assert get_cpu_count(cfg) == 4
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# load_configuration — regressor_params / tuning_grid validatie
# ---------------------------------------------------------------------------

class TestRegressorParamsValidation:
    def test_valid_regressor_params_loads(self, tmp_path):
        cfg = {"model_config": {"regressor_params": {"xgboost": {"max_depth": 3}}}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        result = load_configuration(str(f))
        assert result["model_config"]["regressor_params"]["xgboost"] == {"max_depth": 3}

    def test_regressor_params_not_dict_exits(self, tmp_path):
        cfg = {"model_config": {"regressor_params": [1, 2, 3]}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with pytest.raises(SystemExit) as exc:
            load_configuration(str(f))
        assert exc.value.code == 1

    def test_regressor_params_unknown_regressor_exits(self, tmp_path):
        cfg = {"model_config": {"regressor_params": {"nonexistent": {"max_depth": 3}}}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with pytest.raises(SystemExit) as exc:
            load_configuration(str(f))
        assert exc.value.code == 1

    def test_regressor_params_value_not_dict_exits(self, tmp_path):
        cfg = {"model_config": {"regressor_params": {"xgboost": 5}}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with pytest.raises(SystemExit) as exc:
            load_configuration(str(f))
        assert exc.value.code == 1

    def test_tuning_grid_not_dict_exits(self, tmp_path):
        cfg = {"model_config": {"tuning_grid": [1, 2]}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with pytest.raises(SystemExit) as exc:
            load_configuration(str(f))
        assert exc.value.code == 1

    def test_valid_tuning_grid_loads(self, tmp_path):
        cfg = {"model_config": {"tuning_grid": {"max_depth": [3, 5]}}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        result = load_configuration(str(f))
        assert result["model_config"]["tuning_grid"] == {"max_depth": [3, 5]}


class TestForecasterParamsValidation:
    def test_valid_forecaster_params_loads(self, tmp_path):
        cfg = {"model_config": {"forecaster_params": {"sarima": {"order": [1, 0, 1]}}}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        result = load_configuration(str(f))
        assert result["model_config"]["forecaster_params"]["sarima"]["order"] == [1, 0, 1]

    def test_forecaster_params_not_dict_exits(self, tmp_path):
        cfg = {"model_config": {"forecaster_params": [1, 2]}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with pytest.raises(SystemExit) as exc:
            load_configuration(str(f))
        assert exc.value.code == 1

    def test_forecaster_params_unknown_forecaster_exits(self, tmp_path):
        cfg = {"model_config": {"forecaster_params": {"prophet": {"order": [1, 0, 1]}}}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with pytest.raises(SystemExit) as exc:
            load_configuration(str(f))
        assert exc.value.code == 1

    def test_forecaster_params_value_not_dict_exits(self, tmp_path):
        cfg = {"model_config": {"forecaster_params": {"sarima": 5}}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with pytest.raises(SystemExit) as exc:
            load_configuration(str(f))
        assert exc.value.code == 1

    def test_sarima_tuning_grid_not_dict_exits(self, tmp_path):
        cfg = {"model_config": {"sarima_tuning_grid": [1, 2]}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with pytest.raises(SystemExit) as exc:
            load_configuration(str(f))
        assert exc.value.code == 1

    def test_valid_sarima_tuning_grid_loads(self, tmp_path):
        cfg = {"model_config": {"sarima_tuning_grid": {"order": [[1, 0, 1], [1, 1, 1]]}}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        result = load_configuration(str(f))
        assert result["model_config"]["sarima_tuning_grid"]["order"] == [[1, 0, 1], [1, 1, 1]]
