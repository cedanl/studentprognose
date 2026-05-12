"""Tests for runtime configuration (cpu_count)."""

import json
from unittest.mock import patch

import pytest

from studentprognose.config import get_cpu_count, load_configuration


# ---------------------------------------------------------------------------
# get_cpu_count — resolution
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
        with patch("studentprognose.config.os.cpu_count", return_value=8):
            assert get_cpu_count({"runtime": {"cpu_count": 2}}) == 2

    def test_value_equal_to_available_is_used(self):
        with patch("studentprognose.config.os.cpu_count", return_value=4):
            assert get_cpu_count({"runtime": {"cpu_count": 4}}) == 4

    def test_value_above_available_is_capped_with_warning(self, capsys):
        with patch("studentprognose.config.os.cpu_count", return_value=4):
            result = get_cpu_count({"runtime": {"cpu_count": 16}})
        assert result == 4
        captured = capsys.readouterr()
        assert "Waarschuwing" in captured.out
        assert "16" in captured.out
        assert "4" in captured.out


# ---------------------------------------------------------------------------
# load_configuration — runtime validation
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
