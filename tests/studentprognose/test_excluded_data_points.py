"""Tests for excluded_data_points filter and configuration validation."""

import json

import pandas as pd
import pytest

from studentprognose.data.preprocessing.excluded_data_points import apply_excluded_data_points
from studentprognose.config import load_configuration, _validate_excluded_data_points


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(years, herkomsten=None, examens=None, opleidingen=None):
    n = len(years)
    return pd.DataFrame({
        "Collegejaar": years,
        "Herkomst": herkomsten or ["NL"] * n,
        "Examentype": examens or ["Bachelor"] * n,
        "Croho groepeernaam": opleidingen or ["B Opleiding"] * n,
    })


# ---------------------------------------------------------------------------
# apply_excluded_data_points
# ---------------------------------------------------------------------------

class TestApplyExcludedDataPoints:
    def test_empty_rules_returns_all_rows(self):
        df = _df([2020, 2021, 2022])
        result = apply_excluded_data_points(df, [], predict_year=2024)
        assert len(result) == 3

    def test_year_rule_removes_matching_year(self):
        df = _df([2020, 2021, 2022])
        result = apply_excluded_data_points(df, [{"year": 2020}], predict_year=2024)
        assert 2020 not in result["Collegejaar"].values
        assert len(result) == 2

    def test_year_before_removes_older_rows(self):
        df = _df([2018, 2019, 2020, 2021])
        result = apply_excluded_data_points(df, [{"year_before": 2020}], predict_year=2024)
        assert list(result["Collegejaar"]) == [2020, 2021]

    def test_year_after_removes_newer_rows(self):
        df = _df([2020, 2021, 2022, 2023])
        result = apply_excluded_data_points(df, [{"year_after": 2021}], predict_year=2024)
        assert list(result["Collegejaar"]) == [2020, 2021]

    def test_herkomst_filter(self):
        df = _df([2020, 2021], herkomsten=["NL", "EER"])
        result = apply_excluded_data_points(df, [{"herkomst": "EER"}], predict_year=2024)
        assert list(result["Herkomst"]) == ["NL"]

    def test_herkomst_list_filter(self):
        df = _df([2020, 2021, 2022], herkomsten=["NL", "EER", "Niet-EER"])
        result = apply_excluded_data_points(
            df, [{"herkomst": ["EER", "Niet-EER"]}], predict_year=2024
        )
        assert list(result["Herkomst"]) == ["NL"]

    def test_predict_year_always_protected(self):
        df = _df([2023, 2024])
        result = apply_excluded_data_points(df, [{"year": 2024}], predict_year=2024)
        assert 2024 in result["Collegejaar"].values

    def test_multiple_rules_are_ored(self):
        df = _df([2019, 2020, 2021, 2022])
        rules = [{"year": 2019}, {"year": 2021}]
        result = apply_excluded_data_points(df, rules, predict_year=2024)
        assert list(result["Collegejaar"]) == [2020, 2022]

    def test_keys_within_rule_are_anded(self):
        df = _df([2020, 2021], herkomsten=["NL", "EER"])
        rules = [{"year": 2020, "herkomst": "EER"}]
        result = apply_excluded_data_points(df, rules, predict_year=2024)
        # 2020+EER doesn't exist, so nothing removed
        assert len(result) == 2

    def test_combined_and_and_or(self):
        df = _df([2020, 2020, 2021], herkomsten=["NL", "EER", "NL"])
        # Remove (2020 AND EER)
        rules = [{"year": 2020, "herkomst": "EER"}]
        result = apply_excluded_data_points(df, rules, predict_year=2024)
        assert len(result) == 2
        # The NL-2020 row is kept
        assert any((result["Collegejaar"] == 2020) & (result["Herkomst"] == "NL"))

    def test_missing_column_raises_valueerror(self):
        df = _df([2020, 2021])  # heeft geen "Faculteit" kolom
        with pytest.raises(ValueError, match="bestaan"):
            apply_excluded_data_points(
                df, [{"examentype": "Bachelor"}], predict_year=2024,
                examentype_col="Faculteit",
            )

    def test_missing_herkomst_col_raises(self):
        df = _df([2020, 2021])
        with pytest.raises(ValueError, match="bestaan"):
            apply_excluded_data_points(
                df, [{"herkomst": "NL"}], predict_year=2024,
                herkomst_col="OnbekendeKolom",
            )

    def test_opleiding_filter(self):
        df = pd.DataFrame({
            "Collegejaar": [2020, 2021],
            "Herkomst": ["NL", "NL"],
            "Examentype": ["Bachelor", "Bachelor"],
            "Croho groepeernaam": ["B Psychologie", "B Biologie"],
        })
        result = apply_excluded_data_points(
            df, [{"opleiding": "B Psychologie"}], predict_year=2024
        )
        assert list(result["Croho groepeernaam"]) == ["B Biologie"]


# ---------------------------------------------------------------------------
# _validate_excluded_data_points
# ---------------------------------------------------------------------------

class TestValidateExcludedDataPoints:
    def test_empty_list_is_valid(self):
        _validate_excluded_data_points([], "cfg.json")  # no exception

    def test_valid_rule_passes(self):
        _validate_excluded_data_points([{"year": 2020, "herkomst": "EER"}], "cfg.json")

    def test_non_list_exits(self):
        with pytest.raises(SystemExit):
            _validate_excluded_data_points("not-a-list", "cfg.json")

    def test_non_dict_rule_exits(self):
        with pytest.raises(SystemExit):
            _validate_excluded_data_points(["not-a-dict"], "cfg.json")

    def test_unknown_key_exits(self):
        with pytest.raises(SystemExit):
            _validate_excluded_data_points([{"unknown_key": 2020}], "cfg.json")

    def test_empty_rule_exits(self):
        with pytest.raises(SystemExit):
            _validate_excluded_data_points([{}], "cfg.json")

    def test_non_int_year_exits(self):
        with pytest.raises(SystemExit):
            _validate_excluded_data_points([{"year": "2020"}], "cfg.json")

    def test_non_int_year_before_exits(self):
        with pytest.raises(SystemExit):
            _validate_excluded_data_points([{"year_before": 2020.5}], "cfg.json")


# ---------------------------------------------------------------------------
# load_configuration — excluded_data_points scoping
# ---------------------------------------------------------------------------

class TestLoadConfigurationScoping:
    def test_filtering_json_without_key_loads_without_validation(self, tmp_path):
        cfg = {"filtering": {"programme": [], "herkomst": [], "examentype": []}}
        f = tmp_path / "filtering.json"
        f.write_text(json.dumps(cfg))
        result = load_configuration(str(f))
        assert result == cfg

    def test_configuration_with_valid_excluded_data_points_loads(self, tmp_path):
        cfg = {
            "numerus_fixus": {},
            "excluded_data_points": [{"year": 2020}],
        }
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        result = load_configuration(str(f))
        assert result["excluded_data_points"] == [{"year": 2020}]

    def test_configuration_with_invalid_excluded_data_points_exits(self, tmp_path):
        cfg = {"excluded_data_points": "not-a-list"}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with pytest.raises(SystemExit):
            load_configuration(str(f))
