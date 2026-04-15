import os
import re
import sys
from unittest.mock import patch

import pandas as pd
import pytest

from studentprognose.data.validation import (
    ValidationResult,
    _check_categoricals_per_programme,
    _check_nan_rate,
    _check_telbestand_completeness,
    _handle_result,
    _DEFAULT_VALIDATION_CFG,
    validate_raw_data,
)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

class TestValidationResult:
    def test_defaults_are_empty(self):
        r = ValidationResult()
        assert r.hard_errors == []
        assert r.soft_errors == []
        assert r.warnings == []

    def test_append_to_lists(self):
        r = ValidationResult()
        r.hard_errors.append("fout")
        r.soft_errors.append("probleem")
        r.warnings.append("waarschuwing")
        assert len(r.hard_errors) == 1
        assert len(r.soft_errors) == 1
        assert len(r.warnings) == 1


# ---------------------------------------------------------------------------
# _check_nan_rate
# ---------------------------------------------------------------------------

class TestCheckNanRate:
    def _df(self, values):
        return pd.DataFrame({"kolom": values})

    def test_no_nans_produces_no_entries(self):
        r = ValidationResult()
        _check_nan_rate(self._df([1, 2, 3]), "kolom", "bestand.csv", 0.05, 0.30, r)
        assert not r.warnings and not r.soft_errors

    def test_below_warning_threshold_produces_no_entries(self):
        r = ValidationResult()
        _check_nan_rate(self._df([None] + [1] * 99), "kolom", "bestand.csv", 0.05, 0.30, r)
        assert not r.warnings and not r.soft_errors

    def test_above_warning_threshold_produces_warning(self):
        r = ValidationResult()
        _check_nan_rate(self._df([None] * 10 + [1] * 90), "kolom", "bestand.csv", 0.05, 0.30, r)
        assert len(r.warnings) == 1
        assert not r.soft_errors

    def test_above_error_threshold_produces_soft_error(self):
        r = ValidationResult()
        _check_nan_rate(self._df([None] * 40 + [1] * 60), "kolom", "bestand.csv", 0.05, 0.30, r)
        assert len(r.soft_errors) == 1
        assert not r.warnings

    def test_missing_column_is_ignored(self):
        r = ValidationResult()
        df = pd.DataFrame({"andere_kolom": [1, 2, 3]})
        _check_nan_rate(df, "kolom", "bestand.csv", 0.05, 0.30, r)
        assert not r.warnings and not r.soft_errors

    def test_empty_dataframe_is_ignored(self):
        r = ValidationResult()
        _check_nan_rate(pd.DataFrame({"kolom": []}), "kolom", "bestand.csv", 0.05, 0.30, r)
        assert not r.warnings and not r.soft_errors


# ---------------------------------------------------------------------------
# _check_categoricals_per_programme
# ---------------------------------------------------------------------------

class TestCheckCategoricalsPerProgramme:
    def _df(self, values, programmes=None):
        data = {"Herkomst": values}
        if programmes is not None:
            data["Groepeernaam"] = programmes
        return pd.DataFrame(data)

    def test_all_valid_produces_no_entry(self):
        r = ValidationResult()
        df = self._df(["N", "E", "R", "N"], ["A", "A", "B", "B"])
        _check_categoricals_per_programme(df, "Herkomst", ["N", "E", "R"], "Groepeernaam", "f.csv", r)
        assert not r.soft_errors

    def test_invalid_value_produces_soft_error(self):
        r = ValidationResult()
        df = self._df(["N", "X"], ["A", "B"])
        _check_categoricals_per_programme(df, "Herkomst", ["N", "E", "R"], "Groepeernaam", "f.csv", r)
        assert len(r.soft_errors) == 1
        assert "X" in r.soft_errors[0]
        assert "B" in r.soft_errors[0]

    def test_nan_values_are_ignored(self):
        r = ValidationResult()
        df = self._df(["N", None], ["A", "B"])
        _check_categoricals_per_programme(df, "Herkomst", ["N", "E", "R"], "Groepeernaam", "f.csv", r)
        assert not r.soft_errors

    def test_without_programme_column_still_reports(self):
        r = ValidationResult()
        df = pd.DataFrame({"Herkomst": ["N", "ONBEKEND"]})
        _check_categoricals_per_programme(df, "Herkomst", ["N", "E", "R"], "Groepeernaam", "f.csv", r)
        assert len(r.soft_errors) == 1


# ---------------------------------------------------------------------------
# _check_telbestand_completeness
# ---------------------------------------------------------------------------

class TestCheckTelbestandCompleteness:
    def test_consecutive_weeks_no_warning(self):
        r = ValidationResult()
        files = [f"telbestandY2024W{w:02d}.csv" for w in range(1, 10)]
        _check_telbestand_completeness(files, {}, r)
        assert not r.warnings

    def test_small_gap_no_warning(self):
        r = ValidationResult()
        files = ["telbestandY2024W01.csv", "telbestandY2024W03.csv"]
        _check_telbestand_completeness(files, {}, r)
        assert not r.warnings

    def test_large_gap_produces_warning(self):
        r = ValidationResult()
        files = ["telbestandY2024W01.csv", "telbestandY2024W10.csv"]
        _check_telbestand_completeness(files, {}, r)
        assert len(r.warnings) == 1
        assert "2024" in r.warnings[0]

    def test_multiple_years_checked_independently(self):
        r = ValidationResult()
        files = [
            "telbestandY2023W01.csv", "telbestandY2023W02.csv",
            "telbestandY2024W01.csv", "telbestandY2024W15.csv",
        ]
        _check_telbestand_completeness(files, {}, r)
        assert len(r.warnings) == 1
        assert "2024" in r.warnings[0]


# ---------------------------------------------------------------------------
# _handle_result
# ---------------------------------------------------------------------------

class TestHandleResult:
    def test_no_issues_passes_silently(self, capsys):
        r = ValidationResult()
        _handle_result(r, yes=False)
        out = capsys.readouterr().out
        assert out == ""

    def test_warning_is_printed(self, capsys):
        r = ValidationResult()
        r.warnings.append("let op dit")
        _handle_result(r, yes=False)
        assert "let op dit" in capsys.readouterr().out

    def test_hard_error_exits(self):
        r = ValidationResult()
        r.hard_errors.append("kritieke fout")
        with pytest.raises(SystemExit):
            _handle_result(r, yes=False)

    def test_soft_error_with_yes_does_not_prompt(self, capsys):
        r = ValidationResult()
        r.soft_errors.append("klein probleem")
        _handle_result(r, yes=True)
        assert "--yes" in capsys.readouterr().out

    def test_soft_error_without_yes_prompts_and_j_continues(self, capsys):
        r = ValidationResult()
        r.soft_errors.append("klein probleem")
        with patch("builtins.input", return_value="j"):
            _handle_result(r, yes=False)
        assert "Doorgaan" in capsys.readouterr().out

    def test_soft_error_without_yes_prompts_and_n_exits(self):
        r = ValidationResult()
        r.soft_errors.append("klein probleem")
        with patch("builtins.input", return_value="n"):
            with pytest.raises(SystemExit):
                _handle_result(r, yes=False)

    def test_soft_error_eof_exits_gracefully(self):
        r = ValidationResult()
        r.soft_errors.append("klein probleem")
        with patch("builtins.input", side_effect=EOFError):
            with pytest.raises(SystemExit):
                _handle_result(r, yes=False)


# ---------------------------------------------------------------------------
# validate_raw_data integration (file-based with tmp_path)
# ---------------------------------------------------------------------------

def _make_valid_telbestand(path, year=2024, week=1):
    df = pd.DataFrame({
        "Studiejaar": [year],
        "Isatcode": ["12345"],
        "Groepeernaam": ["B Opleiding"],
        "Aantal": [10],
        "meercode_V": [2],
        "Herinschrijving": ["N"],
        "Hogerejaars": ["N"],
        "Herkomst": ["N"],
    })
    df.to_csv(path, sep=";", index=False)


def _make_configuration(tmp_path, telbestanden_dir, with_individueel=False, with_oktober=False):
    cfg = {
        "paths": {
            "path_raw_telbestanden": str(telbestanden_dir),
            "path_raw_individueel": str(tmp_path / "individuele_aanmelddata.csv") if with_individueel else "nonexistent.csv",
            "path_raw_october": str(tmp_path / "oktober_bestand.xlsx") if with_oktober else "nonexistent.xlsx",
        },
        "validation": {},
    }
    if with_individueel:
        pd.DataFrame({
            "Collegejaar": [2024],
            "Croho": ["12345"],
            "Inschrijfstatus": ["Ingeschreven"],
            "Datum Verzoek Inschr": ["2024-01-01"],
        }).to_csv(tmp_path / "individuele_aanmelddata.csv", sep=";", index=False)
    return cfg


class TestValidateRawDataIntegration:
    def test_valid_data_passes(self, tmp_path, monkeypatch):
        telbestanden_dir = tmp_path / "telbestanden"
        telbestanden_dir.mkdir()
        _make_valid_telbestand(telbestanden_dir / "telbestandY2024W01.csv")
        cfg = _make_configuration(tmp_path, telbestanden_dir)
        monkeypatch.chdir(tmp_path)
        validate_raw_data(cfg, yes=True)

    def test_missing_telbestanden_dir_exits(self, tmp_path, monkeypatch):
        cfg = _make_configuration(tmp_path, tmp_path / "nonexistent")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            validate_raw_data(cfg, yes=True)

    def test_missing_column_in_telbestand_exits(self, tmp_path, monkeypatch):
        telbestanden_dir = tmp_path / "telbestanden"
        telbestanden_dir.mkdir()
        pd.DataFrame({"Studiejaar": [2024], "Isatcode": ["12345"]}).to_csv(
            telbestanden_dir / "telbestandY2024W01.csv", sep=";", index=False
        )
        cfg = _make_configuration(tmp_path, telbestanden_dir)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            validate_raw_data(cfg, yes=True)

    def test_zero_meercode_triggers_soft_error_yes_passes(self, tmp_path, monkeypatch):
        telbestanden_dir = tmp_path / "telbestanden"
        telbestanden_dir.mkdir()
        df = pd.DataFrame({
            "Studiejaar": [2024],
            "Isatcode": ["12345"],
            "Groepeernaam": ["B Opleiding"],
            "Aantal": [10],
            "meercode_V": [0],
            "Herinschrijving": ["N"],
            "Hogerejaars": ["N"],
            "Herkomst": ["N"],
        })
        df.to_csv(telbestanden_dir / "telbestandY2024W01.csv", sep=";", index=False)
        cfg = _make_configuration(tmp_path, telbestanden_dir)
        monkeypatch.chdir(tmp_path)
        validate_raw_data(cfg, yes=True)

    def test_invalid_herkomst_prompts_and_n_exits(self, tmp_path, monkeypatch):
        telbestanden_dir = tmp_path / "telbestanden"
        telbestanden_dir.mkdir()
        df = pd.DataFrame({
            "Studiejaar": [2024],
            "Isatcode": ["12345"],
            "Groepeernaam": ["B Opleiding"],
            "Aantal": [10],
            "meercode_V": [2],
            "Herinschrijving": ["N"],
            "Hogerejaars": ["N"],
            "Herkomst": ["ONBEKEND"],
        })
        df.to_csv(telbestanden_dir / "telbestandY2024W01.csv", sep=";", index=False)
        cfg = _make_configuration(tmp_path, telbestanden_dir)
        monkeypatch.chdir(tmp_path)
        with patch("builtins.input", return_value="n"):
            with pytest.raises(SystemExit):
                validate_raw_data(cfg, yes=False)

    def test_yes_flag_skips_prompt_for_soft_errors(self, tmp_path, monkeypatch):
        telbestanden_dir = tmp_path / "telbestanden"
        telbestanden_dir.mkdir()
        df = pd.DataFrame({
            "Studiejaar": [2024],
            "Isatcode": ["12345"],
            "Groepeernaam": ["B Opleiding"],
            "Aantal": [10],
            "meercode_V": [2],
            "Herinschrijving": ["N"],
            "Hogerejaars": ["N"],
            "Herkomst": ["ONBEKEND"],
        })
        df.to_csv(telbestanden_dir / "telbestandY2024W01.csv", sep=";", index=False)
        cfg = _make_configuration(tmp_path, telbestanden_dir)
        monkeypatch.chdir(tmp_path)
        validate_raw_data(cfg, yes=True)
