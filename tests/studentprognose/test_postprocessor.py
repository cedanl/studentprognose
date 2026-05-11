"""Tests for PostProcessor.prepare_data_for_output_prelim.

Bewaakt het gedrag van de merge met data_studentcount: zowel zonder
Faculteit-kolom (huidige CEDA-default) als mét Faculteit-kolom (Radboud-casus).
"""

import os

import pandas as pd

from studentprognose.output.postprocessor import PostProcessor
from studentprognose.utils.weeks import DataOption


def _make_postprocessor(tmp_path, data_studentcount):
    cfg = {
        "numerus_fixus": {},
        "ensemble_override_cumulative": [],
        "ensemble_weights": {
            "master_week_17_23": {"individual": 0.5, "cumulative": 0.5},
            "week_30_34": {"individual": 0.5, "cumulative": 0.5},
            "week_35_37": {"individual": 0.5, "cumulative": 0.5},
            "default": {"individual": 0.5, "cumulative": 0.5},
        },
    }
    os.makedirs(tmp_path / "data" / "output", exist_ok=True)
    return PostProcessor(
        configuration=cfg,
        data_latest=None,
        ensemble_weights=None,
        data_studentcount=data_studentcount,
        cwd=str(tmp_path),
        data_option=DataOption.CUMULATIVE,
        ci_test_n=None,
    )


def _make_input_data():
    """Predictie-data met Faculteit (zoals uit cumulatieve aanmelddata komt)."""
    return pd.DataFrame({
        "Croho groepeernaam": ["B Bedrijfskunde", "B Bedrijfskunde"],
        "Faculteit":          ["FdM",             "FdM"],
        "Examentype":         ["Bachelor",        "Bachelor"],
        "Collegejaar":        [2024,              2024],
        "Herkomst":           ["NL",              "EER"],
        "Weeknummer":         [10,                10],
        "SARIMA_cumulative":  [100.0,             50.0],
        "SARIMA_individual":  [110.0,             55.0],
        "Voorspelde vooraanmelders": [120.0,      60.0],
    })


class TestStudentcountMergeWithoutFaculteit:
    """Bestaand gedrag — studentcount zonder Faculteit (CEDA-default)."""

    def test_merge_succeeds_and_preserves_faculteit(self, tmp_path):
        studentcount = pd.DataFrame({
            "Croho groepeernaam": ["B Bedrijfskunde", "B Bedrijfskunde"],
            "Collegejaar":        [2024,              2024],
            "Herkomst":           ["NL",              "EER"],
            "Examentype":         ["Bachelor",        "Bachelor"],
            "Aantal_studenten":   [200,               80],
        })
        pp = _make_postprocessor(tmp_path, studentcount)
        pp.prepare_data_for_output_prelim(_make_input_data(), year=2024, week=10)

        assert "Faculteit" in pp.data.columns
        assert "Faculteit_x" not in pp.data.columns
        assert "Faculteit_y" not in pp.data.columns
        assert list(pp.data["Faculteit"]) == ["FdM", "FdM"]
        assert sorted(pp.data["Aantal_studenten"].tolist()) == [80, 200]
        assert len(pp.data) == 2  # geen rij-multiplicatie


class TestStudentcountMergeWithFaculteit:
    """Radboud-casus — studentcount mét Faculteit-kolom."""

    def test_no_suffix_columns_after_merge(self, tmp_path):
        studentcount = pd.DataFrame({
            "Croho groepeernaam": ["B Bedrijfskunde", "B Bedrijfskunde"],
            "Faculteit":          ["FdM",             "FdM"],
            "Collegejaar":        [2024,              2024],
            "Herkomst":           ["NL",              "EER"],
            "Examentype":         ["Bachelor",        "Bachelor"],
            "Aantal_studenten":   [200,               80],
        })
        pp = _make_postprocessor(tmp_path, studentcount)
        pp.prepare_data_for_output_prelim(_make_input_data(), year=2024, week=10)

        assert "Faculteit" in pp.data.columns
        assert "Faculteit_x" not in pp.data.columns
        assert "Faculteit_y" not in pp.data.columns

    def test_multiple_faculteit_rows_sum_not_multiply(self, tmp_path):
        # Dezelfde opleiding bij twee faculteiten — moet worden gesommeerd, niet verdubbeld.
        studentcount = pd.DataFrame({
            "Croho groepeernaam": ["B Bedrijfskunde", "B Bedrijfskunde"],
            "Faculteit":          ["FdM",             "FNWI"],
            "Collegejaar":        [2024,              2024],
            "Herkomst":           ["NL",              "NL"],
            "Examentype":         ["Bachelor",        "Bachelor"],
            "Aantal_studenten":   [120,               80],
        })
        input_data = pd.DataFrame({
            "Croho groepeernaam": ["B Bedrijfskunde"],
            "Faculteit":          ["FdM"],
            "Examentype":         ["Bachelor"],
            "Collegejaar":        [2024],
            "Herkomst":           ["NL"],
            "Weeknummer":         [10],
            "SARIMA_cumulative":  [100.0],
            "SARIMA_individual":  [110.0],
            "Voorspelde vooraanmelders": [120.0],
        })
        pp = _make_postprocessor(tmp_path, studentcount)
        pp.prepare_data_for_output_prelim(input_data, year=2024, week=10)

        assert len(pp.data) == 1  # geen rij-multiplicatie
        assert pp.data["Aantal_studenten"].iloc[0] == 200  # 120 + 80

    def test_faculteit_from_input_data_is_preserved(self, tmp_path):
        # self.data['Faculteit'] moet uit de aanmelddata komen, niet uit studentcount.
        studentcount = pd.DataFrame({
            "Croho groepeernaam": ["B Bedrijfskunde"],
            "Faculteit":          ["ANDERE_FAC"],  # afwijkende waarde
            "Collegejaar":        [2024],
            "Herkomst":           ["NL"],
            "Examentype":         ["Bachelor"],
            "Aantal_studenten":   [200],
        })
        input_data = pd.DataFrame({
            "Croho groepeernaam": ["B Bedrijfskunde"],
            "Faculteit":          ["FdM"],
            "Examentype":         ["Bachelor"],
            "Collegejaar":        [2024],
            "Herkomst":           ["NL"],
            "Weeknummer":         [10],
            "SARIMA_cumulative":  [100.0],
            "SARIMA_individual":  [110.0],
            "Voorspelde vooraanmelders": [120.0],
        })
        pp = _make_postprocessor(tmp_path, studentcount)
        pp.prepare_data_for_output_prelim(input_data, year=2024, week=10)

        assert pp.data["Faculteit"].iloc[0] == "FdM"
