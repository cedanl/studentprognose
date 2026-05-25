"""Tests for PostProcessor.

Bewaakt:
  - het gedrag van de merge met data_studentcount: zowel zonder
    Faculteit-kolom (huidige CEDA-default) als mét Faculteit-kolom
    (Radboud-casus).
  - de doorlopende `_totaal_<sy>_<do>.xlsx`-audittrail (idempotente upsert
    per Collegejaar+Weeknummer).
"""

import os

import pandas as pd

from studentprognose.output.postprocessor import PostProcessor
from studentprognose.utils.weeks import DataOption, StudentYearPrediction


def _make_postprocessor(tmp_path, data_studentcount):
    os.makedirs(tmp_path / "data" / "output", exist_ok=True)
    return PostProcessor(
        configuration={"numerus_fixus": {}},
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


def _make_run_data(year=2024, week=10, sarima_cumulative=(100.0, 50.0)):
    """Bouw een minimale prognose-DataFrame met de standaard sleutelkolommen."""
    return pd.DataFrame({
        "Collegejaar":        [year, year],
        "Weeknummer":         [week, week],
        "Croho groepeernaam": ["B Bedrijfskunde", "B Bedrijfskunde"],
        "Herkomst":           ["NL", "EER"],
        "Examentype":         ["Bachelor", "Bachelor"],
        "Faculteit":          ["FdM", "FdM"],
        "SARIMA_cumulative":  list(sarima_cumulative),
        "Voorspelde vooraanmelders": [120.0, 60.0],
    })


def _audittrail_postprocessor(tmp_path):
    os.makedirs(tmp_path / "data" / "output", exist_ok=True)
    return PostProcessor(
        configuration={"numerus_fixus": {}},
        data_latest=None,
        ensemble_weights=None,
        data_studentcount=None,
        cwd=str(tmp_path),
        data_option=DataOption.CUMULATIVE,
        ci_test_n=None,
    )


class TestSaveTotaalAuditTrail:
    """Bewaakt de idempotente upsert van `_totaal_<sy>_<do>.xlsx`."""

    expected_path_parts = ("data", "output", "_totaal_first-years_cumulatief.xlsx")

    def _path(self, tmp_path):
        return tmp_path.joinpath(*self.expected_path_parts)

    def test_eerste_run_schrijft_alle_rijen_met_run_date(self, tmp_path):
        pp = _audittrail_postprocessor(tmp_path)
        pp.data = _make_run_data()

        pp.save_totaal_audit_trail(StudentYearPrediction.FIRST_YEARS)

        path = self._path(tmp_path)
        assert path.exists()
        result = pd.read_excel(path)
        assert len(result) == 2
        assert "Run_date" in result.columns
        assert result["Run_date"].notna().all()

    def test_zelfde_week_wordt_overschreven_niet_gedupliceerd(self, tmp_path):
        pp1 = _audittrail_postprocessor(tmp_path)
        pp1.data = _make_run_data(week=12, sarima_cumulative=(100.0, 50.0))
        pp1.save_totaal_audit_trail(StudentYearPrediction.FIRST_YEARS)

        # Tweede run met dezelfde (jaar, week, opleiding, herkomst, examentype):
        # waarden moeten overschreven worden.
        pp2 = _audittrail_postprocessor(tmp_path)
        pp2.data = _make_run_data(week=12, sarima_cumulative=(999.0, 888.0))
        pp2.save_totaal_audit_trail(StudentYearPrediction.FIRST_YEARS)

        result = pd.read_excel(self._path(tmp_path))
        assert len(result) == 2  # geen rij-duplicatie
        assert set(result["SARIMA_cumulative"]) == {999.0, 888.0}
        assert 100.0 not in result["SARIMA_cumulative"].values

    def test_andere_week_wordt_toegevoegd(self, tmp_path):
        pp1 = _audittrail_postprocessor(tmp_path)
        pp1.data = _make_run_data(week=11)
        pp1.save_totaal_audit_trail(StudentYearPrediction.FIRST_YEARS)

        pp2 = _audittrail_postprocessor(tmp_path)
        pp2.data = _make_run_data(week=12)
        pp2.save_totaal_audit_trail(StudentYearPrediction.FIRST_YEARS)

        result = pd.read_excel(self._path(tmp_path))
        assert len(result) == 4  # 2 rijen w11 + 2 rijen w12
        assert set(result["Weeknummer"]) == {11, 12}

    def test_schema_drift_crasht_niet(self, tmp_path):
        # Bootst een eerdere run na met een kleiner kolomschema.
        os.makedirs(tmp_path / "data" / "output", exist_ok=True)
        path = self._path(tmp_path)
        pd.DataFrame({
            "Collegejaar":        [2024],
            "Weeknummer":         [10],
            "Croho groepeernaam": ["B Bedrijfskunde"],
            "Herkomst":           ["NL"],
            "Examentype":         ["Bachelor"],
            "SARIMA_cumulative":  [100.0],
            # geen Faculteit, geen Run_date — kolomschema is bewust kleiner
        }).to_excel(path, index=False)

        pp = _audittrail_postprocessor(tmp_path)
        pp.data = _make_run_data(week=11)  # voegt o.a. Faculteit + nieuwe kolommen toe
        pp.save_totaal_audit_trail(StudentYearPrediction.FIRST_YEARS)

        result = pd.read_excel(path)
        # De oude rij blijft bestaan (week 10) plus de twee nieuwe (week 11).
        assert len(result) == 3
        assert {"Faculteit", "Run_date"}.issubset(result.columns)

    def test_geen_data_doet_niets(self, tmp_path):
        pp = _audittrail_postprocessor(tmp_path)
        pp.data = None
        pp.save_totaal_audit_trail(StudentYearPrediction.FIRST_YEARS)
        assert not self._path(tmp_path).exists()

    def test_bestandsnaam_volgt_modus_en_studentjaar(self, tmp_path):
        os.makedirs(tmp_path / "data" / "output", exist_ok=True)
        pp = PostProcessor(
            configuration={"numerus_fixus": {}},
            data_latest=None,
            ensemble_weights=None,
            data_studentcount=None,
            cwd=str(tmp_path),
            data_option=DataOption.BOTH_DATASETS,
            ci_test_n=None,
        )
        pp.data = _make_run_data()
        pp.save_totaal_audit_trail(StudentYearPrediction.VOLUME)

        expected = tmp_path / "data" / "output" / "_totaal_volume_beide.xlsx"
        assert expected.exists()
