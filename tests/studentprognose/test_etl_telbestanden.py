"""Tests voor de ETL-rowbind van telbestanden met configureerbare patronen
en VU-specifieke kolomsets (issue #199)."""

import pandas as pd
import pytest

from studentprognose.data.etl import _rowbind_and_reformat


VU_COLUMNS = [
    "HBO_WO", "Brincode", "Brin_volgnr", "Isatcode", "Type_HO",
    "Opl_vorm", "Voertaal", "Studiejaar", "Fixus", "Maand", "Herkomst",
    "Geslacht", "meercode_V", "meercode_A", "Status", "Hogerejaars",
    "Herinschrijving", "1cHO_L", "1cHO_K", "Aantal", "sw", "jw",
]


def _write_vu_telbestand(directory, filename, week):
    """Schrijf een telbestand met VU-kolomset (zonder Groepeernaam/Faculteit)."""
    row = {
        "HBO_WO": "W",
        "Brincode": "21PB",
        "Brin_volgnr": "00",
        "Isatcode": 56604,
        "Type_HO": "B",
        "Opl_vorm": "VOL",
        "Voertaal": "NL",
        "Studiejaar": 2024,
        "Fixus": "N",
        "Maand": 10,
        "Herkomst": "N",
        "Geslacht": "V",
        "meercode_V": 1.34,
        "meercode_A": 1.0,
        "Status": "I",
        "Hogerejaars": "N",
        "Herinschrijving": "N",
        "1cHO_L": "J",
        "1cHO_K": "N",
        "Aantal": 28,
        "sw": week,
        "jw": 2024,
    }
    df = pd.DataFrame([row], columns=VU_COLUMNS)
    df.to_csv(directory / filename, sep=";", index=False)


class TestRowbindAndReformat:
    def test_default_pattern_reads_studielink_filename(self, tmp_path):
        tel_dir = tmp_path / "telbestanden"
        tel_dir.mkdir()
        _write_vu_telbestand(tel_dir, "telbestandY2024W10.csv", week=10)

        output = tmp_path / "cumulatief.csv"
        _rowbind_and_reformat(str(tel_dir), str(output), {})

        result = pd.read_csv(output, sep=";")
        assert result["Weeknummer"].iloc[0] == 10
        assert result["Collegejaar"].iloc[0] == 2024

    def test_custom_pattern_reads_vu_filename(self, tmp_path):
        tel_dir = tmp_path / "telbestanden"
        tel_dir.mkdir()
        _write_vu_telbestand(tel_dir, "VU_telbestand_2024_W10.csv", week=10)

        config = {
            "telbestand_filename_patterns": ["VU_telbestand_{year}_W{week}"]
        }
        output = tmp_path / "cumulatief.csv"
        _rowbind_and_reformat(str(tel_dir), str(output), config)

        result = pd.read_csv(output, sep=";")
        assert result["Weeknummer"].iloc[0] == 10

    def test_multiple_patterns_processed_together(self, tmp_path):
        tel_dir = tmp_path / "telbestanden"
        tel_dir.mkdir()
        _write_vu_telbestand(tel_dir, "telbestandY2024W10.csv", week=10)
        _write_vu_telbestand(tel_dir, "VU_telbestand_2024_W11.csv", week=11)

        config = {
            "telbestand_filename_patterns": [
                "telbestandY{year}W{week}",
                "VU_telbestand_{year}_W{week}",
            ]
        }
        output = tmp_path / "cumulatief.csv"
        _rowbind_and_reformat(str(tel_dir), str(output), config)

        result = pd.read_csv(output, sep=";")
        assert sorted(result["Weeknummer"].tolist()) == [10, 11]

    def test_unknown_filenames_are_skipped(self, tmp_path, capsys):
        tel_dir = tmp_path / "telbestanden"
        tel_dir.mkdir()
        (tel_dir / "random_file.csv").write_text("x;y\n1;2\n")

        output = tmp_path / "cumulatief.csv"
        _rowbind_and_reformat(str(tel_dir), str(output), {})

        assert not output.exists()
        assert "no telbestand files found" in capsys.readouterr().out

    def test_vu_extra_columns_are_dropped_from_output(self, tmp_path):
        """De ETL-output bevat alleen de kanonieke kolommen, ook bij VU-input."""
        tel_dir = tmp_path / "telbestanden"
        tel_dir.mkdir()
        _write_vu_telbestand(tel_dir, "telbestandY2024W10.csv", week=10)

        output = tmp_path / "cumulatief.csv"
        _rowbind_and_reformat(str(tel_dir), str(output), {})

        result = pd.read_csv(output, sep=";")
        for vu_only in ["HBO_WO", "Opl_vorm", "Voertaal", "Fixus", "Maand",
                        "Geslacht", "meercode_A", "Status", "1cHO_L", "1cHO_K",
                        "sw", "jw", "Brin_volgnr"]:
            assert vu_only not in result.columns, f"{vu_only} mag niet in output staan"

    def test_groepeernaam_falls_back_to_croho(self, tmp_path):
        """Als Groepeernaam ontbreekt (VU-geval), gebruik dan Isatcode/Croho."""
        tel_dir = tmp_path / "telbestanden"
        tel_dir.mkdir()
        _write_vu_telbestand(tel_dir, "telbestandY2024W10.csv", week=10)

        output = tmp_path / "cumulatief.csv"
        _rowbind_and_reformat(str(tel_dir), str(output), {})

        result = pd.read_csv(output, sep=";")
        assert result["Groepeernaam Croho"].iloc[0] == result["Croho"].iloc[0]

    def test_invalid_pattern_raises_clear_error(self, tmp_path):
        tel_dir = tmp_path / "telbestanden"
        tel_dir.mkdir()
        _write_vu_telbestand(tel_dir, "telbestandY2024W10.csv", week=10)
        output = tmp_path / "cumulatief.csv"
        config = {"telbestand_filename_patterns": ["foo_{year}_only"]}

        with pytest.raises(ValueError, match="placeholders"):
            _rowbind_and_reformat(str(tel_dir), str(output), config)
