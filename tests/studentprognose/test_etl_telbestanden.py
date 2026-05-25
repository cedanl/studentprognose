"""Integratietests voor de ETL-rowbind met configureerbare bestandsnaampatronen (issue #199)."""

import pandas as pd
import pytest

from studentprognose.data.etl import _rowbind_and_reformat


def _write_telbestand(path, week):
    """Schrijf een minimaal valide Studielink-telbestand naar ``path``."""
    df = pd.DataFrame(
        [
            {
                "Brincode": "21PB",
                "Studiejaar": 2024,
                "Type_HO": "B",
                "Isatcode": 56604,
                "Groepeernaam": "Test Opleiding",
                "Faculteit": "FacA",
                "Herkomst": "N",
                "Hogerejaars": "N",
                "Herinschrijving": "N",
                "Aantal": 28,
                "meercode_V": 1.34,
                "Status": "V",
            }
        ]
    )
    df.to_csv(path, sep=";", index=False)


class TestRowbindWithFilenamePatterns:
    def test_default_pattern_reads_studielink_filename(self, tmp_path):
        tel_dir = tmp_path / "telbestanden"
        tel_dir.mkdir()
        _write_telbestand(tel_dir / "telbestandY2024W10.csv", week=10)

        output = tmp_path / "cumulatief.csv"
        _rowbind_and_reformat(str(tel_dir), str(output), {})

        result = pd.read_csv(output, sep=";")
        assert result["Weeknummer"].iloc[0] == 10
        assert result["Collegejaar"].iloc[0] == 2024

    def test_custom_pattern_reads_alternative_filename(self, tmp_path):
        tel_dir = tmp_path / "telbestanden"
        tel_dir.mkdir()
        _write_telbestand(tel_dir / "VU_telbestand_2024_W10.csv", week=10)

        config = {"telbestand_filename_patterns": ["VU_telbestand_{year}_W{week}"]}
        output = tmp_path / "cumulatief.csv"
        _rowbind_and_reformat(str(tel_dir), str(output), config)

        result = pd.read_csv(output, sep=";")
        assert result["Weeknummer"].iloc[0] == 10

    def test_multiple_patterns_processed_together(self, tmp_path):
        """Bestanden met verschillende naamconventies worden samengevoegd."""
        tel_dir = tmp_path / "telbestanden"
        tel_dir.mkdir()
        _write_telbestand(tel_dir / "telbestandY2024W10.csv", week=10)
        _write_telbestand(tel_dir / "VU_telbestand_2024_W11.csv", week=11)

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

    def test_invalid_pattern_raises_clear_error(self, tmp_path):
        tel_dir = tmp_path / "telbestanden"
        tel_dir.mkdir()
        _write_telbestand(tel_dir / "telbestandY2024W10.csv", week=10)
        output = tmp_path / "cumulatief.csv"
        config = {"telbestand_filename_patterns": ["foo_{year}_only"]}

        with pytest.raises(ValueError, match="placeholders"):
            _rowbind_and_reformat(str(tel_dir), str(output), config)
