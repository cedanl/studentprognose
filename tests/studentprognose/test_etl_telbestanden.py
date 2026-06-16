"""Integratietests voor de ETL-rowbind met configureerbare bestandsnaampatronen (issue #199)."""

import numpy as np
import pandas as pd
import pytest

from studentprognose.data.etl import _rowbind_and_reformat, _calculate_student_counts
from studentprognose.data.loader import _normalize_programme_code


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

    def test_legacy_default_config_full_output_regression(self, tmp_path):
        """Borgt dat het legacy-pad (lege config) byte-voor-byte gelijk blijft nu
        _rowbind_and_reformat config-gedreven en gedeeld is met het UvA-pad."""
        tel_dir = tmp_path / "telbestanden"
        tel_dir.mkdir()
        _write_telbestand(tel_dir / "telbestandY2024W10.csv", week=10)
        output = tmp_path / "cumulatief.csv"
        _rowbind_and_reformat(str(tel_dir), str(output), {})

        result = pd.read_csv(output, sep=";")
        assert list(result.columns) == _CANONICAL_16
        # Legacy waardevertalingen (uit de defaults).
        assert result["Type hoger onderwijs"].iloc[0] == "Bachelor"
        assert result["Herkomst"].iloc[0] == "NL"
        assert result["Herinschrijving"].iloc[0] == "Nee"
        assert result["Hogerejaars"].iloc[0] == "Nee"
        # Faculteit/Groepeernaam komen uit de bron, NIET de UvA-sentinel/Isatcode.
        assert result["Faculteit"].iloc[0] == "FacA"
        assert result["Groepeernaam Croho"].iloc[0] == "Test Opleiding"
        # Legacy aggregeert niet (default aggregate=False) → rij-aantal behouden.
        assert len(result) == 1


# ---------------------------------------------------------------------------
# UvA Studielink SQL-formaat (issue #231)
# ---------------------------------------------------------------------------

_UVA_COLUMNS = [
    "HBO_WO", "Brincode", "Brin_volgnr", "Isatcode", "Type_HO", "Opl_vorm",
    "Voertaal", "Studiejaar", "Fixus", "Maand", "Herkomst", "Geslacht",
    "meercode_V", "meercode_A", "Status", "Hogerejaars", "Herinschrijving",
    "1cHO_L", "1cHO_K", "Aantal", "etl_ingestion_timestamp", "etl_is_deleted",
]

# Config zoals de repo-root configuration.json het UvA-formaat instelt.
_UVA_CONFIG = {
    "telbestand_filename_patterns": ["telbestand_sl_{date}_v{volgnummer}_{year}"],
    "cumulative_input": {
        "separator": ",",
        "value_maps": {
            "Type hoger onderwijs": {
                "P": "Bachelor", "B": "Bachelor", "M": "Master",
                "A": "Associate degree", "O": "Onbekend",
            },
            "Herkomst": {"N": "NL", "E": "EER", "R": "Niet-EER", "O": "Onbekend"},
        },
        "faculteit_sentinel": "Onbekend",
        "aggregate": True,
        "drop_deleted": True,
    },
}

_CANONICAL_16 = [
    "Korte naam instelling", "Collegejaar", "Weeknummer rapportage", "Weeknummer",
    "Faculteit", "Type hoger onderwijs", "Groepeernaam Croho",
    "Naam Croho opleiding Nederlands", "Croho", "Herinschrijving", "Hogerejaars",
    "Herkomst", "Gewogen vooraanmelders", "Ongewogen vooraanmelders",
    "Aantal aanmelders met 1 aanmelding", "Inschrijvingen",
]


def _uva_row(isat, type_ho, herkomst, voertaal, geslacht, meercode_v, status, aantal, deleted=0):
    return {
        "HBO_WO": "W", "Brincode": "21PE", "Brin_volgnr": 0, "Isatcode": isat,
        "Type_HO": type_ho, "Opl_vorm": 1, "Voertaal": voertaal, "Studiejaar": 2026,
        "Fixus": "N", "Maand": 9, "Herkomst": herkomst, "Geslacht": geslacht,
        "meercode_V": meercode_v, "meercode_A": 0, "Status": status,
        "Hogerejaars": "N", "Herinschrijving": "N", "1cHO_L": 1, "1cHO_K": 1,
        "Aantal": aantal, "etl_ingestion_timestamp": "2026-06-10", "etl_is_deleted": deleted,
    }


class TestRowbindUvaSqlFormat:
    """De UvA SQL-snapshot (komma-gescheiden, 22 kol) → canonieke 16 kolommen."""

    def _run(self, tmp_path, rows, filename="telbestand_sl_20260525_v34_2026.csv"):
        tel_dir = tmp_path / "uva_telbestanden"
        tel_dir.mkdir()
        pd.DataFrame(rows, columns=_UVA_COLUMNS).to_csv(tel_dir / filename, sep=",", index=False)
        output = tmp_path / "cumulatief.csv"
        _rowbind_and_reformat(str(tel_dir), str(output), _UVA_CONFIG)
        return pd.read_csv(output, sep=";")

    def test_outputs_canonical_16_columns(self, tmp_path):
        result = self._run(tmp_path, [_uva_row(30029, "B", "N", 2, "M", 2, "V", 40)])
        assert list(result.columns) == _CANONICAL_16

    def test_week_from_leverdatum(self, tmp_path):
        # Leverdatum 2026-05-25 → ISO-week 22 (niet het volgnummer 34).
        result = self._run(tmp_path, [_uva_row(30029, "B", "N", 2, "M", 2, "V", 40)])
        assert (result["Weeknummer"] == 22).all()
        assert (result["Weeknummer rapportage"] == 22).all()

    def test_aggregates_finer_rows_to_grain(self, tmp_path):
        # Twee rijen, zelfde grain, verschillen in Voertaal/Geslacht → Gewogen optellen.
        rows = [
            _uva_row(30029, "B", "N", 2, "M", 2, "V", 40),  # Gewogen 40/2=20, Ongewogen 40
            _uva_row(30029, "B", "N", 1, "V", 2, "V", 20),  # Gewogen 20/2=10, Ongewogen 20
        ]
        result = self._run(tmp_path, rows)
        assert len(result) == 1
        assert result["Gewogen vooraanmelders"].iloc[0] == 30
        assert result["Ongewogen vooraanmelders"].iloc[0] == 60

    def test_type_ho_associate_degree(self, tmp_path):
        result = self._run(tmp_path, [_uva_row(30029, "A", "N", 2, "M", 1, "V", 25)])
        assert result["Type hoger onderwijs"].iloc[0] == "Associate degree"

    def test_status_a_and_deleted_rows_dropped(self, tmp_path):
        rows = [
            _uva_row(30029, "B", "N", 2, "M", 1, "V", 30),             # behouden
            _uva_row(30029, "B", "N", 2, "M", 1, "A", 99),             # Status A → weg
            _uva_row(30029, "B", "N", 2, "M", 1, "V", 50, deleted=1),  # etl_is_deleted → weg
        ]
        result = self._run(tmp_path, rows)
        assert len(result) == 1
        assert result["Ongewogen vooraanmelders"].iloc[0] == 30

    def test_faculteit_sentinel_and_groepeernaam_is_isatcode(self, tmp_path):
        result = self._run(tmp_path, [_uva_row(30029, "B", "N", 2, "M", 2, "V", 40)])
        assert (result["Faculteit"] == "Onbekend").all()
        assert result["Groepeernaam Croho"].iloc[0] == 30029
        assert result["Naam Croho opleiding Nederlands"].iloc[0] == 30029
        assert result["Croho"].iloc[0] == 30029

    def test_herkomst_mapped_and_no_duplicate_grain(self, tmp_path):
        rows = [
            _uva_row(30029, "B", "N", 2, "M", 2, "V", 40),
            _uva_row(30029, "B", "N", 1, "V", 2, "V", 20),
            _uva_row(50645, "M", "E", 1, "V", 1, "V", 30),
        ]
        result = self._run(tmp_path, rows)
        identity = [c for c in result.columns if c not in (
            "Gewogen vooraanmelders", "Ongewogen vooraanmelders",
            "Aantal aanmelders met 1 aanmelding", "Inschrijvingen")]
        assert not result.duplicated(subset=identity).any()
        assert set(result["Herkomst"]) == {"NL", "EER"}
        assert (result["Type hoger onderwijs"] == "Master").sum() == 1

    def test_inherited_value_maps_applied(self, tmp_path):
        # De UvA-config levert GEEN Herinschrijving/Hogerejaars-map; die erven van de
        # defaults. Load-bearing: de cumulatieve strategie filtert `== "Nee"`.
        result = self._run(tmp_path, [_uva_row(30029, "B", "N", 2, "M", 2, "V", 40)])
        assert (result["Herinschrijving"] == "Nee").all()
        assert (result["Hogerejaars"] == "Nee").all()

    def test_gewogen_is_finite_no_div_by_zero(self, tmp_path):
        # meercode_V == 0 hoort alleen bij Status "A" → uitgefilterd vóór de deling,
        # dus geen inf in Gewogen vooraanmelders.
        rows = [
            _uva_row(30029, "B", "N", 2, "M", 2, "V", 40),
            _uva_row(30029, "B", "N", 2, "M", 0, "A", 99),  # meercode_V=0 + Status A
        ]
        result = self._run(tmp_path, rows)
        assert np.isfinite(result["Gewogen vooraanmelders"]).all()

    def test_unmapped_value_warns_and_passes_through(self, tmp_path, capsys):
        # Onbekende Type_HO-code (niet in de value_map) → waarschuwing + onveranderd
        # doorgegeven (niet NaN).
        result = self._run(tmp_path, [_uva_row(30029, "X", "N", 2, "M", 2, "V", 40)])
        out = capsys.readouterr().out
        assert "niet-gemapte waarde" in out
        assert (result["Type hoger onderwijs"] == "X").all()


# Identity-mapping (canoniek == instellingskolom) voor de oktober-kolommen.
_OKTOBER_COLS = {
    "Collegejaar": "Collegejaar", "Isatcode": "Isatcode",
    "Groepeernaam Croho": "Groepeernaam Croho",
    "Aantal eerstejaars croho": "Aantal eerstejaars croho",
    "EER-NL-nietEER": "EER-NL-nietEER", "Examentype code": "Examentype code",
    "Aantal Hoofdinschrijvingen": "Aantal Hoofdinschrijvingen",
}


class TestStudentCountKeyedOnIsatcode:
    """Regressie (#231/#232): het label (student_count) wordt op de landelijke
    Isatcode gekeyd i.p.v. op de instellingsspecifieke opleidingsnaam, zodat het
    op exact dezelfde sleutel als de cumulatieve features joint."""

    def _make_oktober(self, path, isatcode=56604, naam="B Psychologie", enrolled=120):
        rows = [{
            "Collegejaar": 2024, "Isatcode": isatcode, "Groepeernaam Croho": naam,
            "EER-NL-nietEER": "NL", "Examentype code": "Bachelor eerstejaars",
            "Aantal eerstejaars croho": 1, "Aantal Hoofdinschrijvingen": enrolled,
        }]
        pd.DataFrame(rows).to_excel(path, index=False)

    def _run_etl(self, tmp_path):
        (tmp_path / "data" / "input").mkdir(parents=True)
        okt = tmp_path / "oktober_bestand.xlsx"
        self._make_oktober(okt)
        _calculate_student_counts(str(okt), str(tmp_path), _OKTOBER_COLS)
        return pd.read_excel(tmp_path / "data" / "input" / "student_count_first-years.xlsx")

    def test_programme_key_is_isatcode_not_name(self, tmp_path):
        out = self._run_etl(tmp_path)
        codes = set(out["Croho groepeernaam"].astype(str))
        assert codes == {"56604"}                       # de Isatcode, niet de naam
        assert "B Psychologie" not in codes
        assert out["Examentype"].iloc[0] == "Bachelor"  # "Bachelor eerstejaars" → "Bachelor"
        assert out["Aantal_studenten"].iloc[0] == 120

    def test_label_joins_with_cumulative_features_on_isatcode(self, tmp_path):
        """Kern van de fix: na string-normalisatie (zoals loader + cumulative.py)
        joint het label met een feature-rij die de Isatcode als sleutel draagt."""
        label = self._run_etl(tmp_path)
        _normalize_programme_code(label, "Croho groepeernaam")  # numeriek → "56604"

        # Feature-zijde: cumulatieve sleutel is de Isatcode als string (UvA-formaat).
        features = pd.DataFrame([{
            "Croho groepeernaam": "56604", "Collegejaar": 2024,
            "Herkomst": "NL", "Examentype": "Bachelor",
        }])
        keys = ["Croho groepeernaam", "Collegejaar", "Herkomst", "Examentype"]
        merged = features.merge(label, on=keys, how="inner")
        assert len(merged) == 1
        assert merged["Aantal_studenten"].iloc[0] == 120
