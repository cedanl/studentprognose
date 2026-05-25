"""Tests voor preprocessing-configuratie, met name valid_ingangsdatums."""

import json

import pandas as pd
import pytest

from studentprognose.config import load_configuration
from studentprognose.strategies.individual import preprocess_individual_data


# ---------------------------------------------------------------------------
# Validatie via load_configuration
# ---------------------------------------------------------------------------

class TestValidIngangsdatumsValidation:
    def test_default_loads(self, tmp_path):
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps({}))
        result = load_configuration(str(f))
        assert result["preprocessing"]["individual"]["valid_ingangsdatums"] == [
            "01-09",
            "01-10",
        ]

    def test_custom_list_overrides_default(self, tmp_path):
        f = tmp_path / "configuration.json"
        f.write_text(
            json.dumps(
                {"preprocessing": {"individual": {"valid_ingangsdatums": ["01-02"]}}}
            )
        )
        result = load_configuration(str(f))
        assert result["preprocessing"]["individual"]["valid_ingangsdatums"] == ["01-02"]

    def test_empty_list_exits(self, tmp_path):
        f = tmp_path / "configuration.json"
        f.write_text(
            json.dumps(
                {"preprocessing": {"individual": {"valid_ingangsdatums": []}}}
            )
        )
        with pytest.raises(SystemExit) as exc:
            load_configuration(str(f))
        assert exc.value.code == 1

    def test_non_list_exits(self, tmp_path):
        f = tmp_path / "configuration.json"
        f.write_text(
            json.dumps(
                {"preprocessing": {"individual": {"valid_ingangsdatums": "01-09"}}}
            )
        )
        with pytest.raises(SystemExit) as exc:
            load_configuration(str(f))
        assert exc.value.code == 1

    def test_non_string_element_exits(self, tmp_path):
        f = tmp_path / "configuration.json"
        f.write_text(
            json.dumps(
                {"preprocessing": {"individual": {"valid_ingangsdatums": [1, 2]}}}
            )
        )
        with pytest.raises(SystemExit) as exc:
            load_configuration(str(f))
        assert exc.value.code == 1

    def test_preprocessing_not_a_dict_exits(self, tmp_path):
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps({"preprocessing": "not-a-dict"}))
        with pytest.raises(SystemExit) as exc:
            load_configuration(str(f))
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# Strategy filtert volgens config
# ---------------------------------------------------------------------------

def _row(ingangsdatum: str, sleutel: int = 1) -> dict:
    return {
        "Sleutel": sleutel,
        "Datum Verzoek Inschr": "03-10-2019",
        "Ingangsdatum": ingangsdatum,
        "Collegejaar": 2020,
        "Datum intrekking vooraanmelding": None,
        "Inschrijfstatus": "Ingeschreven",
        "Faculteit": "FSW",
        "Examentype": "Propedeuse Bachelor",
        "Croho": "Psychologie",
        "Croho groepeernaam": "B Psychologie",
        "Eerstejaars croho jaar": 2020,
        "Is eerstejaars croho opleiding": 1,
        "Is hogerejaars": 0,
        "BBC ontvangen": 0,
        "Nationaliteit": "Nederlandse",
        "EER": "N",
        "Aantal studenten": 1,
    }


def _cfg(valid_ingangsdatums=None):
    cfg = {
        "column_roles": {
            "programme": "Croho groepeernaam",
            "academic_year": "Collegejaar",
            "exam_type": "Examentype",
            "origin": "Herkomst",
            "faculty": "Faculteit",
            "week": "Weeknummer",
            "enrollment_status": "Inschrijfstatus",
            "cancellation_date": "Datum intrekking vooraanmelding",
            "student_count": "Aantal_studenten",
        }
    }
    if valid_ingangsdatums is not None:
        cfg["preprocessing"] = {
            "individual": {"valid_ingangsdatums": valid_ingangsdatums}
        }
    return cfg


class TestPreprocessHonorsValidIngangsdatums:
    def test_default_keeps_september_and_october(self):
        df = pd.DataFrame(
            [
                _row("01-09-2020", sleutel=1),
                _row("01-10-2020", sleutel=2),
                _row("01-02-2020", sleutel=3),
            ]
        )
        result = preprocess_individual_data(df, [], _cfg())
        # default = ["01-09", "01-10"] → februari valt weg
        assert len(result) == 2

    def test_custom_list_keeps_february(self):
        df = pd.DataFrame(
            [
                _row("01-09-2020", sleutel=1),
                _row("01-02-2020", sleutel=2),
            ]
        )
        result = preprocess_individual_data(
            df, [], _cfg(valid_ingangsdatums=["01-02"])
        )
        # alleen februari toegelaten
        assert len(result) == 1

    def test_multiple_prefixes(self):
        df = pd.DataFrame(
            [
                _row("01-09-2020", sleutel=1),
                _row("01-02-2020", sleutel=2),
                _row("15-08-2020", sleutel=3),
            ]
        )
        result = preprocess_individual_data(
            df, [], _cfg(valid_ingangsdatums=["01-09", "01-02"])
        )
        assert len(result) == 2

    def test_prefix_matches_only_on_day_month_boundary(self):
        # "01-09-" mag niet matchen op "11-09-2020" — startswith met '-' suffix.
        df = pd.DataFrame(
            [
                _row("01-09-2020", sleutel=1),
                _row("11-09-2020", sleutel=2),
            ]
        )
        result = preprocess_individual_data(df, [], _cfg())
        assert len(result) == 1
