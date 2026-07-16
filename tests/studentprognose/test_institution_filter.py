"""Tests voor de multi-instelling filter (issue #200).

Bewaakt drie lagen:
  * ``apply_institution_filter`` — de rij-filter zelf.
  * ``filter_datasets_by_institution`` — de orchestratie over de dataset-tuple.
  * ``_validate_institution_filter`` — de config-validatie (via load_configuration).
"""

import json
import warnings

import pandas as pd
import pytest

from studentprognose.config import load_configuration
from studentprognose.data.loader import filter_datasets_by_institution
from studentprognose.data.preprocessing.institution_filter import apply_institution_filter


def _df():
    return pd.DataFrame(
        {
            "Korte naam instelling": ["21PC", "21PC", "00IC", "02NR"],
            "Croho groepeernaam": ["A", "B", "A", "C"],
            "waarde": [1, 2, 3, 4],
        }
    )


class TestApplyInstitutionFilter:
    def test_empty_filter_is_noop(self):
        df = _df()
        # Leeg = alle instellingen: exact hetzelfde object terug (geen kopie).
        assert apply_institution_filter(df, []) is df

    def test_none_filter_is_noop(self):
        df = _df()
        assert apply_institution_filter(df, None) is df

    def test_none_dataframe_passthrough(self):
        assert apply_institution_filter(None, ["21PC"]) is None

    def test_single_institution(self):
        result = apply_institution_filter(_df(), ["21PC"])
        assert list(result["waarde"]) == [1, 2]
        # Index is gereset zodat downstream .iloc/.loc voorspelbaar blijft.
        assert list(result.index) == [0, 1]

    def test_multiple_institutions(self):
        result = apply_institution_filter(_df(), ["21PC", "00IC"])
        assert sorted(result["waarde"]) == [1, 2, 3]

    def test_missing_column_passthrough(self):
        # Een spoor zonder instellingskolom (bijv. individueel) wordt niet geraakt.
        df = pd.DataFrame({"Croho groepeernaam": ["A"], "waarde": [1]})
        assert apply_institution_filter(df, ["21PC"]) is df

    def test_int_code_coerced_to_string(self):
        df = pd.DataFrame({"Korte naam instelling": ["123", "456"], "waarde": [1, 2]})
        result = apply_institution_filter(df, [123])
        assert list(result["waarde"]) == [1]

    def test_absent_institution_warns(self):
        with pytest.warns(UserWarning, match="ZZZZ"):
            result = apply_institution_filter(_df(), ["21PC", "ZZZZ"])
        # De aanwezige instelling wordt gewoon behouden.
        assert list(result["waarde"]) == [1, 2]

    def test_no_match_raises(self):
        # De missing-warning vuurt vóór de ValueError; hier gaat het om de fout.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="geen enkele rij"):
                apply_institution_filter(_df(), ["ONBEKEND"])

    def test_custom_column_name(self):
        df = pd.DataFrame({"Brincode": ["21PC", "00IC"], "waarde": [1, 2]})
        result = apply_institution_filter(df, ["00IC"], institution_col="Brincode")
        assert list(result["waarde"]) == [2]


class TestFilterDatasetsByInstitution:
    def _config(self, institutions):
        return {
            "institution_filter": institutions,
            "column_roles": {"institution": "Korte naam instelling"},
        }

    def test_empty_filter_returns_unchanged(self):
        datasets = (_df(), _df(), None, None, None)
        assert filter_datasets_by_institution(datasets, self._config([])) is datasets

    def test_filters_individual_and_cumulative(self):
        individual = _df()
        cumulative = _df()
        datasets = (individual, cumulative, "sc", "latest", "ens")
        out = filter_datasets_by_institution(datasets, self._config(["21PC"]))
        assert list(out[0]["waarde"]) == [1, 2]
        assert list(out[1]["waarde"]) == [1, 2]
        # De overige elementen (student_count, latest, ensemble) blijven ongemoeid.
        assert out[2:] == ("sc", "latest", "ens")

    def test_none_tracks_survive(self):
        # Individueel spoor niet geladen (None) mag geen crash geven.
        datasets = (None, _df(), None, None, None)
        out = filter_datasets_by_institution(datasets, self._config(["00IC"]))
        assert out[0] is None
        assert list(out[1]["waarde"]) == [3]

    def test_falls_back_to_default_column(self):
        # column_roles zonder 'institution' → val terug op "Korte naam instelling".
        cfg = {"institution_filter": ["21PC"], "column_roles": {}}
        datasets = (None, _df(), None, None, None)
        out = filter_datasets_by_institution(datasets, cfg)
        assert list(out[1]["waarde"]) == [1, 2]


class TestInstitutionScopeReport:
    """De run moet altijd melden waarvoor er gerekend wordt (issue #200)."""

    def _config(self, institutions):
        return {
            "institution_filter": institutions,
            "column_roles": {"institution": "Korte naam instelling"},
        }

    def test_reports_no_filter_all_institutions(self, capsys):
        # 4 rijen, 3 unieke instellingen (21PC, 00IC, 02NR).
        datasets = (_df(), _df(), None, None, None)
        filter_datasets_by_institution(datasets, self._config([]))
        out = capsys.readouterr().out
        assert "geen" in out
        assert "alle 3 instellingen" in out
        assert "4 rijen" in out

    def test_reports_active_filter_with_counts(self, capsys):
        datasets = (_df(), _df(), None, None, None)
        filter_datasets_by_institution(datasets, self._config(["21PC"]))
        out = capsys.readouterr().out
        assert "actief" in out
        assert "21PC" in out
        # 2 van de 4 rijen zijn 21PC.
        assert "2 van 4 rijen" in out

    def test_reports_multiple_institutions(self, capsys):
        datasets = (_df(), _df(), None, None, None)
        filter_datasets_by_institution(datasets, self._config(["21PC", "00IC"]))
        out = capsys.readouterr().out
        assert "21PC, 00IC" in out
        assert "3 van 4 rijen" in out

    def test_reports_when_filter_cannot_apply(self, capsys):
        # Cumulatief spoor niet geladen én filter gezet: meld dat 't niet toepast.
        datasets = (_df(), None, None, None, None)
        filter_datasets_by_institution(datasets, self._config(["21PC"]))
        out = capsys.readouterr().out
        assert "niet toegepast" in out
        assert "21PC" in out

    def test_silent_when_no_filter_and_no_institution_column(self, capsys):
        # Individueel-only run zonder filter: geen ruis.
        datasets = (_df(), None, None, None, None)
        filter_datasets_by_institution(datasets, self._config([]))
        assert capsys.readouterr().out == ""

    def test_uses_dutch_thousands_separator(self, capsys):
        big = pd.DataFrame(
            {
                "Korte naam instelling": ["21PC"] * 1500 + ["00IC"] * 500,
                "waarde": range(2000),
            }
        )
        datasets = (None, big, None, None, None)
        filter_datasets_by_institution(datasets, self._config([]))
        out = capsys.readouterr().out
        assert "2.000 rijen" in out


class TestConfigValidation:
    def _write(self, tmp_path, value):
        p = tmp_path / "configuration.json"
        p.write_text(json.dumps({"institution_filter": value}), encoding="utf-8")
        return str(p)

    def test_valid_string_list(self, tmp_path):
        cfg = load_configuration(self._write(tmp_path, ["21PC", "00IC"]))
        assert cfg["institution_filter"] == ["21PC", "00IC"]

    def test_valid_int_and_str_mixed(self, tmp_path):
        cfg = load_configuration(self._write(tmp_path, [123, "21PC"]))
        assert cfg["institution_filter"] == [123, "21PC"]

    def test_default_is_empty_list(self):
        # De packaged/repo-config levert een lege lijst → alle instellingen.
        cfg = load_configuration("configuration/configuration.json")
        assert cfg["institution_filter"] == []

    def test_rejects_non_list(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            load_configuration(self._write(tmp_path, "21PC"))
        assert exc.value.code == 1

    def test_rejects_non_scalar_element(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            load_configuration(self._write(tmp_path, [1.5]))
        assert exc.value.code == 1

    def test_rejects_bool_element(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            load_configuration(self._write(tmp_path, [True]))
        assert exc.value.code == 1
