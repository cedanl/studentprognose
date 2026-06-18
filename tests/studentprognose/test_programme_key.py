"""Tests voor de canonieke programmesleutel-normalisatie (CROHO/Isatcode).

Bewaakt dat de programmesleutel als één dtype door de pipeline loopt:
numerieke isatcodes -> Int64, legacy leesbare namen -> string, en dat de
config-zijde van elke vergelijking (.isin, numerus_fixus) hetzelfde dtype krijgt
zodat merges en filters niet stil leeglopen of crashen op gemengde types.
"""

import json

import pandas as pd
import pytest

from studentprognose.config import load_configuration
from studentprognose.utils.programme_key import (
    merge_on_programme_key,
    normalize_programme_keys,
    normalize_programme_series,
    normalize_programme_value,
    normalize_programme_values,
)


class TestNormalizeProgrammeSeries:
    def test_int_series_becomes_int64(self):
        out = normalize_programme_series(pd.Series([56604, 30008], dtype="int64"))
        assert str(out.dtype) == "Int64"
        assert list(out) == [56604, 30008]

    def test_float_series_drops_dot_zero(self):
        # Excel-round-trip levert numerieke codes vaak als float (56604.0).
        out = normalize_programme_series(pd.Series([56604.0, 30008.0]))
        assert str(out.dtype) == "Int64"
        assert list(out) == [56604, 30008]

    def test_numeric_with_nan_is_nan_safe(self):
        out = normalize_programme_series(pd.Series([56604.0, None]))
        assert str(out.dtype) == "Int64"
        assert out.iloc[0] == 56604
        assert pd.isna(out.iloc[1])

    def test_numeric_strings_become_int64(self):
        # Sommige CSV-exports leveren codes als string.
        out = normalize_programme_series(pd.Series(["56604", "30008"], dtype="object"))
        assert str(out.dtype) == "Int64"
        assert list(out) == [56604, 30008]

    def test_legacy_names_stay_string(self):
        out = normalize_programme_series(pd.Series(["B Psychologie", "M Informatica"]))
        # Niet-numeriek -> string (object of StringDtype, afhankelijk van pandas).
        assert str(out.dtype) != "Int64"
        assert all(isinstance(v, str) for v in out)
        assert list(out) == ["B Psychologie", "M Informatica"]

    def test_none_returns_none(self):
        assert normalize_programme_series(None) is None


class TestNormalizeProgrammeValue:
    @pytest.mark.parametrize("value,expected", [
        ("56604", 56604),
        (56604, 56604),
        ("B Psychologie", "B Psychologie"),
        (None, None),
    ])
    def test_value(self, value, expected):
        assert normalize_programme_value(value) == expected

    def test_values_list(self):
        assert normalize_programme_values(["56604", "B Psych"]) == [56604, "B Psych"]
        assert normalize_programme_values(None) is None

    def test_keys_mapping(self):
        assert normalize_programme_keys({"56604": 100, "B Psych": 50}) == {56604: 100, "B Psych": 50}
        assert normalize_programme_keys({}) == {}
        assert normalize_programme_keys(None) is None


class TestIsinBoundary:
    """Kern van de fix: een Int64-kolom moet matchen met genormaliseerde
    config-waarden (zoals programme_filtering) in .isin — anders loopt het
    filter stil leeg op int-vs-str."""

    def test_int64_column_isin_normalized_values(self):
        col = normalize_programme_series(pd.Series([56604, 30008, 12345], dtype="int64"))
        mask = col.isin(normalize_programme_values(["56604", "30008"]))
        assert list(mask) == [True, True, False]

    def test_string_column_isin_normalized_names(self):
        col = normalize_programme_series(pd.Series(["B Psychologie", "M Informatica"]))
        mask = col.isin(normalize_programme_values(["B Psychologie"]))
        assert list(mask) == [True, False]


class TestNumerusFixusKeysNormalized:
    def test_config_load_normalizes_numerus_fixus_keys(self, tmp_path):
        cfg_path = tmp_path / "configuration.json"
        cfg_path.write_text(json.dumps({"numerus_fixus": {"56604": 100, "B Tand": 60}}))
        cfg = load_configuration(str(cfg_path))
        # Numerieke key -> int (matcht de Int64-kolom), naam-key blijft string.
        assert cfg["numerus_fixus"] == {56604: 100, "B Tand": 60}


class TestMergeOnProgrammeKey:
    """De cross-track merge tussen het individuele spoor (namen) en het
    cumulatieve spoor/label (isatcodes) mag niet crashen op str-vs-Int64; bij
    gelijke sleutel moet hij wél matchen."""

    def _frame(self, key, value):
        return pd.DataFrame({"Croho groepeernaam": [key], "Collegejaar": [2024], "v": [value]})

    def test_mismatched_keyspaces_do_not_crash(self):
        individual = self._frame("B Psychologie", "ind")          # naam (str)
        cumulative = self._frame(56604, "cum")
        cumulative["Croho groepeernaam"] = cumulative["Croho groepeernaam"].astype("Int64")
        merged = merge_on_programme_key(
            individual, cumulative, on=["Croho groepeernaam", "Collegejaar"]
        )
        assert len(merged) == 1                  # left-row behouden
        assert pd.isna(merged["v_y"].iloc[0])    # 0 match -> rechterkolom NaN

    def test_aligned_keyspaces_match(self):
        individual = self._frame(56604, "ind")
        cumulative = self._frame(56604, "cum")
        for df in (individual, cumulative):
            df["Croho groepeernaam"] = df["Croho groepeernaam"].astype("Int64")
        merged = merge_on_programme_key(
            individual, cumulative, on=["Croho groepeernaam", "Collegejaar"]
        )
        assert len(merged) == 1
        assert merged["v_y"].iloc[0] == "cum"    # echte match zodra sleutels gelijk zijn
