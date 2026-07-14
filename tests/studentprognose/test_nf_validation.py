"""Tests voor de numerus_fixus-sleutelguard (#258).

Bewaakt dat een NF-sleutel die niet matcht met de programmakolom niet stil wordt
genegeerd: geen match → harde fout; match in één spoor maar niet het andere →
waarschuwing; geldige match → schoon resultaat.
"""

import pandas as pd
import pytest

from studentprognose.data.nf_validation import (
    NumerusFixusConfigError,
    enforce_numerus_fixus_keys,
    validate_numerus_fixus_keys,
)


class TestValidateNumerusFixusKeys:
    def test_empty_nf_list_is_noop(self):
        result = validate_numerus_fixus_keys(
            {}, {"cumulatief": pd.Series([56604, 56605], dtype="Int64")}
        )
        assert result.hard_errors == []
        assert result.warnings == []

    def test_none_nf_list_is_noop(self):
        result = validate_numerus_fixus_keys(
            None, {"cumulatief": pd.Series([56604], dtype="Int64")}
        )
        assert result.hard_errors == []
        assert result.warnings == []

    def test_no_tracks_is_noop(self):
        result = validate_numerus_fixus_keys({56604: 350}, {})
        assert result.hard_errors == []
        assert result.warnings == []

    def test_matching_isatcode_produces_no_findings(self):
        result = validate_numerus_fixus_keys(
            {56604: 350},
            {"cumulatief": pd.Series([56604, 56605], dtype="Int64")},
        )
        assert result.hard_errors == []
        assert result.warnings == []

    def test_nonmatching_key_is_hard_error(self):
        # Naam-key tegen een numerieke Isatcode-kolom → matcht nooit → harde fout
        # (dit is exact het silent-no-match-scenario uit #258).
        result = validate_numerus_fixus_keys(
            {"B Geneeskunde": 350},
            {"cumulatief": pd.Series([56604, 56605], dtype="Int64")},
        )
        assert len(result.hard_errors) == 1
        assert "B Geneeskunde" in result.hard_errors[0]
        assert result.warnings == []

    def test_wrong_dtype_string_isatcode_is_hard_error(self):
        # Str "56604" != Int64 56604 → geen match → harde fout (dtype-mismatch).
        result = validate_numerus_fixus_keys(
            {"56604": 350},
            {"cumulatief": pd.Series([56604], dtype="Int64")},
        )
        assert len(result.hard_errors) == 1
        assert result.warnings == []

    def test_track_mismatch_produces_warning_not_error(self):
        # Isatcode-key matcht cumulatief (Int64) maar niet individueel (namen).
        # Geen harde fout — de key is geldig voor minstens één spoor — wél een
        # waarschuwing dat NF niet aangrijpt in het individuele spoor (#238).
        result = validate_numerus_fixus_keys(
            {56604: 350},
            {
                "cumulatief": pd.Series([56604, 56605], dtype="Int64"),
                "individueel": pd.Series(["B Geneeskunde", "B Psychologie"]),
            },
        )
        assert result.hard_errors == []
        assert len(result.warnings) == 1
        assert "individueel" in result.warnings[0]

    def test_multiple_keys_mixed_outcomes(self):
        result = validate_numerus_fixus_keys(
            {56604: 350, 99999: 100},
            {"cumulatief": pd.Series([56604, 56605], dtype="Int64")},
        )
        # 56604 matcht, 99999 niet → precies één harde fout.
        assert len(result.hard_errors) == 1
        assert "99999" in result.hard_errors[0]

    def test_none_series_track_is_ignored(self):
        result = validate_numerus_fixus_keys(
            {56604: 350},
            {
                "cumulatief": pd.Series([56604], dtype="Int64"),
                "individueel": None,
            },
        )
        # Individueel spoor niet geladen → geen waarschuwing daarover.
        assert result.hard_errors == []
        assert result.warnings == []


class TestEnforceNumerusFixusKeys:
    def test_hard_error_raises(self):
        with pytest.raises(NumerusFixusConfigError) as exc:
            enforce_numerus_fixus_keys(
                {"B Geneeskunde": 350},
                {"cumulatief": pd.Series([56604], dtype="Int64")},
            )
        assert "B Geneeskunde" in str(exc.value)

    def test_hard_error_is_valueerror_subclass(self):
        # Het API-pad vangt ValueError; de exception moet daaronder vallen.
        assert issubclass(NumerusFixusConfigError, ValueError)

    def test_valid_key_does_not_raise(self):
        enforce_numerus_fixus_keys(
            {56604: 350},
            {"cumulatief": pd.Series([56604], dtype="Int64")},
        )

    def test_track_mismatch_warns_but_does_not_raise(self, capsys):
        enforce_numerus_fixus_keys(
            {56604: 350},
            {
                "cumulatief": pd.Series([56604], dtype="Int64"),
                "individueel": pd.Series(["B Geneeskunde"]),
            },
        )
        out = capsys.readouterr().out
        assert "WAARSCHUWING" in out
        assert "individueel" in out
