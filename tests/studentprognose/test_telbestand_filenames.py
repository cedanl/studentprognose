"""Tests voor configureerbare telbestand-naampatronen (issue #199)."""

import pytest

from studentprognose.config import _validate_telbestand_filename_patterns
from studentprognose.utils.telbestand_filenames import (
    DEFAULT_TELBESTAND_PATTERN,
    compile_patterns,
    match_telbestand,
)


class TestCompilePatterns:
    def test_uses_default_when_config_is_none(self):
        patterns = compile_patterns(None)
        assert len(patterns) == 1

    def test_default_matches_studielink_name(self):
        patterns = compile_patterns(None)
        assert match_telbestand("telbestandY2024W10.csv", patterns) is not None

    def test_uses_default_when_key_missing(self):
        patterns = compile_patterns({})
        assert match_telbestand("telbestandY2024W10.csv", patterns) is not None

    def test_uses_default_when_value_is_empty_list(self):
        patterns = compile_patterns({"telbestand_filename_patterns": []})
        assert match_telbestand("telbestandY2024W10.csv", patterns) is not None

    def test_accepts_string_value(self):
        patterns = compile_patterns(
            {"telbestand_filename_patterns": "foo_{year}_w{week}"}
        )
        match = match_telbestand("foo_2024_w10.csv", patterns)
        assert match.group("year") == "2024"
        assert match.group("week") == "10"

    def test_accepts_list_value(self):
        patterns = compile_patterns(
            {"telbestand_filename_patterns": ["a_{year}_{week}", "b_{year}_{week}"]}
        )
        assert match_telbestand("a_2024_10.csv", patterns) is not None
        assert match_telbestand("b_2024_10.csv", patterns) is not None

    def test_missing_placeholders_raises(self):
        with pytest.raises(ValueError, match="placeholders"):
            compile_patterns({"telbestand_filename_patterns": ["telbestandY2024W10"]})

    def test_missing_year_placeholder_raises(self):
        with pytest.raises(ValueError, match="placeholders"):
            compile_patterns({"telbestand_filename_patterns": ["foo_W{week}"]})

    def test_missing_week_placeholder_raises(self):
        with pytest.raises(ValueError, match="placeholders"):
            compile_patterns({"telbestand_filename_patterns": ["foo_Y{year}"]})

    def test_literal_chars_are_escaped(self):
        """Punten en andere regex-metakarakters moeten letterlijk worden gematcht."""
        patterns = compile_patterns(
            {"telbestand_filename_patterns": ["tel.{year}.{week}"]}
        )
        assert match_telbestand("tel.2024.10.csv", patterns) is not None
        assert match_telbestand("telX2024X10.csv", patterns) is None


class TestMatchTelbestand:
    def test_default_pattern_extracts_year_and_week(self):
        patterns = compile_patterns(None)
        match = match_telbestand("telbestandY2024W10.csv", patterns)
        assert match.group("year") == "2024"
        assert match.group("week") == "10"

    def test_default_pattern_handles_single_digit_week(self):
        patterns = compile_patterns(None)
        match = match_telbestand("telbestandY2024W5.csv", patterns)
        assert match.group("week") == "5"

    def test_returns_none_for_non_matching_name(self):
        patterns = compile_patterns(None)
        assert match_telbestand("random.csv", patterns) is None

    def test_custom_pattern_matches_vu_name(self):
        patterns = compile_patterns(
            {"telbestand_filename_patterns": ["VU_telbestand_{year}_W{week}"]}
        )
        match = match_telbestand("VU_telbestand_2024_W42.csv", patterns)
        assert match.group("year") == "2024"
        assert match.group("week") == "42"

    def test_first_matching_pattern_wins(self):
        patterns = compile_patterns(
            {
                "telbestand_filename_patterns": [
                    "telbestandY{year}W{week}",
                    "foo_{year}_{week}",
                ]
            }
        )
        match = match_telbestand("telbestandY2024W10.csv", patterns)
        assert match.group("year") == "2024"
        assert match.group("week") == "10"

    def test_fallback_to_second_pattern(self):
        patterns = compile_patterns(
            {
                "telbestand_filename_patterns": [
                    "telbestandY{year}W{week}",
                    "foo_{year}_{week}",
                ]
            }
        )
        match = match_telbestand("foo_2024_42.csv", patterns)
        assert match.group("year") == "2024"
        assert match.group("week") == "42"


class TestValidateTelbestandFilenamePatterns:
    def test_empty_config_passes(self):
        _validate_telbestand_filename_patterns({}, "cfg.json")

    def test_default_pattern_passes(self):
        _validate_telbestand_filename_patterns(
            {"telbestand_filename_patterns": [DEFAULT_TELBESTAND_PATTERN]},
            "cfg.json",
        )

    def test_missing_placeholders_exits_with_message(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _validate_telbestand_filename_patterns(
                {"telbestand_filename_patterns": ["telbestandY2024W10"]},
                "cfg.json",
            )
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "Configuratiefout in cfg.json" in out
        assert "telbestand_filename_patterns" in out
