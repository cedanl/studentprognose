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
        assert patterns[0].pattern == DEFAULT_TELBESTAND_PATTERN

    def test_uses_default_when_key_missing(self):
        patterns = compile_patterns({})
        assert len(patterns) == 1
        assert patterns[0].pattern == DEFAULT_TELBESTAND_PATTERN

    def test_uses_default_when_value_is_empty_list(self):
        patterns = compile_patterns({"telbestand_filename_patterns": []})
        assert len(patterns) == 1
        assert patterns[0].pattern == DEFAULT_TELBESTAND_PATTERN

    def test_accepts_string_value(self):
        custom = r"foo_(?P<year>\d{4})_w(?P<week>\d{1,2})"
        patterns = compile_patterns({"telbestand_filename_patterns": custom})
        assert len(patterns) == 1
        assert patterns[0].pattern == custom

    def test_accepts_list_value(self):
        a = r"a_(?P<year>\d{4})W(?P<week>\d{1,2})"
        b = r"b_(?P<year>\d{4})W(?P<week>\d{1,2})"
        patterns = compile_patterns({"telbestand_filename_patterns": [a, b]})
        assert [p.pattern for p in patterns] == [a, b]

    def test_invalid_regex_raises_with_message(self):
        with pytest.raises(ValueError, match="Ongeldig regex-patroon"):
            compile_patterns({"telbestand_filename_patterns": ["[invalid"]})

    def test_missing_named_groups_raises(self):
        with pytest.raises(ValueError, match="mist de named groups"):
            compile_patterns({"telbestand_filename_patterns": [r"telbestandY\d{4}W\d+"]})

    def test_missing_year_group_only_raises(self):
        with pytest.raises(ValueError, match="mist de named groups"):
            compile_patterns(
                {"telbestand_filename_patterns": [r"foo_(?P<week>\d{1,2})"]}
            )


class TestMatchTelbestand:
    def test_default_pattern_matches_studielink_name(self):
        patterns = compile_patterns(None)
        match = match_telbestand("telbestandY2024W10.csv", patterns)
        assert match is not None
        assert match.group("year") == "2024"
        assert match.group("week") == "10"

    def test_returns_none_for_non_matching_name(self):
        patterns = compile_patterns(None)
        assert match_telbestand("random.csv", patterns) is None

    def test_custom_pattern_matches_vu_name(self):
        config = {
            "telbestand_filename_patterns": [
                r"VU_telbestand_(?P<year>\d{4})_W(?P<week>\d{1,2})"
            ]
        }
        patterns = compile_patterns(config)
        match = match_telbestand("VU_telbestand_2024_W42.csv", patterns)
        assert match is not None
        assert match.group("year") == "2024"
        assert match.group("week") == "42"

    def test_first_matching_pattern_wins(self):
        config = {
            "telbestand_filename_patterns": [
                r"telbestandY(?P<year>\d{4})W(?P<week>\d{1,2})",
                r"foo_(?P<year>\d{4})_(?P<week>\d{1,2})",
            ]
        }
        patterns = compile_patterns(config)
        match = match_telbestand("telbestandY2024W10.csv", patterns)
        assert match.group("year") == "2024"
        assert match.group("week") == "10"

    def test_fallback_to_second_pattern(self):
        config = {
            "telbestand_filename_patterns": [
                r"telbestandY(?P<year>\d{4})W(?P<week>\d{1,2})",
                r"foo_(?P<year>\d{4})_(?P<week>\d{1,2})",
            ]
        }
        patterns = compile_patterns(config)
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

    def test_invalid_regex_exits_with_message(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _validate_telbestand_filename_patterns(
                {"telbestand_filename_patterns": ["[invalid"]}, "cfg.json",
            )
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "Configuratiefout in cfg.json" in out
        assert "telbestand_filename_patterns" in out

    def test_missing_named_groups_exits(self):
        with pytest.raises(SystemExit) as exc:
            _validate_telbestand_filename_patterns(
                {"telbestand_filename_patterns": [r"foo(\d{4})W(\d+)"]},
                "cfg.json",
            )
        assert exc.value.code == 1
