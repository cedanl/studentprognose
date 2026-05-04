import pytest
from studentprognose.cli import _expand_slices, parse_args
from studentprognose.utils.weeks import DataOption


class TestExpandSlices:
    def test_single_value(self):
        assert _expand_slices(["10"]) == [10]

    def test_range(self):
        assert _expand_slices(["10", ":", "13"]) == [10, 11, 12, 13]

    def test_compact_range(self):
        assert _expand_slices(["10:13"]) == [10, 11, 12, 13]

    def test_multiple_values(self):
        assert _expand_slices(["5", "10", "15"]) == [5, 10, 15]

    def test_mixed(self):
        assert _expand_slices(["1:3", "10"]) == [1, 2, 3, 10]


class TestParseArgs:
    def test_defaults(self):
        import datetime

        cfg = parse_args(["prog"])
        assert cfg.weeks == [datetime.date.today().isocalendar()[1]]
        assert cfg.noetl is False

    def test_week_flag(self):
        cfg = parse_args(["prog", "-w", "10"])
        assert cfg.weeks == [10]
        assert cfg.weeks_specified is True

    def test_week_range(self):
        cfg = parse_args(["prog", "-w", "10:12"])
        assert cfg.weeks == [10, 11, 12]

    def test_year_flag(self):
        cfg = parse_args(["prog", "-y", "2025"])
        assert cfg.years == [2025]

    def test_noetl_flag(self):
        cfg = parse_args(["prog", "--noetl"])
        assert cfg.noetl is True

    def test_ci_flag(self):
        cfg = parse_args(["prog", "--ci", "test", "5"])
        assert cfg.ci_test_n == 5

    def test_ci_invalid_raises(self):
        with pytest.raises(SystemExit) as exc:
            parse_args(["prog", "--ci", "invalid", "5"])
        assert exc.value.code != 0

    def test_dataset_cumulative(self):
        from studentprognose.utils.weeks import DataOption

        cfg = parse_args(["prog", "-d", "c"])
        assert cfg.data_option == DataOption.CUMULATIVE

    def test_init_command_in_parse_args(self):
        cfg = parse_args(["prog", "init"])
        assert cfg.command == "init"

    def test_default_command_is_none(self):
        cfg = parse_args(["prog"])
        assert cfg.command is None

    def test_init_not_needed_for_flags(self):
        cfg = parse_args(["prog", "-w", "10"])
        assert cfg.command is None

    def test_years_specified_flag(self):
        cfg = parse_args(["prog", "-y", "2024"])
        assert cfg.years_specified is True

    def test_years_not_specified_flag(self):
        cfg = parse_args(["prog"])
        assert cfg.years_specified is False

    def test_yes_flag(self):
        cfg = parse_args(["prog", "--yes"])
        assert cfg.yes is True

    def test_yes_default_is_false(self):
        cfg = parse_args(["prog"])
        assert cfg.yes is False

    def test_noetl_and_yes_combination(self):
        cfg = parse_args(["prog", "--noetl", "--yes"])
        assert cfg.noetl is True
        assert cfg.yes is True

    def test_week_53_accepted(self):
        cfg = parse_args(["prog", "-w", "53"])
        assert cfg.weeks == [53]
        assert cfg.weeks_specified is True

    def test_skipyears_flag(self):
        cfg = parse_args(["prog", "-sk", "2"])
        assert cfg.skip_years == 2

    def test_skipyears_default_is_zero(self):
        cfg = parse_args(["prog"])
        assert cfg.skip_years == 0

    def test_ci_invalid_exits_nonzero(self):
        with pytest.raises(SystemExit) as exc:
            parse_args(["prog", "--ci", "invalid", "5"])
        assert exc.value.code != 0

    def test_dataset_individual(self):
        cfg = parse_args(["prog", "-d", "i"])
        assert cfg.data_option == DataOption.INDIVIDUAL

    def test_dataset_both(self):
        cfg = parse_args(["prog", "-d", "b"])
        assert cfg.data_option == DataOption.BOTH_DATASETS
