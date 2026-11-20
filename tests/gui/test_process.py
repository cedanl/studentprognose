"""Tests voor gui.process — CLI-locator en commando-preview."""

import os

from gui.process import higher_years_source, locate_cli, preview_command, run_conflicts


def test_locate_cli_finds_console_script():
    # Draait in de venv waar studentprognose geïnstalleerd is.
    path = locate_cli()
    assert os.path.isfile(path)
    assert "studentprognose" in os.path.basename(path)


def test_preview_command_formats_readable():
    assert preview_command(["-d", "c", "-w", "6"]) == "studentprognose -d c -w 6"
    assert preview_command(["init"]) == "studentprognose init"
    assert preview_command([]) == "studentprognose"


def test_build_run_args_defaults():
    from gui.process import build_run_args

    # Alleen --yes staat standaard aan.
    assert build_run_args() == ["--yes"]


def test_build_run_args_full():
    from gui.process import build_run_args

    args = build_run_args(
        dataset="Cumulatief",
        cohort="Eerstejaars",
        years="2023 2024",
        weeks="1:38",
        institutions=["00AA"],
        skip_years=2,
        noetl=True,
        dashboard=True,
        no_warnings=True,
        yes=True,
    )
    assert args == [
        "-d",
        "c",
        "-sy",
        "f",
        "-y",
        "2023",
        "2024",
        "-w",
        "1:38",
        "--institution",
        "00AA",
        "-sk",
        "2",
        "--noetl",
        "--dashboard",
        "--no-warnings",
        "--yes",
    ]


def test_build_run_args_splits_ranges_into_tokens():
    from gui.process import build_run_args

    args = build_run_args(weeks="10 : 20", yes=False)
    assert args == ["-w", "10", ":", "20"]


def test_build_run_args_skip_zero_omitted():
    from gui.process import build_run_args

    assert "-sk" not in build_run_args(skip_years=0, yes=False)


def test_build_run_args_cwf():
    from gui.process import build_run_args

    args = build_run_args(dataset="CWF")
    assert args[:2] == ["-d", "cwf"]


def test_wizard_mode_choice_overrides_stored_run_dataset(monkeypatch):
    from types import SimpleNamespace

    from gui.pages import run

    fake_app = SimpleNamespace(storage=SimpleNamespace(general={
        "run_settings": {"dataset": "Cumulatief", "weeks": "16"}
    }))
    monkeypatch.setattr(run, "app", fake_app)
    run.remember_wizard_mode("cwf")
    assert fake_app.storage.general["run_settings"] == {"dataset": "CWF", "weeks": "16"}
    run.remember_wizard_mode("onbekend")
    assert fake_app.storage.general["run_settings"]["dataset"] == "CWF"


# ── Conflicterende run-parameters (#322) ────────────────────────────────────


_CONFIG = {
    "paths": {
        "path_latest_cumulative": "data/input/totaal_cumulatief.xlsx",
        "path_latest_cwf": "data/input/totaal_cwf.xlsx",
    }
}


def _conflicts(tmp_path, **overrides):
    params = {
        "dataset": "Cumulatief",
        "cohort": "Eerstejaars",
        "skip_years": 0,
        "project_dir": str(tmp_path),
        "config": _CONFIG,
    }
    params.update(overrides)
    return run_conflicts(**params)


def test_run_conflicts_none_for_default_run(tmp_path):
    assert _conflicts(tmp_path) == []


def test_run_conflicts_cwf_with_skip_years(tmp_path):
    errors = _conflicts(tmp_path, dataset="CWF", skip_years=2)
    assert len(errors) == 1 and "backtest" in errors[0]


def test_run_conflicts_skip_years_fine_for_cumulative(tmp_path):
    assert _conflicts(tmp_path, skip_years=2) == []


def test_run_conflicts_higher_years_without_source_file(tmp_path):
    errors = _conflicts(tmp_path, cohort="Hogerejaars")
    assert len(errors) == 1
    assert "data/input/totaal_cumulatief.xlsx" in errors[0]


def test_run_conflicts_higher_years_with_source_file(tmp_path):
    src = tmp_path / "data" / "input" / "totaal_cwf.xlsx"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    assert _conflicts(tmp_path, dataset="CWF", cohort="Hogerejaars") == []


def test_run_conflicts_higher_years_impossible_for_both(tmp_path):
    errors = _conflicts(tmp_path, dataset="Beide", cohort="Hogerejaars")
    assert len(errors) == 1 and "Beide" in errors[0]


def test_higher_years_source_reads_config_paths():
    assert higher_years_source("CWF", _CONFIG) == "data/input/totaal_cwf.xlsx"
    assert higher_years_source("Beide", _CONFIG) is None
    assert higher_years_source("Individueel", {}) is None
