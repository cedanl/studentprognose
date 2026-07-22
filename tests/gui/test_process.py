"""Tests voor gui.process — CLI-locator en commando-preview."""

import os

from gui.process import locate_cli, preview_command


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
        institutions=["21PC"],
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
        "21PC",
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
