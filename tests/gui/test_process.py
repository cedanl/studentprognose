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
