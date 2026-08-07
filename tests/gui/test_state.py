"""Tests voor gui.state.AppState — pure padlogica, geen NiceGUI nodig."""

import os

from gui.state import AppState


def test_paths_none_without_project():
    state = AppState()
    assert state.project_dir is None
    assert state.config_path is None
    assert state.filtering_path is None
    assert state.output_dir is None
    assert state.is_initialised is False


def test_derived_paths(tmp_path):
    state = AppState(project_dir=str(tmp_path))
    assert state.config_path == os.path.join(
        str(tmp_path), "configuration", "configuration.json"
    )
    assert state.filtering_path == os.path.join(
        str(tmp_path), "configuration", "filtering", "base.json"
    )
    assert state.output_dir == os.path.join(str(tmp_path), "data", "output")


def test_is_initialised_reflects_config_file(tmp_path):
    state = AppState(project_dir=str(tmp_path))
    assert state.is_initialised is False

    config_file = tmp_path / "configuration" / "configuration.json"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("{}", encoding="utf-8")

    assert state.is_initialised is True
