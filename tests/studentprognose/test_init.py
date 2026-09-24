"""Tests voor `studentprognose init` — vervolgstappen na de demodata-download."""

import os

from studentprognose import init


def _touch_individual(cwd):
    raw = cwd / "data" / "input_raw"
    raw.mkdir(parents=True)
    (raw / "individuele_aanmelddata.csv").write_text("Sleutel\n1")


def test_demo_next_steps_with_individual_data_mentions_both_sporen(tmp_path):
    _touch_individual(tmp_path)

    text = init._demo_next_steps(str(tmp_path))

    assert "-d b" in text
    assert "alleen telbestanden" not in text


def test_demo_next_steps_without_individual_data_falls_back_to_cumulative(tmp_path):
    (tmp_path / "data" / "input_raw").mkdir(parents=True)

    text = init._demo_next_steps(str(tmp_path))

    assert "-d c" in text
    assert "-d b -y" not in text
    assert "alleen telbestanden" in text


def test_run_init_prints_both_sporen_steps_after_full_demo_download(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init, "_ask_demo", lambda: True)

    def fake_download(cwd):
        with open(
            os.path.join(cwd, "data", "input_raw", "individuele_aanmelddata.csv"), "w"
        ) as f:
            f.write("Sleutel\n1")
        return True

    monkeypatch.setattr(init, "_download_demo", fake_download)

    init.run_init()

    out = capsys.readouterr().out
    assert "-d b -y 2024" in out


def test_run_init_without_demo_shows_own_data_steps(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init, "_ask_demo", lambda: False)

    init.run_init()

    out = capsys.readouterr().out
    assert "Volgende stappen" in out
    assert "Demodata staat in" not in out
