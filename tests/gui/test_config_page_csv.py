"""Tests voor het defensief inlezen van telbestand-CSV's op de configuratiepagina.

Telbestanden komen zowel puntkomma-gescheiden (legacy Studielink) als
komma-gescheiden (UvA SQL-export, o.a. de demodataset) voor. De
opleiding-dropdowns moeten beide formaten kunnen lezen, anders blijven ze leeg.

Slaat over als NiceGUI niet geïnstalleerd is (de `gui`-extra is optioneel).
"""

import pytest

pytest.importorskip("nicegui", reason="gui-extra niet geïnstalleerd")

import pandas as pd

from gui.pages.config_page import _ConfigView


def _write_csv(path, sep):
    pd.DataFrame(
        {"Isatcode": [30008, 56604], "Groepeernaam": ["B Psychologie", "B Geneeskunde"]}
    ).to_csv(path, sep=sep, index=False)


def test_detect_csv_sep_semicolon(tmp_path):
    path = tmp_path / "semi.csv"
    _write_csv(path, ";")
    assert _ConfigView._detect_csv_sep(str(path)) == ";"


def test_detect_csv_sep_comma(tmp_path):
    path = tmp_path / "comma.csv"
    _write_csv(path, ",")
    assert _ConfigView._detect_csv_sep(str(path)) == ","


def test_detect_csv_sep_missing_file_falls_back_to_semicolon(tmp_path):
    assert _ConfigView._detect_csv_sep(str(tmp_path / "nope.csv")) == ";"


@pytest.mark.parametrize("sep", [";", ","])
def test_read_table_autodetects_separator(tmp_path, sep):
    path = tmp_path / "tel.csv"
    _write_csv(path, sep)
    # Zonder expliciete sep moet elk formaat correct in >1 kolom worden gelezen.
    df = _ConfigView._read_table(str(path))
    assert df is not None
    assert list(df.columns) == ["Isatcode", "Groepeernaam"]


@pytest.mark.parametrize("sep", [";", ","])
def test_read_table_honours_explicit_separator(tmp_path, sep):
    path = tmp_path / "tel.csv"
    _write_csv(path, sep)
    df = _ConfigView._read_table(str(path), sep=_ConfigView._detect_csv_sep(str(path)))
    assert df is not None
    assert list(df.columns) == ["Isatcode", "Groepeernaam"]


def test_read_table_missing_returns_none(tmp_path):
    assert _ConfigView._read_table(str(tmp_path / "nope.csv")) is None
