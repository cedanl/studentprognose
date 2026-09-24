"""Tests voor scripts.build_demo_data.build_zip."""

import os
import zipfile

import pytest

from scripts.build_demo_data import DEFAULT_MEMBERS, build_zip


def _make_source(tmp_path):
    source = tmp_path / "input_raw"
    (source / "telbestanden").mkdir(parents=True)
    (source / "telbestanden" / "telbestand_sl_1.csv").write_text("a,b")
    (source / "individuele_aanmelddata.csv").write_text("sleutel;status\n1;J")
    (source / "oktober_bestand.xlsx").write_bytes(b"x")
    return source


def test_build_zip_bundles_both_sporen(tmp_path):
    source = _make_source(tmp_path)
    output = tmp_path / "demo-data.zip"

    written = build_zip(str(source), str(output))

    assert written == sorted(
        [
            "telbestanden/telbestand_sl_1.csv",
            "individuele_aanmelddata.csv",
            "oktober_bestand.xlsx",
        ]
    )
    with zipfile.ZipFile(output) as zf:
        assert set(zf.namelist()) == set(written)


def test_build_zip_uses_default_members(tmp_path):
    source = _make_source(tmp_path)
    output = tmp_path / "demo-data.zip"

    build_zip(str(source), str(output))

    assert DEFAULT_MEMBERS == [
        "telbestanden",
        "individuele_aanmelddata.csv",
        "oktober_bestand.xlsx",
    ]


def test_build_zip_missing_member_raises(tmp_path):
    source = _make_source(tmp_path)
    os.remove(source / "oktober_bestand.xlsx")
    output = tmp_path / "demo-data.zip"

    with pytest.raises(FileNotFoundError, match="oktober_bestand.xlsx"):
        build_zip(str(source), str(output))


def test_build_zip_creates_output_dir(tmp_path):
    source = _make_source(tmp_path)
    output = tmp_path / "nested" / "dir" / "demo-data.zip"

    build_zip(str(source), str(output))

    assert output.exists()
