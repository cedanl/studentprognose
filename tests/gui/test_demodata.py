"""Tests voor gui.demodata.extract_zip — uitpakken en zip-slip-bescherming."""

import os
import zipfile

import pytest

from gui.demodata import extract_zip


def _make_zip(path, members: dict[str, bytes]):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def test_extract_creates_files_and_subdirs(tmp_path):
    zip_path = tmp_path / "demo.zip"
    _make_zip(
        zip_path,
        {
            "oktober_bestand.xlsx": b"x",
            "telbestanden/telbestand_sl_1.csv": b"a,b",
        },
    )
    dest = tmp_path / "input_raw"
    extracted = extract_zip(str(zip_path), str(dest))

    assert os.path.isfile(dest / "oktober_bestand.xlsx")
    assert os.path.isfile(dest / "telbestanden" / "telbestand_sl_1.csv")
    assert set(extracted) == {
        "oktober_bestand.xlsx",
        "telbestanden/telbestand_sl_1.csv",
    }


def test_extract_skips_zip_slip(tmp_path):
    zip_path = tmp_path / "evil.zip"
    _make_zip(zip_path, {"../escape.txt": b"nope", "ok.txt": b"yes"})
    dest = tmp_path / "input_raw"
    extracted = extract_zip(str(zip_path), str(dest))

    # Het traversal-lid is overgeslagen; het nette lid is uitgepakt.
    assert extracted == ["ok.txt"]
    assert not os.path.exists(tmp_path / "escape.txt")


def test_bad_zip_raises(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_text("not a zip")
    with pytest.raises(zipfile.BadZipFile):
        extract_zip(str(bad), str(tmp_path / "out"))
