"""Een mislukte of afgebroken run mag de vorige output niet vernietigen (#322)."""

import os

import pandas as pd
import pytest

from studentprognose.output.postprocessor import PostProcessor, write_excel_atomic
from studentprognose.utils.weeks import DataOption, StudentYearPrediction


def _write_previous_output(tmp_path) -> str:
    out_dir = tmp_path / "data" / "output"
    out_dir.mkdir(parents=True)
    path = out_dir / "output_first-years_cumulatief.xlsx"
    pd.DataFrame({"a": [1, 2, 3]}).to_excel(path, index=False)
    return str(path)


def test_check_output_writable_keeps_existing_output(tmp_path, monkeypatch):
    path = _write_previous_output(tmp_path)
    size_before = os.path.getsize(path)
    monkeypatch.chdir(tmp_path)

    PostProcessor.check_output_writable(
        DataOption.CUMULATIVE, StudentYearPrediction.FIRST_YEARS, None
    )

    assert os.path.getsize(path) == size_before
    assert pd.read_excel(path)["a"].tolist() == [1, 2, 3]


def test_check_output_writable_creates_no_empty_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    PostProcessor.check_output_writable(
        DataOption.CWF, StudentYearPrediction.FIRST_YEARS, None
    )

    assert os.listdir(tmp_path / "data" / "output") == []


def test_check_output_writable_exits_1_when_file_is_locked(tmp_path, monkeypatch):
    _write_previous_output(tmp_path)
    monkeypatch.chdir(tmp_path)
    real_open = open

    def locked_open(file, mode="r", *args, **kwargs):
        if str(file).endswith("output_first-years_cumulatief.xlsx") and "a" in mode:
            raise PermissionError("vergrendeld door Excel")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", locked_open)
    with pytest.raises(SystemExit) as exc:
        PostProcessor.check_output_writable(
            DataOption.CUMULATIVE, StudentYearPrediction.FIRST_YEARS, None
        )
    assert exc.value.code == 1


def test_write_excel_atomic_replaces_file(tmp_path):
    path = tmp_path / "out.xlsx"
    write_excel_atomic(pd.DataFrame({"x": [1]}), str(path))
    write_excel_atomic(pd.DataFrame({"x": [2]}), str(path))
    assert pd.read_excel(path)["x"].tolist() == [2]
    assert sorted(os.listdir(tmp_path)) == ["out.xlsx"]


def test_write_excel_atomic_keeps_old_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "out.xlsx"
    write_excel_atomic(pd.DataFrame({"x": [1]}), str(path))

    def boom(self, *args, **kwargs):
        raise RuntimeError("schrijven mislukt halverwege")

    monkeypatch.setattr(pd.DataFrame, "to_excel", boom)
    with pytest.raises(RuntimeError):
        write_excel_atomic(pd.DataFrame({"x": [2]}), str(path))

    monkeypatch.undo()
    assert pd.read_excel(path)["x"].tolist() == [1]
    assert sorted(os.listdir(tmp_path)) == ["out.xlsx"]
