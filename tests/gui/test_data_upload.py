"""Tests voor de upload-validators en afgeleide dekkings-/overlapstatistieken.

Bewaakt de kritische, grootste logicamodule van de GUI: de validatie spiegelt de
pipeline-drempels maar mag nooit ``sys.exit()`` aanroepen, en de bestandsnaam-
sanitering moet padtraversal blokkeren.
"""

import copy

import pandas as pd
import pytest

from gui import data_upload as du
from gui.data_upload import (
    FileCheckResult,
    FileStatus,
    OverlapInfo,
    _build_gui_validation_cfg,
    compute_overlap,
    compute_tel_coverage,
    safe_telbestand_name,
)
from studentprognose.data.validation import _DEFAULT_VALIDATION_CFG

# Kolommen die de GUI-config vereist (Groepeernaam bewust weggelaten).
_TEL_COLS = [
    "Studiejaar",
    "Isatcode",
    "Aantal",
    "meercode_V",
    "Status",
    "Herinschrijving",
    "Hogerejaars",
    "Herkomst",
]


def _tel_csv(rows: int = 3, **overrides) -> str:
    """Bouw een geldige Studielink-telbestand-CSV (puntkomma-gescheiden)."""
    base = {
        "Studiejaar": 2024,
        "Isatcode": 12345,
        "Aantal": 10,
        "meercode_V": 1,
        "Status": "I",
        "Herinschrijving": "N",
        "Hogerejaars": "N",
        "Herkomst": "N",
    }
    base.update(overrides)
    df = pd.DataFrame({col: [base[col]] * rows for col in _TEL_COLS})
    return df.to_csv(sep=";", index=False)


# ── _build_gui_validation_cfg: afgeleid van de canonieke bron ───────────────


def test_gui_cfg_drops_groepeernaam_requirement():
    cfg = _build_gui_validation_cfg()
    assert "Groepeernaam" not in cfg["telbestand"]["required_columns"]
    # Maar de canonieke bron blijft ongewijzigd (diepe kopie, geen mutatie).
    assert "Groepeernaam" in _DEFAULT_VALIDATION_CFG["telbestand"]["required_columns"]


def test_gui_cfg_allows_herkomst_onbekend():
    cfg = _build_gui_validation_cfg()
    assert "O" in cfg["telbestand"]["herkomst_allowed"]
    assert "O" not in _DEFAULT_VALIDATION_CFG["telbestand"]["herkomst_allowed"]


def test_gui_cfg_inherits_thresholds_from_source():
    """Drempels drijven niet uiteen: ze komen één-op-één uit de bron."""
    cfg = _build_gui_validation_cfg()
    for key in ("nan_warning_threshold", "nan_error_threshold",
                "weeknummer_min", "weeknummer_max"):
        assert cfg[key] == _DEFAULT_VALIDATION_CFG[key]


def test_gui_cfg_does_not_mutate_source():
    before = copy.deepcopy(_DEFAULT_VALIDATION_CFG)
    _build_gui_validation_cfg()
    assert _DEFAULT_VALIDATION_CFG == before


# ── safe_telbestand_name: padtraversal-bescherming ──────────────────────────


def test_safe_name_strips_path_components():
    assert safe_telbestand_name("/tmp/foo/telbestandY2024W10.CSV") == "telbestandy2024w10.csv"


def test_safe_name_blocks_traversal():
    assert safe_telbestand_name("../../etc/passwd") == "passwd"


def test_safe_name_handles_backslashes():
    assert safe_telbestand_name(r"..\..\windows\evil.csv") == "evil.csv"


@pytest.mark.parametrize("bad", ["", ".", "..", "/", "   ", "foo/"])
def test_safe_name_rejects_empty_or_dot(bad):
    with pytest.raises(ValueError):
        safe_telbestand_name(bad)


# ── _check_telbestand: happy path + fouten ──────────────────────────────────


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_check_telbestand_valid(tmp_path):
    path = _write(tmp_path, "telbestandY2024W10.csv", _tel_csv())
    res = du._check_telbestand(path, "telbestandY2024W10.csv")
    assert res.status == FileStatus.VALID
    assert res.hard_errors == []
    assert res.row_count == 3


def test_check_telbestand_bad_filename(tmp_path):
    path = _write(tmp_path, "willekeurig.csv", _tel_csv())
    res = du._check_telbestand(path, "willekeurig.csv")
    assert res.status == FileStatus.ERRORS
    assert res.hard_errors


def test_check_telbestand_missing_required_column(tmp_path):
    df = pd.read_csv(pd.io.common.StringIO(_tel_csv()), sep=";")
    df = df.drop(columns=["Herkomst"])
    path = _write(tmp_path, "telbestandY2024W10.csv", df.to_csv(sep=";", index=False))
    res = du._check_telbestand(path, "telbestandY2024W10.csv")
    assert res.status == FileStatus.ERRORS
    assert "Herkomst" in res.missing_required


def test_check_telbestand_herkomst_onbekend_is_accepted(tmp_path):
    """'O' (onbekend) is een bewuste GUI-versoepeling en mag geen soft error zijn."""
    path = _write(tmp_path, "telbestandY2024W10.csv", _tel_csv(Herkomst="O"))
    res = du._check_telbestand(path, "telbestandY2024W10.csv")
    assert res.status == FileStatus.VALID


def test_check_telbestand_invalid_herkomst_is_soft(tmp_path):
    path = _write(tmp_path, "telbestandY2024W10.csv", _tel_csv(Herkomst="Z"))
    res = du._check_telbestand(path, "telbestandY2024W10.csv")
    assert res.status == FileStatus.ERRORS  # soft errors → status ERRORS
    assert any("Herkomst" in m for m in res.soft_errors)


# ── compute_overlap ─────────────────────────────────────────────────────────


def _okt_result(years):
    return FileCheckResult(
        filename="oktober_bestand.xlsx", status=FileStatus.VALID, years=years
    )


def _coverage(years):
    return du.TelCoverage(years=years, present={y: {10} for y in years}, gaps=[], total=len(years))


def test_compute_overlap_intersection_and_range():
    info = compute_overlap(_coverage([2020, 2021, 2022, 2023]), _okt_result([2021, 2022]))
    assert isinstance(info, OverlapInfo)
    assert info.intersection == [2021, 2022]
    assert info.year_range == [2020, 2021, 2022, 2023]


def test_compute_overlap_none_when_missing_side():
    assert compute_overlap(None, _okt_result([2021])) is None
    assert compute_overlap(_coverage([2021]), None) is None


def test_compute_overlap_none_when_oktober_invalid():
    bad = FileCheckResult(
        filename="oktober_bestand.xlsx", status=FileStatus.ERRORS, years=[2021]
    )
    assert compute_overlap(_coverage([2021]), bad) is None


def test_compute_overlap_none_without_oktober_years():
    assert compute_overlap(_coverage([2021]), _okt_result(None)) is None


# ── compute_tel_coverage: gaten binnen het bereik ───────────────────────────


def _valid(name):
    return FileCheckResult(filename=name, status=FileStatus.VALID)


def test_coverage_none_without_valid_files():
    assert compute_tel_coverage({}) is None
    errored = {"telbestandY2024W10.csv": FileCheckResult(
        filename="telbestandY2024W10.csv", status=FileStatus.ERRORS)}
    assert compute_tel_coverage(errored) is None


def test_coverage_single_file_no_gaps():
    cov = compute_tel_coverage({"telbestandY2024W10.csv": _valid("telbestandY2024W10.csv")})
    assert cov is not None
    assert cov.years == [2024]
    assert cov.present == {2024: {10}}
    assert cov.gaps == []
    assert cov.total == 1


def test_coverage_detects_gap_between_weeks_same_year():
    files = {
        "telbestandY2024W10.csv": _valid("telbestandY2024W10.csv"),
        "telbestandY2024W12.csv": _valid("telbestandY2024W12.csv"),
    }
    cov = compute_tel_coverage(files)
    assert cov is not None
    assert (2024, 11) in cov.gaps
    assert cov.total == 2
