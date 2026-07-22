"""Tests voor gui.filtering_io — laden/opslaan, validatie en filterstatistiek."""

import json

import pandas as pd

from gui import filtering_io


def test_load_missing_returns_default(tmp_path):
    data = filtering_io.load_filtering(str(tmp_path / "ontbreekt.json"))
    assert data == filtering_io.DEFAULT_FILTERING


def test_load_fills_missing_keys(tmp_path):
    path = tmp_path / "f.json"
    path.write_text(json.dumps({"filtering": {"programme": ["B X"]}}), encoding="utf-8")
    data = filtering_io.load_filtering(str(path))
    assert data["filtering"]["programme"] == ["B X"]
    assert data["filtering"]["herkomst"] == []
    assert data["filtering"]["examentype"] == []


def test_save_roundtrip(tmp_path):
    path = tmp_path / "f.json"
    data = {"filtering": {"programme": [], "herkomst": ["NL"], "examentype": []}}
    filtering_io.save_filtering(str(path), data)
    assert filtering_io.load_filtering(str(path)) == data


def test_validate_ok():
    data = {"filtering": {"herkomst": ["NL", "EER"], "examentype": ["Bachelor"]}}
    assert filtering_io.validate_filtering(data) == []


def test_validate_rejects_bad_values():
    data = {"filtering": {"herkomst": ["Mars"], "examentype": ["Doctoraat"]}}
    errors = filtering_io.validate_filtering(data)
    assert len(errors) == 2


def _sample_df():
    return pd.DataFrame(
        {
            "Croho groepeernaam": ["B A", "B A", "M B", "B C", "M B"],
            "Herkomst": ["NL", "EER", "NL", "Niet-EER", "NL"],
            "Examentype": ["Bachelor", "Bachelor", "Master", "Bachelor", "Master"],
        }
    )


def _count(df, **kw):
    return filtering_io.count_programmes(
        df,
        programme_col="Croho groepeernaam",
        origin_col="Herkomst",
        exam_col="Examentype",
        programme=kw.get("programme", []),
        herkomst=kw.get("herkomst", []),
        examentype=kw.get("examentype", []),
    )


def test_count_no_filter_returns_all():
    assert _count(_sample_df()) == (3, 3)  # B A, M B, B C


def test_count_examentype_filter():
    # Alleen Master → M B
    assert _count(_sample_df(), examentype=["Master"]) == (1, 3)


def test_count_herkomst_filter():
    # Alleen Niet-EER → B C
    assert _count(_sample_df(), herkomst=["Niet-EER"]) == (1, 3)


def test_count_combined_filter():
    # Bachelor + NL → alleen B A (rij 1)
    assert _count(_sample_df(), examentype=["Bachelor"], herkomst=["NL"]) == (1, 3)


def test_count_programme_filter():
    assert _count(_sample_df(), programme=["B A", "M B"]) == (2, 3)
