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


# ── isatcode_str ──────────────────────────────────────────────────────────────


def test_isatcode_str_int_and_float():
    assert filtering_io.isatcode_str(30008) == "30008"
    # Uit Excel/CSV komt de code soms als float binnen.
    assert filtering_io.isatcode_str(30008.0) == "30008"
    assert filtering_io.isatcode_str("30008.0") == "30008"
    assert filtering_io.isatcode_str("30008") == "30008"


def test_isatcode_str_keeps_legacy_name():
    assert filtering_io.isatcode_str("B Psychologie") == "B Psychologie"


def test_isatcode_str_empty_for_none():
    assert filtering_io.isatcode_str(None) == ""


# ── programme_name_map ──────────────────────────────────────────────────────────


def test_programme_name_map_builds_code_to_name():
    df = pd.DataFrame(
        {
            "Croho groepeernaam": [30008, 30008, 56553.0],
            "groepeernaam_croho": ["B Psychologie", "B Psychologie", "B Gezondheid"],
        }
    )
    mapping = filtering_io.programme_name_map(
        df, code_col="Croho groepeernaam", name_col="groepeernaam_croho"
    )
    assert mapping == {"30008": "B Psychologie", "56553": "B Gezondheid"}


def test_programme_name_map_missing_column_returns_empty():
    df = pd.DataFrame({"Croho groepeernaam": [30008]})
    assert (
        filtering_io.programme_name_map(
            df, code_col="Croho groepeernaam", name_col="groepeernaam_croho"
        )
        == {}
    )


# ── build_programme_options ─────────────────────────────────────────────────────


def test_build_programme_options_labels_and_numeric_sort():
    options = filtering_io.build_programme_options(
        [56553, 30008, 30008.0],
        name_map={"30008": "B Psychologie"},
    )
    # Numeriek gesorteerd, dubbelen samengevoegd, naam als label indien bekend.
    assert list(options.keys()) == ["30008", "56553"]
    assert options["30008"] == "30008 — B Psychologie"
    assert options["56553"] == "56553"


def test_build_programme_options_handles_legacy_names_and_blanks():
    options = filtering_io.build_programme_options([30008, "B A", None, ""])
    assert set(options.keys()) == {"30008", "B A"}
    # Numerieke codes sorteren vóór leesbare namen.
    assert list(options.keys()) == ["30008", "B A"]


def test_count_programme_filter_matches_across_dtype():
    # Datakolom is numeriek (isatcode), filter komt als string binnen: moet toch matchen.
    df = pd.DataFrame(
        {
            "Croho groepeernaam": [30008, 30008, 56553],
            "Herkomst": ["NL", "EER", "NL"],
            "Examentype": ["Bachelor", "Bachelor", "Master"],
        }
    )
    assert _count(df, programme=["30008"]) == (1, 2)
