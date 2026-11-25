"""Tests voor het opzoeken van opleidingsnamen via de DUO-tabel (#326)."""

import warnings

import pandas as pd
import pytest

from studentprognose.data.programme_names import (
    fill_programme_names,
    load_programme_names,
    parse_programme_names,
    warn_missing_programme_names,
)

NAME_COL = "Naam Croho opleiding Nederlands"


# --- Inlezen van de tabel ----------------------------------------------------


def test_parse_takes_name_as_is_including_prefix():
    text = "56604B Psychologie\r\n60220M Publiekrecht\r\n01028Filosofie\r\n"
    names = parse_programme_names(text)
    assert names == {56604: "B Psychologie", 60220: "M Publiekrecht", 1028: "Filosofie"}


def test_parse_strips_only_padding_and_skips_invalid_lines():
    text = "34001B Opleiding tolk-vertaler   \n\nABCDEgeen code\n12345     \n"
    assert parse_programme_names(text) == {34001: "B Opleiding tolk-vertaler"}


def test_bundled_table_loads_with_official_names():
    names = load_programme_names()
    assert len(names) > 4000
    assert names[59320] == "B Security Studies"
    assert names[1028] == "Filosofie"
    # Namen zijn niet uniek: hbo- en wo-psychologie delen dezelfde officiële naam.
    assert names[34930] == names[56604] == "B Psychologie"


def test_empty_path_uses_bundled_table():
    assert load_programme_names("") == load_programme_names(None)


def test_own_table_file_is_used(tmp_path):
    own = tmp_path / "eigen.asc"
    own.write_bytes("99999B Eigen Opleiding é\r\n".encode("utf-8"))
    assert load_programme_names(str(own)) == {99999: "B Eigen Opleiding é"}


def test_own_table_file_in_cp1252_is_decoded(tmp_path):
    own = tmp_path / "eigen.asc"
    own.write_bytes("99999B Thérapie\r\n".encode("cp1252"))
    assert load_programme_names(str(own)) == {99999: "B Thérapie"}


def test_missing_own_table_file_warns_and_falls_back(tmp_path):
    with pytest.warns(UserWarning, match="niet gevonden"):
        names = load_programme_names(str(tmp_path / "bestaat-niet.asc"))
    assert names == load_programme_names()


# --- Naam invullen: volgorde eigen naam -> DUO-naam -> code -------------------


NAMES = {34808: "B Rechtsgeleerdheid", 56604: "B Psychologie"}


def test_placeholder_code_is_replaced_by_duo_name():
    df = pd.DataFrame({"Croho": [34808, 34808], NAME_COL: [34808, 34808]})
    missing = fill_programme_names(df, "Croho", NAME_COL, NAMES)
    assert df[NAME_COL].tolist() == ["B Rechtsgeleerdheid", "B Rechtsgeleerdheid"]
    assert missing == []


def test_placeholder_as_text_or_float_is_also_recognised():
    df = pd.DataFrame({"Croho": ["34808", 56604.0], NAME_COL: ["34808", 56604.0]})
    fill_programme_names(df, "Croho", NAME_COL, NAMES)
    assert df[NAME_COL].tolist() == ["B Rechtsgeleerdheid", "B Psychologie"]


def test_empty_name_is_filled():
    df = pd.DataFrame({"Croho": [34808, 56604], NAME_COL: [None, ""]})
    fill_programme_names(df, "Croho", NAME_COL, NAMES)
    assert df[NAME_COL].tolist() == ["B Rechtsgeleerdheid", "B Psychologie"]


def test_own_institution_name_takes_precedence():
    df = pd.DataFrame({"Croho": [56604], NAME_COL: ["Psychologie (UvA)"]})
    missing = fill_programme_names(df, "Croho", NAME_COL, NAMES)
    assert df[NAME_COL].tolist() == ["Psychologie (UvA)"]
    assert missing == []


def test_unknown_code_keeps_code_and_is_reported():
    df = pd.DataFrame({"Croho": [11111, 34808, 11111], NAME_COL: [11111, 34808, 11111]})
    missing = fill_programme_names(df, "Croho", NAME_COL, NAMES)
    assert df[NAME_COL].tolist() == [11111, "B Rechtsgeleerdheid", 11111]
    assert missing == [11111]


def test_code_column_is_never_changed():
    df = pd.DataFrame({"Croho": pd.array([34808, 56604], dtype="Int64"), NAME_COL: [34808, 56604]})
    before = df["Croho"].copy()
    fill_programme_names(df, "Croho", NAME_COL, NAMES)
    pd.testing.assert_series_equal(df["Croho"], before)


def test_missing_columns_is_a_no_op():
    df = pd.DataFrame({"Croho": [34808]})
    assert fill_programme_names(df, "Croho", NAME_COL, NAMES) == []
    assert list(df.columns) == ["Croho"]


def test_non_numeric_code_is_left_alone():
    df = pd.DataFrame({"Croho": ["B Psychologie"], NAME_COL: ["B Psychologie"]})
    assert fill_programme_names(df, "Croho", NAME_COL, NAMES) == []
    assert df[NAME_COL].tolist() == ["B Psychologie"]


# --- Waarschuwing ------------------------------------------------------------


def test_warning_lists_codes_and_truncates():
    with pytest.warns(UserWarning, match=r"12 code\(s\): 1, 2, .* en 2 meer"):
        warn_missing_programme_names(list(range(1, 13)))


def test_no_warning_when_nothing_missing():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_missing_programme_names([])



# --- Integratie: load_data vult de naam bij het inlezen ------------------------


def test_load_data_fills_official_name_and_keeps_code(tmp_path):
    from studentprognose.config import load_defaults
    from studentprognose.data.etl import _OUTPUT_COLUMNS
    from studentprognose.data.loader import load_data
    from studentprognose.utils.weeks import DataOption

    row = {c: "" for c in _OUTPUT_COLUMNS}
    row.update({
        "Korte naam instelling": "21PG", "Collegejaar": 2024, "Weeknummer rapportage": 10,
        "Weeknummer": 10, "Faculteit": "Onbekend", "Type hoger onderwijs": "Bachelor",
        "Groepeernaam Croho": 59320, "Naam Croho opleiding Nederlands": 59320, "Croho": 59320,
        "Herinschrijving": "Nee", "Hogerejaars": "Nee", "Herkomst": "NL",
        "Gewogen vooraanmelders": 5.0, "Ongewogen vooraanmelders": 5,
    })
    # load_data slaat de tweede regel over (skiprows=[1]): schrijf een dummyregel.
    frame = pd.DataFrame([row, row])
    path = tmp_path / "cumulatief.csv"
    frame.to_csv(path, sep=";", index=False)

    cfg = load_defaults()
    cfg["paths"].update({
        "path_cumulative": str(path), "path_cumulative_new": "", "path_latest_cumulative": "",
        "path_ensemble_weights": "", "path_student_count_first-years": "",
    })
    _, data_cumulative, *_ = load_data(cfg, DataOption.CUMULATIVE)

    assert data_cumulative[NAME_COL].tolist() == ["B Security Studies"]
    # De sleutel waarop de pipeline koppelt blijft de code.
    assert data_cumulative["Groepeernaam Croho"].tolist() == [59320]
    assert data_cumulative["Croho"].tolist() == [59320]


def test_default_config_uses_bundled_table():
    from studentprognose.config import load_defaults

    assert load_defaults()["paths"]["path_programme_names"] == ""


# --- Deel 2: etiketten voor uitvoer, dashboard en GUI --------------------------


from studentprognose.data.programme_names import (  # noqa: E402
    build_programme_names,
    programme_label,
    programme_name_column,
)


def test_programme_label_formats_name_and_code():
    assert programme_label(56604, NAMES) == "B Psychologie (56604)"
    assert programme_label("56604", NAMES) == "B Psychologie (56604)"
    assert programme_label(56604.0, NAMES) == "B Psychologie (56604)"


def test_programme_label_falls_back_to_code_or_legacy_name():
    assert programme_label(11111, NAMES) == "11111"
    assert programme_label("B Psychologie", NAMES) == "B Psychologie"
    assert programme_label(None, NAMES) == ""


def test_programme_name_column_maps_codes_and_keeps_legacy_names():
    codes = pd.Series([56604, 11111, "B Eigen"], dtype=object)
    assert programme_name_column(codes, NAMES).tolist() == ["B Psychologie", None, "B Eigen"]


def test_build_programme_names_prefers_own_institution_names():
    cumulative = pd.DataFrame(
        {"Croho": [56604, 59320, 34808], NAME_COL: ["Psychologie (UvA)", 59320, "34808"]}
    )
    names = build_programme_names({"paths": {}}, cumulative)
    assert names[56604] == "Psychologie (UvA)"  # eigen naam wint
    assert names[59320] == "B Security Studies"  # placeholder -> DUO
    assert len(names) > 4000


def test_build_programme_names_without_data_is_duo_table():
    assert build_programme_names({"paths": {"path_programme_names": ""}}) == load_programme_names()


# --- Uitvoerbestanden ----------------------------------------------------------


def _postprocessor():
    from studentprognose.config import load_defaults
    from studentprognose.output.postprocessor import PostProcessor
    from studentprognose.utils.weeks import DataOption

    return PostProcessor(load_defaults(), None, None, None, ".", DataOption.CUMULATIVE, None)


def test_output_gets_name_column_next_to_code_without_changing_data():
    pp = _postprocessor()
    pp.programme_names = NAMES
    data = pd.DataFrame(
        {"Collegejaar": [2024, 2024], "Croho groepeernaam": [56604, 11111], "Prognose": [1, 2]}
    )
    out = pp._with_programme_names(data)
    assert list(out.columns) == ["Collegejaar", "Croho groepeernaam", "Opleidingsnaam", "Prognose"]
    assert out["Opleidingsnaam"].tolist() == ["B Psychologie", None]
    assert "Opleidingsnaam" not in data.columns  # alleen bij het wegschrijven


def test_output_name_column_is_rebuilt_for_older_rows():
    pp = _postprocessor()
    pp.programme_names = NAMES
    data = pd.DataFrame(
        {"Croho groepeernaam": [56604, 34808], "Opleidingsnaam": ["oud", None]}
    )
    out = pp._with_programme_names(data)
    assert out["Opleidingsnaam"].tolist() == ["B Psychologie", "B Rechtsgeleerdheid"]


def test_output_uses_duo_table_when_pipeline_set_no_names():
    pp = _postprocessor()
    out = pp._with_programme_names(pd.DataFrame({"Croho groepeernaam": [59320]}))
    assert out["Opleidingsnaam"].tolist() == ["B Security Studies"]
