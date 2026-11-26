"""Tests voor het opzoeken van instellingsnamen bij Brincodes (#328)."""

from studentprognose.data.institution_names import (
    institution_label,
    institution_options,
    load_institution_names,
    parse_institution_names,
)


def test_bundled_table_contains_known_institutions():
    names = load_institution_names()
    assert names["21PK"] == "Universiteit van Amsterdam"
    assert names["28DN"] == "Hogeschool van Amsterdam"
    assert names["21PC"] == "Rijksuniversiteit Groningen"
    assert len(names) > 40


def test_parse_skips_rows_without_code_or_name():
    text = "Brincode;Instellingsnaam;Soort\n21pk ;Universiteit van Amsterdam;wo\n;Leeg;hbo\n99XX;;hbo\n"
    assert parse_institution_names(text) == {"21PK": "Universiteit van Amsterdam"}


def test_label_falls_back_to_code():
    names = {"21PK": "Universiteit van Amsterdam"}
    assert institution_label("21PK", names) == "Universiteit van Amsterdam"
    assert institution_label("21pk", names) == "Universiteit van Amsterdam"
    assert institution_label("99ZZ", names) == "99ZZ"


def test_options_keep_code_as_key_sorted_by_name():
    names = {"21PK": "Universiteit van Amsterdam", "28DN": "Hogeschool van Amsterdam"}
    options = institution_options(["21PK", "99ZZ", "28DN"], names)
    assert list(options) == ["28DN", "21PK", "99ZZ"]
    assert options["28DN"] == "Hogeschool van Amsterdam"
    assert options["99ZZ"] == "99ZZ (naam onbekend)"


def test_options_disambiguate_duplicate_names():
    names = {"AA11": "Hogeschool X", "BB22": "Hogeschool X"}
    options = institution_options(["AA11", "BB22", "AA11"], names)
    assert options == {"AA11": "Hogeschool X (AA11)", "BB22": "Hogeschool X (BB22)"}
