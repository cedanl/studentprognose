"""Tests voor scripts.fill_missing_origins (demodata met alle herkomsten)."""

import pandas as pd

from scripts.fill_missing_origins import (
    ORIGINS,
    fill_missing_origins,
    fill_oktober,
    fill_telbestanden,
    origin_weights,
)

_TEL_COLUMNS = [
    "Brincode", "Isatcode", "Type_HO", "Studiejaar", "Herkomst",
    "Status", "Hogerejaars", "Herinschrijving", "Aantal",
]


def test_origin_weights_are_reproducible_and_nl_is_largest():
    w = origin_weights("50700")
    assert w == origin_weights(50700)
    assert w["NL"] == 1.0
    assert 0 < w["Niet-EER"] < 1 and 0 < w["EER"] < 1


def test_fill_adds_only_missing_origins_scaled_from_source():
    df = pd.DataFrame({
        "prog": ["A", "A", "B"],
        "origin": ["NL", "NL", "EER"],
        "n": [100, 40, 10],
    })
    weights = {"NL": 1.0, "EER": 0.2, "Niet-EER": 0.1}
    new = fill_missing_origins(df, ["prog"], "origin", "n", ORIGINS, lambda _k: weights)

    a = new[new.prog == "A"]
    assert sorted(a.origin.unique()) == ["EER", "Niet-EER"]
    assert a[a.origin == "EER"].n.tolist() == [20, 8]
    assert a[a.origin == "Niet-EER"].n.tolist() == [10, 4]
    # B heeft alleen EER: NL wordt opgeschaald, Niet-EER afgeschaald.
    b = new[new.prog == "B"].set_index("origin").n
    assert b["NL"] == 50 and b["Niet-EER"] == 5


def test_fill_never_goes_below_one_and_is_idempotent():
    df = pd.DataFrame({"prog": ["A"], "origin": ["NL"], "n": [1]})
    weights = {"NL": 1.0, "EER": 0.01, "Niet-EER": 0.01}
    new = fill_missing_origins(df, ["prog"], "origin", "n", ORIGINS, lambda _k: weights)
    assert new.n.tolist() == [1, 1]
    again = fill_missing_origins(
        pd.concat([df, new]), ["prog"], "origin", "n", ORIGINS, lambda _k: weights
    )
    assert again.empty


def _write_tel(path, rows):
    pd.DataFrame(rows, columns=_TEL_COLUMNS).to_csv(path, index=False)


def test_fill_telbestanden_per_programme_year_across_weeks(tmp_path):
    tel = tmp_path / "telbestanden"
    tel.mkdir()
    _write_tel(tel / "w1.csv", [["21PK", "50700", "B", "2025", "N", "V", "N", "N", "100"]])
    _write_tel(tel / "w2.csv", [
        ["21PK", "50700", "B", "2025", "N", "V", "N", "N", "200"],
        ["21PK", "50700", "B", "2025", "N", "I", "N", "N", "50"],
    ])
    added = fill_telbestanden(str(tel))
    assert added == 6  # 3 bronrijen × 2 ontbrekende herkomsten

    w2 = pd.read_csv(tel / "w2.csv", dtype=str)
    assert sorted(w2.Herkomst.unique()) == ["E", "N", "R"]
    assert w2.columns.tolist() == _TEL_COLUMNS
    # Bestaande rijen blijven onaangeroerd bovenaan staan.
    assert w2.iloc[:2].Aantal.tolist() == ["200", "50"]
    assert fill_telbestanden(str(tel)) == 0


def test_fill_telbestanden_keeps_crlf_line_endings(tmp_path):
    tel = tmp_path / "telbestanden"
    tel.mkdir()
    path = tel / "w1.csv"
    pd.DataFrame(
        [["21PK", "50700", "B", "2025", "N", "V", "N", "N", "100"]], columns=_TEL_COLUMNS
    ).to_csv(path, index=False, lineterminator="\r\n")
    fill_telbestanden(str(tel))
    raw = path.read_bytes()
    assert raw.count(b"\r\n") == 4  # kop + 3 rijen
    assert b"\n" not in raw.replace(b"\r\n", b"")


def _row(origin, status, n, hoger="N"):
    return ["21PK", "50700", "B", "2025", origin, status, hoger, "N", str(n)]


def test_cancellations_and_higher_years_do_not_count_as_present(tmp_path):
    """Een herkomst met alleen Status A of hogerejaars telt de pipeline niet mee."""
    tel = tmp_path / "telbestanden"
    tel.mkdir()
    _write_tel(tel / "w1.csv", [
        _row("N", "V", 100),
        _row("E", "A", 30),           # annulering: telt niet
        _row("R", "V", 30, hoger="J"),  # hogerejaars: telt niet
    ])
    assert fill_telbestanden(str(tel)) == 2
    w1 = pd.read_csv(tel / "w1.csv", dtype=str)
    added = w1.iloc[3:]
    assert sorted(added.Herkomst) == ["E", "R"]
    assert (added.Status == "V").all() and (added.Hogerejaars == "N").all()


def test_missing_week_uses_observed_ratio_between_origins(tmp_path):
    """EER ontbreekt in week 2; schaal met de EER/NL-verhouding uit week 1."""
    tel = tmp_path / "telbestanden"
    tel.mkdir()
    _write_tel(tel / "w1.csv", [_row("N", "V", 100), _row("E", "V", 50), _row("R", "V", 10)])
    _write_tel(tel / "w2.csv", [_row("N", "V", 200), _row("R", "V", 20)])
    assert fill_telbestanden(str(tel)) == 1
    w2 = pd.read_csv(tel / "w2.csv", dtype=str)
    eer = w2[w2.Herkomst == "E"]
    assert eer.Aantal.tolist() == ["100"]  # 200 × (50 / 100)


def test_ratio_is_pooled_over_years_of_a_programme():
    """2025 mist EER helemaal; gebruik de EER/NL-verhouding uit 2024."""
    df = pd.DataFrame({
        "prog": ["A", "A", "A"],
        "year": [2024, 2024, 2025],
        "origin": ["NL", "EER", "NL"],
        "n": [100, 30, 200],
    })
    weights = {"NL": 1.0, "EER": 0.9, "Niet-EER": 0.1}
    new = fill_missing_origins(
        df, ["prog", "year"], "origin", "n", ORIGINS, lambda _k: weights, ratio_cols=["prog"]
    )
    eer_2025 = new[(new.year == 2025) & (new.origin == "EER")]
    assert eer_2025.n.tolist() == [60]  # 200 × 30/100, niet × 0.9


def test_fill_oktober_completes_each_programme_year(tmp_path):
    path = tmp_path / "oktober_bestand.xlsx"
    pd.DataFrame({
        "Collegejaar": [2025],
        "Isatcode": [50700],
        "Groepeernaam Croho": [50700],
        "EER-NL-nietEER": ["NL"],
        "Examentype code": ["Bachelor eerstejaars"],
        "Aantal eerstejaars croho": [1],
        "Aantal Hoofdinschrijvingen": [448],
    }).to_excel(path, index=False)

    assert fill_oktober(str(path)) == 2
    okt = pd.read_excel(path)
    assert sorted(okt["EER-NL-nietEER"]) == ["EER", "NL", "Niet-EER"]
    assert (okt["Aantal Hoofdinschrijvingen"] >= 1).all()
    assert fill_oktober(str(path)) == 0
