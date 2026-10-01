"""Tests voor de dashboardpagina's rond modelperformance (issues #295 en #297).

Bewaakt de foutaggregatie (opleidingsniveau, grootteklassen, examentype, herkomst,
naïeve baseline), dat de fout altijd hoort bij de getoonde prognose, en dat de
pagina als zelfstandig HTML-bestand wordt weggeschreven, zonder dode
navigatielinks.
"""

import json

import numpy as np
import pandas as pd
import pytest

from studentprognose.output import performance_page
from studentprognose.output import model_performance as mp
from studentprognose.output.dashboard import DashboardBuilder
from studentprognose.utils.weeks import DataOption, StudentYearPrediction


def _row(year, prog, herkomst, week, actual, xgb, ratio, exam="Bachelor"):
    return {
        "Collegejaar": year,
        "Croho groepeernaam": prog,
        "Herkomst": herkomst,
        "Examentype": exam,
        "Weeknummer": week,
        "Aantal_studenten": actual,
        "SARIMA_cumulative": xgb,
        "Prognose_ratio": ratio,
    }


@pytest.fixture
def output():
    return pd.DataFrame(
        [
            # A: twee herkomstgroepen, samen 100 werkelijk / 110 voorspeld
            _row(2024, "A", "NL", 10, 60, 66.0, 60.0),
            _row(2024, "A", "EER", 10, 40, 44.0, 40.0),
            # latere week zonder voorspelling mag niet meetellen
            _row(2024, "A", "NL", 11, 60, np.nan, 61.0),
            # B: master, klein
            _row(2024, "B", "NL", 10, 10, 5.0, 12.0, exam="Master"),
            # C: numerus fixus
            _row(2024, "C", "NL", 10, 50, 80.0, 50.0),
            # D: één herkomstgroep zonder XGBoost-voorspelling -> onvolledig
            _row(2024, "D", "NL", 10, 30, 30.0, 30.0),
            _row(2024, "D", "EER", 10, 20, np.nan, 20.0),
            # E: geen realisatie -> niet evalueerbaar
            _row(2024, "E", "NL", 10, np.nan, 10.0, 10.0),
        ]
    )


@pytest.fixture
def studentcount():
    return pd.DataFrame(
        {
            "Collegejaar": [2023, 2023, 2023, 2024],
            "Croho groepeernaam": ["A", "A", "B", "A"],
            "Herkomst": ["NL", "EER", "NL", "NL"],
            "Examentype": ["Bachelor", "Bachelor", "Master", "Bachelor"],
            "Aantal_studenten": [50, 40, 8, 60],
        }
    )


def test_units_are_programme_level_at_predict_week(output, studentcount):
    units = mp.build_evaluation_units(output, 10, studentcount, {"C": 60})
    a = units.set_index("Croho groepeernaam").loc["A"]
    assert a["Aantal_studenten"] == 100
    assert a["SARIMA_cumulative"] == pytest.approx(110.0)
    assert a["Prognose_ratio"] == pytest.approx(100.0)
    assert set(units["Croho groepeernaam"]) == {"A", "B", "D"}, (
        "NF (C) en zonder realisatie (E) eruit"
    )


def test_incomplete_herkomst_prediction_is_not_summed(output, studentcount):
    units = mp.build_evaluation_units(output, 10, studentcount).set_index(
        "Croho groepeernaam"
    )
    assert np.isnan(units.loc["D", "SARIMA_cumulative"]), (
        "deelsom mag niet tegen volledige realisatie"
    )
    assert units.loc["D", "Prognose_ratio"] == pytest.approx(50.0)


def test_naive_baseline_is_previous_year_actual(output, studentcount):
    units = mp.build_evaluation_units(output, 10, studentcount).set_index(
        "Croho groepeernaam"
    )
    assert units.loc["A", mp.NAIVE] == 90
    assert units.loc["B", mp.NAIVE] == 8
    assert np.isnan(units.loc["D", mp.NAIVE])


def test_naive_baseline_missing_without_studentcount(output):
    units = mp.build_evaluation_units(output, 10, None)
    assert units[mp.NAIVE].isna().all()


def test_empty_when_no_realisation(output):
    live = output.assign(Aantal_studenten=np.nan)
    assert mp.build_evaluation_units(live, 10).empty


@pytest.mark.parametrize(
    ("actual", "label"),
    [
        (1, "< 25"),
        (24, "< 25"),
        (25, "25–49"),
        (99, "50–99"),
        (100, "100–249"),
        (250, "≥ 250"),
    ],
)
def test_size_class_boundaries(actual, label):
    assert str(mp.size_class(pd.Series([actual]))[0]) == label


def test_error_metrics_values():
    m = mp.error_metrics(np.array([100.0, 10.0]), np.array([110.0, 5.0]))
    assert m["n"] == 2
    assert m["wape"] == pytest.approx(15 / 110)
    assert m["mape"] == pytest.approx((0.10 + 0.50) / 2)
    assert m["bias"] == pytest.approx((115 - 110) / 110)
    assert m["within"] == pytest.approx(0.5)


def test_comparable_models_drops_low_coverage():
    units = pd.DataFrame(
        {
            "Aantal_studenten": [10, 20, 30, 40],
            "SARIMA_cumulative": [1, 2, 3, 4],
            "Prognose_ratio": [1, np.nan, np.nan, np.nan],
            mp.NAIVE: [1, 2, 3, np.nan],
        }
    )
    assert mp.comparable_models(units) == ["SARIMA_cumulative", mp.NAIVE]


def test_summarise_by_exam_type_uses_common_population(output, studentcount):
    units = mp.build_evaluation_units(output, 10, studentcount)
    models = mp.comparable_models(units)
    pop = mp.common_population(units, models)
    summary = mp.summarise(pop, models, by="Examentype")
    assert set(summary["group"]) == {"Bachelor", "Master"}
    # D valt af (geen XGBoost, geen baseline): Bachelor bevat alleen A.
    bachelor = summary[
        (summary["group"] == "Bachelor") & (summary["model"] == "SARIMA_cumulative")
    ].iloc[0]
    assert bachelor["n"] == 1
    assert bachelor["wape"] == pytest.approx(0.10)


def test_summarise_skips_empty_size_classes(output, studentcount):
    units = mp.build_evaluation_units(output, 10, studentcount)
    summary = mp.summarise(
        units, ["SARIMA_cumulative"], by="Grootteklasse", order=mp.SIZE_LABELS
    )
    assert list(summary["group"]) == ["< 25", "50–99", "100–249"]


# ── Pagina ────────────────────────────────────────────────────────────


def _cumulative(years=(2023, 2024)):
    rows = []
    for y in years:
        for w in (8, 9, 10, 11):
            rows.append(
                {
                    "Collegejaar": y,
                    "Croho groepeernaam": "A",
                    "Weeknummer": w,
                    "Examentype": "Bachelor",
                    "Faculteit": "FdM",
                    "Herkomst": "NL",
                    "Gewogen vooraanmelders": 10.0 * w,
                }
            )
    return pd.DataFrame(rows)


def _builder(tmp_path, data, studentcount, option=DataOption.CUMULATIVE):
    return DashboardBuilder(
        data=data,
        data_option=option,
        numerus_fixus_list={},
        student_year_prediction=StudentYearPrediction.FIRST_YEARS,
        ci_test_n=None,
        cwd=str(tmp_path),
        predict_week=10,
        data_cumulative=_cumulative(),
        data_studentcount=studentcount,
    )


def test_payload_is_strict_json(output, studentcount):
    payload = performance_page.build_payload(
        output,
        _cumulative(),
        studentcount,
        2024,
        10,
        [str(w) for w in range(1, 53)],
        38,
    )
    text = json.dumps(payload, allow_nan=False)  # NaN zou hier falen
    assert payload["performance"]["units"]
    assert payload["trend"]["programmes"][0]["code"] == "A"
    assert "</script" not in text.lower()


def test_cumulative_page_written_without_dead_nav_link(tmp_path, output, studentcount):
    b = _builder(tmp_path, output, studentcount)
    b.build_and_save()
    page = (
        tmp_path / "data/output/visualisations/cumulative/dashboard.html"
    ).read_text("utf-8")
    assert "__DATA__" not in page and "/*__PLOTLY__*/" not in page
    assert "../individual/dashboard.html" not in page, (
        "-d c bouwt geen individuele pagina"
    )
    assert "../final/dashboard.html" in page
    final = (tmp_path / "data/output/visualisations/final/dashboard.html").read_text(
        "utf-8"
    )
    assert "../individual/dashboard.html" not in final


def test_cumulative_page_without_realisation_still_renders(tmp_path, output):
    live = output.assign(Aantal_studenten=np.nan)
    b = _builder(tmp_path, live, None)
    b._save_cumulative()
    page = (
        tmp_path / "data/output/visualisations/cumulative/dashboard.html"
    ).read_text("utf-8")
    data = json.loads(
        page.split('type="application/json">', 1)[1].split("</script>", 1)[0]
    )
    assert data["performance"]["units"] == []
    assert data["trend"]["programmes"]


# ── Betrouwbaarheid en tabel per opleiding ────────────────────────────


@pytest.mark.parametrize(
    ("wape", "n_years", "beats_naive", "expected"),
    [
        (0.05, 3, True, "hoog"),
        (0.10, 2, None, "hoog"),  # geen baseline beschikbaar telt niet tegen
        (0.05, 1, True, "middel"),  # één jaar is te weinig bewijs voor "hoog"
        (0.05, 3, False, "middel"),  # slechter dan naïef
        (0.25, 4, True, "middel"),
        (0.26, 4, True, "laag"),
        (np.nan, 0, None, "onbekend"),
    ],
)
def test_trust_level(wape, n_years, beats_naive, expected):
    assert mp.trust_level(wape, n_years, beats_naive) == expected


def _units_for_trust():
    rows = []
    for y, xgb, naive in [(2022, 102, 90), (2023, 97, 120), (2024, 101, np.nan)]:
        rows.append(
            {
                "Collegejaar": y,
                "Croho groepeernaam": "A",
                "Examentype": "Bachelor",
                "Aantal_studenten": 100.0,
                "SARIMA_cumulative": float(xgb),
                "Prognose_ratio": np.nan,
                mp.NAIVE: naive,
            }
        )
    return pd.DataFrame(rows)


def test_programme_summary_scores_each_model_on_its_own_years():
    s = mp.programme_summary(
        _units_for_trust(), ["SARIMA_cumulative", "Prognose_ratio", mp.NAIVE]
    ).iloc[0]
    assert s["n_SARIMA_cumulative"] == 3
    assert s["wape_SARIMA_cumulative"] == pytest.approx(6 / 300)
    assert s["trust_SARIMA_cumulative"] == "hoog"
    # Ratio heeft geen enkele voorspelling: onbekend, maar de opleiding blijft in beeld.
    assert s["n_Prognose_ratio"] == 0
    assert s["trust_Prognose_ratio"] == "onbekend"
    assert s[f"trust_{mp.NAIVE}"] is None


def test_programme_summary_compares_with_naive_on_shared_years():
    units = _units_for_trust()
    # Op de gedeelde jaren (2022, 2023) is naïef foutloos, dus beter dan XGBoost.
    units[mp.NAIVE] = [100.0, 100.0, np.nan]
    s = mp.programme_summary(units, ["SARIMA_cumulative", mp.NAIVE]).iloc[0]
    assert s["trust_SARIMA_cumulative"] == "middel", (
        "slechter dan naïef mag geen 'hoog' zijn"
    )


def test_current_predictions_sums_herkomst_for_prediction_year(output):
    live = output.assign(Collegejaar=2025)
    cur = mp.current_predictions(live, 2025, 10, {"C": 60}).set_index(
        "Croho groepeernaam"
    )
    assert cur.loc["A", "SARIMA_cumulative"] == pytest.approx(110.0)
    assert np.isnan(cur.loc["D", "SARIMA_cumulative"])
    assert cur.loc["D", "Prognose_ratio"] == pytest.approx(50.0)
    assert "C" not in cur.index


def test_payload_table_includes_unevaluated_programmes(output, studentcount):
    # E heeft een prognose maar geen realisatie: in de tabel als 'onbekend'.
    data = output.copy()
    data.loc[data["Croho groepeernaam"] == "E", "Collegejaar"] = 2024
    payload = performance_page.build_payload(
        data,
        _cumulative(),
        studentcount,
        2024,
        10,
        [str(w) for w in range(1, 53)],
        38,
    )
    rows = {r["p"]: r for r in payload["performance"]["programmes"]["alle"]}
    # Zonder evaluatie geen modelcijfers; de pagina toont dan "onbekend".
    assert rows["E"]["m"] == {}
    assert rows["A"]["m"]["SARIMA_cumulative"]["t"] in mp.TRUST_LEVELS
    assert (
        payload["performance"]["current"]["A|Bachelor"]["m"]["SARIMA_cumulative"]
        == 110.0
    )


# ── Individuele pagina en eindoverzicht ───────────────────────────────


def test_track_models_final_prefers_filled_ensemble():
    data = pd.DataFrame(
        {
            "Weighted_ensemble_prediction": [np.nan],
            "Ensemble_prediction": [5.0],
            "SARIMA_cumulative": [4.0],
            "SARIMA_individual": [np.nan],
            "Prognose_ratio": [3.0],
        }
    )
    assert performance_page.track_models("final", data) == [
        "Ensemble_prediction",
        "SARIMA_cumulative",
        "Prognose_ratio",
    ]
    assert performance_page.track_models("individual", data) == []


def test_final_payload_has_forecast_overview_with_numerus_fixus(output, studentcount):
    data = output.assign(Weighted_ensemble_prediction=output["SARIMA_cumulative"])
    payload = performance_page.build_payload(
        data,
        _cumulative(),
        studentcount,
        2024,
        10,
        [str(w) for w in range(1, 53)],
        38,
        numerus_fixus={"C": 60},
        track="final",
    )
    fc = payload["forecast"]
    # A (110) + B (5) + C (80, NF telt mee in de prognose) + E (10); D is onvolledig.
    assert fc["total"] == pytest.approx(205.0)
    assert fc["nf"] == [{"p": "C", "cap": 60.0, "prognose": 80.0, "prev": None}]
    assert payload["performance"]["current"]["C|Bachelor"]["nf"] is True
    assert payload["trend"]["programmes"] == []
    assert (
        payload["performance"]["tableModels"][0]["key"]
        == "Weighted_ensemble_prediction"
    )


def test_individual_payload_trend_uses_xgboost_curve(output, studentcount):
    data = output.assign(SARIMA_individual=output["SARIMA_cumulative"])
    curve = pd.DataFrame(
        {
            "Collegejaar": [2024] * 4,
            "Croho groepeernaam": ["A", "A", "A", "A"],
            "Herkomst": ["NL", "EER", "NL", "EER"],
            "Examentype": ["Bachelor"] * 4,
            "Faculteit": ["FdM"] * 4,
            "Weeknummer": [9, 9, 10, 10],
            "XGBoost_cumulative": [10.0, 5.0, 20.0, 8.0],
        }
    )
    payload = performance_page.build_payload(
        data,
        None,
        studentcount,
        2024,
        10,
        [str(w) for w in range(1, 53)],
        38,
        track="individual",
        xgboost_curve=curve,
    )
    prog = payload["trend"]["programmes"][0]
    assert prog["code"] == "A"
    # Reeksen per herkomst; de pagina telt ze op voor "alle herkomsten".
    assert prog["series"]["NL"]["2024"][8:10] == [10.0, 20.0]
    assert prog["series"]["EER"]["2024"][8:10] == [5.0, 8.0]
    assert prog["predicted"] == {"NL": 66.0, "EER": 44.0}
    assert payload["meta"]["page"]["title"] == "Individueel model"


def test_builder_skips_individual_page_without_individual_predictions(
    tmp_path, output, studentcount
):
    data = output.assign(SARIMA_individual=np.nan)
    b = _builder(tmp_path, data, studentcount, option=DataOption.BOTH_DATASETS)
    b.build_and_save()
    vis = tmp_path / "data/output/visualisations"
    assert not (vis / "individual").exists()
    assert "../individual/dashboard.html" not in (
        vis / "final/dashboard.html"
    ).read_text("utf-8")


def test_both_mode_builds_all_three_pages(tmp_path, output, studentcount):
    """Standaardmodus (beide) met voorspellingen van beide sporen: drie pagina's."""
    data = output.assign(
        SARIMA_individual=output["SARIMA_cumulative"],
        Weighted_ensemble_prediction=output["SARIMA_cumulative"],
    )
    _builder(
        tmp_path, data, studentcount, option=DataOption.BOTH_DATASETS
    ).build_and_save()
    vis = tmp_path / "data/output/visualisations"
    assert sorted(p.name for p in vis.iterdir()) == [
        "cumulative",
        "final",
        "individual",
    ]
    final = (vis / "final/dashboard.html").read_text("utf-8")
    for page in ("individual", "cumulative", "final"):
        assert f"../{page}/dashboard.html" in final


# ── Herkomst (#297) ───────────────────────────────────────────────────


def _unrealised_output():
    """F: NL met realisatie, EER voorspeld maar zonder realisatie (geen EER-studenten)."""
    return pd.DataFrame(
        [
            _row(2024, "F", "NL", 10, 262, 308.0, 250.0),
            _row(2024, "F", "EER", 10, np.nan, 231.0, 20.0),
            _row(2024, "F", "Niet-EER", 10, np.nan, np.nan, np.nan),
        ]
    )


def test_unrealised_origin_prediction_counts_against_zero():
    units = mp.build_evaluation_units(_unrealised_output(), 10).set_index(
        "Croho groepeernaam"
    )
    f = units.loc["F"]
    assert f["Aantal_studenten"] == 262
    assert f["SARIMA_cumulative"] == pytest.approx(539.0), (
        "de EER-voorspelling hoort bij de opleiding, ook zonder EER-studenten"
    )
    assert f["Prognose_ratio"] == pytest.approx(270.0)
    assert f[mp.UNREALISED] == 1, "Niet-EER zonder voorspelling telt niet als gat"


def test_evaluated_prediction_equals_shown_prognose():
    """De WAPE moet gemeten zijn op precies de prognose die ernaast staat."""
    data = pd.concat([_unrealised_output()], ignore_index=True)
    units = mp.build_evaluation_units(data, 10).set_index("Croho groepeernaam")
    cur = mp.current_predictions(data, 2024, 10).set_index("Croho groepeernaam")
    for m in ("SARIMA_cumulative", "Prognose_ratio"):
        assert units.loc["F", m] == pytest.approx(cur.loc["F", m])


def test_payload_wape_matches_table_prognose():
    payload = performance_page.build_payload(
        _unrealised_output(), None, None, 2024, 10, [str(w) for w in range(1, 53)], 38
    )
    perf = payload["performance"]
    row = next(r for r in perf["programmes"]["alle"] if r["p"] == "F")
    prognose = perf["current"]["F|Bachelor"]["m"]["SARIMA_cumulative"]
    assert prognose == pytest.approx(539.0)
    assert row["m"]["SARIMA_cumulative"]["w"] == pytest.approx(
        abs(539 - 262) / 262, abs=1e-4
    )
    assert perf["nUnrealised"] == 1


def test_units_by_origin_keep_origin_and_naive_per_origin(output, studentcount):
    units = mp.build_evaluation_units(
        output, 10, studentcount, {"C": 60}, by_origin=True
    ).set_index(["Croho groepeernaam", "Herkomst"])
    assert units.loc[("A", "NL"), "SARIMA_cumulative"] == pytest.approx(66.0)
    assert units.loc[("A", "EER"), "Aantal_studenten"] == 40
    assert units.loc[("A", "NL"), mp.NAIVE] == 50, "baseline: NL-realisatie vorig jaar"
    assert units.loc[("A", "EER"), mp.NAIVE] == 40
    # D-EER mist alleen in XGBoost; per herkomst is D-NL wel volledig.
    assert units.loc[("D", "NL"), "SARIMA_cumulative"] == pytest.approx(30.0)
    assert np.isnan(units.loc[("D", "EER"), "SARIMA_cumulative"])


def test_by_origin_without_origin_column_is_empty(output):
    units = mp.build_evaluation_units(
        output.drop(columns="Herkomst"), 10, by_origin=True
    )
    assert units.empty


def test_origin_order_is_fixed_then_alphabetical():
    assert mp.origin_order(["Niet-EER", "Onbekend", "NL", "EER", None, "NL"]) == [
        "NL",
        "EER",
        "Niet-EER",
        "Onbekend",
    ]


def test_payload_has_view_per_origin(output, studentcount):
    payload = performance_page.build_payload(
        output,
        _cumulative(),
        studentcount,
        2024,
        10,
        [str(w) for w in range(1, 53)],
        38,
    )
    assert payload["meta"]["origins"] == ["NL", "EER"]
    views = payload["performanceByOrigin"]
    assert set(views) == {"NL", "EER"}
    eer = views["EER"]["current"]["A|Bachelor"]
    assert eer["m"]["SARIMA_cumulative"] == pytest.approx(44.0)
    assert eer["prev"] == 40, "naïef per herkomst"
    # De herkomstweergave heeft dezelfde opbouw als 'alle herkomsten'.
    assert set(views["NL"]) == set(payload["performance"])


def test_payload_error_by_origin_in_all_view(output, studentcount):
    payload = performance_page.build_payload(
        output,
        _cumulative(),
        studentcount,
        2024,
        10,
        [str(w) for w in range(1, 53)],
        38,
    )
    rows = payload["performance"]["summaries"]["alle"]["origin"]
    xgb = {r["group"]: r for r in rows if r["model"] == "SARIMA_cumulative"}
    assert list(xgb) == ["NL", "EER"]
    # NL: A 66/60, B 5/10 -> |6|+|5| / 70 ; D valt af (geen XGBoost voor D-EER
    # is per herkomst geen probleem, maar de ratio/naïef-populatie bepaalt mee).
    assert xgb["NL"]["n"] >= 2
    assert xgb["EER"]["wape"] == pytest.approx(0.1)
    # Per herkomst bestaat de uitsplitsing niet (alles is daar al één herkomst).
    assert "origin" not in payload["performanceByOrigin"]["NL"]["summaries"]["alle"]


def test_final_forecast_per_origin_adds_up(output, studentcount):
    data = output.assign(Weighted_ensemble_prediction=output["SARIMA_cumulative"])
    payload = performance_page.build_payload(
        data,
        _cumulative(),
        studentcount,
        2024,
        10,
        [str(w) for w in range(1, 53)],
        38,
        numerus_fixus={"C": 60},
        track="final",
    )
    fc = payload["forecast"]
    bars = {r["h"]: r["prognose"] for r in fc["herkomst"]}
    assert bars == {h: fc["byOrigin"][h]["total"] for h in fc["byOrigin"]}
    assert fc["byOrigin"]["EER"]["total"] == pytest.approx(44.0)


def test_payload_without_origin_column_has_no_origin_views(output, studentcount):
    payload = performance_page.build_payload(
        output.drop(columns="Herkomst"),
        _cumulative().drop(columns="Herkomst"),
        studentcount.drop(columns="Herkomst"),
        2024,
        10,
        [str(w) for w in range(1, 53)],
        38,
    )
    assert payload["meta"]["origins"] == []
    assert payload["performanceByOrigin"] == {}
    assert "origin" not in payload["performance"]["summaries"]["alle"]
    # Trend valt terug op één reeks zonder herkomst.
    assert list(payload["trend"]["programmes"][0]["series"]) == [""]


def test_page_has_origin_filter(tmp_path, output, studentcount):
    _builder(tmp_path, output, studentcount).build_and_save()
    page = (
        tmp_path / "data/output/visualisations/cumulative/dashboard.html"
    ).read_text("utf-8")
    assert 'id="origin-filter"' in page
    assert "Fout naar herkomst" in page
    assert 'id="expand-all"' in page, "uitklapbare herkomstrijen in de tabel"


def test_origin_views_add_up_to_programme_prognose(output, studentcount):
    """De uitklaprijen per herkomst tellen op tot de opleidingsrij erboven."""
    payload = performance_page.build_payload(
        output,
        _cumulative(),
        studentcount,
        2024,
        10,
        [str(w) for w in range(1, 53)],
        38,
    )
    total = payload["performance"]["current"]["A|Bachelor"]
    parts = [
        v["current"]["A|Bachelor"] for v in payload["performanceByOrigin"].values()
    ]
    for m in ("SARIMA_cumulative", "Prognose_ratio"):
        assert sum(p["m"][m] for p in parts) == pytest.approx(total["m"][m])
    assert sum(p["a"] for p in parts) == total["a"]
    assert sum(p["prev"] for p in parts) == total["prev"]
