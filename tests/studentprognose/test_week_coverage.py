"""Regressietests voor #303: ontbrekende weken in het voorspeljaar.

Drie fouten samen lieten de instroomvoorspelling (XGBoost) instorten zodra het
voorspeljaar weken miste aan het begin van het academisch jaar:

1. de ETL-interpolatie berekende ontbrekende weken wel, maar voegde ze nooit
   toe (left-merge op bestaande rijen);
2. de ETL zocht gaten in kalendervolgorde (1..52) in plaats van in de volgorde
   van het academisch jaar;
3. lege vroege weken gingen als 0 naar de regressor, terwijl die 0 in de
   trainingsjaren nooit voorkwam.

Daarnaast: de startweek werd altijd met 0 overschreven, en er was geen
waarschuwing voor een eindweek die niet bij de data past of voor een
voorspeljaar met ontbrekende weken.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from studentprognose.data.etl import _interpolate_missing_weeks
from studentprognose.data.prediction_validator import (
    _check_final_week,
    _check_week_coverage,
    _weekly_totals,
)
from studentprognose.data.transforms import ensure_start_week
from studentprognose.models.regressors import covered_feature_columns
from studentprognose.models.xgboost_regressor import predict_with_xgboost
from studentprognose.utils.weeks import get_all_weeks_ordered

_ORDER_38 = [int(w) for w in get_all_weeks_ordered(38)]


# ---------------------------------------------------------------------------
# ETL-interpolatie
# ---------------------------------------------------------------------------

def _etl_rows(weeks, year=2026, value=lambda w: 10.0 * _ORDER_38.index(w), **overrides):
    rows = []
    for w in weeks:
        row = {
            "Korte naam instelling": "X", "Collegejaar": year, "Weeknummer rapportage": w,
            "Weeknummer": w, "Faculteit": "F", "Type hoger onderwijs": "Bachelor",
            "Groepeernaam Croho": "P", "Naam Croho opleiding Nederlands": "P", "Croho": 1,
            "Herinschrijving": "Nee", "Hogerejaars": "Nee", "Herkomst": "NL",
            "Gewogen vooraanmelders": value(w), "Ongewogen vooraanmelders": value(w),
            "Aantal aanmelders met 1 aanmelding": None, "Inschrijvingen": None,
        }
        row.update(overrides)
        rows.append(row)
    return rows


def _run_etl_interpolation(tmp_path, rows, final_week=38):
    path = tmp_path / "cum.csv"
    pd.DataFrame(rows).to_csv(path, sep=";", index=False)
    _interpolate_missing_weeks(str(path), final_week=final_week)
    return pd.read_csv(path, sep=";")


def _weeks(df, **filters):
    for col, val in filters.items():
        df = df[df[col] == val]
    return sorted(df["Weeknummer"])


class TestEtlInterpolation:
    def test_missing_week_is_added_with_interpolated_value(self, tmp_path):
        """Fout 1: week 40 werd berekend maar nooit als rij toegevoegd."""
        out = _run_etl_interpolation(tmp_path, _etl_rows([38, 39, 41, 42]))
        assert 40 in _weeks(out)
        row = out[out["Weeknummer"] == 40].iloc[0]
        expected = 10.0 * _ORDER_38.index(40)
        assert row["Gewogen vooraanmelders"] == pytest.approx(expected)
        assert row["Weeknummer rapportage"] == 40
        assert row["Faculteit"] == "F" and row["Collegejaar"] == 2026

    def test_gap_across_new_year_is_filled(self, tmp_path):
        """Academische volgorde: 51 → 2 is één gat (52 en 1 ontbreken)."""
        out = _run_etl_interpolation(tmp_path, _etl_rows([50, 51, 2, 3]))
        assert _weeks(out) == [1, 2, 3, 50, 51, 52]

    def test_no_interpolation_across_the_academic_year_boundary(self, tmp_path):
        """Fout 2: met weken 41..52 en 1..35 zag de ETL één gat "35 → 41". Weken
        vóór de eerste (39-40) en ná de laatste snapshot (36-38) horen leeg te blijven."""
        present = [w for w in _ORDER_38 if w not in (36, 37, 38, 39, 40)]
        out = _run_etl_interpolation(tmp_path, _etl_rows(present))
        assert set(range(1, 53)) - set(_weeks(out)) == {36, 37, 38, 39, 40}

    def test_order_follows_final_academic_week(self, tmp_path):
        """Bij eindweek 38 liggen 36-37 tussen 35 en 38; bij eindweek 36 ligt 35-36
        aan het eind en 37-38 aan het begin van het jaar."""
        rows = _etl_rows([34, 35, 38, 39], value=lambda w: float(w))
        assert {36, 37} <= set(_weeks(_run_etl_interpolation(tmp_path, rows, final_week=38)))
        out_36 = _run_etl_interpolation(tmp_path, rows, final_week=36)
        assert not {36, 37} & set(_weeks(out_36))

    def test_only_first_year_rows_are_filled(self, tmp_path):
        rows = _etl_rows([39, 41], Hogerejaars="Ja") + _etl_rows([39, 41])
        out = _run_etl_interpolation(tmp_path, rows)
        assert _weeks(out, Hogerejaars="Ja") == [39, 41]
        assert _weeks(out, Hogerejaars="Nee") == [39, 40, 41]

    def test_series_are_kept_apart(self, tmp_path):
        """Elke reeks (opleiding/herkomst/jaar) wordt los aangevuld."""
        rows = _etl_rows([39, 41], Herkomst="NL") + _etl_rows([39, 40, 41], Herkomst="EER") \
            + _etl_rows([39, 41], year=2025)
        out = _run_etl_interpolation(tmp_path, rows)
        assert _weeks(out, Herkomst="NL", Collegejaar=2026) == [39, 40, 41]
        assert _weeks(out, Herkomst="EER") == [39, 40, 41]
        assert _weeks(out, Collegejaar=2025) == [39, 40, 41]

    def test_empty_key_and_empty_measure_survive(self, tmp_path):
        out = _run_etl_interpolation(tmp_path, _etl_rows([39, 41], Faculteit=None))
        row = out[out["Weeknummer"] == 40].iloc[0]
        assert pd.isna(row["Faculteit"])
        assert pd.isna(row["Inschrijvingen"])

    def test_week_53_is_left_alone(self, tmp_path):
        out = _run_etl_interpolation(tmp_path, _etl_rows([51, 52], value=float) + _etl_rows([53, 2], value=float))
        assert 53 in _weeks(out) and 1 in _weeks(out)

    def test_complete_data_is_unchanged(self, tmp_path):
        rows = _etl_rows([39, 40, 41])
        out = _run_etl_interpolation(tmp_path, rows)
        assert len(out) == 3


# ---------------------------------------------------------------------------
# XGBoost-features
# ---------------------------------------------------------------------------

def _wide(years, missing_in=None, missing_weeks=()):
    rows = []
    for yr in years:
        row = {"Collegejaar": yr, "Croho groepeernaam": "P", "Herkomst": "NL",
               "Examentype": "Bachelor", "Faculteit": "F"}
        for i, w in enumerate(_ORDER_38):
            row[str(w)] = 0.0 if w == 39 else 10.0 + i
        if yr == missing_in:
            for w in missing_weeks:
                row[str(w)] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


class TestCoveredFeatureColumns:
    def test_drops_early_weeks_missing_in_one_year(self):
        train = _wide(range(2018, 2026))
        test = _wide([2026], missing_in=2026, missing_weeks=(40,))
        cols = covered_feature_columns([train, test])
        assert "39" not in cols and "40" not in cols
        assert "41" in cols and "38" in cols and "Herkomst" in cols

    def test_keeps_all_weeks_when_every_year_is_covered(self):
        """Week 39 (startweek, altijd 0) valt weg; verder blijft alles staan."""
        cols = covered_feature_columns([_wide(range(2018, 2026)), _wide([2026])])
        weeks = [c for c in cols if c.isdigit()]
        assert weeks == [str(w) for w in _ORDER_38[1:]]

    def test_missing_week_value_no_longer_changes_the_prediction(self):
        """Kern van #303: of week 40 in het voorspeljaar 0 is of een echte waarde
        heeft, mag de voorspelling niet meer veranderen."""
        rng = np.random.default_rng(0)
        programmes = [f"P{i}" for i in range(12)]
        scale = {(p, yr): rng.uniform(1, 20) for p in programmes for yr in range(2016, 2027)}

        def rows(yr, p, missing=()):
            row = _wide([yr], missing_in=yr, missing_weeks=missing).assign(**{"Croho groepeernaam": p})
            weeks = [str(w) for w in _ORDER_38]
            row[weeks] = row[weeks] * scale[(p, yr)]
            return row

        train = pd.concat([rows(yr, p) for yr in range(2016, 2026) for p in programmes])
        count = pd.DataFrame([
            {"Croho groepeernaam": p, "Collegejaar": yr, "Herkomst": "NL",
             "Examentype": "Bachelor", "Aantal_studenten": round(30 * scale[(p, yr)])}
            for yr in range(2016, 2026) for p in programmes
        ])
        complete = pd.concat([rows(2026, p) for p in programmes])
        gap = pd.concat([rows(2026, p, missing=(40,)) for p in programmes])

        pred_complete, _ = predict_with_xgboost(train, complete, count)
        pred_gap, _ = predict_with_xgboost(train, gap, count)
        assert pred_gap.tolist() == pred_complete.tolist()
        # En de voorspelling zelf is zinnig: dicht bij 30 × schaal.
        expected = np.array([30 * scale[(p, 2026)] for p in programmes])
        assert np.median(np.abs(pred_complete - expected) / expected) < 0.25


# ---------------------------------------------------------------------------
# Startweek
# ---------------------------------------------------------------------------

class TestEnsureStartWeek:
    def test_adds_zero_column_when_absent(self):
        df = ensure_start_week(pd.DataFrame({"40": [1.0]}), final_week=38)
        assert df["39"].tolist() == [0]

    def test_keeps_real_values(self):
        df = ensure_start_week(pd.DataFrame({"39": [5.0], "40": [7.0]}), final_week=38)
        assert df["39"].tolist() == [5.0]


# ---------------------------------------------------------------------------
# Waarschuwingen vóór de voorspelling
# ---------------------------------------------------------------------------

def _long(years, weeks_by_year=None, value=lambda pos: 10.0 * pos, order=_ORDER_38):
    rows = []
    for yr in years:
        weeks = (weeks_by_year or {}).get(yr, order)
        for w in weeks:
            rows.append({"Collegejaar": yr, "Weeknummer": w, "Hogerejaars": "Nee",
                         "Gewogen vooraanmelders": value(order.index(w))})
    return pd.DataFrame(rows)


class TestFinalWeekCheck:
    def test_warns_when_data_runs_past_final_week(self):
        """Eindweek 36 terwijl de cyclus tot 38 loopt: week 37 bevat dan al de
        eindstand van het jaar."""
        order_36 = [int(w) for w in get_all_weeks_ordered(36)]
        # Echte cyclus: 39 → 38. Op de 36-volgorde staan 37/38 vooraan met de eindstand.
        value = lambda pos: 100.0 if order_36[pos] in (37, 38) else 2.0 * pos
        totals = _weekly_totals(_long(range(2020, 2026), value=value, order=order_36))
        with pytest.warns(UserWarning, match="final_academic_week=36"):
            _check_final_week(totals, 2026, final_week=36)

    def test_silent_for_a_normal_start_week(self):
        totals = _weekly_totals(_long(range(2020, 2026)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_final_week(totals, 2026, final_week=38)


    def test_silent_when_years_have_no_applicants(self):
        totals = pd.DataFrame({39: [0.0, 0.0], 40: [0.0, 0.0]}, index=[2024, 2025])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_final_week(totals, 2026, final_week=38)


class TestWeekCoverageCheck:
    def test_warns_for_missing_weeks_before_predict_week(self):
        weeks_2026 = [w for w in _ORDER_38 if w not in (39, 40, 5)]
        totals = _weekly_totals(_long(range(2020, 2027), {2026: weeks_2026}))
        with pytest.warns(UserWarning, match=r"mist week 39, 40, 5 vóór de voorspelweek 10"):
            _check_week_coverage(totals, 2026, 10, final_week=38)

    def test_ignores_weeks_after_predict_week_and_unusual_weeks(self):
        """Weken ná de voorspelweek tellen niet; een week die ook in de meeste
        eerdere jaren ontbreekt evenmin."""
        hist = {yr: [w for w in _ORDER_38 if w != 40] for yr in range(2020, 2026)}
        hist[2026] = [w for w in _ORDER_38 if w not in (40, 20)]
        totals = _weekly_totals(_long(range(2020, 2027), hist))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_week_coverage(totals, 2026, 10, final_week=38)

    def test_only_first_year_rows_count(self):
        data = _long(range(2020, 2027), {2026: [w for w in _ORDER_38 if w != 41]})
        extra = data[(data.Collegejaar == 2026) & (data.Weeknummer == 42)].assign(
            Weeknummer=41, Hogerejaars="Ja")
        totals = _weekly_totals(pd.concat([data, extra]))
        assert np.isnan(totals.loc[2026, 41])


def test_run_etl_passes_final_academic_week(tmp_path, monkeypatch):
    """De ETL interpoleert in de volgorde die bij de ingestelde eindweek hoort."""
    from studentprognose.config import load_defaults
    from studentprognose.data import etl

    (tmp_path / "data" / "input").mkdir(parents=True)
    (tmp_path / "data" / "input" / "vooraanmeldingen_cumulatief.csv").write_text("x\n")
    monkeypatch.chdir(tmp_path)
    seen = {}
    monkeypatch.setattr(etl, "_interpolate_missing_weeks",
                        lambda path, final_week: seen.setdefault("final_week", final_week))
    config = load_defaults()
    config.setdefault("model_config", {})["final_academic_week"] = 36
    etl.run_etl(config)
    assert seen["final_week"] == 36
