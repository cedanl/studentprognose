"""Regressietests voor #318: een run voor een afgerond jaar mag niet in de toekomst kijken.

Kern-invariant: een voorspelling voor (jaar, week) hangt niet af van de waarnemingen
van dat jaar ná de voorspelweek. Een run op een afgerond jaar moet dus exact
hetzelfde opleveren als dezelfde run waarin die latere weken (nog) niet bestaan,
zoals in een live run.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import studentprognose.strategies.cumulative as cumulative_module
from studentprognose.config import load_defaults
from studentprognose.data.transforms import drop_weeks_after_predict_week
from studentprognose.strategies.cumulative import (
    CumulativeStrategy,
    _add_engineered_features,
    _lag_reference_week,
)
from studentprognose.utils.weeks import DataOption, get_all_weeks_ordered, week_sort_key

FINAL_WEEK = 38
PREDICT_YEAR = 2024


# ---------------------------------------------------------------------------
# Synthetische data
# ---------------------------------------------------------------------------


def _synthetic_data(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cumulatieve data (loader-formaat) en studentaantallen voor 2018-2024.

    De instroom hangt vooral af van de groei ná week 16: die zit niet in de weken
    t/m 16. Een model dat de echte latere weken ziet, voorspelt daardoor merkbaar
    anders dan een model dat ze niet ziet. Zo is een lek in de test zichtbaar.
    """
    rng = np.random.default_rng(seed)
    weeks = [int(w) for w in get_all_weeks_ordered(FINAL_WEEK)]
    cut = weeks.index(16)
    rows, counts = [], []
    for programme, exam_type in (("B Een", "Bachelor"), ("B Twee", "Bachelor"), ("M Drie", "Master")):
        for origin in ("NL", "EER"):
            for year in range(2018, PREDICT_YEAR + 1):
                early = float(rng.uniform(50, 150))
                late = float(rng.uniform(0, 400))
                for k, week in enumerate(weeks):
                    if k <= cut:
                        weighted = early * (k + 1) / (cut + 1)
                    else:
                        weighted = early + late * (k - cut) / (len(weeks) - 1 - cut)
                    rows.append({
                        "Korte naam instelling": "XX",
                        "Collegejaar": year,
                        "Weeknummer rapportage": week,
                        "Weeknummer": week,
                        "Faculteit": "FAC",
                        "Type hoger onderwijs": exam_type,
                        "Groepeernaam Croho": programme,
                        "Naam Croho opleiding Nederlands": programme,
                        "Croho": programme,
                        "Herinschrijving": "Nee",
                        "Hogerejaars": "Nee",
                        "Herkomst": origin,
                        "Gewogen vooraanmelders": weighted,
                        "Ongewogen vooraanmelders": weighted * 1.6,
                        "Aantal aanmelders met 1 aanmelding": weighted * 0.5,
                        "Inschrijvingen": 0.0,
                    })
                counts.append({
                    "Collegejaar": year,
                    "Croho groepeernaam": programme,
                    "Herkomst": origin,
                    "Examentype": exam_type,
                    "Aantal_studenten": int(0.8 * (early + late)),
                })
    return pd.DataFrame(rows), pd.DataFrame(counts)


def _as_live(data: pd.DataFrame, year: int, week: int) -> pd.DataFrame:
    """Dezelfde data zoals een live run in (year, week) hem zou zien."""
    later = data["Collegejaar"].eq(year) & (
        data["Weeknummer"].map(lambda w: week_sort_key(w, FINAL_WEEK))
        > week_sort_key(week, FINAL_WEEK)
    )
    return data.loc[~later].reset_index(drop=True)


def _config() -> dict:
    cfg = load_defaults()
    cfg["runtime"] = {**cfg.get("runtime", {}), "cpu_count": 1}
    cfg["numerus_fixus"] = {}
    cfg["model_config"] = {**cfg.get("model_config", {}), "final_academic_week": FINAL_WEEK}
    return cfg


def _predict(data, counts, week, cfg=None) -> pd.Series:
    strategy = CumulativeStrategy(
        data.copy(), counts.copy(), cfg or _config(), None, None, "/tmp",
        DataOption.CUMULATIVE, None,
    )
    strategy.preprocess()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = strategy.predict_nr_of_students(PREDICT_YEAR, week, 0)
    out = out.dropna(subset=["SARIMA_cumulative"])
    return out.set_index(["Croho groepeernaam", "Herkomst"])["SARIMA_cumulative"].sort_index()


@pytest.fixture(scope="module")
def synthetic():
    return _synthetic_data()


# ---------------------------------------------------------------------------
# drop_weeks_after_predict_week
# ---------------------------------------------------------------------------


def _long(rows):
    return pd.DataFrame(rows, columns=["Collegejaar", "Weeknummer"])


class TestDropWeeksAfterPredictWeek:
    def test_spring_week_keeps_autumn_and_drops_later_weeks(self):
        data = _long([(2024, 40), (2024, 52), (2024, 16), (2024, 17), (2024, 38)])
        result = drop_weeks_after_predict_week(data, 2024, 16, FINAL_WEEK)
        assert sorted(result["Weeknummer"]) == [16, 40, 52]

    def test_autumn_week_drops_spring_weeks_of_same_year(self):
        # Week 41 ligt aan het begin van het collegejaar: week 5 en 30 zijn toekomst.
        data = _long([(2024, 39), (2024, 41), (2024, 42), (2024, 5), (2024, 30)])
        result = drop_weeks_after_predict_week(data, 2024, 41, FINAL_WEEK)
        assert sorted(result["Weeknummer"]) == [39, 41]

    def test_training_years_stay_complete(self):
        data = _long([(2023, 17), (2023, 38), (2024, 17)])
        result = drop_weeks_after_predict_week(data, 2024, 16, FINAL_WEEK)
        assert result.values.tolist() == [[2023, 17], [2023, 38]]

    def test_later_years_are_truncated_too(self):
        data = _long([(2025, 16), (2025, 17)])
        result = drop_weeks_after_predict_week(data, 2024, 16, FINAL_WEEK)
        assert result["Weeknummer"].tolist() == [16]

    def test_iso_week_53_lies_between_52_and_1(self):
        data = _long([(2024, 52), (2024, 53), (2024, 1)])
        assert drop_weeks_after_predict_week(data, 2024, 53, FINAL_WEEK)[
            "Weeknummer"
        ].tolist() == [52, 53]
        assert drop_weeks_after_predict_week(data, 2024, 52, FINAL_WEEK)[
            "Weeknummer"
        ].tolist() == [52]

    def test_final_week_drops_nothing(self):
        data = _long([(2024, 39), (2024, 1), (2024, 38)])
        result = drop_weeks_after_predict_week(data, 2024, FINAL_WEEK, FINAL_WEEK)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Lag-features in academische volgorde
# ---------------------------------------------------------------------------


class TestLagReferenceWeek:
    @pytest.mark.parametrize(
        ("predict_week", "lag", "final_week", "expected"),
        [
            (10, 2, 38, 8),
            (10, 5, 38, 5),
            (3, 5, 38, 50),   # telt terug over de jaarwisseling
            (40, 2, 38, 39),  # niet week 38: die ligt aan het eind van het collegejaar
            (40, 5, 38, 39),  # stopt bij de startweek
            (37, 2, 36, 37),  # final_week 36: startweek 37
            (41, 5, 36, 37),
            (53, 2, 38, 51),
            (1, 2, 38, 51),
        ],
    )
    def test_reference_week(self, predict_week, lag, final_week, expected):
        assert _lag_reference_week(predict_week, lag, final_week) == expected

    @pytest.mark.parametrize("predict_week", [37, 38, 40, 45, 52, 1, 3, 20, 36])
    @pytest.mark.parametrize("lag", [2, 5])
    def test_never_after_predict_week(self, predict_week, lag):
        ref = _lag_reference_week(predict_week, lag, 36)
        assert week_sort_key(ref, 36) <= week_sort_key(predict_week, 36)


def _features_long(rows):
    defaults = {
        "Collegejaar": 2024, "Faculteit": "FacA", "Herkomst": "NL",
        "Examentype": "Bachelor", "Croho groepeernaam": "B Opleiding",
        "Gewogen vooraanmelders": 0.0, "Ongewogen vooraanmelders": 0.0,
        "Aantal aanmelders met 1 aanmelding": 0.0, "Inschrijvingen": 0.0,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _features_wide():
    return pd.DataFrame([{
        "Collegejaar": 2024, "Faculteit": "FacA", "Herkomst": "NL",
        "Examentype": "Bachelor", "Croho groepeernaam": "B Opleiding",
    }])


class TestEngineeredFeaturesNoLookahead:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_autumn_lag_ignores_end_of_same_year(self):
        # Week 35 en 36 zijn het éínde van collegejaar 2024: toekomst vanaf week 37.
        long = _features_long([
            {"Weeknummer": 35, "Gewogen vooraanmelders": 999.0},
            {"Weeknummer": 36, "Gewogen vooraanmelders": 999.0},
            {"Weeknummer": 37, "Gewogen vooraanmelders": 4.0},
            {"Weeknummer": 39, "Gewogen vooraanmelders": 6.0},
        ])
        result = _add_engineered_features(_features_wide(), long, 39, final_week=36)
        assert result.loc[0, "Gewogen_t-2"] == pytest.approx(4.0)
        assert result.loc[0, "Gewogen_t-5"] == pytest.approx(4.0)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_week1_fallback_not_used_before_week1(self):
        # In week 40 ligt week 1 nog in de toekomst: geen terugval op week 1.
        long = _features_long([
            {"Weeknummer": 1, "Gewogen vooraanmelders": 999.0},
            {"Weeknummer": 40, "Gewogen vooraanmelders": 6.0},
        ])
        result = _add_engineered_features(_features_wide(), long, 40, final_week=38)
        assert result.loc[0, "Gewogen_t-2"] == pytest.approx(0.0)
        assert result.loc[0, "Gewogen_t-5"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Persistentie bij ontbrekende forecast
# ---------------------------------------------------------------------------


def _fill_strategy(predict_week: int = 36) -> CumulativeStrategy:
    strategy = CumulativeStrategy.__new__(CumulativeStrategy)
    strategy.final_academic_week = FINAL_WEEK
    strategy.predict_week = predict_week
    return strategy


def _fill_data():
    return pd.DataFrame({"Collegejaar": [2024, 2023], "36": [10.0, 1.0], "37": [0.0, 2.0], "38": [0.0, 3.0]})


class TestFillFutureWeeks:
    def test_empty_forecast_persists_predict_week_value(self):
        data = _fill_data()
        mask = data["Collegejaar"] == 2024
        assert _fill_strategy()._fill_future_weeks(data, mask, []) is False
        assert data.loc[0, ["37", "38"]].tolist() == [10.0, 10.0]
        # Andere jaren blijven onaangeroerd.
        assert data.loc[1, ["36", "37", "38"]].tolist() == [1.0, 2.0, 3.0]

    def test_full_forecast_is_used(self):
        data = _fill_data()
        mask = data["Collegejaar"] == 2024
        assert _fill_strategy()._fill_future_weeks(data, mask, [12.0, 15.0]) is True
        assert data.loc[0, ["37", "38"]].tolist() == [12.0, 15.0]

    def test_nan_in_forecast_carries_last_value_forward(self):
        data = _fill_data()
        mask = data["Collegejaar"] == 2024
        assert _fill_strategy()._fill_future_weeks(data, mask, [12.0, np.nan]) is False
        assert data.loc[0, ["37", "38"]].tolist() == [12.0, 12.0]

    def test_iso_week_53_fills_spring_weeks(self):
        # Week 53 valt tussen 52 en 1; de weken daarna (1 t/m 38) krijgen persistentie.
        data = pd.DataFrame({"Collegejaar": [2024], "53": [7.0], "1": [0.0], "38": [0.0]})
        mask = data["Collegejaar"] == 2024
        assert _fill_strategy(predict_week=53)._fill_future_weeks(data, mask, []) is False
        assert data.loc[0, ["1", "38"]].tolist() == [7.0, 7.0]

    def test_missing_week_column_is_skipped(self):
        # Week 37 bestaat niet als kolom; de forecast voor 38 hoort op 38.
        data = _fill_data().drop(columns="37")
        mask = data["Collegejaar"] == 2024
        assert _fill_strategy()._fill_future_weeks(data, mask, [12.0, 15.0]) is True
        assert data.loc[0, "38"] == 15.0


# ---------------------------------------------------------------------------
# Strategie: afgerond jaar == live
# ---------------------------------------------------------------------------


class TestStrategyHasNoLookahead:
    @pytest.mark.parametrize("week", [16, 41])
    def test_failed_sarima_does_not_leak_real_future_weeks(self, synthetic, monkeypatch, week):
        data, counts = synthetic
        monkeypatch.setattr(cumulative_module, "predict_with_sarima_cumulative", lambda *a, **k: [])
        completed = _predict(data, counts, week)
        live = _predict(_as_live(data, PREDICT_YEAR, week), counts, week)
        assert len(completed) == 6
        pd.testing.assert_series_equal(completed, live)

    @pytest.mark.parametrize("week", [16, 41])
    @pytest.mark.parametrize("sarima_fails", [True, False])
    def test_regressor_features_identical_for_completed_and_live(
        self, synthetic, monkeypatch, week, sarima_fails
    ):
        # Strenger dan de uitkomst: de volledige feature-matrix van het voorspeljaar
        # (weekkolommen én lag-features) moet gelijk zijn. In week 41 wees lag 5
        # voorheen naar week 36, het eind van hetzelfde collegejaar.
        data, counts = synthetic
        forecast = (lambda *a, **k: []) if sarima_fails else (lambda *a, **k: np.full(a[4], 50.0))
        monkeypatch.setattr(cumulative_module, "predict_with_sarima_cumulative", forecast)
        original = cumulative_module.predict_with_xgboost

        def features_for(run_data):
            seen = []

            def spy(train, test, *args, **kwargs):
                seen.append(test.copy())
                return original(train, test, *args, **kwargs)

            monkeypatch.setattr(cumulative_module, "predict_with_xgboost", spy)
            _predict(run_data, counts, week)
            return (
                pd.concat(seen)
                .sort_values(["Croho groepeernaam", "Herkomst"])
                .reset_index(drop=True)
            )

        completed = features_for(data)
        live = features_for(_as_live(data, PREDICT_YEAR, week))
        assert len(completed) == 6
        pd.testing.assert_frame_equal(completed, live)

    def test_real_sarima_is_identical_for_completed_and_live(self, synthetic):
        data, counts = synthetic
        completed = _predict(data, counts, 16)
        live = _predict(_as_live(data, PREDICT_YEAR, 16), counts, 16)
        assert len(completed) == 6
        pd.testing.assert_series_equal(completed, live)

    def test_failed_sarima_warns_with_combinations(self, synthetic, monkeypatch):
        data, counts = synthetic
        monkeypatch.setattr(cumulative_module, "predict_with_sarima_cumulative", lambda *a, **k: [])
        strategy = CumulativeStrategy(
            data.copy(), counts.copy(), _config(), None, None, "/tmp",
            DataOption.CUMULATIVE, None,
        )
        strategy.preprocess()
        with pytest.warns(UserWarning, match=r"SARIMA gaf voor 6 combinatie\(s\)"):
            strategy.predict_nr_of_students(PREDICT_YEAR, 16, 0)

    def test_dashboard_keeps_full_data(self, synthetic, monkeypatch):
        # Alleen de modelinput wordt afgekapt; het dashboard toont de echte curve.
        data, counts = synthetic
        monkeypatch.setattr(cumulative_module, "predict_with_sarima_cumulative", lambda *a, **k: [])
        strategy = CumulativeStrategy(
            data.copy(), counts.copy(), _config(), None, None, "/tmp",
            DataOption.CUMULATIVE, None,
        )
        strategy.preprocess()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            strategy.predict_nr_of_students(PREDICT_YEAR, 16, 0)
        dashboard = strategy.get_dashboard_data()["data_cumulative"]
        assert 30 in set(dashboard.loc[dashboard["Collegejaar"] == PREDICT_YEAR, "Weeknummer"])


# ---------------------------------------------------------------------------
# Tuning ziet alleen jaren vóór het voorspeljaar
# ---------------------------------------------------------------------------


def _fake_tuning(monkeypatch, seen: dict) -> None:
    """Vervang de tuning door een spion die bijhoudt welke jaren hij krijgt."""
    from studentprognose.models import tuning

    def fake_tune_regressor(full_data, *args, **kwargs):
        seen["regressor"] = set(full_data["Collegejaar"].astype(int))
        return {"best_params": None, "results": [], "n_candidates": 0, "best_mape": np.nan,
                "target": "regressor", "model": "xgboost", "regressor_name": "xgboost"}

    def fake_tune_sarima(data_cumulative, *args, **kwargs):
        seen["sarima"] = set(data_cumulative["Collegejaar"].astype(int))
        return {"best_params": None, "results": [], "n_candidates": 0, "best_mape": np.nan,
                "target": "sarima", "model": "sarima"}

    monkeypatch.setattr(tuning, "tune_regressor", fake_tune_regressor)
    monkeypatch.setattr(tuning, "tune_sarima", fake_tune_sarima)
    monkeypatch.setattr(tuning, "format_tuning_results", lambda result: "")
    monkeypatch.setattr(cumulative_module, "predict_with_sarima_cumulative", lambda *a, **k: [])


def _tuning_config() -> dict:
    cfg = _config()
    cfg["model_config"]["tune_targets"] = {"regressor": None, "sarima": None}
    return cfg


class TestTuningSeesOnlyPastYears:
    def test_tuning_input_excludes_predict_year(self, synthetic, monkeypatch):
        seen: dict = {}
        _fake_tuning(monkeypatch, seen)
        data, counts = synthetic
        _predict(data, counts, 16, cfg=_tuning_config())

        assert seen["regressor"] and max(seen["regressor"]) < PREDICT_YEAR
        assert seen["sarima"] and max(seen["sarima"]) < PREDICT_YEAR

    @pytest.mark.filterwarnings("ignore:SARIMA gaf voor:UserWarning")
    def test_warns_when_earlier_year_follows_tuning(self, synthetic, monkeypatch):
        _fake_tuning(monkeypatch, {})
        data, counts = synthetic
        strategy = CumulativeStrategy(
            data.copy(), counts.copy(), _tuning_config(), None, None, "/tmp",
            DataOption.CUMULATIVE, None,
        )
        strategy.preprocess()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            strategy.predict_nr_of_students(PREDICT_YEAR, 16, 0)
        with pytest.warns(UserWarning, match="geen eerlijke terugblik"):
            strategy.predict_nr_of_students(PREDICT_YEAR - 1, 16, 0)
