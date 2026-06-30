"""Contract tests for strategy preprocess() return values."""

import pandas as pd

from studentprognose.strategies.individual import IndividualStrategy
from studentprognose.strategies.cumulative import CumulativeStrategy
from studentprognose.utils.weeks import DataOption


def _cfg():
    return {
        "numerus_fixus": {},
        "ensemble_weights": {
            "master_week_17_23": {"individual": 0.5, "cumulative": 0.5},
            "week_30_34": {"individual": 0.5, "cumulative": 0.5},
            "week_35_37": {"individual": 0.5, "cumulative": 0.5},
            "default": {"individual": 0.5, "cumulative": 0.5},
        },
        "column_roles": {
            "programme": "Croho groepeernaam",
            "academic_year": "Collegejaar",
            "exam_type": "Examentype",
            "origin": "Herkomst",
            "faculty": "Faculteit",
            "week": "Weeknummer",
            "enrollment_status": "Inschrijfstatus",
            "cancellation_date": "Datum intrekking vooraanmelding",
            "student_count": "Aantal_studenten",
        },
    }


def _minimal_individual_df():
    return pd.DataFrame({
        "Sleutel": [1000],
        "Datum Verzoek Inschr": ["03-10-2019"],
        "Ingangsdatum": ["01-09-2020"],
        "Collegejaar": [2020],
        "Datum intrekking vooraanmelding": [None],
        "Inschrijfstatus": ["Ingeschreven"],
        "Faculteit": ["FSW"],
        "Examentype": ["Propedeuse Bachelor"],
        "Croho": ["Psychologie"],
        "Croho groepeernaam": ["B Psychologie"],
        "Eerstejaars croho jaar": [2020],
        "Is eerstejaars croho opleiding": [1],
        "Is hogerejaars": [0],
        "BBC ontvangen": [0],
        "Nationaliteit": ["Nederlandse"],
        "EER": ["N"],
        "Aantal studenten": [1],
    })


class TestIndividualStrategyPreprocess:
    def test_preprocess_returns_none(self):
        # preprocess() wordt in de pipeline opgeslagen als `data_cumulative`.
        # Als het een DataFrame teruggeeft, crashen downstream functies die
        # cumulatieve kolommen verwachten (zoals run_pre_prediction_checks).
        strategy = IndividualStrategy(
            _minimal_individual_df(), _cfg(),
            None, None, None, "/tmp", DataOption.INDIVIDUAL, None,
        )
        assert strategy.preprocess() is None

    def test_preprocess_sets_backup(self):
        strategy = IndividualStrategy(
            _minimal_individual_df(), _cfg(),
            None, None, None, "/tmp", DataOption.INDIVIDUAL, None,
        )
        strategy.preprocess()
        assert strategy.data_individual_backup is not None


def _cfg_cumulative():
    cfg = _cfg()
    cfg["model_config"] = {
        "min_training_year": 2016,
        "cumulative_timeseries": "sarima",
        "cumulative_regressor": "xgboost",
    }
    return cfg


def _minimal_cumulative_df(programme=30029):
    """16-koloms format zoals de loader levert; Groepeernaam Croho als int (UvA-Isatcode)."""
    return pd.DataFrame({
        "Korte naam instelling": ["21PE", "21PE"],
        "Collegejaar": [2023, 2024],
        "Weeknummer rapportage": [10, 10],
        "Weeknummer": [10, 10],
        "Faculteit": ["Onbekend", "Onbekend"],
        "Type hoger onderwijs": ["Bachelor", "Bachelor"],
        "Groepeernaam Croho": [programme, programme],
        "Naam Croho opleiding Nederlands": [programme, programme],
        "Croho": [programme, programme],
        "Herinschrijving": ["Nee", "Nee"],
        "Hogerejaars": ["Nee", "Nee"],
        "Herkomst": ["NL", "NL"],
        "Gewogen vooraanmelders": [20.0, 25.0],
        "Ongewogen vooraanmelders": [40, 50],
        "Aantal aanmelders met 1 aanmelding": [None, None],
        "Inschrijvingen": [None, None],
    })


class TestCumulativeStrategyPreprocess:
    def test_croho_groepeernaam_normalized_to_int64(self):
        # Regressie: de UvA-Isatcode moet als één canoniek dtype door de pipeline.
        # preprocess normaliseert de numerieke Isatcode naar Int64 (NaN-veilig,
        # geen float-`.0`-staart), zodat de merge met data_studentcount in het
        # ratio-model (ratio.py) op gelijk dtype joint i.p.v. int-vs-str te botsen.
        strategy = CumulativeStrategy(
            _minimal_cumulative_df(programme=30029), None, _cfg_cumulative(),
            None, None, "/tmp", DataOption.CUMULATIVE, None,
        )
        result = strategy.preprocess()
        assert str(result["Croho groepeernaam"].dtype) == "Int64"
        assert set(result["Croho groepeernaam"]) == {30029}

    def test_set_filtering_normalizes_programme_keys(self):
        # programme_filtering moet hetzelfde dtype krijgen als de Int64-kolom,
        # anders loopt het .isin-filter in de strategie stil leeg op int-vs-str.
        strategy = CumulativeStrategy(
            _minimal_cumulative_df(programme=30029), None, _cfg_cumulative(),
            None, None, "/tmp", DataOption.CUMULATIVE, None,
        )
        strategy.set_filtering(["30029", "B Tand"], [], [])
        assert strategy.programme_filtering == [30029, "B Tand"]


def _tune_full_data(years):
    """Wide-format feature-matrix voor de regressor-tuning (incl. weekkolommen)."""
    import numpy as np

    from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK
    from studentprognose.utils.weeks import get_weeks_list

    week_cols = [str(w) for w in get_weeks_list(FINAL_ACADEMIC_WEEK, FINAL_ACADEMIC_WEEK)]
    rng = np.random.default_rng(0)
    rows = []
    for y in years:
        for prog in ["A", "B"]:
            base = 100 + (y - 2016) * 20 + (10 if prog == "B" else 0)
            row = {
                "Collegejaar": y, "Croho groepeernaam": prog, "Herkomst": "NL",
                "Examentype": "Bachelor", "Faculteit": "F",
                "Gewogen_t-2": base + rng.normal(0, 2),
                "Gewogen_t-5": base * 0.8 + rng.normal(0, 2),
                "Gewogen_acceleration": rng.normal(0, 1),
                "exclusivity_ratio": 0.5 + rng.normal(0, 0.05),
            }
            for i, wk in enumerate(week_cols):
                row[wk] = base * (i + 1) / len(week_cols) + rng.normal(0, 1)
            rows.append(row)
    return pd.DataFrame(rows)


def _tune_studentcount(years):
    rows = []
    for y in years:
        for prog in ["A", "B"]:
            base = 100 + (y - 2016) * 20 + (10 if prog == "B" else 0)
            rows.append({
                "Croho groepeernaam": prog, "Collegejaar": y, "Herkomst": "NL",
                "Examentype": "Bachelor", "Aantal_studenten": base,
            })
    return pd.DataFrame(rows)


class TestCumulativeStrategyTuning:
    def test_tune_flags_default_off(self):
        strategy = CumulativeStrategy(
            _minimal_cumulative_df(), None, _cfg_cumulative(),
            None, None, "/tmp", DataOption.CUMULATIVE, None,
        )
        assert strategy._tune is False
        assert strategy._tuned is False

    def test_tune_flags_read_from_config(self):
        cfg = _cfg_cumulative()
        cfg["model_config"]["tune_hyperparameters"] = True
        cfg["model_config"]["tuning_grid"] = {"max_depth": [2, 4]}
        strategy = CumulativeStrategy(
            _minimal_cumulative_df(), None, cfg,
            None, None, "/tmp", DataOption.CUMULATIVE, None,
        )
        assert strategy._tune is True
        assert strategy._tuning_grid == {"max_depth": [2, 4]}

    def test_tune_regressor_injects_best_params(self):
        years = [2016, 2017, 2018, 2019, 2020, 2021]
        cfg = _cfg_cumulative()
        cfg["model_features"] = {"regressor": {"categorical": ["Examentype", "Faculteit", "Croho groepeernaam", "Herkomst"]}}
        strategy = CumulativeStrategy(
            _minimal_cumulative_df(), _tune_studentcount(years), cfg,
            None, None, "/tmp", DataOption.CUMULATIVE, None,
        )
        strategy._tuning_grid = {"max_depth": [2, 4]}
        strategy._tune_regressor(_tune_full_data(years))

        injected = cfg["model_config"]["regressor_params"]["xgboost"]
        assert injected in [{"max_depth": 2}, {"max_depth": 4}]
        # De actieve regressor draagt de getunede parameter.
        assert strategy._regressor._model.get_params()["max_depth"] == injected["max_depth"]

    def test_tune_regressor_warns_on_too_few_years(self, recwarn):
        years = [2016, 2017]
        cfg = _cfg_cumulative()
        strategy = CumulativeStrategy(
            _minimal_cumulative_df(), _tune_studentcount(years), cfg,
            None, None, "/tmp", DataOption.CUMULATIVE, None,
        )
        before = strategy._regressor
        strategy._tune_regressor(_tune_full_data(years))
        # Geen params geïnjecteerd, regressor ongewijzigd, waarschuwing gegeven.
        assert "regressor_params" not in cfg["model_config"]
        assert strategy._regressor is before
        assert any(issubclass(w.category, UserWarning) for w in recwarn.list)

    def test_tune_targets_select_both_traps(self):
        cfg = _cfg_cumulative()
        cfg["model_config"]["tune_targets"] = {
            "regressor": {"max_depth": [2]},
            "sarima": {"order": [(1, 0, 1)]},
        }
        strategy = CumulativeStrategy(
            _minimal_cumulative_df(), None, cfg,
            None, None, "/tmp", DataOption.CUMULATIVE, None,
        )
        assert strategy._tune is True
        assert strategy._tune_sarima is True
        assert strategy._tuning_grid == {"max_depth": [2]}
        assert strategy._sarima_tuning_grid == {"order": [(1, 0, 1)]}

    def test_tune_targets_sarima_only(self):
        cfg = _cfg_cumulative()
        cfg["model_config"]["tune_targets"] = {"sarima": None}
        strategy = CumulativeStrategy(
            _minimal_cumulative_df(), None, cfg,
            None, None, "/tmp", DataOption.CUMULATIVE, None,
        )
        # Alleen SARIMA aan; de regressor-trap blijft uit.
        assert strategy._tune is False
        assert strategy._tune_sarima is True

    def test_tune_sarima_injects_best_orders(self):
        years = [2016, 2017, 2018, 2019, 2020, 2021]
        cfg = _cfg_cumulative()
        strategy = CumulativeStrategy(
            _minimal_cumulative_df(), None, cfg,
            None, None, "/tmp", DataOption.CUMULATIVE, None,
        )
        # Klein grid voor snelheid; lange curve-data en predict_week handmatig zetten
        # (zoals predict_nr_of_students dat in productie doet vóór de tuning-stap).
        strategy._sarima_tuning_grid = {"order": [(1, 0, 1), (1, 1, 1)], "seasonal_order": [(1, 1, 0, 52)]}
        strategy.data_cumulative = _tune_cumulative_long(years)
        strategy.predict_week = 12
        before_factory = strategy._forecaster_factory
        strategy._tune_sarima_orders()

        injected = cfg["model_config"]["forecaster_params"]["sarima"]
        assert "order" in injected and "seasonal_order" in injected
        # De factory is vervangen en bouwt een SARIMA met de getunede ordes.
        assert strategy._forecaster_factory is not before_factory
        forecaster = strategy._forecaster_factory()
        assert forecaster.order == tuple(injected["order"])

    def test_tune_sarima_warns_on_too_few_years(self, recwarn):
        years = [2016, 2017]
        cfg = _cfg_cumulative()
        strategy = CumulativeStrategy(
            _minimal_cumulative_df(), None, cfg,
            None, None, "/tmp", DataOption.CUMULATIVE, None,
        )
        strategy._sarima_tuning_grid = {"order": [(1, 0, 1)], "seasonal_order": [(1, 1, 0, 52)]}
        strategy.data_cumulative = _tune_cumulative_long(years)
        strategy.predict_week = 12
        before_factory = strategy._forecaster_factory
        strategy._tune_sarima_orders()
        # Geen ordes geïnjecteerd, factory ongewijzigd, waarschuwing gegeven.
        assert "forecaster_params" not in cfg["model_config"]
        assert strategy._forecaster_factory is before_factory
        assert any(issubclass(w.category, UserWarning) for w in recwarn.list)


def _tune_cumulative_long(years):
    """Lang-format cumulatieve curve (met ``ts``) voor SARIMA-tuning."""
    weeks = list(range(35, 53)) + list(range(1, 21))
    rows = []
    for y in years:
        for prog in ["A", "B"]:
            base = 100 + (y - 2016) * 10 + (5 if prog == "B" else 0)
            for i, wk in enumerate(weeks):
                rows.append({
                    "Collegejaar": y, "Faculteit": "F", "Herkomst": "NL",
                    "Examentype": "Bachelor", "Croho groepeernaam": prog,
                    "ts": float(base + 5 * i), "Weeknummer": int(wk),
                })
    return pd.DataFrame(rows)
