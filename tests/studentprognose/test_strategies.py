"""Contract tests for strategy preprocess() return values."""

import pandas as pd
import pytest

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
