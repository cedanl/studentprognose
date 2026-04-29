"""Contract tests for strategy preprocess() return values."""

import pandas as pd
import pytest

from studentprognose.strategies.individual import IndividualStrategy
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
