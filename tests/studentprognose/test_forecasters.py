import numpy as np
import pandas as pd
import pytest

from studentprognose.models.base import BaseForecaster
from studentprognose.models.sarima import (
    SARIMAForecaster,
    predict_with_sarima_cumulative,
    predict_with_sarima_individual,
)
from studentprognose.models.forecasters import (
    ETSForecaster,
    ThetaForecaster,
    AutoARIMAForecaster,
)
from studentprognose.utils.weeks import get_all_weeks_ordered


def _seasonal_series(n_years: int = 5, season_length: int = 52) -> np.ndarray:
    """Genereer synthetische seizoensgebonden tijdreeks."""
    n = n_years * season_length
    trend = np.linspace(100, 200, n)
    season = 20 * np.sin(2 * np.pi * np.arange(n) / season_length)
    noise = np.random.default_rng(42).normal(0, 2, n)
    return (trend + season + noise).astype(np.float64)


FORECASTER_CLASSES = [
    SARIMAForecaster,
    ETSForecaster,
    ThetaForecaster,
    AutoARIMAForecaster,
]


@pytest.mark.parametrize("cls", FORECASTER_CLASSES, ids=lambda c: c.__name__)
def test_forecaster_implements_base(cls):
    assert issubclass(cls, BaseForecaster)


@pytest.mark.parametrize("cls", FORECASTER_CLASSES, ids=lambda c: c.__name__)
def test_fit_forecast_returns_array(cls):
    ts = _seasonal_series()
    steps = 10
    model = cls()
    model.fit(ts)
    result = model.forecast(steps=steps)

    assert isinstance(result, np.ndarray)
    assert len(result) == steps
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("cls", [ETSForecaster, ThetaForecaster])
def test_exog_ignored_gracefully(cls):
    """ETS en Theta ondersteunen geen exogene variabelen — moeten exog negeren."""
    ts = _seasonal_series()
    model = cls()
    exog = np.random.default_rng(42).normal(0, 1, len(ts))
    model.fit(ts, exog=exog)
    result = model.forecast(steps=5)
    assert len(result) == 5


def test_auto_arima_supports_exog():
    """AutoARIMA moet exogene variabelen accepteren."""
    ts = _seasonal_series()
    n = len(ts)
    exog = np.random.default_rng(42).normal(0, 1, n)

    model = AutoARIMAForecaster()
    model.fit(ts, exog=exog)

    exog_future = np.random.default_rng(99).normal(0, 1, 5)
    result = model.forecast(steps=5, exog=exog_future)
    assert len(result) == 5


def test_forecaster_predictions_reasonable():
    """Voorspellingen moeten in dezelfde orde van grootte liggen als de trainingsdata."""
    ts = _seasonal_series()
    model = ETSForecaster()
    model.fit(ts)
    pred = model.forecast(steps=52)

    assert np.mean(pred) > 50
    assert np.mean(pred) < 500


def _empty_individual_wide_frame(programme: str, herkomst: str, examentype: str) -> pd.DataFrame:
    """Bouw een minimale wide-format frame met alle weekkolommen op 0.

    Bootst de output van ``IndividualStrategy._transform_data_individual`` na voor een
    (programme × herkomst × examentype)-combinatie zonder enige historische aanmelding.
    """
    group_cols = {
        "Collegejaar": [2024],
        "Faculteit": ["FSW"],
        "Herkomst": [herkomst],
        "Examentype": [examentype],
        "Croho groepeernaam": [programme],
    }
    week_cols = {w: [0.0] for w in get_all_weeks_ordered()}
    return pd.DataFrame({**group_cols, **week_cols})


def test_predict_individual_returns_empty_for_all_zero_history(capsys):
    """SARIMA mag geen 0 retourneren voor combinaties zonder historische aanmeldingen.

    Regressie voor issue #196: een all-zero training-tijdreeks leidde stil tot een
    forecast-vector van nullen, waardoor ``SARIMA_individual = 0`` in de output stond
    in plaats van expliciet ``NaN``.
    """
    programme, herkomst, examentype = "B Sociologie", "Niet-EER", "Bachelor"
    frame = _empty_individual_wide_frame(programme, herkomst, examentype)
    row = {"Croho groepeernaam": programme, "Herkomst": herkomst, "Examentype": examentype}

    result = predict_with_sarima_individual(
        frame, row,
        predict_year=2024, predict_week=12, max_year=2024,
        numerus_fixus_list=[], already_printed=True,
    )

    assert result == []
    assert "Individual SARIMA skipped" in capsys.readouterr().out


def test_predict_cumulative_returns_empty_for_all_zero_history(capsys):
    """Cumulatieve variant moet dezelfde guard hebben — geen silent 0-voorspellingen."""
    programme, herkomst, examentype = "B Sociologie", "Niet-EER", "Bachelor"
    weeks = [int(w) for w in get_all_weeks_ordered()]
    long_frame = pd.DataFrame({
        "Collegejaar": [2024] * len(weeks),
        "Faculteit": ["FSW"] * len(weeks),
        "Herkomst": [herkomst] * len(weeks),
        "Examentype": [examentype] * len(weeks),
        "Croho groepeernaam": [programme] * len(weeks),
        "Weeknummer": weeks,
        "ts": [0.0] * len(weeks),
    })
    row = {"Croho groepeernaam": programme, "Herkomst": herkomst, "Examentype": examentype}

    result = predict_with_sarima_cumulative(
        long_frame, row,
        predict_year=2024, predict_week=12, pred_len=26,
        already_printed=True, min_training_year=2016,
    )

    assert result == []
    assert "Cumulative SARIMA skipped" in capsys.readouterr().out
