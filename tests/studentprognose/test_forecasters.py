import numpy as np
import pytest

from studentprognose.models.base import BaseForecaster
from studentprognose.models.sarima import SARIMAForecaster
from studentprognose.models.forecasters import (
    ETSForecaster,
    ThetaForecaster,
    AutoARIMAForecaster,
)
from studentprognose.utils.constants import SARIMA_ORDER, SARIMA_SEASONAL_ORDER


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


def test_sarima_short_series_falls_back_to_nonseasonal():
    """Regressie (#231): de native statsforecast-backend crasht het PROCES
    (SIGSEGV/SIGABRT door heap-corruptie) bij ARIMA(1,1,1)x52 op een reeks korter
    dan twee volledige seizoenen. SARIMAForecaster valt dan terug op een niet-
    seizoensmodel. Zonder die guard haalt deze test het pytest-proces onderuit
    (exitcode 134)."""
    rng = np.random.default_rng(0)
    n = 93  # < 2 * 52: precies in het empirische crash-bereik
    ts = (np.cumsum(np.abs(rng.standard_normal(n))) + np.arange(n)).astype(np.float64)

    model = SARIMAForecaster(order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER)
    model.fit(ts)

    assert model._model.season_length == 1  # seizoenscomponent uitgeschakeld
    pred = model.forecast(steps=10)
    assert len(pred) == 10
    assert np.all(np.isfinite(pred))


def test_sarima_long_series_keeps_seasonal_model():
    """Lange reeksen (>= 2 seizoenen) houden het volledige seizoensmodel intact."""
    ts = _seasonal_series(n_years=5)  # 260 >= 2 * 52

    model = SARIMAForecaster(order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER)
    model.fit(ts)

    assert model._model.season_length == SARIMA_SEASONAL_ORDER[3]  # 52 behouden
