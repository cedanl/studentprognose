import numpy as np
from statsforecast.models import AutoETS, AutoTheta, AutoARIMA

from studentprognose.models.base import BaseForecaster
from studentprognose.utils.constants import WEEKS_PER_YEAR


class ETSForecaster(BaseForecaster):
    """Exponential smoothing via statsforecast AutoETS.

    Selecteert automatisch error-, trend- en seizoenscomponenten.
    Stabieler dan SARIMA bij korte reeksen (<10 jaar) doordat er geen
    vaste ARIMA-ordes gekozen hoeven te worden.
    """

    def __init__(self, season_length: int = WEEKS_PER_YEAR, model: str = "ZZZ"):
        self.season_length = season_length
        self.model_spec = model
        self._model: AutoETS | None = None

    def fit(self, ts_data: np.ndarray, exog: np.ndarray | None = None) -> "ETSForecaster":
        self._model = AutoETS(season_length=self.season_length, model=self.model_spec)
        self._model.fit(y=ts_data.astype(np.float64))
        return self

    def forecast(self, steps: int, exog: np.ndarray | None = None) -> np.ndarray:
        result = self._model.predict(h=steps)
        return result["mean"]


class ThetaForecaster(BaseForecaster):
    """Theta-methode via statsforecast AutoTheta.

    Zeer simpel model dat competitief is bij korte reeksen (M3-competitie).
    Goede robuuste baseline met weinig tuning.
    """

    def __init__(self, season_length: int = WEEKS_PER_YEAR):
        self.season_length = season_length
        self._model: AutoTheta | None = None

    def fit(self, ts_data: np.ndarray, exog: np.ndarray | None = None) -> "ThetaForecaster":
        self._model = AutoTheta(season_length=self.season_length)
        self._model.fit(y=ts_data.astype(np.float64))
        return self

    def forecast(self, steps: int, exog: np.ndarray | None = None) -> np.ndarray:
        result = self._model.predict(h=steps)
        return result["mean"]


class AutoARIMAForecaster(BaseForecaster):
    """Automatische ARIMA orde-selectie via statsforecast AutoARIMA.

    In tegenstelling tot de vaste (1,0,1)×(1,1,1,52) ordes van SARIMAForecaster
    selecteert dit model automatisch de optimale p,d,q en P,D,Q via AICc-minimalisatie.
    Ondersteunt exogene variabelen.
    """

    def __init__(self, season_length: int = WEEKS_PER_YEAR):
        self.season_length = season_length
        self._model: AutoARIMA | None = None

    def fit(self, ts_data: np.ndarray, exog: np.ndarray | None = None) -> "AutoARIMAForecaster":
        self._model = AutoARIMA(season_length=self.season_length)
        X = exog.reshape(-1, 1) if exog is not None else None
        self._model.fit(y=ts_data.astype(np.float64), X=X)
        return self

    def forecast(self, steps: int, exog: np.ndarray | None = None) -> np.ndarray:
        X = exog.reshape(-1, 1) if exog is not None else None
        result = self._model.predict(h=steps, X=X)
        return result["mean"]
