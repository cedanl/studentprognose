from abc import ABC, abstractmethod
import numpy as np


class BaseForecaster(ABC):
    """Base class for time-series forecasting models."""

    @abstractmethod
    def fit(self, ts_data: np.ndarray, exog: np.ndarray | None = None) -> "BaseForecaster":
        ...

    @abstractmethod
    def forecast(self, steps: int, exog: np.ndarray | None = None) -> np.ndarray:
        ...
