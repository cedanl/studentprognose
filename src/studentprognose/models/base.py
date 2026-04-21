from abc import ABC, abstractmethod
import numpy as np


class BaseForecaster(ABC):
    """Abstracte basisklasse voor tijdreeksmodellen.

    Momenteel is `SARIMAForecaster` de enige implementatie. De abstractie is
    intentioneel: toekomstige modellen (bijv. Prophet, ETS) kunnen dezelfde
    interface implementeren zodat ze uitwisselbaar zijn in de ensemble-pipeline.
    """

    @abstractmethod
    def fit(self, ts_data: np.ndarray, exog: np.ndarray | None = None) -> "BaseForecaster":
        ...

    @abstractmethod
    def forecast(self, steps: int, exog: np.ndarray | None = None) -> np.ndarray:
        ...
