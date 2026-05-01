from abc import ABC, abstractmethod

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from studentprognose.utils.weeks import get_weeks_list


CATEGORICAL_COLS = ["Examentype", "Faculteit", "Croho groepeernaam", "Herkomst"]


class BaseRegressor(ABC):
    """Abstracte basisklasse voor regressiemodellen (cumulatief spoor, stap 2)."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseRegressor":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @property
    def feature_importances_(self) -> np.ndarray | None:
        """Feature importances als het model die ondersteunt, anders None."""
        return None


class XGBoostRegressor(BaseRegressor):
    """XGBoost gradient boosting regressor (huidige default)."""

    def __init__(self, learning_rate: float = 0.25):
        from xgboost import XGBRegressor
        self._model = XGBRegressor(learning_rate=learning_rate)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostRegressor":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return self._model.feature_importances_


class RidgeRegressor(BaseRegressor):
    """Ridge (L2) regressie — stabiel bij weinig data en multicollineariteit."""

    def __init__(self, alpha: float = 1.0):
        from sklearn.linear_model import Ridge
        self._model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class RandomForestRegressor(BaseRegressor):
    """Random Forest — robuust bij kleine datasets, ingebouwde feature importance."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        from sklearn.ensemble import RandomForestRegressor as RF
        self._model = RF(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return self._model.feature_importances_


def build_preprocessor(
    extra_numeric_cols: list[str] | None = None,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Bouw de gedeelde preprocessing pipeline voor alle regressors.

    Returns:
        (preprocessor, numeric_cols, categorical_cols)
    """
    numeric_cols = (
        ["Collegejaar"]
        + [str(x) for x in get_weeks_list(38)]
        + (extra_numeric_cols or [])
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )

    return preprocessor, numeric_cols, CATEGORICAL_COLS
