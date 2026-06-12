from abc import ABC, abstractmethod

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from studentprognose.utils.weeks import get_weeks_list
from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK


_DEFAULT_CATEGORICAL_COLS = ["Examentype", "Faculteit", "Croho groepeernaam", "Herkomst"]


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


class GradientBoostingRegressor(BaseRegressor):
    """Sklearn Gradient Boosting — geen extra dependency nodig."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        from sklearn.ensemble import GradientBoostingRegressor as GBR
        self._model = GBR(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return self._model.feature_importances_


class ExtraTreesRegressor(BaseRegressor):
    """Extra Trees — variant op Random Forest met random splits, sneller."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        from sklearn.ensemble import ExtraTreesRegressor as ETR
        self._model = ETR(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtraTreesRegressor":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return self._model.feature_importances_


def build_preprocessor(
    extra_numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    year_col: str = "Collegejaar",
    final_week: int = FINAL_ACADEMIC_WEEK,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Bouw de gedeelde preprocessing pipeline voor alle regressors.

    Args:
        extra_numeric_cols: Extra numerieke kolommen (bijv. engineered features).
        categorical_cols: Categorische kolommen voor one-hot encoding.
            Standaard uit ``_DEFAULT_CATEGORICAL_COLS``; configureerbaar via
            ``config["model_features"]["regressor"]["categorical"]``.
        year_col: Naam van de jaar-kolom (standaard ``"Collegejaar"``).
        final_week: Laatste week van het academisch jaar (default 38; UvA 36).
            Bepaalt welke weekkolommen als numerieke features worden gebruikt,
            zodat ze aansluiten op de wide-format van ``transform_data``.

    Returns:
        (preprocessor, numeric_cols, categorical_cols)
    """
    if categorical_cols is None:
        categorical_cols = list(_DEFAULT_CATEGORICAL_COLS)

    numeric_cols = (
        [year_col]
        + [str(x) for x in get_weeks_list(final_week, final_week)]
        + (extra_numeric_cols or [])
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols
