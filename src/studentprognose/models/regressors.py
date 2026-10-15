from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
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
    """XGBoost gradient boosting regressor (huidige default).

    Accepteert willekeurige XGBoost-hyperparameters als keyword-argumenten
    (bijv. ``n_estimators``, ``max_depth``, ``subsample``). Niet-opgegeven
    parameters vallen terug op de XGBoost-defaults; ``learning_rate`` heeft
    een eigen default van ``0.25``. Zo kunnen getunede parameters (zie
    ``models/tuning.py`` en ``model_config.regressor_params``) ongewijzigd
    worden doorgegeven.
    """

    def __init__(self, **params):
        from xgboost import XGBRegressor
        self._model = XGBRegressor(**{"learning_rate": 0.25, **params})

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

    def __init__(self, **params):
        from sklearn.linear_model import Ridge
        self._model = Ridge(**{"alpha": 1.0, **params})

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class RandomForestRegressor(BaseRegressor):
    """Random Forest — robuust bij kleine datasets, ingebouwde feature importance."""

    def __init__(self, **params):
        from sklearn.ensemble import RandomForestRegressor as RF
        self._model = RF(**{"n_estimators": 100, "random_state": 42, **params})

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

    def __init__(self, **params):
        from sklearn.ensemble import GradientBoostingRegressor as GBR
        self._model = GBR(**{"n_estimators": 100, "random_state": 42, **params})

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

    def __init__(self, **params):
        from sklearn.ensemble import ExtraTreesRegressor as ETR
        self._model = ETR(**{"n_estimators": 100, "random_state": 42, **params})

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtraTreesRegressor":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return self._model.feature_importances_


def covered_feature_columns(
    frames: list[pd.DataFrame],
    final_week: int = FINAL_ACADEMIC_WEEK,
    year_col: str = "Collegejaar",
) -> list[str]:
    """Kolommen voor de regressor, zonder vroege weken die niet in elk jaar data hebben.

    Een weekkolom aan het begin van het academisch jaar telt alleen mee als
    feature als **elk** collegejaar in ``frames`` (training én voorspeljaar)
    vanaf die week data heeft. Mist één jaar zijn eerste weken — een gemist
    telbestand, of een aanmeldcyclus die later opende — dan staan daar nullen
    die in de andere jaren nooit voorkomen. De regressor ziet het jaar dan als
    iets buiten zijn trainingsdata en voorspelt onbetrouwbaar (#303). Die vroege
    weken bevatten nauwelijks informatie over de instroom, dus ze weglaten kost
    weinig.

    Args:
        frames: Train- en testdata (wide format, één weekkolom per week).
        final_week: Laatste week van het academisch jaar.
        year_col: Naam van de jaar-kolom.

    Returns:
        De kolommen van ``frames[0]`` minus de weggelaten vroege weken.
    """
    columns = list(frames[0].columns)
    week_cols = [str(w) for w in get_weeks_list(final_week, final_week) if str(w) in columns]
    if not week_cols:
        return columns

    data = pd.concat([f[[year_col] + week_cols] for f in frames], ignore_index=True)
    filled = data.groupby(year_col)[week_cols].sum().gt(0)
    filled = filled[filled.any(axis=1)]
    if filled.empty:
        return columns

    # Eerste week met data per jaar (week_cols staat in academische volgorde);
    # de laatste daarvan is de eerste week die in élk jaar gevuld is.
    first_filled = filled.to_numpy().argmax(axis=1).max()
    dropped = set(week_cols[:first_filled])
    return [c for c in columns if c not in dropped]


def build_preprocessor(
    extra_numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    year_col: str = "Collegejaar",
    final_week: int = FINAL_ACADEMIC_WEEK,
    available_cols: list[str] | None = None,
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

    week_cols = [str(x) for x in get_weeks_list(final_week, final_week)]
    if available_cols is not None:
        avail = set(available_cols)
        week_cols = [w for w in week_cols if w in avail]

    numeric_cols = (
        [year_col]
        + week_cols
        + (extra_numeric_cols or [])
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols
