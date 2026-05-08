from abc import ABC, abstractmethod

import numpy as np


class BaseClassifier(ABC):
    """Abstracte basisklasse voor classificatiemodellen (individueel spoor)."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseClassifier":
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return None


class XGBoostClassifier(BaseClassifier):
    """XGBoost binaire classifier (huidige default)."""

    def __init__(self):
        from xgboost import XGBClassifier
        self._model = XGBClassifier(objective="binary:logistic", eval_metric="auc")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifier":
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return self._model.feature_importances_


class RandomForestClassifier(BaseClassifier):
    """Random Forest classifier — robuust bij kleine datasets."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        from sklearn.ensemble import RandomForestClassifier as RFC
        self._model = RFC(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return self._model.feature_importances_


class LogisticRegressionClassifier(BaseClassifier):
    """Logistic Regression — lineaire baseline voor vergelijking."""

    def __init__(self, max_iter: int = 1000):
        from sklearn.linear_model import LogisticRegression
        self._model = LogisticRegression(max_iter=max_iter)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionClassifier":
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)


class GradientBoostingClassifier(BaseClassifier):
    """Sklearn Gradient Boosting — trager dan XGBoost, maar geen extra dependency."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        from sklearn.ensemble import GradientBoostingClassifier as GBC
        self._model = GBC(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifier":
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return self._model.feature_importances_


class ExtraTreesClassifier(BaseClassifier):
    """Extra Trees — variant op Random Forest met random splits, sneller."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        from sklearn.ensemble import ExtraTreesClassifier as ETC
        self._model = ETC(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtraTreesClassifier":
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return self._model.feature_importances_
