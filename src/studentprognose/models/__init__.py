from studentprognose.models.base import BaseForecaster
from studentprognose.models.classifiers import (
    BaseClassifier,
    XGBoostClassifier,
    RandomForestClassifier,
    LogisticRegressionClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from studentprognose.models.regressors import (
    BaseRegressor,
    XGBoostRegressor,
    RidgeRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from studentprognose.models.sarima import SARIMAForecaster
from studentprognose.models.forecasters import (
    ETSForecaster,
    ThetaForecaster,
    AutoARIMAForecaster,
)

FORECASTER_REGISTRY: dict[str, type[BaseForecaster]] = {
    "sarima": SARIMAForecaster,
    "ets": ETSForecaster,
    "theta": ThetaForecaster,
    "auto_arima": AutoARIMAForecaster,
}

REGRESSOR_REGISTRY: dict[str, type[BaseRegressor]] = {
    "xgboost": XGBoostRegressor,
    "ridge": RidgeRegressor,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "extra_trees": ExtraTreesRegressor,
}

CLASSIFIER_REGISTRY: dict[str, type[BaseClassifier]] = {
    "xgboost": XGBoostClassifier,
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegressionClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "extra_trees": ExtraTreesClassifier,
}


def create_forecaster(config: dict) -> BaseForecaster:
    """Maak een tijdreeks-forecaster op basis van model_config.

    De gekozen forecaster (``model_config.cumulative_timeseries``) wordt
    geïnstantieerd met de bijbehorende parameters uit
    ``model_config.forecaster_params[<naam>]`` (leeg = modeldefaults). Voor
    SARIMA zijn dat ``order`` en ``seasonal_order``. Dit is het reproduceerbare
    pad: getunede ordes (zie ``models/tuning.py``) worden hier vastgelegd en
    ingelezen.
    """
    model_config = config.get("model_config", {})
    name = model_config.get("cumulative_timeseries", "sarima")
    if name not in FORECASTER_REGISTRY:
        raise ValueError(
            f"Onbekend tijdreeksmodel: '{name}'. "
            f"Geldige opties: {', '.join(FORECASTER_REGISTRY)}"
        )
    params = model_config.get("forecaster_params", {}).get(name, {})
    return FORECASTER_REGISTRY[name](**params)


def create_regressor(config: dict) -> BaseRegressor:
    """Maak een regressor op basis van model_config.

    De gekozen regressor (``model_config.cumulative_regressor``) wordt
    geïnstantieerd met de bijbehorende hyperparameters uit
    ``model_config.regressor_params[<naam>]`` (leeg = modeldefaults). Dit is
    het reproduceerbare pad: getunede parameters (zie ``models/tuning.py``)
    worden hier vastgelegd en ingelezen.
    """
    model_config = config.get("model_config", {})
    name = model_config.get("cumulative_regressor", "xgboost")
    if name not in REGRESSOR_REGISTRY:
        raise ValueError(
            f"Onbekend regressiemodel: '{name}'. "
            f"Geldige opties: {', '.join(REGRESSOR_REGISTRY)}"
        )
    params = model_config.get("regressor_params", {}).get(name, {})
    return REGRESSOR_REGISTRY[name](**params)


def create_classifier(config: dict) -> BaseClassifier:
    """Maak een classifier op basis van model_config."""
    name = config.get("model_config", {}).get("individual_classifier", "xgboost")
    if name not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Onbekend classificatiemodel: '{name}'. "
            f"Geldige opties: {', '.join(CLASSIFIER_REGISTRY)}"
        )
    return CLASSIFIER_REGISTRY[name]()
