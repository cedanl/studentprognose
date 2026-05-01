from studentprognose.models.base import BaseForecaster
from studentprognose.models.regressors import BaseRegressor


def create_forecaster(config: dict) -> BaseForecaster:
    """Maak een tijdreeks-forecaster op basis van model_config."""
    from studentprognose.models.sarima import SARIMAForecaster
    from studentprognose.models.forecasters import (
        ETSForecaster,
        ThetaForecaster,
        AutoARIMAForecaster,
    )

    registry: dict[str, type[BaseForecaster]] = {
        "sarima": SARIMAForecaster,
        "ets": ETSForecaster,
        "theta": ThetaForecaster,
        "auto_arima": AutoARIMAForecaster,
    }

    name = config.get("model_config", {}).get("cumulative_timeseries", "sarima")
    if name not in registry:
        raise ValueError(
            f"Onbekend tijdreeksmodel: '{name}'. "
            f"Geldige opties: {', '.join(registry)}"
        )
    return registry[name]()


def create_regressor(config: dict) -> BaseRegressor:
    """Maak een regressor op basis van model_config."""
    from studentprognose.models.regressors import (
        XGBoostRegressor,
        RidgeRegressor,
        RandomForestRegressor,
    )

    registry: dict[str, type[BaseRegressor]] = {
        "xgboost": XGBoostRegressor,
        "ridge": RidgeRegressor,
        "random_forest": RandomForestRegressor,
    }

    name = config.get("model_config", {}).get("cumulative_regressor", "xgboost")
    if name not in registry:
        raise ValueError(
            f"Onbekend regressiemodel: '{name}'. "
            f"Geldige opties: {', '.join(registry)}"
        )
    return registry[name]()
