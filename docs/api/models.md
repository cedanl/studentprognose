# Modellen

## Base

::: studentprognose.models.base

## SARIMA

::: studentprognose.models.sarima
    options:
      members:
        - SARIMAForecaster
        - predict_with_sarima_cumulative
        - predict_with_sarima_individual

## XGBoost Classifier

::: studentprognose.models.xgboost_classifier
    options:
      members:
        - predict_applicant

## XGBoost Regressor

::: studentprognose.models.xgboost_regressor
    options:
      members:
        - predict_with_xgboost

## Ratio-model

::: studentprognose.models.ratio
    options:
      members:
        - predict_with_ratio
