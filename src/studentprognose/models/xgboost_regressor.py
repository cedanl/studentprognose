import numpy as np

from studentprognose.models.importance import extract_grouped_importance
from studentprognose.models.regressors import (
    BaseRegressor,
    XGBoostRegressor,
    build_preprocessor,
)


def predict_with_xgboost(
    train,
    test,
    data_studentcount,
    extra_numeric_cols: list[str] | None = None,
    regressor: BaseRegressor | None = None,
) -> tuple[np.ndarray, dict[str, float] | None]:
    """Train een regressor om studentaantallen te voorspellen uit cumulatieve vooraanmelddata.

    Returns:
        (predictions, importance_dict) — predictions afgerond op integers,
        importance_dict is None als het model geen feature importances ondersteunt.
    """
    if data_studentcount is None:
        return np.nan, None

    train = train.merge(
        data_studentcount[
            [
                "Croho groepeernaam",
                "Collegejaar",
                "Herkomst",
                "Examentype",
                "Aantal_studenten",
            ]
        ],
        on=["Croho groepeernaam", "Collegejaar", "Herkomst", "Examentype"],
    )

    train = train.drop_duplicates(ignore_index=True)

    if train.empty:
        return np.full(len(test), np.nan), None

    X_train = train.drop(columns=["Aantal_studenten"])
    y_train = train["Aantal_studenten"]

    if extra_numeric_cols:
        missing = [c for c in extra_numeric_cols if c not in X_train.columns]
        if missing:
            raise ValueError(
                f"predict_with_xgboost: kolommen in extra_numeric_cols ontbreken in de "
                f"trainingsdata: {missing}. Controleer of _add_engineered_features is "
                f"aangeroepen vóór predict_with_xgboost."
            )

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(extra_numeric_cols)

    X_train_transformed = preprocessor.fit_transform(X_train)
    test_transformed = preprocessor.transform(test)

    if regressor is None:
        regressor = XGBoostRegressor()

    regressor.fit(X_train_transformed, y_train)

    importance = None
    if regressor.feature_importances_ is not None:
        importance = extract_grouped_importance(
            regressor, preprocessor, numeric_cols, categorical_cols
        )

    predictions = regressor.predict(test_transformed)
    predictions = np.rint(predictions).astype(int)

    return predictions, importance
