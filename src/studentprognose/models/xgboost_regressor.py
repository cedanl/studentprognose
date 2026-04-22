import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from studentprognose.utils.weeks import get_weeks_list
from studentprognose.models.importance import extract_grouped_importance
from studentprognose.tuning.cache import load_cached_params

DEFAULT_REGRESSOR_HP = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 50,
}


def _resolve_hyperparameters(configuration):
    cached = load_cached_params("xgboost_regressor")
    if cached is not None:
        return cached["params"]
    model_config = configuration.get("model_config", {}) if configuration else {}
    return model_config.get("hyperparameters", {}).get(
        "xgboost_regressor", DEFAULT_REGRESSOR_HP
    )


def predict_with_xgboost(train, test, data_studentcount, configuration=None) -> tuple:
    """Train an XGBoost regressor to predict student counts from cumulative pre-application data.

    Returns:
        tuple: (rounded integer predictions or np.nan, feature importance dict, training log dict).
    """
    if data_studentcount is not None:
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
    else:
        return np.nan, None, {}

    train.drop_duplicates(inplace=True, ignore_index=True)

    if train.empty:
        return np.full(len(test), np.nan), None, {}

    hp = _resolve_hyperparameters(configuration)

    # Temporal validation split: hold out most recent training year
    val_df = None
    train_years = sorted(train["Collegejaar"].unique())
    if len(train_years) >= 3:
        val_year = train_years[-1]
        val_df = train[train["Collegejaar"] == val_year].copy()
        train = train[train["Collegejaar"] < val_year]

    X_train = train.drop(["Aantal_studenten"], axis=1)
    y_train = train.pop("Aantal_studenten")

    numeric_cols = ["Collegejaar"] + [str(x) for x in get_weeks_list(38)]
    categorical_cols = ["Examentype", "Faculteit", "Croho groepeernaam", "Herkomst"]

    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_cols),
            ("categorical", categorical_transformer, categorical_cols),
        ]
    )

    X_train = preprocessor.fit_transform(X_train)
    test_transformed = preprocessor.transform(test)

    X_val, y_val = None, None
    if val_df is not None and not val_df.empty:
        X_val = preprocessor.transform(val_df.drop(["Aantal_studenten"], axis=1))
        y_val = val_df["Aantal_studenten"]

    model = XGBRegressor(
        n_estimators=hp.get("n_estimators", 500),
        learning_rate=hp.get("learning_rate", 0.05),
        max_depth=hp.get("max_depth", 6),
        subsample=hp.get("subsample", 0.8),
        colsample_bytree=hp.get("colsample_bytree", 0.8),
        min_child_weight=hp.get("min_child_weight", 5),
        reg_alpha=hp.get("reg_alpha", 0.1),
        reg_lambda=hp.get("reg_lambda", 1.0),
        early_stopping_rounds=hp.get("early_stopping_rounds", 50),
        random_state=42,
        verbosity=0,
    )

    if X_val is not None and X_val.shape[0] > 0:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)

    importance = extract_grouped_importance(
        model, preprocessor, numeric_cols, categorical_cols
    )

    training_log = {
        "best_iteration": getattr(model, "best_iteration", None),
        "eval_results": model.evals_result() if model.evals_result() else None,
        "hyperparameters": hp,
    }

    predictions = model.predict(test_transformed)

    for i in range(len(predictions)):
        predictions[i] = int(round(predictions[i], 0))

    return predictions, importance, training_log
