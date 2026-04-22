import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from studentprognose.utils.weeks import get_weeks_list
from studentprognose.models.importance import extract_grouped_importance
from studentprognose.tuning.cache import load_cached_params

DEFAULT_STATUS_MAP = {
    "Ingeschreven": 1,
    "Geannuleerd": 0,
    "Uitgeschreven": 1,
    "Verzoek tot inschrijving": 0,
    "Studie gestaakt": 0,
    "Aanmelding vervolgen": 0,
}

MIN_TRAINING_YEAR = 2016

DEFAULT_CLASSIFIER_HP = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": None,
    "early_stopping_rounds": 50,
}


def _resolve_hyperparameters(configuration):
    cached = load_cached_params("xgboost_classifier")
    if cached is not None:
        return cached["params"]
    model_config = configuration.get("model_config", {}) if configuration else {}
    return model_config.get("hyperparameters", {}).get(
        "xgboost_classifier", DEFAULT_CLASSIFIER_HP
    )


def predict_applicant(
    data, predict_year, predict_week, max_year, data_cumulative=None, configuration=None
) -> np.ndarray | float:
    """Train an XGBoost classifier to predict individual applicant enrollment probability.

    Returns:
        tuple: (predicted enrollment probabilities or np.nan, feature importance dict, training log dict).
    """
    data_train = data[data["Weeknummer"].isin(get_weeks_list(predict_week))]

    if data_cumulative is not None:
        data_train = _create_ratio(data_train)
    data_train = data_train.replace([np.inf, -np.inf], np.nan)

    model_config = configuration.get("model_config", {}) if configuration else {}
    status_map = model_config.get("status_mapping", DEFAULT_STATUS_MAP)
    min_year = model_config.get("min_training_year", MIN_TRAINING_YEAR)
    hp = _resolve_hyperparameters(configuration)

    if predict_year == max_year:
        train = data_train[
            (data_train["Collegejaar"] < predict_year)
            & (data_train["Collegejaar"] >= min_year)
        ]
        test = data_train[
            (data_train["Collegejaar"] == predict_year)
            & (data_train["Weeknummer"].isin(get_weeks_list(predict_week)))
        ]
    else:
        train = data_train[
            (data_train["Collegejaar"] != predict_year)
            & (data_train["Collegejaar"] >= min_year)
            & (data_train["Collegejaar"] != max_year)
        ]
        test = data_train[
            (data_train["Collegejaar"] == predict_year)
            & (data_train["Weeknummer"].isin(get_weeks_list(predict_week)))
        ]

    if int(predict_week) <= 38:
        train = train[
            (train["Datum intrekking vooraanmelding"].isna())
            | (
                (train["Datum intrekking vooraanmelding"] >= int(predict_week))
                & (train["Datum intrekking vooraanmelding"] < 39)
            )
        ]
    elif int(predict_week) > 38:
        train = train[
            (train["Datum intrekking vooraanmelding"].isna())
            | (
                (train["Datum intrekking vooraanmelding"] > int(predict_week))
                | (train["Datum intrekking vooraanmelding"] < 39)
            )
        ]

    train["Inschrijfstatus"] = train["Inschrijfstatus"].map(status_map)

    # Temporal validation split: hold out most recent training year
    val = None
    train_years = sorted(train["Collegejaar"].unique())
    if len(train_years) >= 3:
        val_year = train_years[-1]
        val = train[train["Collegejaar"] == val_year]
        train = train[train["Collegejaar"] < val_year]

    X_train = train.drop(["Inschrijfstatus"], axis=1)
    y_train = train.pop("Inschrijfstatus")

    X_test = test.drop(["Inschrijfstatus"], axis=1)
    y_test = test.pop("Inschrijfstatus")

    X_val, y_val = None, None
    if val is not None:
        X_val = val.drop(["Inschrijfstatus"], axis=1)
        y_val = val.pop("Inschrijfstatus")

    if len(X_test) == 0:
        return np.nan, None, {}

    numeric_cols = ["Collegejaar", "Sleutel_count", "is_numerus_fixus"]

    categorical_cols = [
        "Examentype",
        "Faculteit",
        "Croho groepeernaam",
        "Deadlineweek",
        "Herkomst",
        "Weeknummer",
        "Opleiding",
        "Type vooropleiding",
        "Nationaliteit",
        "EER",
        "Geslacht",
        "Plaats code eerste vooropleiding",
        "Studieadres postcode",
        "Studieadres land",
        "Geverifieerd adres plaats",
        "Geverifieerd adres land",
        "Geverifieerd adres postcode",
        "School code eerste vooropleiding",
        "School eerste vooropleiding",
        "Land code eerste vooropleiding",
    ]

    if data_cumulative is not None:
        numeric_cols = numeric_cols + [
            "Gewogen vooraanmelders",
            "Ongewogen vooraanmelders",
            "Aantal aanmelders met 1 aanmelding",
            "Inschrijvingen",
        ]

    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_cols),
            ("categorical", categorical_transformer, categorical_cols),
        ]
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    if X_val is not None:
        X_val = preprocessor.transform(X_val)

    # Auto-calculate scale_pos_weight from class distribution
    spw = hp.get("scale_pos_weight")
    if spw is None:
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        spw = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric=hp.get("eval_metric", "auc"),
        n_estimators=hp.get("n_estimators", 500),
        learning_rate=hp.get("learning_rate", 0.05),
        max_depth=hp.get("max_depth", 6),
        subsample=hp.get("subsample", 0.8),
        colsample_bytree=hp.get("colsample_bytree", 0.8),
        min_child_weight=hp.get("min_child_weight", 5),
        reg_alpha=hp.get("reg_alpha", 0.1),
        reg_lambda=hp.get("reg_lambda", 1.0),
        scale_pos_weight=spw,
        early_stopping_rounds=hp.get("early_stopping_rounds", 50),
        random_state=42,
        verbosity=0,
    )

    if X_val is not None and X_val.shape[0] > 0:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print(
            f"  XGBoost classifier: stopped at {model.best_iteration}/{hp.get('n_estimators', 500)} trees"
        )
    else:
        model.fit(X_train, y_train)

    importance = extract_grouped_importance(
        model, preprocessor, numeric_cols, categorical_cols
    )

    training_log = {
        "best_iteration": getattr(model, "best_iteration", None),
        "eval_results": model.evals_result() if model.evals_result() else None,
        "hyperparameters": hp,
        "scale_pos_weight": spw,
    }

    voorspellingen = model.predict_proba(X_test)[:, 1]

    predicties = np.zeros(len(voorspellingen))

    for i, (voorspelling, real) in enumerate(zip(voorspellingen, y_test)):
        if real == "Geannuleerd" and test["Datum intrekking vooraanmelding"].iloc[
            i
        ] in get_weeks_list(predict_week):
            pred = 0
        else:
            pred = voorspelling

        predicties[i] = pred

    return predicties, importance, training_log


def _create_ratio(data):
    data = data.copy()

    def fix_zero(number):
        if number == 0:
            return np.nan
        else:
            return number

    data["Ongewogen vooraanmelders"] = data["Ongewogen vooraanmelders"].apply(fix_zero)

    data["Gewogen vooraanmelders"] = (
        data["Gewogen vooraanmelders"] / data["Ongewogen vooraanmelders"]
    )
    data["Gewogen vooraanmelders"] = data["Gewogen vooraanmelders"].fillna(1)

    data["Aantal aanmelders met 1 aanmelding"] = (
        data["Aantal aanmelders met 1 aanmelding"] / data["Ongewogen vooraanmelders"]
    )
    data["Aantal aanmelders met 1 aanmelding"] = data[
        "Aantal aanmelders met 1 aanmelding"
    ].fillna(1)

    data["Inschrijvingen"] = data["Inschrijvingen"] / data["Ongewogen vooraanmelders"]
    data["Inschrijvingen"] = data["Inschrijvingen"].fillna(1)

    return data
