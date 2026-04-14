import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.utils.weeks import get_weeks_list

DEFAULT_STATUS_MAP = {
    "Ingeschreven": 1,
    "Geannuleerd": 0,
    "Uitgeschreven": 1,
    "Verzoek tot inschrijving": 0,
    "Studie gestaakt": 0,
    "Aanmelding vervolgen": 0,
}

MIN_TRAINING_YEAR = 2016


def predict_applicant(data, predict_year, predict_week, max_year, data_cumulative=None, configuration=None):
    """
    Train an XGBoost classifier to predict individual applicant enrollment probability.

    Returns:
        np.ndarray: predicted enrollment probabilities, or np.nan if no test data.
    """
    data_train = data[data["Weeknummer"].isin(get_weeks_list(predict_week))]

    if data_cumulative is not None:
        data_train = _create_ratio(data_train)
    data_train = data_train.replace([np.inf, -np.inf], np.nan)

    model = XGBClassifier(objective="binary:logistic", eval_metric="auc")

    model_config = configuration.get("model_config", {}) if configuration else {}
    status_map = model_config.get("status_mapping", DEFAULT_STATUS_MAP)
    min_year = model_config.get("min_training_year", MIN_TRAINING_YEAR)

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

    X_train = train.drop(["Inschrijfstatus"], axis=1)
    y_train = train.pop("Inschrijfstatus")

    X_test = test.drop(["Inschrijfstatus"], axis=1)
    y_test = test.pop("Inschrijfstatus")

    if len(X_test) == 0:
        return np.nan

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

    model.fit(X_train, y_train)

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

    return predicties


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
