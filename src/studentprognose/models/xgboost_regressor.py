import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from studentprognose.utils.weeks import get_weeks_list


def predict_with_xgboost(train, test, data_studentcount) -> np.ndarray | float:
    """
    Train an XGBoost regressor to predict student counts from cumulative pre-application data.

    Returns:
        np.ndarray: rounded integer predictions, or np.nan if no student count data.
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
        return np.nan

    train.drop_duplicates(inplace=True, ignore_index=True)

    if train.empty:
        return np.full(len(test), np.nan)

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
    test = preprocessor.transform(test)

    model = XGBRegressor(learning_rate=0.25)

    model.fit(X_train, y_train)

    predictions = model.predict(test)

    for i in range(len(predictions)):
        predictions[i] = int(round(predictions[i], 0))

    return predictions
