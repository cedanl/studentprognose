import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_curve as sklearn_roc_curve
from sklearn.preprocessing import OneHotEncoder

from studentprognose.benchmark.metrics import accuracy, auc_roc, f1, mae
from studentprognose.benchmark.splitter import time_series_split
from studentprognose.models.xgboost_classifier import DEFAULT_STATUS_MAP
from studentprognose.utils.weeks import get_weeks_list

NUMERIC_COLS = ["Collegejaar", "Sleutel_count", "is_numerus_fixus"]

CATEGORICAL_COLS = [
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


def evaluate_classifier_model(
    data_individual: pd.DataFrame,
    classifier_factory: callable,
    predict_week: int,
    config: dict,
    min_training_year: int = 2016,
    min_train_years: int = 3,
) -> tuple[pd.DataFrame, list[dict]]:
    """Evalueer een classificatiemodel op individuele data via cross-validatie.

    Returns:
        Tuple van (metrics_df, roc_curves) waar roc_curves een lijst is van
        dicts met test_year, fpr en tpr arrays.
    """
    model_config = config.get("model_config", {})
    status_map = model_config.get("status_mapping", DEFAULT_STATUS_MAP)

    week_filter = get_weeks_list(predict_week)
    data_filtered = data_individual[data_individual["Weeknummer"].isin(week_filter)].copy()

    splits = time_series_split(data_filtered, min_training_year, min_train_years)
    results = []
    roc_curves = []

    for train_df, test_df in splits:
        test_year = int(test_df["Collegejaar"].iloc[0])

        train_df = _apply_cancellation_filter(train_df, predict_week)

        train_df = train_df.copy()
        train_df["Inschrijfstatus"] = train_df["Inschrijfstatus"].map(status_map)
        train_df = train_df[train_df["Inschrijfstatus"].notna()]

        test_df = test_df.copy()
        test_df["Inschrijfstatus"] = test_df["Inschrijfstatus"].map(status_map)
        test_df = test_df[test_df["Inschrijfstatus"].notna()]

        if train_df.empty or test_df.empty:
            continue

        y_train = train_df["Inschrijfstatus"].values.astype(int)
        y_test = test_df["Inschrijfstatus"].values.astype(int)

        X_train = train_df.drop(columns=["Inschrijfstatus"])
        X_test = test_df.drop(columns=["Inschrijfstatus"])

        numeric_cols = [c for c in NUMERIC_COLS if c in X_train.columns]
        categorical_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", "passthrough", numeric_cols),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ]
        )

        try:
            X_train_t = preprocessor.fit_transform(X_train)
            X_test_t = preprocessor.transform(X_test)
        except ValueError:
            continue

        t0 = time.perf_counter()
        try:
            clf = classifier_factory()
            clf.fit(X_train_t, y_train)
            proba = clf.predict_proba(X_test_t)
            if proba.ndim == 2:
                proba_positive = proba[:, 1]
            else:
                proba_positive = proba
            elapsed = time.perf_counter() - t0
        except Exception:
            continue

        y_pred = (proba_positive >= 0.5).astype(int)

        agg_mae_val = _compute_aggregate_mae(test_df, proba_positive, y_test)

        results.append({
            "test_year": test_year,
            "accuracy": accuracy(y_test, y_pred),
            "auc_roc": auc_roc(y_test, proba_positive),
            "f1": f1(y_test, y_pred),
            "aggregate_mae": agg_mae_val,
            "train_time_s": elapsed,
            "n_train": len(y_train),
            "n_test": len(y_test),
        })

        if len(np.unique(y_test)) >= 2:
            fpr, tpr, _ = sklearn_roc_curve(y_test, proba_positive)
            roc_curves.append({
                "test_year": test_year,
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
            })

    return pd.DataFrame(results), roc_curves


def _apply_cancellation_filter(train_df: pd.DataFrame, predict_week: int) -> pd.DataFrame:
    """Filter geannuleerde aanmeldingen op basis van intrekkingsdatum."""
    if int(predict_week) <= 38:
        return train_df[
            (train_df["Datum intrekking vooraanmelding"].isna())
            | (
                (train_df["Datum intrekking vooraanmelding"] >= int(predict_week))
                & (train_df["Datum intrekking vooraanmelding"] < 39)
            )
        ]
    else:
        return train_df[
            (train_df["Datum intrekking vooraanmelding"].isna())
            | (
                (train_df["Datum intrekking vooraanmelding"] > int(predict_week))
                | (train_df["Datum intrekking vooraanmelding"] < 39)
            )
        ]


def _compute_aggregate_mae(
    test_df: pd.DataFrame,
    proba_positive: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Bereken MAE op geaggregeerd niveau (programme × herkomst × examentype)."""
    group_cols = ["Croho groepeernaam", "Herkomst", "Examentype"]
    available_cols = [c for c in group_cols if c in test_df.columns]
    if not available_cols:
        return mae(np.array([y_test.sum()]), np.array([proba_positive.sum()]))

    agg = test_df[available_cols].copy()
    agg["predicted"] = proba_positive
    agg["actual"] = y_test

    grouped = agg.groupby(available_cols).agg(
        predicted=("predicted", "sum"),
        actual=("actual", "sum"),
    ).reset_index()

    return mae(grouped["actual"].values, grouped["predicted"].values)
