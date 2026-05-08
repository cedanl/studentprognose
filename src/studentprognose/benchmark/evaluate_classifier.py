import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_curve as sklearn_roc_curve
from sklearn.preprocessing import OneHotEncoder

from studentprognose.benchmark.metrics import accuracy, auc_roc, f1, mae
from studentprognose.benchmark.splitter import time_series_split
from studentprognose.config import get_columns, get_model_features
from studentprognose.models.xgboost_classifier import DEFAULT_STATUS_MAP
from studentprognose.utils.weeks import get_weeks_list


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
    c = get_columns(config)
    mf = get_model_features(config)
    clf_features = mf.get("classifier", {})
    configured_numeric = clf_features.get("numeric", [])
    configured_categorical = clf_features.get("categorical", [])

    model_config = config.get("model_config", {})
    status_map = model_config.get("status_mapping", DEFAULT_STATUS_MAP)

    week_filter = get_weeks_list(predict_week)
    data_filtered = data_individual[data_individual[c.week].isin(week_filter)].copy()

    splits = time_series_split(data_filtered, min_training_year, min_train_years, year_col=c.academic_year)
    results = []
    roc_curves = []

    for train_df, test_df in splits:
        test_year = int(test_df[c.academic_year].iloc[0])

        train_df = _apply_cancellation_filter(train_df, predict_week, c.cancellation_date)

        train_df = train_df.copy()
        train_df[c.enrollment_status] = train_df[c.enrollment_status].map(status_map)
        train_df = train_df[train_df[c.enrollment_status].notna()]

        test_df = test_df.copy()
        test_df[c.enrollment_status] = test_df[c.enrollment_status].map(status_map)
        test_df = test_df[test_df[c.enrollment_status].notna()]

        if train_df.empty or test_df.empty:
            continue

        y_train = train_df[c.enrollment_status].values.astype(int)
        y_test = test_df[c.enrollment_status].values.astype(int)

        X_train = train_df.drop(columns=[c.enrollment_status])
        X_test = test_df.drop(columns=[c.enrollment_status])

        numeric_cols = [col for col in configured_numeric if col in X_train.columns]
        categorical_cols = [col for col in configured_categorical if col in X_train.columns]

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

        agg_mae_val = _compute_aggregate_mae(
            test_df, proba_positive, y_test,
            group_cols=[c.programme, c.origin, c.exam_type],
        )

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


def _apply_cancellation_filter(
    train_df: pd.DataFrame,
    predict_week: int,
    cancellation_col: str,
) -> pd.DataFrame:
    """Filter geannuleerde aanmeldingen op basis van intrekkingsdatum."""
    if int(predict_week) <= 38:
        return train_df[
            (train_df[cancellation_col].isna())
            | (
                (train_df[cancellation_col] >= int(predict_week))
                & (train_df[cancellation_col] < 39)
            )
        ]
    else:
        return train_df[
            (train_df[cancellation_col].isna())
            | (
                (train_df[cancellation_col] > int(predict_week))
                | (train_df[cancellation_col] < 39)
            )
        ]


def _compute_aggregate_mae(
    test_df: pd.DataFrame,
    proba_positive: np.ndarray,
    y_test: np.ndarray,
    group_cols: list[str],
) -> float:
    """Bereken MAE op geaggregeerd niveau (programme x herkomst x examentype)."""
    available_cols = [col for col in group_cols if col in test_df.columns]
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
