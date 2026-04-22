from __future__ import annotations

import itertools
import random
from collections.abc import Callable

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

PARAM_GRID = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 5, 10],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [1.0, 5.0, 10.0],
}


def _generate_param_combinations(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _sample_param_combinations(
    grid: dict, n_iter: int | None = None, seed: int = 42
) -> list[dict]:
    all_combos = _generate_param_combinations(grid)
    if n_iter is None or n_iter >= len(all_combos):
        return all_combos
    rng = random.Random(seed)
    return rng.sample(all_combos, n_iter)


def _build_expanding_folds(
    years: np.ndarray, min_train_years: int = 2
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build expanding-window folds from sorted year labels.

    Each fold trains on all years up to val_year-1 and validates on val_year.
    Requires at least min_train_years in the training set.
    """
    unique_years = sorted(set(years))
    folds = []
    for i in range(min_train_years, len(unique_years)):
        val_year = unique_years[i]
        train_mask = np.array([y < val_year for y in years])
        val_mask = np.array([y == val_year for y in years])
        if train_mask.sum() > 0 and val_mask.sum() > 0:
            folds.append((train_mask, val_mask))
    return folds


def tune_xgboost_classifier(
    X: np.ndarray,
    y: np.ndarray,
    years: np.ndarray,
    scale_pos_weight: float = 1.0,
    n_iter: int | None = None,
    on_progress: Callable[[], None] | None = None,
) -> tuple[dict, float, int, int]:
    """Randomized search with expanding-window CV for XGBoost classifier.

    Args:
        X: Feature matrix (already preprocessed).
        y: Binary target.
        years: Collegejaar for each row (for temporal splitting).
        scale_pos_weight: Class weight ratio.
        n_iter: Number of random parameter combinations to try.
            None means exhaustive grid search.
        on_progress: Called after each combination is evaluated.

    Returns:
        (best_params, best_score, n_folds, n_combinations)
    """
    folds = _build_expanding_folds(years)
    if not folds:
        return {}, float("inf"), 0, 0

    combinations = _sample_param_combinations(PARAM_GRID, n_iter)
    best_score = float("inf")
    best_params = {}

    for params in combinations:
        fold_scores = []
        for train_mask, val_mask in folds:
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                n_estimators=500,
                early_stopping_rounds=50,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                verbosity=0,
                **params,
            )
            model.fit(
                X[train_mask],
                y[train_mask],
                eval_set=[(X[val_mask], y[val_mask])],
                verbose=False,
            )
            preds = model.predict_proba(X[val_mask])[:, 1]
            mape = _cohort_mape(y[val_mask], preds)
            fold_scores.append(mape)

        avg_score = np.mean(fold_scores)
        if avg_score < best_score:
            best_score = avg_score
            best_params = params

        if on_progress:
            on_progress()

    return best_params, best_score, len(folds), len(combinations)


def tune_xgboost_regressor(
    X: np.ndarray,
    y: np.ndarray,
    years: np.ndarray,
    n_iter: int | None = None,
    on_progress: Callable[[], None] | None = None,
) -> tuple[dict, float, int, int]:
    """Randomized search with expanding-window CV for XGBoost regressor.

    Args:
        n_iter: Number of random parameter combinations to try.
            None means exhaustive grid search.
        on_progress: Called after each combination is evaluated.

    Returns:
        (best_params, best_score, n_folds, n_combinations)
    """
    folds = _build_expanding_folds(years)
    if not folds:
        return {}, float("inf"), 0, 0

    combinations = _sample_param_combinations(PARAM_GRID, n_iter)
    best_score = float("inf")
    best_params = {}

    for params in combinations:
        fold_scores = []
        for train_mask, val_mask in folds:
            model = XGBRegressor(
                n_estimators=500,
                early_stopping_rounds=50,
                random_state=42,
                verbosity=0,
                **params,
            )
            model.fit(
                X[train_mask],
                y[train_mask],
                eval_set=[(X[val_mask], y[val_mask])],
                verbose=False,
            )
            preds = model.predict(X[val_mask])
            mae = np.mean(np.abs(y[val_mask] - preds))
            fold_scores.append(mae)

        avg_score = np.mean(fold_scores)
        if avg_score < best_score:
            best_score = avg_score
            best_params = params

        if on_progress:
            on_progress()

    return best_params, best_score, len(folds), len(combinations)


def _cohort_mape(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """MAPE at the cohort level: compare sum of predicted probabilities to sum of actuals."""
    actual_total = y_true.sum()
    predicted_total = y_pred_proba.sum()
    if actual_total == 0:
        return 0.0
    return abs(actual_total - predicted_total) / actual_total
