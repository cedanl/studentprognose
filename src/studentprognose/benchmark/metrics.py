import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error. NaN-waarden en nullen in y_true worden uitgesloten."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    if not valid.any():
        return np.nan
    return float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error. NaN-waarden worden uitgesloten."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        return np.nan
    return float(np.mean(np.abs(y_true[valid] - y_pred[valid])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error. NaN-waarden worden uitgesloten."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        return np.nan
    return float(np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2)))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy. Retourneert NaN bij lege input."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return np.nan
    return float(accuracy_score(y_true, y_pred))


def auc_roc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """AUC-ROC. Retourneert NaN als slechts één klasse aanwezig is."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_proba))


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1-score (binary). Retourneert NaN bij lege input."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return np.nan
    return float(f1_score(y_true, y_pred, zero_division=0.0))
