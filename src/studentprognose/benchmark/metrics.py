import numpy as np


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
