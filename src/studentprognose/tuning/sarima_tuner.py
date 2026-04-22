import itertools
import warnings

import numpy as np
import statsmodels.api as sm
from numpy.linalg import LinAlgError

from studentprognose.utils.constants import WEEKS_PER_YEAR


def find_best_sarima_order(
    ts_data: np.ndarray,
    max_p: int = 2,
    max_d: int = 1,
    max_q: int = 2,
    max_P: int = 1,
    max_D: int = 1,
    max_Q: int = 1,
    seasonal_period: int = WEEKS_PER_YEAR,
    criterion: str = "aic",
    exog: np.ndarray | None = None,
) -> tuple[tuple, tuple, float]:
    """Grid search over SARIMA orders, select by AIC or BIC.

    Returns:
        (order, seasonal_order, best_criterion_value) or (None, None, inf) if all fail.
    """
    if ts_data.size < 2 * seasonal_period:
        return None, None, float("inf")

    p_range = range(max_p + 1)
    d_range = range(max_d + 1)
    q_range = range(max_q + 1)
    P_range = range(max_P + 1)
    D_range = range(max_D + 1)
    Q_range = range(max_Q + 1)

    best_score = float("inf")
    best_order = None
    best_seasonal = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for p, d, q, P, D, Q in itertools.product(
            p_range, d_range, q_range, P_range, D_range, Q_range
        ):
            order = (p, d, q)
            seasonal_order = (P, D, Q, seasonal_period)
            try:
                model = sm.tsa.statespace.SARIMAX(
                    ts_data,
                    order=order,
                    seasonal_order=seasonal_order,
                    exog=exog,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                results = model.fit(disp=0, maxiter=50)
                score = results.aic if criterion == "aic" else results.bic
                if score < best_score:
                    best_score = score
                    best_order = order
                    best_seasonal = seasonal_order
            except (LinAlgError, ValueError, np.linalg.LinAlgError, RuntimeError):
                continue

    return best_order, best_seasonal, best_score
