import time

import numpy as np
import pandas as pd

from studentprognose.benchmark.metrics import mape, mae, rmse
from studentprognose.benchmark.splitter import time_series_split
from studentprognose.config import get_columns
from studentprognose.models.sarima import _get_transformed_data
from studentprognose.utils.weeks import compute_pred_len, get_all_weeks_valid


def evaluate_timeseries_model(
    data_cumulative: pd.DataFrame,
    forecaster_factory: callable,
    predict_week: int,
    config: dict,
    min_training_year: int = 2016,
    min_train_years: int = 3,
) -> pd.DataFrame:
    """Evalueer een tijdreeksmodel op de cumulatieve curves via cross-validatie.

    Meet de kwaliteit van de tijdreeksextrapolatie (stap 1 in de pipeline),
    los van de XGBoost-regressie.

    Args:
        data_cumulative: Preprocessed cumulatieve data (lang formaat).
        forecaster_factory: Callable die een vers BaseForecaster-object retourneert.
        predict_week: Week waarvandaan voorspeld wordt.
        config: Configuratie-dict met column_roles.
        min_training_year: Vroegste trainingsjaar.
        min_train_years: Minimaal aantal trainingsjaren.

    Returns:
        DataFrame met per (programme, herkomst, examentype, test_year)
        de metrics MAPE, MAE, RMSE en trainingstijd.
    """
    c = get_columns(config)
    pred_len = compute_pred_len(predict_week)

    group_keys = data_cumulative.groupby(
        [c.programme, c.origin, c.exam_type]
    ).first()[[]].reset_index()

    results = []

    for _, group_row in group_keys.iterrows():
        programme = group_row[c.programme]
        herkomst = group_row[c.origin]
        examentype = group_row[c.exam_type]

        subset = data_cumulative[
            (data_cumulative[c.programme] == programme)
            & (data_cumulative[c.origin] == herkomst)
            & (data_cumulative[c.exam_type] == examentype)
        ]

        wide = _get_transformed_data(subset.copy(), min_training_year)
        if wide.empty:
            continue
        wide["39"] = 0

        splits = time_series_split(wide, min_training_year, min_train_years, year_col=c.academic_year)

        for train_wide, test_wide in splits:
            test_year = int(test_wide[c.academic_year].iloc[0])

            train_for_ts = pd.concat([train_wide, test_wide])
            ts_data = train_for_ts.loc[
                :, get_all_weeks_valid(train_for_ts.columns)
            ].values.flatten()

            if len(ts_data) <= pred_len:
                continue

            ts_train = ts_data[:-pred_len]
            ts_actual = ts_data[-pred_len:]

            t0 = time.perf_counter()
            try:
                model = forecaster_factory()
                model.fit(np.array(ts_train))
                pred = model.forecast(steps=pred_len)
                elapsed = time.perf_counter() - t0
            except Exception:
                elapsed = time.perf_counter() - t0
                results.append({
                    "programme": programme,
                    "herkomst": herkomst,
                    "examentype": examentype,
                    "test_year": test_year,
                    "mape": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "train_time_s": elapsed,
                    "converged": False,
                })
                continue

            pred = np.asarray(pred)
            results.append({
                "programme": programme,
                "herkomst": herkomst,
                "examentype": examentype,
                "test_year": test_year,
                "mape": mape(ts_actual, pred),
                "mae": mae(ts_actual, pred),
                "rmse": rmse(ts_actual, pred),
                "train_time_s": elapsed,
                "converged": True,
            })

    return pd.DataFrame(results)
