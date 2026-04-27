import numpy as np
from numpy import linalg as LA
import statsmodels.api as sm

from studentprognose.models.base import BaseForecaster
from studentprognose.utils.weeks import get_all_weeks_valid, compute_pred_len
from studentprognose.utils.constants import (
    FINAL_ACADEMIC_WEEK,
    SARIMA_ORDER, SARIMA_ORDER_INDIVIDUAL, SARIMA_SEASONAL_ORDER, SARIMA_SEASONAL_ORDER_ALT,
    SARIMA_BACHELOR_DEADLINE_WEEKS,
)


class SARIMAForecaster(BaseForecaster):
    """Unified SARIMA forecaster used by both individual and cumulative strategies."""

    def __init__(self, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER):
        self.order = order
        self.seasonal_order = seasonal_order
        self._results = None

    def fit(self, ts_data, exog=None):
        model = sm.tsa.statespace.SARIMAX(
            ts_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            exog=exog,
        )
        self._results = model.fit(disp=0)
        return self

    def forecast(self, steps, exog=None):
        return self._results.forecast(steps=steps, exog=exog)


def create_time_series(data, pred_len):
    ts_data = data.loc[:, get_all_weeks_valid(data.columns)].values.flatten()
    ts_data = ts_data[:-pred_len]
    return np.array(ts_data)


def predict_with_sarima_cumulative(data_cumulative, row, predict_year, predict_week, pred_len, skip_years=0, already_printed=False, min_training_year: int = 2016) -> list:
    """
    Predicts pre-registrations with SARIMA per programme/origin/week for cumulative data.

    Returns:
        list: predictions for each future week, or empty list on error.
    """
    programme = row["Croho groepeernaam"]
    herkomst = row["Herkomst"]
    examentype = row["Examentype"]

    if not already_printed:
        print(
            f"Prediction for {programme}, {herkomst}, year: {predict_year}, week: {predict_week}"
        )

    data_cumulative = data_cumulative.astype(
        {"Weeknummer": "int32", "Collegejaar": "int32"}
    )
    data = _get_transformed_data(data_cumulative.copy(deep=True), min_training_year)

    data = data[
        (data["Herkomst"] == herkomst)
        & (data["Collegejaar"] <= predict_year - skip_years)
        & (data["Croho groepeernaam"] == programme)
        & (data["Examentype"] == examentype)
    ]

    data["39"] = 0

    ts_data = create_time_series(data, pred_len)

    try:
        model = SARIMAForecaster(order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER)
        model.fit(ts_data)
        pred = model.forecast(steps=pred_len)
        return pred

    except (LA.LinAlgError, IndexError, ValueError) as error:
        print(f"Cumulative sarima error on: {programme}, {herkomst}")
        print(error)
        return []


def predict_with_sarima_individual(data_individual, row, predict_year, predict_week, max_year, numerus_fixus_list, data_exog=None, already_printed=False) -> list:
    """
    Predicts nr of students with SARIMA per programme/origin/week for individual data.

    Returns:
        list: predictions for each future week, or empty list on error.
    """
    from studentprognose.data.transforms import transform_data

    data = data_individual.copy()
    programme = row["Croho groepeernaam"]
    herkomst = row["Herkomst"]
    examentype = row["Examentype"]

    if not already_printed:
        print(
            f"Prediction for {programme}, {herkomst}, {examentype}, year: {predict_year}, week: {predict_week}"
        )

    def filter_data(data, programme, herkomst, examentype, jaar, max_year):
        data = data[data["Herkomst"] == herkomst]
        if jaar != max_year:
            data = data[data["Collegejaar"] <= jaar]
        data = data[data["Croho groepeernaam"] == programme]
        data = data[data["Examentype"] == examentype]
        return data

    if data_exog is not None:
        data_exog = filter_data(
            data_exog, programme, herkomst, examentype, predict_year, max_year
        )
    data = filter_data(data, programme, herkomst, examentype, predict_year, max_year)

    def deadline_week(weeknummer, croho, examentype):
        if (
            weeknummer in [16, 17]
            and examentype == "Bachelor"
            and croho not in numerus_fixus_list
        ):
            return 1
        elif (
            weeknummer in [1, 2]
            and examentype == "Bachelor"
            and croho in numerus_fixus_list
        ):
            return 1
        else:
            return 0

    if data_exog is not None:
        data_exog["Deadline"] = data_exog.apply(
            lambda x: deadline_week(x["Weeknummer"], x["Croho groepeernaam"], x["Examentype"]),
            axis=1,
        )

    try:
        if data_exog is not None:
            data_exog = transform_data(data_exog, "Deadline")

        if predict_week == FINAL_ACADEMIC_WEEK:
            ts_data = data.loc[:, get_all_weeks_valid(data.columns)].values.flatten()
            try:
                return [ts_data[-1]]
            except IndexError:
                return []

        pred_len = compute_pred_len(int(predict_week))

        def create_exogenous(data, pred_len):
            exg_data = data.loc[:, get_all_weeks_valid(data.columns)].values.flatten()
            exg_data_train = exg_data[:-pred_len]
            exg_data_test = exg_data[-pred_len:]
            return np.array(exg_data_train), np.array(exg_data_test)

        ts_data = create_time_series(data, pred_len)

        if data_exog is not None:
            exogenous_train_1, exg_data_test_1 = create_exogenous(data_exog, pred_len)
        else:
            exogenous_train_1 = None

        if ts_data.size == 0:
            return []

        try:
            if programme.startswith("B") and predict_week in SARIMA_BACHELOR_DEADLINE_WEEKS:
                model = SARIMAForecaster(order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER)
            else:
                model = SARIMAForecaster(order=SARIMA_ORDER_INDIVIDUAL, seasonal_order=SARIMA_SEASONAL_ORDER_ALT)

            model.fit(ts_data, exog=exogenous_train_1)

            if data_exog is not None:
                pred = model.forecast(steps=pred_len, exog=exg_data_test_1)
            else:
                pred = model.forecast(steps=pred_len)

            return pred
        except (LA.LinAlgError, IndexError, ValueError) as error:
            print(f"Individual error on: {programme}, {herkomst}")
            print(error)
            return []
    except KeyError as error:
        print(f"Individual key error on: {programme}, {herkomst}")
        print(error)
        return []


def _get_transformed_data(data, min_training_year: int = 2016):
    """Helper to transform cumulative data for SARIMA.

    Args:
        data: Cumulative pre-application data.
        min_training_year: Earliest academic year included in training. Should be
            read from ``model_config.min_training_year`` in the caller's configuration.
    """
    from studentprognose.data.transforms import transform_data

    data = data.drop_duplicates()
    data = data[data["Collegejaar"] >= min_training_year]
    data = transform_data(data, "ts")
    return data
