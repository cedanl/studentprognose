import numpy as np
import pandas as pd
import pytest

from studentprognose.benchmark.metrics import mape, mae, rmse
from studentprognose.benchmark.splitter import time_series_split


class TestMetrics:
    def test_mape_known_values(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 180, 330])
        result = mape(y_true, y_pred)
        expected = np.mean([0.10, 0.10, 0.10])
        assert abs(result - expected) < 1e-10

    def test_mae_known_values(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 180, 330])
        result = mae(y_true, y_pred)
        expected = np.mean([10, 20, 30])
        assert abs(result - expected) < 1e-10

    def test_rmse_known_values(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        result = rmse(y_true, y_pred)
        expected = np.sqrt(np.mean([100.0, 100.0]))
        assert abs(result - expected) < 1e-10

    def test_mape_skips_zeros_in_denominator(self):
        y_true = np.array([0, 100, 200])
        y_pred = np.array([10, 110, 210])
        result = mape(y_true, y_pred)
        expected = np.mean([0.10, 0.05])
        assert abs(result - expected) < 1e-10

    def test_metrics_handle_nan(self):
        y_true = np.array([100, np.nan, 300])
        y_pred = np.array([110, 200, np.nan])
        assert mape(y_true, y_pred) == pytest.approx(0.10)
        assert mae(y_true, y_pred) == pytest.approx(10.0)
        assert rmse(y_true, y_pred) == pytest.approx(10.0)

    def test_all_nan_returns_nan(self):
        y_true = np.array([np.nan])
        y_pred = np.array([np.nan])
        assert np.isnan(mape(y_true, y_pred))
        assert np.isnan(mae(y_true, y_pred))
        assert np.isnan(rmse(y_true, y_pred))


class TestSplitter:
    def _make_data(self, years):
        rows = []
        for y in years:
            for prog in ["A", "B"]:
                rows.append({"Collegejaar": y, "Croho groepeernaam": prog, "val": y * 10})
        return pd.DataFrame(rows)

    def test_splits_correct_years(self):
        data = self._make_data([2016, 2017, 2018, 2019, 2020])
        splits = time_series_split(data, min_training_year=2016, min_train_years=3)

        assert len(splits) == 2

        train0, test0 = splits[0]
        assert sorted(train0["Collegejaar"].unique()) == [2016, 2017, 2018]
        assert test0["Collegejaar"].unique() == [2019]

        train1, test1 = splits[1]
        assert sorted(train1["Collegejaar"].unique()) == [2016, 2017, 2018, 2019]
        assert test1["Collegejaar"].unique() == [2020]

    def test_no_leakage(self):
        data = self._make_data([2016, 2017, 2018, 2019, 2020])
        splits = time_series_split(data, min_training_year=2016, min_train_years=3)

        for train_df, test_df in splits:
            test_year = test_df["Collegejaar"].iloc[0]
            assert train_df["Collegejaar"].max() < test_year

    def test_too_few_years_returns_empty(self):
        data = self._make_data([2016, 2017, 2018])
        splits = time_series_split(data, min_training_year=2016, min_train_years=3)
        assert len(splits) == 0

    def test_min_training_year_filters(self):
        data = self._make_data([2010, 2015, 2016, 2017, 2018, 2019])
        splits = time_series_split(data, min_training_year=2016, min_train_years=3)

        for train_df, _ in splits:
            assert train_df["Collegejaar"].min() >= 2016


class TestModelConfig:
    def test_create_forecaster_default(self):
        from studentprognose.models import create_forecaster
        from studentprognose.models.sarima import SARIMAForecaster

        f = create_forecaster({})
        assert isinstance(f, SARIMAForecaster)

    def test_create_forecaster_ets(self):
        from studentprognose.models import create_forecaster
        from studentprognose.models.forecasters import ETSForecaster

        f = create_forecaster({"model_config": {"cumulative_timeseries": "ets"}})
        assert isinstance(f, ETSForecaster)

    def test_create_forecaster_invalid_raises(self):
        from studentprognose.models import create_forecaster

        with pytest.raises(ValueError, match="Onbekend tijdreeksmodel"):
            create_forecaster({"model_config": {"cumulative_timeseries": "nonexistent"}})

    def test_create_regressor_default(self):
        from studentprognose.models import create_regressor
        from studentprognose.models.regressors import XGBoostRegressor

        r = create_regressor({})
        assert isinstance(r, XGBoostRegressor)

    def test_create_regressor_ridge(self):
        from studentprognose.models import create_regressor
        from studentprognose.models.regressors import RidgeRegressor

        r = create_regressor({"model_config": {"cumulative_regressor": "ridge"}})
        assert isinstance(r, RidgeRegressor)

    def test_create_regressor_invalid_raises(self):
        from studentprognose.models import create_regressor

        with pytest.raises(ValueError, match="Onbekend regressiemodel"):
            create_regressor({"model_config": {"cumulative_regressor": "nonexistent"}})
