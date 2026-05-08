import numpy as np
import pandas as pd
import pytest

from studentprognose.benchmark.metrics import mape, mae, rmse, accuracy, auc_roc, f1
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


class TestClassificationMetrics:
    def test_accuracy_known_values(self):
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0])
        assert accuracy(y_true, y_pred) == pytest.approx(0.8)

    def test_accuracy_empty_returns_nan(self):
        assert np.isnan(accuracy(np.array([]), np.array([])))

    def test_auc_roc_known_values(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.6, 0.9])
        assert auc_roc(y_true, y_proba) == pytest.approx(1.0)

    def test_auc_roc_single_class_returns_nan(self):
        y_true = np.array([1, 1, 1])
        y_proba = np.array([0.5, 0.6, 0.7])
        assert np.isnan(auc_roc(y_true, y_proba))

    def test_f1_known_values(self):
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0])
        result = f1(y_true, y_pred)
        assert 0.0 < result < 1.0

    def test_f1_empty_returns_nan(self):
        assert np.isnan(f1(np.array([]), np.array([])))


class TestClassifierModelConfig:
    def test_create_classifier_default(self):
        from studentprognose.models import create_classifier
        from studentprognose.models.classifiers import XGBoostClassifier

        c = create_classifier({})
        assert isinstance(c, XGBoostClassifier)

    def test_create_classifier_random_forest(self):
        from studentprognose.models import create_classifier
        from studentprognose.models.classifiers import RandomForestClassifier

        c = create_classifier({"model_config": {"individual_classifier": "random_forest"}})
        assert isinstance(c, RandomForestClassifier)

    def test_create_classifier_logistic_regression(self):
        from studentprognose.models import create_classifier
        from studentprognose.models.classifiers import LogisticRegressionClassifier

        c = create_classifier({"model_config": {"individual_classifier": "logistic_regression"}})
        assert isinstance(c, LogisticRegressionClassifier)

    def test_create_classifier_invalid_raises(self):
        from studentprognose.models import create_classifier

        with pytest.raises(ValueError, match="Onbekend classificatiemodel"):
            create_classifier({"model_config": {"individual_classifier": "nonexistent"}})


class TestBenchmarkValidation:
    def test_benchmark_without_d_flag_exits(self):
        from studentprognose.cli import parse_args

        cfg = parse_args(["prog", "benchmark"])
        assert cfg.command == "benchmark"
        assert cfg.dataset_specified is False

    def test_benchmark_with_d_both_exits(self):
        from studentprognose.cli import parse_args
        from studentprognose.utils.weeks import DataOption

        cfg = parse_args(["prog", "benchmark", "-d", "b"])
        assert cfg.command == "benchmark"
        assert cfg.data_option == DataOption.BOTH_DATASETS
        assert cfg.dataset_specified is True

    def test_benchmark_with_d_cumulative(self):
        from studentprognose.cli import parse_args
        from studentprognose.utils.weeks import DataOption

        cfg = parse_args(["prog", "benchmark", "-d", "c"])
        assert cfg.command == "benchmark"
        assert cfg.data_option == DataOption.CUMULATIVE
        assert cfg.dataset_specified is True

    def test_benchmark_with_d_individual(self):
        from studentprognose.cli import parse_args
        from studentprognose.utils.weeks import DataOption

        cfg = parse_args(["prog", "benchmark", "-d", "i"])
        assert cfg.command == "benchmark"
        assert cfg.data_option == DataOption.INDIVIDUAL
        assert cfg.dataset_specified is True
