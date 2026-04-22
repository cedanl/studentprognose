import os
import tempfile

import numpy as np

from studentprognose.tuning.cache import (
    load_cached_params,
    save_cached_params,
    load_sarima_order,
    save_sarima_orders,
)
from studentprognose.tuning.xgboost_tuner import (
    _generate_param_combinations,
    _sample_param_combinations,
    _build_expanding_folds,
    _cohort_mape,
)
from studentprognose.tuning.sarima_tuner import find_best_sarima_order
from studentprognose.models.xgboost_classifier import (
    _resolve_hyperparameters,
    DEFAULT_CLASSIFIER_HP,
)
from studentprognose.models.xgboost_regressor import (
    _resolve_hyperparameters as _resolve_regressor_hp,
    DEFAULT_REGRESSOR_HP,
)
from studentprognose.models.sarima import _resolve_sarima_orders
from studentprognose.cli import parse_args


# ── Cache tests ──────────────────────────────────────────────────


class TestCache:
    def setup_method(self):
        import studentprognose.tuning.cache as cache_mod

        self._orig_path = cache_mod.CACHE_PATH
        self._tmpdir = tempfile.mkdtemp()
        cache_mod.CACHE_PATH = os.path.join(self._tmpdir, "tuning_cache.json")

    def teardown_method(self):
        import studentprognose.tuning.cache as cache_mod

        cache_mod.CACHE_PATH = self._orig_path

    def test_empty_cache_returns_none(self):
        assert load_cached_params("xgboost_classifier") is None

    def test_save_and_load_xgboost(self):
        params = {"max_depth": 6, "learning_rate": 0.05}
        save_cached_params("xgboost_classifier", params, 0.12, 4, 100)
        result = load_cached_params("xgboost_classifier")
        assert result is not None
        assert result["params"] == params
        assert result["cv_score"] == 0.12
        assert result["n_folds"] == 4

    def test_save_and_load_sarima_order(self):
        orders = {
            "B Test|NL|Bachelor|cumulative": {
                "order": [1, 0, 1],
                "seasonal_order": [1, 1, 1, 52],
                "aic": 1234.56,
            }
        }
        save_sarima_orders(orders)
        result = load_sarima_order("B Test", "NL", "Bachelor", "cumulative")
        assert result is not None
        assert result["order"] == [1, 0, 1]
        assert result["aic"] == 1234.56

    def test_missing_sarima_order_returns_none(self):
        assert load_sarima_order("Nonexistent", "NL", "Bachelor", "cumulative") is None

    def test_multiple_models_coexist(self):
        save_cached_params("xgboost_classifier", {"a": 1}, 0.1, 3, 50)
        save_cached_params("xgboost_regressor", {"b": 2}, 5.0, 4, 60)
        assert load_cached_params("xgboost_classifier")["params"] == {"a": 1}
        assert load_cached_params("xgboost_regressor")["params"] == {"b": 2}


# ── XGBoost tuner utility tests ──────────────────────────────────


class TestXGBoostTunerUtils:
    def test_param_combinations(self):
        grid = {"a": [1, 2], "b": [3, 4]}
        combos = _generate_param_combinations(grid)
        assert len(combos) == 4
        assert {"a": 1, "b": 3} in combos
        assert {"a": 2, "b": 4} in combos

    def test_sample_param_combinations_subset(self):
        grid = {"a": [1, 2, 3], "b": [4, 5, 6]}
        sampled = _sample_param_combinations(grid, n_iter=3)
        assert len(sampled) == 3
        all_combos = _generate_param_combinations(grid)
        for combo in sampled:
            assert combo in all_combos

    def test_sample_param_combinations_none_returns_all(self):
        grid = {"a": [1, 2], "b": [3, 4]}
        sampled = _sample_param_combinations(grid, n_iter=None)
        assert len(sampled) == 4

    def test_sample_param_combinations_exceeds_total(self):
        grid = {"a": [1, 2], "b": [3, 4]}
        sampled = _sample_param_combinations(grid, n_iter=100)
        assert len(sampled) == 4

    def test_sample_param_combinations_deterministic(self):
        grid = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
        run1 = _sample_param_combinations(grid, n_iter=5, seed=42)
        run2 = _sample_param_combinations(grid, n_iter=5, seed=42)
        assert run1 == run2

    def test_expanding_folds_basic(self):
        years = np.array([2016, 2016, 2017, 2017, 2018, 2018, 2019, 2019])
        folds = _build_expanding_folds(years, min_train_years=2)
        assert len(folds) == 2
        train_mask, val_mask = folds[0]
        assert val_mask.sum() == 2
        assert train_mask.sum() == 4

    def test_expanding_folds_too_few_years(self):
        years = np.array([2016, 2016])
        folds = _build_expanding_folds(years, min_train_years=2)
        assert len(folds) == 0

    def test_cohort_mape_perfect(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1.0, 1.0, 0.0, 0.0])
        assert _cohort_mape(y_true, y_pred) == 0.0

    def test_cohort_mape_imperfect(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        assert _cohort_mape(y_true, y_pred) == 0.0

    def test_cohort_mape_zero_actual(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0.5, 0.5, 0.5])
        assert _cohort_mape(y_true, y_pred) == 0.0


# ── SARIMA tuner tests ───────────────────────────────────────────


class TestSARIMATuner:
    def test_short_series_returns_none(self):
        ts = np.random.randn(50)
        order, seasonal, aic = find_best_sarima_order(ts, seasonal_period=52)
        assert order is None

    def test_valid_series_returns_order(self):
        np.random.seed(42)
        n = 52 * 4
        ts = np.sin(np.linspace(0, 8 * np.pi, n)) + np.random.randn(n) * 0.3
        order, seasonal, aic = find_best_sarima_order(
            ts,
            max_p=1,
            max_d=1,
            max_q=1,
            max_P=0,
            max_D=0,
            max_Q=0,
            seasonal_period=52,
        )
        assert order is not None
        assert len(order) == 3
        assert aic < float("inf")


# ── Hyperparameter resolution tests ─────────────────────────────


class TestHyperparameterResolution:
    def setup_method(self):
        import studentprognose.tuning.cache as cache_mod

        self._orig_path = cache_mod.CACHE_PATH
        self._tmpdir = tempfile.mkdtemp()
        cache_mod.CACHE_PATH = os.path.join(self._tmpdir, "tuning_cache.json")

    def teardown_method(self):
        import studentprognose.tuning.cache as cache_mod

        cache_mod.CACHE_PATH = self._orig_path

    def test_falls_back_to_defaults_without_config(self):
        hp = _resolve_hyperparameters({})
        assert hp == DEFAULT_CLASSIFIER_HP

    def test_reads_from_config(self):
        config = {
            "model_config": {
                "hyperparameters": {"xgboost_classifier": {"n_estimators": 999}}
            }
        }
        hp = _resolve_hyperparameters(config)
        assert hp["n_estimators"] == 999

    def test_cache_takes_priority_over_config(self):
        save_cached_params("xgboost_classifier", {"n_estimators": 42}, 0.1, 3, 50)
        config = {
            "model_config": {
                "hyperparameters": {"xgboost_classifier": {"n_estimators": 999}}
            }
        }
        hp = _resolve_hyperparameters(config)
        assert hp["n_estimators"] == 42

    def test_regressor_falls_back_to_defaults(self):
        hp = _resolve_regressor_hp({})
        assert hp == DEFAULT_REGRESSOR_HP


# ── SARIMA order resolution tests ────────────────────────────────


class TestSARIMAOrderResolution:
    def setup_method(self):
        import studentprognose.tuning.cache as cache_mod

        self._orig_path = cache_mod.CACHE_PATH
        self._tmpdir = tempfile.mkdtemp()
        cache_mod.CACHE_PATH = os.path.join(self._tmpdir, "tuning_cache.json")

    def teardown_method(self):
        import studentprognose.tuning.cache as cache_mod

        cache_mod.CACHE_PATH = self._orig_path

    def test_fallback_cumulative(self):
        order, seasonal = _resolve_sarima_orders(
            "B Test", "NL", "Bachelor", "cumulative"
        )
        assert order == (1, 0, 1)
        assert seasonal == (1, 1, 1, 52)

    def test_fallback_individual_bachelor_deadline(self):
        order, seasonal = _resolve_sarima_orders(
            "B Test", "NL", "Bachelor", "individual", predict_week=17
        )
        assert order == (1, 0, 1)
        assert seasonal == (1, 1, 1, 52)

    def test_fallback_individual_non_deadline(self):
        order, seasonal = _resolve_sarima_orders(
            "M Test", "NL", "Master", "individual", predict_week=12
        )
        assert order == (1, 1, 1)
        assert seasonal == (1, 1, 0, 52)

    def test_cache_overrides_fallback(self):
        orders = {
            "B Test|NL|Bachelor|cumulative": {
                "order": [2, 1, 0],
                "seasonal_order": [0, 1, 1, 52],
                "aic": 500.0,
            }
        }
        save_sarima_orders(orders)
        order, seasonal = _resolve_sarima_orders(
            "B Test", "NL", "Bachelor", "cumulative"
        )
        assert order == (2, 1, 0)
        assert seasonal == (0, 1, 1, 52)


# ── CLI tests ────────────────────────────────────────────────────


class TestCLITuneFlag:
    def test_tune_flag_default_false(self):
        cfg = parse_args(["prog", "-w", "12", "-y", "2024"])
        assert cfg.tune is False

    def test_tune_flag_enabled(self):
        cfg = parse_args(["prog", "--tune", "-w", "12", "-y", "2024"])
        assert cfg.tune is True
