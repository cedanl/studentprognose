import numpy as np
import pandas as pd
import pytest

from studentprognose.models.regressors import (
    BaseRegressor,
    XGBoostRegressor,
    RidgeRegressor,
    RandomForestRegressor,
    build_preprocessor,
)


REGRESSOR_CLASSES = [XGBoostRegressor, RidgeRegressor, RandomForestRegressor]


@pytest.mark.parametrize("cls", REGRESSOR_CLASSES, ids=lambda c: c.__name__)
def test_regressor_implements_base(cls):
    assert issubclass(cls, BaseRegressor)


@pytest.mark.parametrize("cls", REGRESSOR_CLASSES, ids=lambda c: c.__name__)
def test_fit_predict_returns_array(cls):
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (50, 5))
    y = rng.normal(100, 10, 50)

    model = cls()
    model.fit(X, y)
    predictions = model.predict(X[:10])

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 10
    assert np.all(np.isfinite(predictions))


def test_xgboost_has_feature_importances():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (50, 5))
    y = rng.normal(100, 10, 50)

    model = XGBoostRegressor()
    model.fit(X, y)
    assert model.feature_importances_ is not None
    assert len(model.feature_importances_) == 5


def test_random_forest_has_feature_importances():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (50, 5))
    y = rng.normal(100, 10, 50)

    model = RandomForestRegressor()
    model.fit(X, y)
    assert model.feature_importances_ is not None
    assert len(model.feature_importances_) == 5


def test_ridge_no_feature_importances():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (50, 5))
    y = rng.normal(100, 10, 50)

    model = RidgeRegressor()
    model.fit(X, y)
    assert model.feature_importances_ is None


def test_build_preprocessor_columns():
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(
        extra_numeric_cols=["feat_a", "feat_b"]
    )
    assert "Collegejaar" in numeric_cols
    assert "feat_a" in numeric_cols
    assert "feat_b" in numeric_cols
    assert "Examentype" in categorical_cols


def test_build_preprocessor_no_extras():
    preprocessor, numeric_cols, categorical_cols = build_preprocessor()
    assert "Collegejaar" in numeric_cols
    assert len(categorical_cols) == 4


def test_vectorized_rounding():
    """predict_with_xgboost moet np.rint gebruiken (vectorized) i.p.v. loop."""
    from studentprognose.models.xgboost_regressor import predict_with_xgboost
    from studentprognose.utils.weeks import get_weeks_list

    rng = np.random.default_rng(42)
    n = 10
    weeks = [str(w) for w in get_weeks_list(38)]

    base = {
        "Croho groepeernaam": ["A"] * n,
        "Collegejaar": list(range(2016, 2016 + n)),
        "Herkomst": ["NL"] * n,
        "Examentype": ["Bachelor"] * n,
        "Faculteit": ["F1"] * n,
    }
    for w in weeks:
        base[w] = rng.normal(50, 5, n)

    train = pd.DataFrame(base)
    test = train.iloc[-1:].copy()

    studentcount = pd.DataFrame({
        "Croho groepeernaam": ["A"] * n,
        "Collegejaar": list(range(2016, 2016 + n)),
        "Herkomst": ["NL"] * n,
        "Examentype": ["Bachelor"] * n,
        "Aantal_studenten": rng.integers(50, 200, n),
    })

    predictions, _ = predict_with_xgboost(train.iloc[:9], test, studentcount)
    if not np.isnan(predictions).all():
        assert predictions.dtype in (np.int32, np.int64, int)
