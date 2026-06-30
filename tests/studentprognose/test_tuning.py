"""Tests voor hyperparameter tuning van de cumulatieve regressor."""

import numpy as np
import pandas as pd
import pytest

from studentprognose.config import load_defaults
from studentprognose.models.tuning import (
    DEFAULT_PARAM_GRIDS,
    expand_grid,
    tune_regressor,
)
from studentprognose.strategies.cumulative import ENGINEERED_FEATURE_COLS
from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK
from studentprognose.utils.weeks import get_weeks_list

# Weekkolommen die build_preprocessor als numerieke features verwacht (zoals de
# productie-pivot ze oplevert). De regressor-evaluatie passeert geen
# available_cols, dus ze moeten allemaal aanwezig zijn in de feature-matrix.
_WEEK_COLS = [str(w) for w in get_weeks_list(FINAL_ACADEMIC_WEEK, FINAL_ACADEMIC_WEEK)]


def _make_full_data(years):
    """Bouw een wide-format feature-matrix zoals het cumulatieve spoor die voedt."""
    rng = np.random.default_rng(0)
    rows = []
    for y in years:
        for prog in ["A", "B"]:
            # Deterministisch signaal + lichte ruis, zodat de regressor iets te
            # leren heeft en MAPE-waarden eindig en vergelijkbaar zijn.
            base = 100 + (y - 2016) * 20 + (10 if prog == "B" else 0)
            row = {
                "Collegejaar": y,
                "Croho groepeernaam": prog,
                "Herkomst": "NL",
                "Examentype": "Bachelor",
                "Faculteit": "F",
                "Gewogen_t-2": base + rng.normal(0, 2),
                "Gewogen_t-5": base * 0.8 + rng.normal(0, 2),
                "Gewogen_acceleration": rng.normal(0, 1),
                "exclusivity_ratio": 0.5 + rng.normal(0, 0.05),
            }
            for i, wk in enumerate(_WEEK_COLS):
                row[wk] = base * (i + 1) / len(_WEEK_COLS) + rng.normal(0, 1)
            rows.append(row)
    full_data = pd.DataFrame(rows)
    assert set(ENGINEERED_FEATURE_COLS).issubset(full_data.columns)
    return full_data


def _make_studentcount(years):
    rows = []
    for y in years:
        for prog in ["A", "B"]:
            base = 100 + (y - 2016) * 20 + (10 if prog == "B" else 0)
            rows.append({
                "Croho groepeernaam": prog,
                "Collegejaar": y,
                "Herkomst": "NL",
                "Examentype": "Bachelor",
                "Aantal_studenten": base,
            })
    return pd.DataFrame(rows)


class TestExpandGrid:
    def test_empty_grid_yields_single_default_candidate(self):
        assert expand_grid({}) == [{}]

    def test_cartesian_product(self):
        grid = {"a": [1, 2], "b": [3, 4, 5]}
        candidates = expand_grid(grid)
        assert len(candidates) == 6
        assert {"a": 1, "b": 3} in candidates
        assert {"a": 2, "b": 5} in candidates

    def test_default_xgboost_grid_size(self):
        # 3 learning_rates × 2 n_estimators × 2 max_depth
        assert len(expand_grid(DEFAULT_PARAM_GRIDS["xgboost"])) == 12


class TestTuneRegressor:
    def test_returns_best_params_with_enough_years(self):
        years = [2016, 2017, 2018, 2019, 2020, 2021]
        config = load_defaults()
        result = tune_regressor(
            _make_full_data(years),
            _make_studentcount(years),
            config,
            regressor_name="xgboost",
            min_training_year=2016,
        )
        assert result["regressor_name"] == "xgboost"
        assert result["n_candidates"] == 12
        assert isinstance(result["best_params"], dict)
        assert np.isfinite(result["best_mape"])
        # De winnaar is één van de geëvalueerde kandidaten.
        assert result["best_params"] in [r["params"] for r in result["results"]]

    def test_results_sorted_finite_first(self):
        years = [2016, 2017, 2018, 2019, 2020, 2021]
        config = load_defaults()
        result = tune_regressor(
            _make_full_data(years), _make_studentcount(years), config,
            regressor_name="ridge", min_training_year=2016,
        )
        mapes = [r["mean_mape"] for r in result["results"]]
        finite = [m for m in mapes if np.isfinite(m)]
        # Eindige scores komen eerst en zijn oplopend gesorteerd.
        assert finite == sorted(finite)
        assert mapes[: len(finite)] == finite

    def test_custom_grid_limits_candidates(self):
        years = [2016, 2017, 2018, 2019, 2020]
        config = load_defaults()
        result = tune_regressor(
            _make_full_data(years), _make_studentcount(years), config,
            regressor_name="xgboost", grid={"max_depth": [2, 4]},
            min_training_year=2016,
        )
        assert result["n_candidates"] == 2
        assert result["best_params"] in [{"max_depth": 2}, {"max_depth": 4}]

    def test_too_few_years_returns_none(self):
        years = [2016, 2017]  # < min_train_years → geen splits
        config = load_defaults()
        result = tune_regressor(
            _make_full_data(years), _make_studentcount(years), config,
            regressor_name="xgboost", min_training_year=2016,
        )
        assert result["best_params"] is None
        assert np.isnan(result["best_mape"])

    def test_unknown_regressor_raises(self):
        config = load_defaults()
        with pytest.raises(ValueError, match="Onbekend regressiemodel"):
            tune_regressor(pd.DataFrame(), pd.DataFrame(), config, regressor_name="nonexistent")

    def test_defaults_to_config_regressor(self):
        years = [2016, 2017, 2018, 2019, 2020]
        config = load_defaults()
        config["model_config"]["cumulative_regressor"] = "ridge"
        result = tune_regressor(
            _make_full_data(years), _make_studentcount(years), config,
            min_training_year=2016,
        )
        assert result["regressor_name"] == "ridge"
        assert result["n_candidates"] == len(expand_grid(DEFAULT_PARAM_GRIDS["ridge"]))
