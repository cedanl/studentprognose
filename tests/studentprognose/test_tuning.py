"""Tests voor tuning van het cumulatieve spoor (regressor stap 2, SARIMA stap 1)."""

import json

import numpy as np
import pandas as pd
import pytest

from studentprognose.config import load_defaults
from studentprognose.models.tuning import (
    DEFAULT_PARAM_GRIDS,
    DEFAULT_SARIMA_GRID,
    expand_grid,
    format_tuning_results,
    sarima_config_snippet,
    tune_regressor,
    tune_sarima,
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


def _make_cumulative_long(years):
    """Lang-format cumulatieve curve (met ``ts``) zoals tune_sarima die voedt.

    Eén stijgende curve per (programma × jaar) over de gevulde weken; genoeg
    historie voor meerdere tijdreeks-splits.
    """
    weeks = list(range(35, 53)) + list(range(1, 21))
    rows = []
    for y in years:
        for prog in ["A", "B"]:
            base = 100 + (y - 2016) * 10 + (5 if prog == "B" else 0)
            for i, wk in enumerate(weeks):
                rows.append({
                    "Collegejaar": y,
                    "Faculteit": "F",
                    "Herkomst": "NL",
                    "Examentype": "Bachelor",
                    "Croho groepeernaam": prog,
                    "ts": float(base + 5 * i),
                    "Weeknummer": int(wk),
                })
    return pd.DataFrame(rows)


def _cfg_sarima():
    return {
        "column_roles": {
            "programme": "Croho groepeernaam",
            "origin": "Herkomst",
            "exam_type": "Examentype",
            "academic_year": "Collegejaar",
        },
        "model_config": {"final_academic_week": 36, "cumulative_timeseries": "sarima"},
    }


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


class TestFormatTuningResults:
    """Bewaakt de render-laag die zowel het CLI- als het API-pad gebruiken."""

    def _result(self, best_params):
        return {
            "regressor_name": "xgboost",
            "best_params": best_params,
            "best_mape": 0.1424 if best_params else float("nan"),
            "n_candidates": 2,
            "results": [
                {"params": {"max_depth": 5}, "mean_mape": 0.1424, "n_evals": 12},
                {"params": {"max_depth": 3}, "mean_mape": 0.1587, "n_evals": 12},
            ],
        }

    def test_marks_best_row_with_checkmark(self):
        out = format_tuning_results(self._result({"max_depth": 5}))
        lines = [ln for ln in out.splitlines() if "max_depth" in ln and "regressor_params" not in ln]
        winner = next(ln for ln in lines if '"max_depth": 5' in ln)
        loser = next(ln for ln in lines if '"max_depth": 3' in ln)
        # Precies één ✓, en die staat op de winnaar.
        assert out.count("✓") == 1
        assert "✓" in winner
        assert "✓" not in loser

    def test_includes_config_snippet(self):
        out = format_tuning_results(self._result({"max_depth": 5}))
        assert "regressor_params" in out
        assert '"max_depth": 5' in out

    def test_no_checkmark_when_no_valid_winner(self):
        result = self._result(None)
        result["results"] = [{"params": {}, "mean_mape": float("nan"), "n_evals": 0}]
        out = format_tuning_results(result)
        assert "✓" not in out
        assert "Geen geldige resultaten" in out

    def test_empty_results(self):
        result = self._result(None)
        result["results"] = []
        assert format_tuning_results(result) == "Geen kandidaten geëvalueerd."

    def test_sarima_target_uses_forecaster_params_snippet(self):
        # Een sarima-resultaat moet het forecaster_params-snippet tonen (niet
        # regressor_params) en het juiste model-label, met ✓ op de winnaar.
        result = {
            "target": "sarima",
            "model": "sarima",
            "best_params": {"order": (1, 1, 1), "seasonal_order": (1, 1, 0, 52)},
            "best_mape": 0.10,
            "n_candidates": 2,
            "results": [
                {"params": {"order": (1, 1, 1), "seasonal_order": (1, 1, 0, 52)},
                 "mean_mape": 0.10, "n_evals": 6},
                {"params": {"order": (1, 0, 1), "seasonal_order": (1, 1, 0, 52)},
                 "mean_mape": 0.13, "n_evals": 6},
            ],
        }
        out = format_tuning_results(result)
        assert "forecaster_params" in out
        assert "regressor_params" not in out
        assert "Beste parameters voor 'sarima'" in out
        assert out.count("✓") == 1
        # Orde-arrays blijven op één regel in het snippet.
        assert '"order": [1, 1, 1]' in out


class TestSarimaConfigSnippet:
    def test_inline_arrays_and_structure(self):
        snippet = sarima_config_snippet({"order": (1, 0, 1), "seasonal_order": (1, 1, 1, 52)})
        assert '"order": [1, 0, 1]' in snippet
        assert '"seasonal_order": [1, 1, 1, 52]' in snippet
        assert "forecaster_params" in snippet
        # Geldige JSON na de inleidende tekstregel.
        body = snippet[snippet.index("{"):]
        parsed = json.loads(body)
        assert parsed["model_config"]["forecaster_params"]["sarima"]["order"] == [1, 0, 1]


class TestTuneSarima:
    def test_returns_sarima_result_shape(self):
        years = [2016, 2017, 2018, 2019, 2020, 2021]
        grid = {"order": [(1, 0, 1), (1, 1, 1)], "seasonal_order": [(1, 1, 0, 52)]}
        result = tune_sarima(
            _make_cumulative_long(years), 12, _cfg_sarima(),
            grid=grid, min_training_year=2016,
        )
        assert result["target"] == "sarima"
        assert result["model"] == "sarima"
        assert result["n_candidates"] == 2
        assert isinstance(result["best_params"], dict)
        assert "order" in result["best_params"]
        # Winnaar zit in de gerangschikte resultaten.
        assert result["best_params"] in [r["params"] for r in result["results"]]

    def test_results_sorted_by_mape(self):
        years = [2016, 2017, 2018, 2019, 2020, 2021]
        result = tune_sarima(
            _make_cumulative_long(years), 12, _cfg_sarima(), min_training_year=2016,
        )
        finite = [r["mean_mape"] for r in result["results"] if np.isfinite(r["mean_mape"])]
        assert finite == sorted(finite)

    def test_default_grid_used_when_none(self):
        years = [2016, 2017, 2018, 2019, 2020, 2021]
        result = tune_sarima(
            _make_cumulative_long(years), 12, _cfg_sarima(), min_training_year=2016,
        )
        assert result["n_candidates"] == len(expand_grid(DEFAULT_SARIMA_GRID))

    def test_too_few_years_yields_no_winner(self):
        years = [2016, 2017]  # < min_train_years → geen splits
        result = tune_sarima(
            _make_cumulative_long(years), 12, _cfg_sarima(), min_training_year=2016,
        )
        assert result["best_params"] is None
        assert np.isnan(result["best_mape"])
