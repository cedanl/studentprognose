"""Tests for _add_engineered_features in strategies/cumulative.py."""

import warnings

import numpy as np
import pandas as pd
import pytest

from studentprognose.strategies.cumulative import _add_engineered_features, _GROUP_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_long(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal long-format data_cumulative from a list of row dicts."""
    defaults = {
        "Collegejaar": 2024,
        "Faculteit": "FacA",
        "Herkomst": "NL",
        "Examentype": "Bachelor",
        "Croho groepeernaam": "B Opleiding",
        "Weeknummer": 10,
        "Gewogen vooraanmelders": 0.0,
        "Ongewogen vooraanmelders": 0.0,
        "Aantal aanmelders met 1 aanmelding": 0.0,
        "Inschrijvingen": 0.0,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _make_wide(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal wide-format full_data (one row per group)."""
    defaults = {
        "Collegejaar": 2024,
        "Faculteit": "FacA",
        "Herkomst": "NL",
        "Examentype": "Bachelor",
        "Croho groepeernaam": "B Opleiding",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


# ---------------------------------------------------------------------------
# Lagged features
# ---------------------------------------------------------------------------

class TestLaggedFeatures:
    def test_lag2_correct_week(self):
        long = _make_long([
            {"Weeknummer": 8, "Gewogen vooraanmelders": 50.0},
            {"Weeknummer": 10, "Gewogen vooraanmelders": 70.0},
        ])
        wide = _make_wide([{}])

        result = _add_engineered_features(wide, long, predict_week=10)

        assert result.loc[0, "Gewogen_t-2"] == pytest.approx(50.0)

    def test_lag5_correct_week(self):
        long = _make_long([
            {"Weeknummer": 5, "Gewogen vooraanmelders": 30.0},
            {"Weeknummer": 10, "Gewogen vooraanmelders": 70.0},
        ])
        wide = _make_wide([{}])

        result = _add_engineered_features(wide, long, predict_week=10)

        assert result.loc[0, "Gewogen_t-5"] == pytest.approx(30.0)

    def test_missing_ref_week_falls_back_to_week1(self):
        """Reference week absent → use week-1 value."""
        long = _make_long([
            {"Weeknummer": 1, "Gewogen vooraanmelders": 5.0},
            {"Weeknummer": 10, "Gewogen vooraanmelders": 70.0},
            # week 8 (lag 2) and week 5 (lag 5) are absent
        ])
        wide = _make_wide([{}])

        result = _add_engineered_features(wide, long, predict_week=10)

        assert result.loc[0, "Gewogen_t-2"] == pytest.approx(5.0)
        assert result.loc[0, "Gewogen_t-5"] == pytest.approx(5.0)

    def test_missing_ref_and_week1_fills_zero_with_warning(self):
        """No reference week and no week-1 data → 0 with UserWarning."""
        long = _make_long([
            {"Weeknummer": 10, "Gewogen vooraanmelders": 70.0},
        ])
        wide = _make_wide([{}])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _add_engineered_features(wide, long, predict_week=10)

        assert result.loc[0, "Gewogen_t-2"] == pytest.approx(0.0)
        assert result.loc[0, "Gewogen_t-5"] == pytest.approx(0.0)
        assert any("week-1 fallback" in str(w.message) for w in caught)

    def test_predict_week_1_clamps_ref_to_week1(self):
        """predict_week=1 → max(1-2,1)=1 → lag uses week 1."""
        long = _make_long([
            {"Weeknummer": 1, "Gewogen vooraanmelders": 99.0},
        ])
        wide = _make_wide([{}])

        result = _add_engineered_features(wide, long, predict_week=1)

        assert result.loc[0, "Gewogen_t-2"] == pytest.approx(99.0)
        assert result.loc[0, "Gewogen_t-5"] == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# Acceleration
# ---------------------------------------------------------------------------

class TestAcceleration:
    def test_acceleration_formula(self):
        """Gewogen_acceleration = (curr - t-2) - (t-2 - t-5)."""
        # curr=70, t-2=50, t-5=30 → (70-50) - (50-30) = 20 - 20 = 0
        long = _make_long([
            {"Weeknummer": 5, "Gewogen vooraanmelders": 30.0},
            {"Weeknummer": 8, "Gewogen vooraanmelders": 50.0},
            {"Weeknummer": 10, "Gewogen vooraanmelders": 70.0},
        ])
        wide = _make_wide([{}])

        result = _add_engineered_features(wide, long, predict_week=10)

        assert result.loc[0, "Gewogen_acceleration"] == pytest.approx(0.0)

    def test_acceleration_positive_when_accelerating(self):
        """Larger gap in second interval → positive acceleration."""
        # curr=100, t-2=40, t-5=30 → (100-40) - (40-30) = 60 - 10 = 50
        long = _make_long([
            {"Weeknummer": 5, "Gewogen vooraanmelders": 30.0},
            {"Weeknummer": 8, "Gewogen vooraanmelders": 40.0},
            {"Weeknummer": 10, "Gewogen vooraanmelders": 100.0},
        ])
        wide = _make_wide([{}])

        result = _add_engineered_features(wide, long, predict_week=10)

        assert result.loc[0, "Gewogen_acceleration"] == pytest.approx(50.0)

    def test_acceleration_negative_when_decelerating(self):
        """Smaller gap vs first interval → negative acceleration."""
        # curr=35, t-2=30, t-5=10 → (35-30) - (30-10) = 5 - 20 = -15
        long = _make_long([
            {"Weeknummer": 5, "Gewogen vooraanmelders": 10.0},
            {"Weeknummer": 8, "Gewogen vooraanmelders": 30.0},
            {"Weeknummer": 10, "Gewogen vooraanmelders": 35.0},
        ])
        wide = _make_wide([{}])

        result = _add_engineered_features(wide, long, predict_week=10)

        assert result.loc[0, "Gewogen_acceleration"] == pytest.approx(-15.0)


# ---------------------------------------------------------------------------
# Exclusivity ratio
# ---------------------------------------------------------------------------

class TestExclusivityRatio:
    def test_ratio_computed_correctly(self):
        long = _make_long([
            {
                "Weeknummer": 10,
                "Gewogen vooraanmelders": 100.0,
                "Ongewogen vooraanmelders": 200.0,
                "Aantal aanmelders met 1 aanmelding": 50.0,
            }
        ])
        wide = _make_wide([{}])

        result = _add_engineered_features(wide, long, predict_week=10)

        # 50 / (200 + 1e-8) ≈ 0.25
        assert result.loc[0, "exclusivity_ratio"] == pytest.approx(50.0 / (200.0 + 1e-8))

    def test_ratio_zero_unweighted_does_not_divide_by_zero(self):
        long = _make_long([
            {
                "Weeknummer": 10,
                "Gewogen vooraanmelders": 5.0,
                "Ongewogen vooraanmelders": 0.0,
                "Aantal aanmelders met 1 aanmelding": 0.0,
            }
        ])
        wide = _make_wide([{}])

        result = _add_engineered_features(wide, long, predict_week=10)

        assert np.isfinite(result.loc[0, "exclusivity_ratio"])

    def test_ratio_missing_week_fills_zero(self):
        """No predict_week rows in long → ratio = 0 / (0 + EPS) ≈ 0."""
        long = _make_long([
            {"Weeknummer": 9, "Ongewogen vooraanmelders": 100.0}
        ])
        wide = _make_wide([{}])

        result = _add_engineered_features(wide, long, predict_week=10)

        assert result.loc[0, "exclusivity_ratio"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Output columns
# ---------------------------------------------------------------------------

class TestOutputColumns:
    def test_all_four_features_present(self):
        long = _make_long([
            {"Weeknummer": 5, "Gewogen vooraanmelders": 10.0},
            {"Weeknummer": 8, "Gewogen vooraanmelders": 20.0},
            {"Weeknummer": 10, "Gewogen vooraanmelders": 30.0,
             "Ongewogen vooraanmelders": 25.0,
             "Aantal aanmelders met 1 aanmelding": 5.0},
        ])
        wide = _make_wide([{}])

        result = _add_engineered_features(wide, long, predict_week=10)

        for col in ["Gewogen_t-2", "Gewogen_t-5", "Gewogen_acceleration", "exclusivity_ratio"]:
            assert col in result.columns, f"missing column: {col}"

    def test_no_intermediate_columns_leaked(self):
        """Helper columns like _gewogen_curr must not appear in the output."""
        long = _make_long([{"Weeknummer": 10, "Gewogen vooraanmelders": 30.0}])
        wide = _make_wide([{}])

        result = _add_engineered_features(wide, long, predict_week=10)

        for col in result.columns:
            assert not col.startswith("_"), f"leaked helper column: {col}"

    def test_multiple_groups_independent(self):
        """Each group gets its own lag values, not cross-contaminated."""
        long = _make_long([
            {"Croho groepeernaam": "B Opleiding A", "Weeknummer": 8, "Gewogen vooraanmelders": 10.0},
            {"Croho groepeernaam": "B Opleiding A", "Weeknummer": 10, "Gewogen vooraanmelders": 20.0},
            {"Croho groepeernaam": "B Opleiding B", "Weeknummer": 8, "Gewogen vooraanmelders": 80.0},
            {"Croho groepeernaam": "B Opleiding B", "Weeknummer": 10, "Gewogen vooraanmelders": 90.0},
        ])
        wide = _make_wide([
            {"Croho groepeernaam": "B Opleiding A"},
            {"Croho groepeernaam": "B Opleiding B"},
        ])

        result = _add_engineered_features(wide, long, predict_week=10)

        row_a = result[result["Croho groepeernaam"] == "B Opleiding A"].iloc[0]
        row_b = result[result["Croho groepeernaam"] == "B Opleiding B"].iloc[0]
        assert row_a["Gewogen_t-2"] == pytest.approx(10.0)
        assert row_b["Gewogen_t-2"] == pytest.approx(80.0)
