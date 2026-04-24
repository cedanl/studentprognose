"""Tests for pre- and post-prediction data quality checks."""

import warnings

import pandas as pd
import pytest

from studentprognose.data.prediction_validator import (
    run_pre_prediction_checks,
    _check_decimal_integrity,
    _check_empty_data,
    _check_historical_realism,
)
from studentprognose.output.validator import (
    _check_trend_realism,
    _check_numerus_fixus_caps,
)
from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cumulative(year, week, programme="B Opleiding", herkomst="NL",
                     examentype="Bachelor", gewogen=100.0):
    return pd.DataFrame({
        "Collegejaar": [year],
        "Weeknummer": [week],
        "Croho groepeernaam": [programme],
        "Herkomst": [herkomst],
        "Examentype": [examentype],
        "Gewogen vooraanmelders": [gewogen],
        "Ongewogen vooraanmelders": [gewogen],
        "Aantal aanmelders met 1 aanmelding": [gewogen],
        "Inschrijvingen": [gewogen],
    })


def _make_predictions(year, week, programme="B Opleiding", herkomst="NL",
                      examentype="Bachelor", prediction=100.0):
    return pd.DataFrame({
        "Collegejaar": [year],
        "Weeknummer": [week],
        "Croho groepeernaam": [programme],
        "Herkomst": [herkomst],
        "Examentype": [examentype],
        "Ensemble_prediction": [prediction],
    })


# ---------------------------------------------------------------------------
# _check_decimal_integrity
# ---------------------------------------------------------------------------

class TestCheckDecimalIntegrity:
    def test_numeric_column_passes(self):
        df = _make_cumulative(2024, 10)
        _check_decimal_integrity(df, 2024, 10)  # no exception

    def test_comma_decimal_exits(self):
        df = _make_cumulative(2024, 10)
        df["Gewogen vooraanmelders"] = ["100,5"]
        with pytest.raises(SystemExit):
            _check_decimal_integrity(df, 2024, 10)

    def test_non_numeric_string_exits(self):
        df = _make_cumulative(2024, 10)
        df["Gewogen vooraanmelders"] = df["Gewogen vooraanmelders"].astype(str)
        df["Gewogen vooraanmelders"] = ["abc"]
        with pytest.raises(SystemExit):
            _check_decimal_integrity(df, 2024, 10)

    def test_missing_column_is_skipped(self):
        df = pd.DataFrame({"Collegejaar": [2024]})
        _check_decimal_integrity(df, 2024, 10)  # no exception


# ---------------------------------------------------------------------------
# _check_empty_data
# ---------------------------------------------------------------------------

class TestCheckEmptyData:
    def test_non_empty_passes(self):
        df = _make_cumulative(2024, 10)
        _check_empty_data(df, 2024, 10)  # no exception

    def test_empty_dataframe_exits(self):
        df = pd.DataFrame()
        with pytest.raises(SystemExit):
            _check_empty_data(df, 2024, 10)


# ---------------------------------------------------------------------------
# _check_historical_realism
# ---------------------------------------------------------------------------

class TestCheckHistoricalRealism:
    def _make_pair(self, curr_val, last_val):
        curr = _make_cumulative(2024, 10, gewogen=curr_val)
        last = _make_cumulative(2023, 10, gewogen=last_val)
        return curr, last

    def test_normal_deviation_passes_silently(self):
        curr, last = self._make_pair(100, 105)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_historical_realism(curr, last, {}, 2024, 10)
        assert not w

    def test_moderate_deviation_warns(self):
        curr, last = self._make_pair(50, 100)  # 50% relative, 50 absolute
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_historical_realism(curr, last, {}, 2024, 10)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)

    def test_extreme_deviation_hard_stops(self):
        curr, last = self._make_pair(5, 100)  # 95% relative, 95 absolute — above max(25, 70)
        with pytest.raises(SystemExit):
            _check_historical_realism(curr, last, {}, 2024, 10)

    def test_extreme_deviation_with_yes_warns_instead(self):
        curr, last = self._make_pair(5, 100)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_historical_realism(curr, last, {}, 2024, 10, yes=True)
        assert len(w) == 1
        assert "--yes" in str(w[0].message).lower() or "[--yes]" in str(w[0].message)

    def test_nf_bachelor_is_skipped(self):
        curr, last = self._make_pair(5, 100)
        nf_list = {"B Opleiding": 50}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_historical_realism(curr, last, nf_list, 2024, 10)
        assert not w

    def test_empty_last_year_passes(self):
        curr = _make_cumulative(2024, 10)
        last = pd.DataFrame()
        _check_historical_realism(curr, last, {}, 2024, 10)  # no exception

    def test_yes_flag_propagates_through_run_pre_prediction_checks(self):
        data = pd.concat([
            _make_cumulative(2024, 10, gewogen=5),
            _make_cumulative(2023, 10, gewogen=100),
        ])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_pre_prediction_checks(data, 2024, 10, {}, yes=True)
        assert len(w) == 1


# ---------------------------------------------------------------------------
# _check_trend_realism (post-prediction)
# ---------------------------------------------------------------------------

class TestCheckTrendRealism:
    def test_large_yoy_deviation_warns(self):
        curr_preds = _make_predictions(2024, 10, prediction=5)
        last_year_cum = _make_cumulative(2023, 10, gewogen=100)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_trend_realism(curr_preds, curr_preds, last_year_cum, 2024, 10)
        assert len(w) >= 1

    def test_normal_yoy_deviation_no_warn(self):
        curr_preds = _make_predictions(2024, 10, prediction=100)
        last_year_cum = _make_cumulative(2023, 10, gewogen=105)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_trend_realism(curr_preds, curr_preds, last_year_cum, 2024, 10)
        assert not w

    def test_wow_guard_skips_first_week_of_cycle(self):
        """Week FINAL_ACADEMIC_WEEK+1 should not compare WoW against week FINAL_ACADEMIC_WEEK."""
        first_cycle_week = FINAL_ACADEMIC_WEEK + 1
        data = pd.concat([
            _make_predictions(2024, first_cycle_week, prediction=100),
            _make_predictions(2024, FINAL_ACADEMIC_WEEK, prediction=1),  # would trigger WoW warn
        ])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_trend_realism(
                data[data["Weeknummer"] == first_cycle_week],
                data,
                None,
                2024,
                first_cycle_week,
            )
        assert not any("week-op-week" in str(warning.message) for warning in w)

    def test_wow_guard_skips_week_1(self):
        data = pd.concat([
            _make_predictions(2024, 1, prediction=100),
        ])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_trend_realism(
                data[data["Weeknummer"] == 1],
                data,
                None,
                2024,
                1,
            )
        assert not any("week-op-week" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# _check_numerus_fixus_caps
# ---------------------------------------------------------------------------

class TestCheckNumerusFixusCaps:
    def test_below_cap_no_warn(self):
        preds = _make_predictions(2024, 10, prediction=40)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_numerus_fixus_caps(preds, {"B Opleiding": 50}, 2024, 10)
        assert not w

    def test_above_cap_warns(self):
        preds = _make_predictions(2024, 10, prediction=60)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_numerus_fixus_caps(preds, {"B Opleiding": 50}, 2024, 10)
        assert len(w) == 1

    def test_premaster_excluded_from_cap(self):
        preds = _make_predictions(2024, 10, examentype="Pre-master", prediction=200)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_numerus_fixus_caps(preds, {"B Opleiding": 50}, 2024, 10)
        assert not w

    def test_empty_nf_list_skips_check(self):
        preds = _make_predictions(2024, 10, prediction=9999)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_numerus_fixus_caps(preds, {}, 2024, 10)
        assert not w
