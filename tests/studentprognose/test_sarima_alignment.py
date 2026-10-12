"""Regressietests voor #301: de SARIMA-reeks moet precies op ``predict_week``
eindigen en de forecast moet bij de juiste kalenderweken terechtkomen, ook als
er weekkolommen ontbreken.

Vóór de fix werd de reeks afgekapt met ``compute_pred_len`` (alle kalenderweken
tot ``final_week``), terwijl de reeks alleen de bestaande weekkolommen bevat.
Bij ontbrekende weken eindigde de trainingsreeks daardoor vóór ``predict_week``,
schoof de forecast op, en kon bij vroege weken (horizon ~49) de afkapping zelfs
in het vorige collegejaar vallen.
"""

import numpy as np
import pandas as pd
import pytest

from studentprognose.benchmark.evaluate_ts import evaluate_timeseries_model
from studentprognose.data.transforms import transform_data
from studentprognose.models.base import BaseForecaster
from studentprognose.models.sarima import (
    align_forecast_to_calendar,
    predict_with_sarima_cumulative,
    split_time_series,
)
from studentprognose.output.postprocessor import PostProcessor
from studentprognose.utils.weeks import DataOption, compute_pred_len, get_forecast_weeks

_ROW = {"Croho groepeernaam": "TEST", "Herkomst": "NL", "Examentype": "Bachelor"}
_YEARS = (2020, 2021, 2022, 2023, 2024)

# Legacy-kalender (final_week 38) waarin weken 40-42 en 35-38 in ÁLLE jaren
# ontbreken; week 39 wordt als reset-week geïnjecteerd. 45 kolommen per jaar.
_WEEKS_WITH_GAPS = list(range(43, 53)) + list(range(1, 35))


def _value(year: int, week: int) -> float:
    """Unieke, herleidbare waarde per (jaar, week)."""
    return year * 1000.0 + week


def _long_df(weeks, years=_YEARS) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Collegejaar": yr, "Faculteit": "F", "Herkomst": "NL",
            "Examentype": "Bachelor", "Croho groepeernaam": "TEST",
            "ts": _value(yr, wk), "Weeknummer": wk,
        }
        for yr in years
        for wk in weeks
    ])


class _Recorder(BaseForecaster):
    """Legt de trainingsreeks en horizon vast; forecast = 1, 2, 3, ..."""

    def __init__(self, log):
        self.season_length = 52
        self.log = log

    def fit(self, ts_data, exog=None):
        self.log["train"] = np.asarray(ts_data)
        return self

    def forecast(self, steps, exog=None):
        self.log["steps"] = steps
        return np.arange(1, steps + 1, dtype=np.float64)


def _predict(df, year, week, final_week=38):
    log = {}
    pred = predict_with_sarima_cumulative(
        df, _ROW, year, week, compute_pred_len(week, final_week),
        already_printed=True, forecaster_factory=lambda: _Recorder(log),
        final_week=final_week,
    )
    return np.asarray(pred), log


class TestGetForecastWeeks:
    def test_only_existing_columns_after_predict_week(self):
        cols = [str(w) for w in [39] + _WEEKS_WITH_GAPS]
        weeks = get_forecast_weeks(cols, 10)
        assert weeks[0] == "11" and weeks[-1] == "34"
        assert len(weeks) == 24  # i.p.v. compute_pred_len(10) == 28

    def test_missing_predict_week_raises(self):
        with pytest.raises(ValueError, match="Voorspelweek 41"):
            get_forecast_weeks([str(w) for w in _WEEKS_WITH_GAPS], 41)


class TestCumulativeAlignment:
    def test_training_series_ends_exactly_at_predict_week(self):
        _, log = _predict(_long_df(_WEEKS_WITH_GAPS), 2024, 10)
        assert log["train"][-1] == _value(2024, 10)
        assert log["steps"] == 24

    def test_forecast_lands_on_the_right_calendar_weeks(self):
        pred, _ = _predict(_long_df(_WEEKS_WITH_GAPS), 2024, 10)
        assert len(pred) == compute_pred_len(10)  # één waarde per kalenderweek 11..38
        assert pred[0] == 1  # eerste forecast-stap = week 11
        assert pred[34 - 11] == 24  # laatste stap = week 34
        assert np.isnan(pred[35 - 11:]).all()  # weken 35-38 bestaan niet in de data

    def test_early_week_does_not_cut_into_previous_year(self):
        """Week 43 heeft horizon 47 kalenderweken, meer dan de 45 kolommen per
        jaar: de oude afkapping viel in het vorige collegejaar."""
        pred, log = _predict(_long_df(_WEEKS_WITH_GAPS), 2024, 43)
        assert log["train"][-1] == _value(2024, 43)
        assert log["steps"] == 43  # wk44-52 (9) + wk1-34 (34)
        assert np.nanmin(pred) > 0

    def test_full_calendar_unchanged(self):
        """Zonder ontbrekende weken is de horizon gewoon compute_pred_len."""
        weeks = list(range(39, 53)) + list(range(1, 39))
        pred, log = _predict(_long_df(weeks), 2024, 10)
        assert log["steps"] == compute_pred_len(10)
        assert log["train"][-1] == _value(2024, 10)
        assert not np.isnan(pred).any()

    def test_missing_predict_year_row_returns_empty(self):
        """Zonder rij voor het voorspeljaar mag er niet stil in het vorige jaar
        worden afgekapt."""
        df = _long_df(_WEEKS_WITH_GAPS, years=(2020, 2021, 2022, 2023))
        pred, log = _predict(df, 2024, 10)
        assert len(pred) == 0 and "train" not in log

    def test_skip_years_cuts_in_the_skipped_year(self):
        log = {}
        predict_with_sarima_cumulative(
            _long_df(_WEEKS_WITH_GAPS), _ROW, 2024, 10, compute_pred_len(10),
            skip_years=1, already_printed=True,
            forecaster_factory=lambda: _Recorder(log),
        )
        assert log["train"][-1] == _value(2023, 10)

    def test_last_existing_week_returns_all_nan(self):
        """Voorspelweek = laatste bestaande week: niets te voorspellen."""
        pred, log = _predict(_long_df(_WEEKS_WITH_GAPS), 2024, 34)
        assert len(pred) == compute_pred_len(34) and np.isnan(pred).all()
        assert "train" not in log


class TestInteriorGaps:
    def _wide(self, interpolate):
        df = pd.DataFrame([
            {"Collegejaar": 2024, "Faculteit": "F", "Herkomst": "NL",
             "Examentype": "Bachelor", "Croho groepeernaam": "TEST",
             "ts": v, "Weeknummer": w}
            for w, v in [(1, 10.0), (2, 20.0), (4, 40.0), (5, 50.0)]
        ] + [
            # andere opleiding levert de kolommen 3 en 6 aan de union
            {"Collegejaar": 2024, "Faculteit": "F", "Herkomst": "NL",
             "Examentype": "Bachelor", "Croho groepeernaam": "OTHER",
             "ts": 1.0, "Weeknummer": w}
            for w in (1, 2, 3, 4, 5, 6)
        ])
        wide = transform_data(df, "ts", interpolate_interior_gaps=interpolate)
        return wide[wide["Croho groepeernaam"] == "TEST"].iloc[0]

    def test_interior_gap_is_interpolated_trailing_stays_zero(self):
        row = self._wide(interpolate=True)
        assert row["3"] == 30.0
        assert row["6"] == 0.0

    def test_default_keeps_zero_fill(self):
        """Niet-cumulatieve aanroepers (bijv. de Deadline-indicator) blijven 0 vullen."""
        assert self._wide(interpolate=False)["3"] == 0.0

    def test_cumulative_series_has_no_false_zero(self):
        weeks_2024 = [w for w in _WEEKS_WITH_GAPS if w != 5]  # telbestand wk5 ontbreekt
        df = pd.concat([
            _long_df(_WEEKS_WITH_GAPS, years=(2020, 2021, 2022, 2023)),
            _long_df(weeks_2024, years=(2024,)),
        ])
        _, log = _predict(df, 2024, 10)
        train_2024 = log["train"][-(1 + 10 + 10):]  # wk39 + wk43-52 + wk1-10
        assert (train_2024[1:] > 0).all()
        assert _value(2024, 4) < train_2024[-6] < _value(2024, 6)


class TestSplitAndAlignHelpers:
    def test_split_requires_predict_year_as_last_row(self):
        wide = pd.DataFrame({"Collegejaar": [2023], "1": [1.0], "2": [2.0]})
        with pytest.raises(ValueError, match="2024"):
            split_time_series(wide, 2024, 1)

    def test_align_places_values_by_week(self):
        aligned = align_forecast_to_calendar([7.0, 8.0], ["37", "38"], 35)
        assert aligned.tolist()[1:] == [7.0, 8.0] and np.isnan(aligned[0])


def test_benchmark_uses_same_cut_as_production():
    log = {}
    config = {
        "column_roles": {
            "programme": "Croho groepeernaam", "origin": "Herkomst",
            "exam_type": "Examentype", "academic_year": "Collegejaar",
        },
    }
    evaluate_timeseries_model(
        _long_df(_WEEKS_WITH_GAPS), lambda: _Recorder(log), predict_week=10,
        config=config, min_training_year=2016, min_train_years=3,
    )
    assert log["steps"] == 24
    assert log["train"][-1] == _value(2024, 10)


def test_postprocessor_skips_weeks_without_forecast(tmp_path):
    (tmp_path / "data" / "output").mkdir(parents=True)
    pp = PostProcessor(
        configuration={"numerus_fixus": {}}, data_latest=None, ensemble_weights=None,
        data_studentcount=None, cwd=str(tmp_path),
        data_option=DataOption.CUMULATIVE, ci_test_n=None,
    )
    data = pd.DataFrame([{
        "Collegejaar": 2024, "Faculteit": "F", "Examentype": "Bachelor",
        "Herkomst": "NL", "Croho groepeernaam": "TEST", "Weeknummer": 35,
    }])
    out = pp.add_predicted_preregistrations(data, [[np.nan, 7.0, 8.0]])
    added = out.iloc[1:]
    assert added["Weeknummer"].tolist() == [37, 38]
    assert added["Voorspelde vooraanmelders"].tolist() == [7.0, 8.0]


class TestIndividualAlignment:
    """Het individuele spoor gebruikt dezelfde helper. Zijn data heeft altijd een
    volledig weekrooster, dus de horizon blijft daar compute_pred_len."""

    def _wide(self, years):
        from studentprognose.utils.weeks import get_all_weeks_ordered

        rows = []
        for yr in years:
            row = {"Collegejaar": yr, "Faculteit": "F", "Herkomst": "NL",
                   "Examentype": "Bachelor", "Croho groepeernaam": "M TEST"}
            for i, wk in enumerate(get_all_weeks_ordered()):
                row[wk] = float(10 * i)
            rows.append(row)
        return pd.DataFrame(rows)

    def test_full_grid_forecasts_every_calendar_week(self):
        from studentprognose.models.sarima import predict_with_sarima_individual

        row = {"Croho groepeernaam": "M TEST", "Herkomst": "NL", "Examentype": "Bachelor"}
        pred = np.asarray(predict_with_sarima_individual(
            self._wide(range(2019, 2025)), row, 2024, 10, 2024, [], already_printed=True,
        ))
        assert len(pred) == compute_pred_len(10)
        assert np.isfinite(pred).all()

    def test_missing_predict_year_row_returns_empty(self):
        from studentprognose.models.sarima import predict_with_sarima_individual

        row = {"Croho groepeernaam": "M TEST", "Herkomst": "NL", "Examentype": "Bachelor"}
        pred = predict_with_sarima_individual(
            self._wide(range(2019, 2024)), row, 2024, 10, 2023, [], already_printed=True,
        )
        assert len(pred) == 0
