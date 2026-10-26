"""Tests voor het CWF-model (confidence-weighted forecast, issue #308).

Dekt het puntmodel, het predictie-interval, het betrouwbaarheidslabel, de
datavoorbereiding (vulgraad zonder look-ahead, herkomst Onbekend), het
strategie-contract, de configuratievalidatie en de pipeline-integratie. Alle data
is synthetisch.
"""

import json
from importlib import resources

import numpy as np
import pandas as pd
import pytest

from studentprognose.cli import PipelineConfig
from studentprognose.config import cwf_params_errors, load_configuration
from studentprognose.models.cwf import (
    LABEL_GREEN,
    LABEL_ORANGE,
    LABEL_RED,
    RELIABILITY_LABELS,
    ConfidenceWeightedForecast,
    CWFParams,
    reliability_label,
)
from studentprognose.models.cwf_interval import PredictionIntervalForecaster
from studentprognose.output import postprocessor as pp_module
from studentprognose.strategies import create_strategy
from studentprognose.strategies.cwf import (
    CWF_LOWER_COL,
    CWF_OUTPUT_COLUMNS,
    CWF_PREDICTION_COL,
    CWF_RELIABILITY_COL,
    CWF_UPPER_COL,
    CWFStrategy,
    apply_numerus_fixus_caps,
    build_cwf_history,
    redistribute_unknown_origin,
)
from studentprognose.utils.weeks import DataOption

YEARS = list(range(2019, 2026))
WEEKS = [8, 9, 10, 11, 12]


# ---------------------------------------------------------------------------
# Synthetische data
# ---------------------------------------------------------------------------


def _cfg(**cwf_params):
    cfg = {
        "numerus_fixus": {},
        "model_config": {"min_training_year": 2016},
        "column_roles": {"programme": "Croho groepeernaam"},
        "runtime": {"cpu_count": 1},
    }
    if cwf_params:
        cfg["model_config"]["cwf_params"] = cwf_params
    return cfg


def _synthetic(programmes=("Opleiding A", "Opleiding B", "Opleiding C", "Opleiding D"), seed=1):
    """Cumulatieve data (loaderformaat) en studentaantallen met een stabiele ratio."""
    rng = np.random.default_rng(seed)
    cum_rows, count_rows = [], []
    for p_idx, programme in enumerate(programmes):
        for origin in ("NL", "EER"):
            base = (60 if origin == "NL" else 20) * (1 + 0.3 * p_idx)
            for year in YEARS:
                final = base * rng.uniform(0.9, 1.1)
                count_rows.append({
                    "Collegejaar": year,
                    "Croho groepeernaam": programme,
                    "Herkomst": origin,
                    "Examentype": "Bachelor",
                    "Aantal_studenten": round(final),
                })
                for i, week in enumerate(WEEKS):
                    # Aanmeldingen groeien door de weken heen naar ~2x de instroom.
                    weighted = final * 2 * (0.4 + 0.15 * i)
                    cum_rows.append({
                        "Korte naam instelling": "INST",
                        "Collegejaar": year,
                        "Weeknummer rapportage": week,
                        "Weeknummer": week,
                        "Faculteit": "Faculteit X",
                        "Type hoger onderwijs": "Bachelor",
                        "Groepeernaam Croho": programme,
                        "Naam Croho opleiding Nederlands": programme,
                        "Croho": programme,
                        "Herinschrijving": "Nee",
                        "Hogerejaars": "Nee",
                        "Herkomst": origin,
                        "Gewogen vooraanmelders": weighted,
                        "Ongewogen vooraanmelders": weighted * 1.3,
                        "Aantal aanmelders met 1 aanmelding": weighted * 0.5,
                        "Inschrijvingen": 0.0,
                    })
    return pd.DataFrame(cum_rows), pd.DataFrame(count_rows)


def _strategy(cfg=None, cum=None, counts=None):
    if cum is None or counts is None:
        cum, counts = _synthetic()
    strategy = CWFStrategy(
        cum, counts, cfg or _cfg(), None, None, "/tmp", DataOption.CWF, None,
    )
    strategy.preprocess()
    return strategy


# ---------------------------------------------------------------------------
# Puntmodel
# ---------------------------------------------------------------------------


class TestRatioFactor:
    def test_constant_ratios_give_that_ratio_and_zero_spread(self):
        model = ConfidenceWeightedForecast()
        factor, confidence, stability, cv = model.estimate_factor(
            [0.5, 0.5, 0.5, 0.5], [2021, 2022, 2023, 2024]
        )
        assert factor == pytest.approx(0.5)
        assert cv == pytest.approx(0.0)
        assert stability == pytest.approx(1.0)
        assert confidence == pytest.approx(1.0)

    def test_recent_years_weigh_more(self):
        model = ConfidenceWeightedForecast()
        factor, *_ = model.estimate_factor([0.4, 0.8], [2020, 2024])
        assert factor > np.sqrt(0.4 * 0.8)

    def test_few_years_reduce_confidence(self):
        model = ConfidenceWeightedForecast()
        _, conf_one, *_ = model.estimate_factor([0.5], [2024])
        _, conf_four, *_ = model.estimate_factor([0.5, 0.52, 0.48, 0.5], [2021, 2022, 2023, 2024])
        assert conf_one < conf_four

    def test_empty_input_gives_nan(self):
        factor, confidence, _, _ = ConfidenceWeightedForecast().estimate_factor([], [])
        assert np.isnan(factor)
        assert confidence == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            ConfidenceWeightedForecast().estimate_factor([0.5, 0.6], [2024])


class TestCountPrior:
    def test_single_point_is_returned(self):
        assert ConfidenceWeightedForecast().get_prior([42], [2024]) == 42

    def test_strong_trend_is_extrapolated_damped(self):
        prior = ConfidenceWeightedForecast().get_prior([10, 20, 30, 40], [2021, 2022, 2023, 2024])
        assert prior == pytest.approx(40 + 0.5 * 10)

    def test_outlier_last_year_uses_weighted_mean(self):
        model = ConfidenceWeightedForecast()
        prior = model.get_prior([50, 51, 49, 50, 120], [2020, 2021, 2022, 2023, 2024])
        assert 50 < prior < 120

    def test_predefined_prior_only_without_own_history(self):
        model = ConfidenceWeightedForecast()
        assert model.get_prior([10], [2024], pre_defined_prior=99, has_own_history=False) == 99
        assert model.get_prior([10], [2024], pre_defined_prior=99, has_own_history=True) == 10


class TestShrinkage:
    def test_equal_weight_at_k_own_years(self):
        model = ConfidenceWeightedForecast(CWFParams(pool_shrinkage_k=3.0))
        shrunk = model._shrink_toward_pool(0.4, 0.9, n_own=3)
        assert shrunk == pytest.approx(np.sqrt(0.4 * 0.9))

    def test_missing_pool_keeps_own_factor(self):
        assert ConfidenceWeightedForecast()._shrink_toward_pool(0.4, np.nan, 3) == 0.4


def _own(fill_rate=1.0, ratio=0.5, counts=(50, 52, 48, 50)):
    years = [2021, 2022, 2023, 2024][-len(counts):]
    counts = np.asarray(counts, dtype=float)
    pipeline = counts / ratio
    return pd.DataFrame({
        "Collegejaar": years,
        "Croho groepeernaam": "Opleiding A",
        "Aantal_studenten": counts,
        "Pipeline": pipeline,
        "Ratio": counts / pipeline,
        "Vulgraad": fill_rate,
    })


class TestPredict:
    def test_full_season_follows_ratio_signal(self):
        out = ConfidenceWeightedForecast().predict(_own(fill_rate=1.0), 200.0, np.nan)
        assert out["prediction"] == 100
        assert out["signal_weight"] == pytest.approx(1.0)

    def test_empty_season_follows_prior(self):
        out = ConfidenceWeightedForecast().predict(_own(fill_rate=0.0), 400.0, np.nan)
        assert out["prediction"] == 50
        assert out["signal_weight"] == pytest.approx(0.0)

    def test_prediction_is_integer(self):
        out = ConfidenceWeightedForecast().predict(_own(fill_rate=0.5), 123.0, np.nan)
        assert isinstance(out["prediction"], int)


# ---------------------------------------------------------------------------
# Interval
# ---------------------------------------------------------------------------


class TestInterval:
    def setup_method(self):
        self.model = ConfidenceWeightedForecast()
        self.interval = PredictionIntervalForecaster(self.model)

    def test_kish_effective_n(self):
        assert self.interval._kish_effective_n([1, 1, 1, 1]) == pytest.approx(4)
        assert self.interval._kish_effective_n([10, 1]) < 2

    def test_error_interval_contains_anchor(self):
        lower, upper = self.interval.error_interval([0.1, -0.05, 0.08, -0.12], [1, 1, 1, 1], 100)
        assert lower < 100 < upper

    def test_error_interval_needs_two_errors(self):
        assert all(np.isnan(self.interval.error_interval([0.1], [1], 100)))

    def test_yoy_interval_needs_two_changes(self):
        assert all(np.isnan(self.interval.yoy_log_change_interval([50, 55], [2023, 2024], 50)))

    def test_count_noise_interval_is_poisson_width(self):
        lower, upper = self.interval.count_noise_interval(100)
        assert lower < 100 < upper
        assert upper - lower == pytest.approx(2 * 1.96 * 10, abs=2)

    def test_prediction_interval_brackets_prediction(self):
        own = _own(fill_rate=0.8)
        full = own[["Collegejaar", "Aantal_studenten"]]
        lower, upper = self.interval.prediction_interval(
            50, "Opleiding A", own, own.iloc[0:0], full, current_fill_rate=0.8,
        )
        assert lower <= 50 <= upper


# ---------------------------------------------------------------------------
# Betrouwbaarheidslabel
# ---------------------------------------------------------------------------


class TestReliabilityLabel:
    params = CWFParams()

    def _label(self, **kw):
        base = {
            "prediction": 100, "fill_rate": 0.95, "own_error_count": 4,
            "own_worst_error": 0.1, "own_mean_error": 0.05,
            "used_pool_fallback": False, "exam_type": "Bachelor",
        }
        base.update(kw)
        return reliability_label(self.params, **base)

    def test_green_when_late_and_accurate(self):
        assert self._label() == LABEL_GREEN

    def test_pool_fallback_is_red(self):
        assert self._label(used_pool_fallback=True) == LABEL_RED

    def test_early_cycle_is_red(self):
        assert self._label(fill_rate=0.1) == LABEL_RED

    def test_too_few_own_years_is_red(self):
        assert self._label(own_error_count=2) == LABEL_RED

    def test_large_errors_are_red(self):
        assert self._label(own_worst_error=0.9) == LABEL_RED

    def test_mid_cycle_is_orange(self):
        assert self._label(fill_rate=0.5) == LABEL_ORANGE

    def test_pre_master_has_stricter_green(self):
        assert self._label(exam_type="Pre-master", own_worst_error=0.12) == LABEL_ORANGE

    def test_hysteresis_keeps_green(self):
        assert self._label(own_worst_error=0.18) == LABEL_ORANGE
        assert self._label(own_worst_error=0.18, previous_label=LABEL_GREEN) == LABEL_GREEN

    def test_hysteresis_keeps_orange_above_red_threshold(self):
        assert self._label(fill_rate=0.5, own_worst_error=0.75) == LABEL_RED
        assert (
            self._label(fill_rate=0.5, own_worst_error=0.75, previous_label=LABEL_ORANGE)
            == LABEL_ORANGE
        )


# ---------------------------------------------------------------------------
# Parameters en configuratie
# ---------------------------------------------------------------------------


class TestParams:
    def test_bundled_config_matches_dataclass_defaults(self):
        bundled = json.loads(
            resources.files("studentprognose.configuration")
            .joinpath("configuration.json")
            .read_text(encoding="utf-8")
        )
        assert bundled["model_config"]["cwf_params"] == CWFParams.defaults_as_dict()

    def test_from_config_converts_year_keys(self):
        params = CWFParams.from_config(_cfg(year_weight_overrides={"2021": 0.5}))
        assert params.year_weight_overrides == {2021: 0.5}

    def test_defaults_are_valid(self):
        assert cwf_params_errors(CWFParams.defaults_as_dict()) == []

    @pytest.mark.parametrize(
        "params",
        [
            {"bestaat_niet": 1},
            {"confidence_level": 1.5},
            {"max_group_programmes": 2.5},
            {"cap_smooth": "ja"},
            {"training_start_year": "2020"},
            {"year_weight_overrides": {"twintig": 1}},
            {"predefined_priors": {"Opleiding A": {"Bachelor": {"NL": -1}}}},
        ],
    )
    def test_invalid_params_give_errors(self, params):
        assert cwf_params_errors(params)

    def test_load_configuration_exits_on_invalid_params(self, tmp_path):
        path = tmp_path / "configuration.json"
        path.write_text(json.dumps({"model_config": {"cwf_params": {"ratio_floor": 0}}}))
        with pytest.raises(SystemExit) as exc:
            load_configuration(str(path))
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# Datavoorbereiding
# ---------------------------------------------------------------------------


class TestDataPreparation:
    def test_fill_rate_has_no_look_ahead(self):
        cum, counts = _synthetic()
        strategy = _strategy(cum=cum.copy(), counts=counts)
        changed = cum.copy()
        changed.loc[changed["Collegejaar"] == 2025, "Gewogen vooraanmelders"] *= 5
        other = _strategy(cum=changed, counts=counts)

        def _fill(s):
            h = s.history
            return h[h["Collegejaar"] == 2025].sort_values(
                ["Croho groepeernaam", "Herkomst", "Weeknummer"]
            )["Vulgraad"].to_numpy()

        np.testing.assert_allclose(_fill(strategy), _fill(other))

    def test_fill_rate_is_between_zero_and_one(self):
        fill = _strategy().history["Vulgraad"]
        assert fill.between(0, 1).all()

    def test_unknown_origin_is_redistributed_with_whole_headcounts(self):
        df = pd.DataFrame({
            "Collegejaar": [2024, 2024],
            "Weeknummer": [10, 10],
            "Faculteit": ["F", "F"],
            "Examentype": ["Bachelor", "Bachelor"],
            "Herkomst": ["NL", "Onbekend"],
            "Croho groepeernaam": ["Opleiding A", "Opleiding A"],
            "Gewogen vooraanmelders": [10.0, 3.0],
            "Ongewogen vooraanmelders": [12.0, 5.0],
            "Aantal aanmelders met 1 aanmelding": [6.0, 2.0],
            "Inschrijvingen": [0.0, 4.0],
        })
        out = redistribute_unknown_origin(df)
        assert set(out["Herkomst"]) == {"NL", "EER", "Niet-EER"}
        for col in ("Gewogen vooraanmelders", "Ongewogen vooraanmelders", "Inschrijvingen"):
            assert out[col].sum() == pytest.approx(df[col].sum())
        assert (out["Ongewogen vooraanmelders"] % 1 == 0).all()
        assert out.loc[out["Herkomst"] == "NL", "Gewogen vooraanmelders"].iloc[0] == pytest.approx(11)

    def test_history_has_ratio_and_pipeline(self):
        cum, counts = _synthetic()
        from studentprognose.strategies.cumulative import prepare_cumulative_data

        history = build_cwf_history(prepare_cumulative_data(cum), counts, 38, 0.8)
        row = history.iloc[0]
        assert row["Pipeline"] == pytest.approx(row["Gewogen vooraanmelders"] + row["Inschrijvingen"])
        assert row["Ratio"] == pytest.approx(row["Aantal_studenten"] / row["Pipeline"])


# ---------------------------------------------------------------------------
# Strategie-contract
# ---------------------------------------------------------------------------


class TestCWFStrategyContract:
    def test_preprocess_returns_standard_cumulative_frame(self):
        strategy = CWFStrategy(*_synthetic(), _cfg(), None, None, "/tmp", DataOption.CWF, None)
        result = strategy.preprocess()
        assert isinstance(result, pd.DataFrame)
        # Geen modelkolommen: de postprocessor merget dit frame met de studentaantallen.
        assert "Aantal_studenten" not in result.columns
        assert "Gewogen vooraanmelders" in result.columns

    def test_predict_returns_frame_with_cwf_columns(self):
        result = _strategy().predict_nr_of_students(2025, 10, 0)
        assert isinstance(result, pd.DataFrame)
        for col in CWF_OUTPUT_COLUMNS:
            assert col in result.columns
        assert len(result) == 8

    def test_predictions_are_consistent(self):
        result = _strategy().predict_nr_of_students(2025, 10, 0)
        assert (result[CWF_LOWER_COL] <= result[CWF_PREDICTION_COL]).all()
        assert (result[CWF_PREDICTION_COL] <= result[CWF_UPPER_COL]).all()
        assert (result[CWF_PREDICTION_COL] % 1 == 0).all()
        assert result[CWF_RELIABILITY_COL].isin(RELIABILITY_LABELS).all()

    def test_predictions_are_close_on_stable_data(self):
        cum, counts = _synthetic()
        result = _strategy(cum=cum, counts=counts).predict_nr_of_students(2025, 12, 0)
        actual = result.merge(counts, on=["Collegejaar", "Croho groepeernaam", "Herkomst", "Examentype"])
        mape = ((actual[CWF_PREDICTION_COL] - actual["Aantal_studenten"]).abs()
                / actual["Aantal_studenten"]).mean()
        assert mape < 0.15

    def test_unknown_week_returns_none(self):
        assert _strategy().predict_nr_of_students(2025, 30, 0) is None

    def test_skip_years_is_rejected(self):
        with pytest.raises(ValueError):
            _strategy().predict_nr_of_students(2025, 10, 1)

    def test_new_programme_uses_predefined_prior_and_is_red(self):
        cum, counts = _synthetic()
        new = cum[(cum["Groepeernaam Croho"] == "Opleiding A") & (cum["Collegejaar"] == 2025)].copy()
        new["Groepeernaam Croho"] = "Opleiding Nieuw"
        cum = pd.concat([cum, new], ignore_index=True)
        cfg = _cfg(predefined_priors={"Opleiding Nieuw": {"Bachelor": {"NL": 40, "EER": 15}}})
        result = _strategy(cfg=cfg, cum=cum, counts=counts).predict_nr_of_students(2025, 10, 0)
        rows = result[result["Croho groepeernaam"] == "Opleiding Nieuw"]
        assert len(rows) == 2
        assert rows[CWF_PREDICTION_COL].notna().all()
        assert (rows[CWF_RELIABILITY_COL] == LABEL_RED).all()

    def test_programme_without_own_counts_borrows_from_pool(self):
        cum, counts = _synthetic()
        counts = counts[counts["Croho groepeernaam"] != "Opleiding A"]
        strategy = _strategy(cum=cum, counts=counts)
        with pytest.warns(UserWarning, match="geen eigen historie"):
            result = strategy.predict_nr_of_students(2025, 10, 0)
        rows = result[result["Croho groepeernaam"] == "Opleiding A"]
        assert rows[CWF_PREDICTION_COL].notna().all()
        assert (rows[CWF_RELIABILITY_COL] == LABEL_RED).all()

    def test_combination_without_any_counts_gets_no_prediction(self):
        cum, counts = _synthetic()
        counts = counts[counts["Herkomst"] != "EER"]
        strategy = _strategy(cum=cum, counts=counts)
        with pytest.warns(UserWarning, match="geen enkele gerealiseerde instroom"):
            result = strategy.predict_nr_of_students(2025, 10, 0)
        eer = result[result["Herkomst"] == "EER"]
        assert eer[CWF_PREDICTION_COL].isna().all()
        assert (eer[CWF_RELIABILITY_COL] == LABEL_RED).all()
        assert result.loc[result["Herkomst"] == "NL", CWF_PREDICTION_COL].notna().all()

    def test_previous_week_label_feeds_hysteresis(self):
        strategy = _strategy()
        first = strategy.predict_nr_of_students(2025, 10, 0)
        strategy.postprocessor.data_latest = first.assign(cwf_betrouwbaarheid=LABEL_GREEN)
        labels = strategy._previous_labels(2025, 11)
        assert len(labels) == len(first)
        assert set(labels.values()) == {LABEL_GREEN}
        # De eerste week van het academisch jaar heeft geen vorige week.
        assert strategy._previous_labels(2025, strategy.final_academic_week + 1) == {}

    def test_filtering_limits_combinations(self):
        strategy = _strategy()
        strategy.set_filtering(["Opleiding B"], ["NL"], [])
        result = strategy.predict_nr_of_students(2025, 10, 0)
        assert list(result["Croho groepeernaam"]) == ["Opleiding B"]
        assert list(result["Herkomst"]) == ["NL"]


class TestNumerusFixus:
    def test_caps_total_and_upper_bounds(self):
        result = pd.DataFrame({
            "Croho groepeernaam": ["Opleiding A"] * 3,
            "Examentype": ["Bachelor"] * 3,
            "Herkomst": ["NL", "EER", "Niet-EER"],
            CWF_PREDICTION_COL: [80.0, 30.0, 10.0],
            CWF_LOWER_COL: [70.0, 25.0, 8.0],
            CWF_UPPER_COL: [95.0, 40.0, 14.0],
        })
        capped = apply_numerus_fixus_caps(result, {"Opleiding A": 100})
        assert capped[CWF_PREDICTION_COL].sum() == 100
        assert capped[CWF_UPPER_COL].sum() <= 100
        assert (capped[CWF_LOWER_COL] <= capped[CWF_PREDICTION_COL]).all()
        assert (capped[CWF_PREDICTION_COL] <= capped[CWF_UPPER_COL]).all()

    def test_under_cap_is_untouched(self):
        result = pd.DataFrame({
            "Croho groepeernaam": ["Opleiding A"],
            "Examentype": ["Bachelor"],
            "Herkomst": ["NL"],
            CWF_PREDICTION_COL: [50.0],
            CWF_LOWER_COL: [40.0],
            CWF_UPPER_COL: [60.0],
        })
        pd.testing.assert_frame_equal(apply_numerus_fixus_caps(result, {"Opleiding A": 100}), result)


# ---------------------------------------------------------------------------
# Pipeline-integratie
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def test_output_column_lists_are_in_sync(self):
        assert pp_module.CWF_OUTPUT_COLUMNS == CWF_OUTPUT_COLUMNS

    def test_create_strategy_returns_cwf_strategy(self):
        cum, counts = _synthetic()
        cfg = PipelineConfig(data_option=DataOption.CWF)
        strategy = create_strategy(cfg, (None, cum, counts, None, None), _cfg(), "/tmp")
        assert isinstance(strategy, CWFStrategy)

    def test_create_strategy_requires_student_counts(self):
        cum, _ = _synthetic()
        cfg = PipelineConfig(data_option=DataOption.CWF)
        with pytest.raises(ValueError):
            create_strategy(cfg, (None, cum, None, None, None), _cfg(), "/tmp")

    def test_create_strategy_rejects_skip_years(self):
        cum, counts = _synthetic()
        cfg = PipelineConfig(data_option=DataOption.CWF, skip_years=1)
        with pytest.raises(ValueError):
            create_strategy(cfg, (None, cum, counts, None, None), _cfg(), "/tmp")

    def test_filename_suffix(self):
        assert DataOption.CWF.filename_suffix == "cwf"

    def test_api_run_returns_cwf_columns(self, tmp_path):
        from studentprognose import run_pipeline_from_dataframes
        from studentprognose.config import load_defaults

        cum, counts = _synthetic()
        configuration = load_defaults()
        configuration["runtime"]["cpu_count"] = 1
        result = run_pipeline_from_dataframes(
            year=2025,
            week=10,
            data_cumulative=cum,
            data_student_numbers=counts,
            configuration=configuration,
            dataset=DataOption.CWF,
            cwd=str(tmp_path),
            save_output=False,
        )
        rows = result[(result["Collegejaar"] == 2025) & (result["Weeknummer"] == 10)]
        assert rows[CWF_PREDICTION_COL].notna().all()
        assert "MAE_Prognose_cwf" in result.columns
        assert "SARIMA_cumulative" not in result.columns

    def test_api_rejects_tune_for_cwf(self, tmp_path):
        from studentprognose import run_pipeline_from_dataframes

        cum, counts = _synthetic()
        with pytest.raises(ValueError, match="CWF"):
            run_pipeline_from_dataframes(
                year=2025, week=10, data_cumulative=cum, data_student_numbers=counts,
                dataset=DataOption.CWF, cwd=str(tmp_path), save_output=False, tune=True,
            )


class TestBenchmark:
    def test_backtest_covers_completed_years_and_summarises(self):
        from studentprognose.benchmark.evaluate_cwf import backtest_cwf, summarise_cwf

        cum, counts = _synthetic()
        results = backtest_cwf(cum, counts, _cfg(), predict_week=10)
        assert sorted(results["test_year"].unique()) == [2021, 2022, 2023, 2024, 2025]
        assert {"Aantal_studenten", "Naief_vorig_jaar", CWF_PREDICTION_COL} <= set(results.columns)

        by_year, by_label = summarise_cwf(results)
        assert by_year.iloc[0]["test_year"] == "alle"
        assert by_year.iloc[0]["n"] == len(results)
        assert 0 <= by_year.iloc[0]["coverage"] <= 1
        assert set(by_label["label"]) <= set(RELIABILITY_LABELS)

    def test_report_is_written(self, tmp_path):
        from studentprognose.benchmark.evaluate_cwf import backtest_cwf, summarise_cwf
        from studentprognose.benchmark.report import generate_cwf_report

        cum, counts = _synthetic()
        by_year, by_label = summarise_cwf(backtest_cwf(cum, counts, _cfg(), predict_week=10))
        generate_cwf_report(by_year, by_label, 10, str(tmp_path))
        html = (tmp_path / "benchmark" / "benchmark_cwf.html").read_text("utf-8")
        assert "Benchmark — CWF-model" in html
        assert "Per betrouwbaarheidslabel" in html
