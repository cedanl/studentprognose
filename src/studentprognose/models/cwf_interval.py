"""Predictie-interval voor het CWF-model.

Het interval is een blend van twee onafhankelijke schattingen:

- een **jaar-op-jaar-interval** uit de historische veranderingen in de
  gerealiseerde instroom, gelezen als percentage (log-veranderingen) én als
  aantal studenten;
- een **foutinterval** uit hoe ver de ratioschatter er bij leave-one-out
  backtests op eerdere jaren naast zat, voor deze combinatie zelf en voor
  vergelijkbare opleidingen uit de pool.

Vroeg in de aanmeldcyclus domineert het jaar-op-jaar-interval (de ratio heeft
dan weinig signaal), later het foutinterval.

Gebaseerd op het model van Radboud Institutional Research
(MIT-licentie, © 2026 Radboud Institutional Research; zie ``NOTICE``).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

from studentprognose.models.cwf import (
    FILL_RATE_COL,
    INPUT_VALUE_COL,
    PROGRAMME_COL,
    RATIO_COL,
    STUDENT_COUNT_COL,
    YEAR_COL,
    ConfidenceWeightedForecast,
    CWFParams,
)

logger = logging.getLogger(__name__)


class PredictionIntervalForecaster:
    """Bouwt het predictie-interval rond een CWF-puntvoorspelling.

    Args:
        forecaster: Het puntmodel; zijn ``estimate_factor`` en
            ``_select_reference_group`` worden hergebruikt voor de backtests.
        params: Modelparameters; ``None`` gebruikt de defaults.
    """

    # Maakt de MAD een consistente schatter van de std bij normaliteit.
    MAD_SCALE = 1.4826

    def __init__(
        self, forecaster: ConfidenceWeightedForecast, params: CWFParams | None = None
    ):
        p = params or CWFParams()
        self.forecaster = forecaster
        # Gewicht van een eigen backtest-fout t.o.v. een poolfout.
        self.own_error_weight = p.own_error_weight
        # Verbreding van het foutinterval: zonder schaal dekt de MAD-marge zijn
        # nominale 95% niet.
        self.margin_scale = p.margin_scale
        # Poolfouten worden op dit kwantiel van |fout| gewinsoriseerd, zodat één
        # wild jaar van één vergelijkbare opleiding het interval niet bepaalt.
        self.pool_outlier_quantile = p.pool_outlier_quantile
        # Alleen fouten met minstens dit gewicht mogen de cap volledig zetten.
        self.cap_min_weight = p.cap_min_weight
        self.yoy_margin_scale = p.yoy_margin_scale
        # Ondergrens van de jaar-op-jaar-marge: de Poisson-ruis van het
        # verwachte aantal (z / sqrt(n) relatief, z · sqrt(n) absoluut).
        self.yoy_poisson_floor_z = p.yoy_poisson_floor_z
        # Aandeel van de absolute (studenten-)grens in het jaar-op-jaar-interval;
        # de rest is de relatieve (log-)grens.
        self.yoy_absolute_weight = p.yoy_absolute_weight
        self.pool_min_candidates = p.pool_min_candidates
        self.pool_winsor_min_n = p.pool_winsor_min_n
        # Zonder eigen backtest-fouten wordt het blendgewicht tot deze macht
        # verheven: het poolfoutinterval telt dan pas laat in de cyclus mee.
        self.pool_only_blend_power = p.pool_only_blend_power
        self.pool_yoy_min_candidates = p.pool_yoy_min_candidates
        # Eigen fouten krijgen dezelfde recency-decay en jaargewichten als de
        # ratio's in het puntmodel.
        self.own_error_recency = p.own_error_recency
        # Elke fout telt in de cap mee naar rato van zijn gewicht, zodat de grens
        # niet verspringt zodra een gewicht over ``cap_min_weight`` gaat.
        self.cap_smooth = p.cap_smooth
        # De gewogen mediaan van |fout| wordt niet lager dan dit veelvoud van de
        # mediaan van de poolfouten: drie toevallig kleine eigen fouten mogen het
        # interval niet laten instorten.
        self.pool_spread_floor = p.pool_spread_floor
        # Een log-verandering tussen twee kleine aantallen is vooral ruis; zijn
        # invloed op de cap schaalt met min(N_t-1, N_t) / dit aantal.
        self.yoy_level_weight_count = p.yoy_level_weight_count
        # Spreidingsvloer en cap-vloer komen uit de dichtstbijzijnde
        # pool-opleidingen, aangevuld tot dit aantal op similariteit.
        self.pool_floor_min_candidates = int(p.pool_floor_min_candidates)

    # ------------------------------------------------------------------
    # Jaar-op-jaar-interval
    # ------------------------------------------------------------------

    @staticmethod
    def _own_year_series(counts, years):
        """(counts, years) met alleen eindige, positieve aantallen, gesorteerd op jaar."""
        counts = np.asarray(counts, dtype=float)
        years = np.asarray(years, dtype=float)
        valid = np.isfinite(counts) & np.isfinite(years) & (counts > 0)
        counts, years = counts[valid], years[valid]
        order = np.argsort(years)
        return counts[order], years[order]

    @staticmethod
    def _yoy_changes(counts, years, transform=np.log):
        """Veranderingen tussen opeenvolgende jaren op de ``transform``-schaal.

        Niet-aansluitende jaren (een gat in de reeks) worden overgeslagen.
        """
        changes = [
            transform(counts[i]) - transform(counts[i - 1])
            for i in range(1, len(years))
            if years[i] - years[i - 1] == 1
        ]
        return np.asarray(changes, dtype=float)

    @staticmethod
    def _yoy_change_levels(counts, years):
        """Het kleinste van de twee aantallen achter elke verandering."""
        levels = [
            min(counts[i], counts[i - 1])
            for i in range(1, len(years))
            if years[i] - years[i - 1] == 1
        ]
        return np.asarray(levels, dtype=float)

    def _yoy_margin(self, changes, floor, confidence_level=0.95, weights=None):
        """Robuuste marge uit een steekproef jaar-op-jaar-veranderingen.

        ``margin = min(t · MAD · sqrt(1 + 1/n), grootste afwijking)``, daarna
        minimaal ``floor``. ``weights`` (in [0, 1]) wegen alleen de cap. Geeft
        NaN bij minder dan 2 veranderingen.
        """
        changes = np.asarray(changes, dtype=float)
        if weights is None:
            changes = changes[np.isfinite(changes)]
            if len(changes) < 2:
                return np.nan
            med = np.median(changes)
            abs_devs = np.abs(changes - med)
            cap = np.max(abs_devs)
        else:
            weights = np.asarray(weights, dtype=float)
            valid = np.isfinite(changes) & np.isfinite(weights)
            changes, weights = changes[valid], np.clip(weights[valid], 0.0, 1.0)
            if len(changes) < 2:
                return np.nan
            med = np.median(changes)
            abs_devs = np.abs(changes - med)
            cap = np.max(weights * abs_devs)

        n_eff = float(len(changes))
        mad = self.MAD_SCALE * np.median(abs_devs)
        alpha = 1.0 - confidence_level
        t_crit = tdist.ppf(1.0 - alpha / 2.0, df=n_eff - 1)
        margin = self.yoy_margin_scale * t_crit * mad * np.sqrt(1 + 1.0 / n_eff)
        return max(min(margin, cap), floor)

    def _yoy_interval_from_series(self, series, anchor, confidence_level=0.95):
        """Jaar-op-jaar-interval rond ``anchor`` uit één of meer (counts, years)-reeksen.

        Het gemiddelde (gewicht ``yoy_absolute_weight``) van een relatieve grens
        ``[a·e^-m, a·e^m]`` en een absolute grens ``[max(0, a-m), a+m]``.
        Geeft ``(nan, nan)`` bij minder dan 2 veranderingen of een ongeldig anker.
        """
        if not np.isfinite(anchor) or anchor <= 0:
            return np.nan, np.nan

        log_changes, abs_changes, levels = [], [], []
        for counts, years in series:
            c, y = self._own_year_series(counts, years)
            log_changes.append(self._yoy_changes(c, y, np.log))
            abs_changes.append(self._yoy_changes(c, y, lambda x: x))
            levels.append(self._yoy_change_levels(c, y))
        empty = np.array([], dtype=float)
        log_changes = np.concatenate(log_changes) if log_changes else empty
        abs_changes = np.concatenate(abs_changes) if abs_changes else empty
        levels = np.concatenate(levels) if levels else empty

        log_weights = None
        if self.yoy_level_weight_count > 0 and len(levels) == len(log_changes):
            log_weights = np.minimum(1.0, levels / self.yoy_level_weight_count)
            if np.all(log_weights >= 1.0):
                log_weights = None

        m_log = self._yoy_margin(
            log_changes,
            self.yoy_poisson_floor_z / np.sqrt(anchor),
            confidence_level,
            weights=log_weights,
        )
        m_abs = self._yoy_margin(
            abs_changes,
            self.yoy_poisson_floor_z * np.sqrt(anchor),
            confidence_level,
        )
        if not (np.isfinite(m_log) and np.isfinite(m_abs)):
            return np.nan, np.nan

        lower_log, upper_log = anchor * np.exp(-m_log), anchor * np.exp(m_log)
        lower_abs, upper_abs = max(0.0, anchor - m_abs), anchor + m_abs

        w = min(max(self.yoy_absolute_weight, 0.0), 1.0)
        return w * lower_abs + (1 - w) * lower_log, w * upper_abs + (1 - w) * upper_log

    def count_noise_interval(self, anchor):
        """Interval uit alleen de Poisson-telruis rond ``anchor``.

        Terugval wanneer er te weinig historie is voor zowel het jaar-op-jaar- als
        het foutinterval: dezelfde ondergrenzen die het jaar-op-jaar-interval
        gebruikt (relatief ``z / sqrt(a)``, absoluut ``z · sqrt(a)``), gemiddeld
        met ``yoy_absolute_weight``. Zegt alleen iets over toevalsvariatie, niet
        over modelfouten, en is dus een ondergrens voor de werkelijke onzekerheid.
        """
        if not np.isfinite(anchor) or anchor <= 0:
            return np.nan, np.nan
        m_log = self.yoy_poisson_floor_z / np.sqrt(anchor)
        m_abs = self.yoy_poisson_floor_z * np.sqrt(anchor)
        w = min(max(self.yoy_absolute_weight, 0.0), 1.0)
        lower = w * max(0.0, anchor - m_abs) + (1 - w) * anchor * np.exp(-m_log)
        upper = w * (anchor + m_abs) + (1 - w) * anchor * np.exp(m_log)
        return np.round(lower, 0), np.round(upper, 0)

    def yoy_log_change_interval(self, counts, years, anchor, confidence_level=0.95):
        """Jaar-op-jaar-interval uit de eigen instroomhistorie, rond ``anchor``."""
        return self._yoy_interval_from_series(
            [(counts, years)],
            anchor,
            confidence_level=confidence_level,
        )

    # ------------------------------------------------------------------
    # Leave-one-out backtest-fouten
    # ------------------------------------------------------------------

    def _leave_one_out_log_errors(self, years, counts, inputs):
        """Leave-one-out log-fouten van de ratioschatter op één reeks.

        Voor elk jaar wordt de ratiofactor geschat uit alle andere jaren;
        ``fout_t = log(N_t / (pipeline_t · R_(-t)))``. Pas vanaf 3 geldige rijen:
        met 2 zijn de twee fouten elkaars exacte tegengestelde.

        Returns:
            ``(errors, years)``.
        """
        years = np.asarray(years, dtype=float)
        counts = np.asarray(counts, dtype=float)
        inputs = np.asarray(inputs, dtype=float)

        n = len(years)
        valid_rows = (
            np.isfinite(years)
            & np.isfinite(counts)
            & (counts > 0)
            & np.isfinite(inputs)
            & (inputs > 0)
        )
        if valid_rows.sum() < 3:
            return np.array([], dtype=float), np.array([], dtype=float)

        ratios = np.full(n, np.nan)
        ratios[valid_rows] = counts[valid_rows] / inputs[valid_rows]

        errors, error_years = [], []
        for i in range(n):
            if not valid_rows[i]:
                continue
            held_out = np.arange(n) != i
            other_ratios, other_years = ratios[held_out], years[held_out]
            valid = np.isfinite(other_ratios) & np.isfinite(other_years)
            if not valid.any():
                continue

            factor, _, _, _ = self.forecaster.estimate_factor(
                other_ratios[valid], other_years[valid]
            )
            if not np.isfinite(factor) or factor <= 0:
                continue
            predicted = inputs[i] * factor
            if not np.isfinite(predicted) or predicted <= 0:
                continue

            errors.append(np.log(counts[i] / predicted))
            error_years.append(years[i])

        return np.asarray(errors, dtype=float), np.asarray(error_years, dtype=float)

    def _own_prediction_errors(self, own, programme=None, recency_alpha=None):
        """Eigen leave-one-out-fouten, gewogen met de vulgraad (en recency).

        Een jaar waarvan de pipeline in deze week nog nauwelijks bestond, heeft
        een lage vulgraad en dus weinig gewicht: zijn fout is vooral ruis van de
        kleine noemer.

        Returns:
            ``(errors, years, weights)``.
        """
        errors, error_years = self._leave_one_out_log_errors(
            own[YEAR_COL].to_numpy(dtype=float),
            own[STUDENT_COUNT_COL].to_numpy(dtype=float),
            own[INPUT_VALUE_COL].to_numpy(dtype=float),
        )

        fill_rate_by_year = dict(
            zip(
                own[YEAR_COL].to_numpy(dtype=float),
                own[FILL_RATE_COL].to_numpy(dtype=float),
            )
        )
        weights = np.array(
            [fill_rate_by_year.get(year, np.nan) for year in error_years], dtype=float
        )
        weights = np.where(np.isfinite(weights), np.clip(weights, 0.0, 1.0), 0.0)

        if self.own_error_recency and len(error_years) > 0:
            alpha = (
                self.forecaster.variance_alpha
                if recency_alpha is None
                else recency_alpha
            )
            overrides = self.forecaster.year_weight_overrides or {}
            delta = np.max(error_years) - error_years
            recency = np.exp(-alpha * delta) * np.array(
                [overrides.get(int(y), 1.0) for y in error_years], dtype=float
            )
            weights = weights * recency

        return errors, error_years, weights

    def _select_pool_candidates(
        self,
        programme,
        own,
        pool,
        anchor_ratio=np.nan,
        anchor_size=np.nan,
        min_candidates=None,
    ):
        """De pool-opleidingen die vergelijkbaar zijn met deze combinatie.

        Het doel van de similariteit is de eigen ratiofactor en mediane grootte,
        of — zonder eigen historie — het anker uit de ingestelde prior.

        Returns:
            ``(years, counts, inputs)``, één entry per geselecteerde opleiding.
        """
        if pool is None or pool.empty or programme is None:
            return [], [], []

        if np.isfinite(anchor_ratio) and np.isfinite(anchor_size) and anchor_size > 0:
            own_ratio, own_size = anchor_ratio, anchor_size
        else:
            own_ratio, _, _, _ = self.forecaster.estimate_factor(
                own[RATIO_COL].to_numpy(dtype=float),
                own[YEAR_COL].to_numpy(dtype=float),
            )
            own_counts = own[STUDENT_COUNT_COL].to_numpy(dtype=float)
            own_size = (
                np.nanmedian(own_counts) if np.any(np.isfinite(own_counts)) else np.nan
            )

        _, selected_years, selected_counts, selected_inputs = (
            self.forecaster._select_reference_group(
                programme=programme,
                programme_ratio=own_ratio,
                programme_size=own_size,
                programmes_group=pool[PROGRAMME_COL].to_numpy(dtype=object),
                years_group=pool[YEAR_COL].to_numpy(dtype=float),
                counts_group=pool[STUDENT_COUNT_COL].to_numpy(dtype=float),
                input_values_group=pool[INPUT_VALUE_COL].to_numpy(dtype=float),
                min_candidates=(
                    self.pool_min_candidates
                    if min_candidates is None
                    else min_candidates
                ),
            )
        )
        return selected_years, selected_counts, selected_inputs

    def _pool_prediction_errors(
        self,
        programme,
        own,
        pool,
        anchor_ratio=np.nan,
        anchor_size=np.nan,
        min_candidates=None,
    ):
        """Leave-one-out-fouten van alle vergelijkbare pool-opleidingen, samengevoegd
        en gewinsoriseerd op ``pool_outlier_quantile``.

        Returns:
            ``(errors, years)``.
        """
        selected_years, selected_counts, selected_inputs = self._select_pool_candidates(
            programme,
            own,
            pool,
            anchor_ratio=anchor_ratio,
            anchor_size=anchor_size,
            min_candidates=min_candidates,
        )
        if not selected_years:
            return np.array([], dtype=float), np.array([], dtype=float)

        per_candidate = [
            self._leave_one_out_log_errors(y, c, i)
            for y, c, i in zip(selected_years, selected_counts, selected_inputs)
        ]
        errors = np.concatenate([e for e, _ in per_candidate])
        error_years = np.concatenate([y for _, y in per_candidate])

        finite = errors[np.isfinite(errors)]
        if self.pool_outlier_quantile < 1.0 and len(finite) >= self.pool_winsor_min_n:
            cap = np.quantile(np.abs(finite), self.pool_outlier_quantile)
            errors = np.clip(errors, -cap, cap)

        return errors, error_years

    @staticmethod
    def _weighted_median(values, weights):
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        order = np.argsort(values)
        values, weights = values[order], weights[order]
        cum_weights = np.cumsum(weights)
        idx = min(np.searchsorted(cum_weights, weights.sum() / 2.0), len(values) - 1)
        return values[idx]

    @staticmethod
    def _kish_effective_n(weights):
        """Kish' effectieve steekproefgrootte ``(Σw)² / Σw²``."""
        weights = np.asarray(weights, dtype=float)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            return 0.0
        return (weight_sum**2) / np.sum(weights**2)

    def error_interval(
        self,
        errors,
        weights,
        anchor,
        confidence_level=0.95,
        spread_floor=np.nan,
        cap_floor=np.nan,
    ) -> tuple[float, float]:
        """Foutinterval rond ``anchor`` uit gewogen leave-one-out log-fouten.

        ``mad = 1.4826 · gewogen mediaan(|fout|)`` (minimaal ``spread_floor``),
        ``n`` = Kish' effectieve n, ``marge = min(margin_scale · t · mad ·
        sqrt(1 + 1/n), cap)`` met als cap de grootste (gewogen) waargenomen fout
        (minimaal ``cap_floor``). Spreiding wordt gemeten rond nul, niet rond de
        mediaan: een structureel scheve schatter krijgt zo een breder interval.

        Returns:
            ``(lower, upper)``, of ``(nan, nan)`` bij te weinig fouten of n_eff < 2.
        """
        if not np.isfinite(anchor) or anchor <= 0:
            return np.nan, np.nan

        errors = np.asarray(errors, dtype=float)
        weights = np.asarray(weights, dtype=float)
        valid = np.isfinite(errors) & np.isfinite(weights) & (weights > 0)
        errors, weights = errors[valid], weights[valid]
        if len(errors) < 2:
            return np.nan, np.nan

        abs_errors = np.abs(errors)
        median_abs = self._weighted_median(abs_errors, weights)
        if np.isfinite(spread_floor):
            median_abs = max(median_abs, spread_floor)
        mad = self.MAD_SCALE * median_abs

        n_eff = self._kish_effective_n(weights)
        if n_eff < 2:
            return np.nan, np.nan

        alpha = 1.0 - confidence_level
        t_crit = tdist.ppf(1.0 - alpha / 2.0, df=n_eff - 1)
        margin_log = self.margin_scale * t_crit * mad * np.sqrt(1 + 1.0 / n_eff)

        # t_crit explodeert als n_eff → 1; de marge mag niet verder reiken dan
        # de grootste fout die echt is waargenomen.
        if self.cap_smooth:
            eligibility = np.minimum(1.0, weights / max(self.cap_min_weight, 1e-12))
            cap = np.max(eligibility * abs_errors)
        else:
            reliable = weights >= self.cap_min_weight
            cap = np.max(abs_errors[reliable] if reliable.any() else abs_errors)
        if np.isfinite(cap_floor):
            cap = max(cap, cap_floor)
        margin_log = min(margin_log, cap)

        return anchor * np.exp(-margin_log), anchor * np.exp(margin_log)

    def own_error_track_record(self, programme, own) -> tuple[int, float, float]:
        """Eigen trackrecord voor het betrouwbaarheidslabel.

        Returns:
            ``(aantal, grootste, gemiddelde)`` van de eigen leave-one-out-fouten
            met positief gewicht; het gemiddelde is gewogen zoals het puntmodel
            zijn jaren weegt. ``(0, nan, nan)`` zonder eigen historie.
        """
        if own is None or own.empty:
            return 0, np.nan, np.nan

        errors, _, fill_weights = self._own_prediction_errors(own, programme)
        weights = self.own_error_weight * fill_weights
        valid = np.isfinite(errors) & np.isfinite(weights) & (weights > 0)
        if not valid.any():
            return 0, np.nan, np.nan

        _, _, mean_weights = self._own_prediction_errors(
            own, programme, recency_alpha=self.forecaster.mean_alpha
        )
        mean_valid = valid & np.isfinite(mean_weights) & (mean_weights > 0)
        weighted_mean = (
            float(
                np.sum(mean_weights[mean_valid] * np.abs(errors[mean_valid]))
                / np.sum(mean_weights[mean_valid])
            )
            if mean_valid.any()
            else np.nan
        )
        return int(valid.sum()), float(np.max(np.abs(errors[valid]))), weighted_mean

    # ------------------------------------------------------------------
    # Blend en publieke ingang
    # ------------------------------------------------------------------

    @staticmethod
    def _blend_intervals(yoy_lower, yoy_upper, error_lower, error_upper, fill_rate):
        """``grens = f · fout + (1 - f) · jaar-op-jaar``; valt terug op de kant
        die eindig is als de andere ontbreekt."""
        yoy_ok = np.isfinite(yoy_lower) and np.isfinite(yoy_upper)
        error_ok = np.isfinite(error_lower) and np.isfinite(error_upper)
        if not yoy_ok and not error_ok:
            return np.nan, np.nan
        if not yoy_ok:
            return error_lower, error_upper
        if not error_ok:
            return yoy_lower, yoy_upper

        f = 0.0 if not np.isfinite(fill_rate) else min(max(fill_rate, 0.0), 1.0)
        return f * error_lower + (1 - f) * yoy_lower, f * error_upper + (
            1 - f
        ) * yoy_upper

    def prediction_interval(
        self,
        prediction: float,
        programme: object,
        own: pd.DataFrame,
        pool: pd.DataFrame,
        full_own: pd.DataFrame,
        confidence_level: float = 0.95,
        current_fill_rate: float = np.nan,
        pre_defined_prior: float = np.nan,
        current_student_pipeline: float = np.nan,
    ) -> tuple[float, float]:
        """Predictie-interval rond ``prediction``.

        Args:
            prediction: De puntvoorspelling (het anker van beide intervallen).
            programme: Opleidingssleutel.
            own: Eigen historie in de voorspelweek; leeg zonder eigen historie.
            pool: Historie van de vergelijkingspool.
            full_own: Alle gerealiseerde instroomaantallen van deze combinatie
                (``Collegejaar``, ``Aantal_studenten``), ook van jaren vóór de
                trainingsperiode.
            confidence_level: Nominale dekking.
            current_fill_rate: Blendgewicht tussen fout- en jaar-op-jaar-interval.
            pre_defined_prior: Anker voor de poolselectie zonder eigen historie.
            current_student_pipeline: Huidige pipeline, voor datzelfde anker.

        Returns:
            ``(lower, upper)`` afgerond, of ``(nan, nan)``.
        """
        lower, upper = self.yoy_log_change_interval(
            full_own[STUDENT_COUNT_COL].to_numpy(dtype=float),
            full_own[YEAR_COL].to_numpy(dtype=float),
            anchor=prediction,
            confidence_level=confidence_level,
        )

        anchor_ratio = anchor_size = np.nan
        if (
            own.empty
            and np.isfinite(pre_defined_prior)
            and pre_defined_prior > 0
            and np.isfinite(current_student_pipeline)
            and current_student_pipeline > 0
        ):
            anchor_ratio = pre_defined_prior / current_student_pipeline
            anchor_size = pre_defined_prior

        if not (np.isfinite(lower) and np.isfinite(upper)):
            # Minder dan 2 eigen veranderingen: leen die van vergelijkbare opleidingen.
            pool_years, pool_counts, _ = self._select_pool_candidates(
                programme,
                own,
                pool,
                anchor_ratio=anchor_ratio,
                anchor_size=anchor_size,
                min_candidates=max(
                    self.pool_min_candidates, self.pool_yoy_min_candidates
                ),
            )
            lower, upper = self._yoy_interval_from_series(
                list(zip(pool_counts, pool_years)),
                prediction,
                confidence_level=confidence_level,
            )

        if own.empty:
            own_errors = own_fill_weights = np.array([], dtype=float)
        else:
            own_errors, _, own_fill_weights = self._own_prediction_errors(
                own, programme
            )
        pool_errors, _ = self._pool_prediction_errors(
            programme,
            own,
            pool,
            anchor_ratio=anchor_ratio,
            anchor_size=anchor_size,
        )

        combined_errors = np.concatenate([own_errors, pool_errors])
        combined_weights = np.concatenate(
            [
                self.own_error_weight * own_fill_weights,
                np.ones(len(pool_errors)),
            ]
        )

        spread_floor = cap_floor = np.nan
        floor_errors = pool_errors
        if self.pool_floor_min_candidates > 0:
            floor_errors, _ = self._pool_prediction_errors(
                programme,
                own,
                pool,
                anchor_ratio=anchor_ratio,
                anchor_size=anchor_size,
                min_candidates=self.pool_floor_min_candidates,
            )
        finite_pool = floor_errors[np.isfinite(floor_errors)]
        if self.pool_spread_floor > 0 and len(finite_pool) >= self.pool_winsor_min_n:
            spread_floor = self.pool_spread_floor * float(
                np.median(np.abs(finite_pool))
            )
        if (
            self.pool_floor_min_candidates > 0
            and len(finite_pool) >= self.pool_winsor_min_n
        ):
            cap_floor = float(np.max(np.abs(finite_pool)))

        error_lower, error_upper = self.error_interval(
            combined_errors,
            combined_weights,
            anchor=prediction,
            confidence_level=confidence_level,
            spread_floor=spread_floor,
            cap_floor=cap_floor,
        )

        blend_weight = current_fill_rate
        if len(own_errors) == 0 and np.isfinite(blend_weight):
            blend_weight = (
                min(max(blend_weight, 0.0), 1.0) ** self.pool_only_blend_power
            )

        blended_lower, blended_upper = self._blend_intervals(
            lower,
            upper,
            error_lower,
            error_upper,
            blend_weight,
        )
        logger.debug(
            "CWF interval | %s | yoy=[%s, %s] error=[%s, %s] w=%s -> [%s, %s]",
            programme,
            lower,
            upper,
            error_lower,
            error_upper,
            blend_weight,
            blended_lower,
            blended_upper,
        )

        if not np.isfinite(blended_lower) or not np.isfinite(blended_upper):
            return np.nan, np.nan
        return np.round(blended_lower, 0), np.round(blended_upper, 0)
