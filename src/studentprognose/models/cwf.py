"""Confidence-weighted forecast (CWF): puntvoorspelling en betrouwbaarheidslabel.

Het CWF-model voorspelt de eerstejaarsinstroom per opleiding × herkomst ×
examentype rechtstreeks uit de cumulatieve vooraanmeldingen:

    voorspelling = w · (pipeline · R) + (1 - w) · prior

met ``pipeline`` = gewogen vooraanmelders + inschrijvingen in de voorspelweek,
``R`` de recency-gewogen (geometrisch gemiddelde) ratio instroom/pipeline in
dezelfde week van eerdere jaren, ``prior`` een count-prior op basis van de
gerealiseerde instroom en ``w`` de vulgraad (hoe ver de aanmeldcyclus
historisch gezien al is) geschaald met de betrouwbaarheid van ``R``. Bij weinig
eigen historie wordt ``R`` naar vergelijkbare opleidingen ("de pool")
getrokken. Het predictie-interval staat in :mod:`studentprognose.models.cwf_interval`.

Gebaseerd op het model van Radboud Institutional Research
(https://github.com/radboudir/studentprognose, MIT-licentie,
© 2026 Radboud Institutional Research).
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Werkkolommen van de CWF-historieframes (één rij per opleiding × jaar).
YEAR_COL = "Collegejaar"
PROGRAMME_COL = "Croho groepeernaam"
STUDENT_COUNT_COL = "Aantal_studenten"
RATIO_COL = "Ratio"
FILL_RATE_COL = "Vulgraad"
INPUT_VALUE_COL = "Pipeline"

# Betrouwbaarheidslabels zoals ze in de output komen.
LABEL_GREEN = "Groen"
LABEL_ORANGE = "Oranje"
LABEL_RED = "Rood"
RELIABILITY_LABELS = (LABEL_GREEN, LABEL_ORANGE, LABEL_RED)


@dataclass
class CWFParams:
    """Alle instelbare parameters van het CWF-model.

    De defaults zijn ook de waarden in ``model_config.cwf_params`` van de
    ingebakken ``configuration.json``. Zie ``docs/configuratie-referentie.md``
    voor de betekenis per parameter.
    """

    # --- Puntvoorspelling (ConfidenceWeightedForecast) ---
    cv_lambda: float = 0.3
    max_history_years: int = 4
    ratio_tolerance: float = 0.15
    size_factor: float = 2.0
    max_group_programmes: int = 15
    trend_r2_threshold: float = 0.85
    trend_damping: float = 0.5
    outlier_z_threshold: float = 2.5
    min_relative_std: float = 0.08
    mean_alpha: float = 0.8
    variance_alpha: float = 0.15
    ratio_floor: float = 1e-3
    pool_shrinkage_k: float = 3.0
    min_trusted_confidence: float = 0.3
    ratio_trend_damping: float = 0.15
    year_weight_overrides: dict = field(default_factory=dict)

    # --- Predictie-interval (PredictionIntervalForecaster) ---
    confidence_level: float = 0.95
    margin_scale: float = 1.6
    own_error_weight: float = 5.0
    pool_outlier_quantile: float = 0.9
    cap_min_weight: float = 1.0
    yoy_margin_scale: float = 1.0
    yoy_poisson_floor_z: float = 1.96
    yoy_absolute_weight: float = 0.5
    pool_min_candidates: int = 0
    pool_winsor_min_n: int = 3
    pool_only_blend_power: float = 2.0
    pool_yoy_min_candidates: int = 3
    own_error_recency: bool = True
    cap_smooth: bool = True
    pool_spread_floor: float = 1.0
    yoy_level_weight_count: float = 5.0
    pool_floor_min_candidates: int = 2

    # --- Data en nabewerking (CWFStrategy) ---
    training_start_year: int | None = None
    fill_rate_recency_alpha: float = 0.8
    redistribute_unknown_origin: bool = True
    upper_bound_cap_fill_rate: float = 0.98
    upper_bound_cap_exam_types: list = field(
        default_factory=lambda: ["Bachelor", "Master"]
    )
    interval_min_half_width: float = 2.0
    predefined_priors: dict = field(default_factory=dict)

    # --- Betrouwbaarheidslabel ---
    reliability_orange_min_fill: float = 0.2
    reliability_green_min_fill: float = 0.9
    reliability_min_own_years: int = 3
    reliability_green_max_own_error: float = 0.15
    reliability_green_max_own_error_by_exam_type: dict = field(
        default_factory=lambda: {"Pre-master": 0.10}
    )
    reliability_red_min_own_error: float = 0.7
    reliability_green_max_own_mean_error: float = 0.075
    reliability_green_leave_own_error_add: float = 0.05
    reliability_red_leave_own_error_add: float = 0.10
    reliability_green_leave_own_mean_error_add: float = 0.025

    @classmethod
    def from_config(cls, configuration: dict | None) -> CWFParams:
        """Bouw de parameters uit ``configuration["model_config"]["cwf_params"]``.

        Ontbrekende sleutels krijgen hun default; onbekende sleutels worden door
        ``config._validate_cwf_params`` al bij het laden afgewezen en hier
        genegeerd.
        """
        raw = ((configuration or {}).get("model_config", {}) or {}).get(
            "cwf_params"
        ) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        params = cls(**{k: v for k, v in raw.items() if k in known})
        # JSON-sleutels zijn altijd strings; jaren moeten als int matchen.
        params.year_weight_overrides = {
            int(year): float(weight)
            for year, weight in params.year_weight_overrides.items()
        }
        return params

    @classmethod
    def defaults_as_dict(cls) -> dict:
        """De defaultwaarden als dict (zelfde vorm als ``cwf_params`` in de config)."""
        return dataclasses.asdict(cls())


class ConfidenceWeightedForecast:
    """Ratio-gebaseerde puntvoorspelling met pool-shrinkage en count-prior.

    De ratiofactor is een recency-gewogen geometrisch gemiddelde van de eigen
    historische ratio's: ratio's zijn strikt positief en multiplicatief, dus één
    uitschieter trekt een geometrisch gemiddelde minder scheef dan een
    rekenkundig gemiddelde, en de spreiding (``cv``, de gewogen std van de
    log-ratio's) is schaalvrij.

    Args:
        params: Modelparameters; ``None`` gebruikt de defaults.
    """

    def __init__(self, params: CWFParams | None = None):
        p = params or CWFParams()
        self.cv_lambda = p.cv_lambda
        self.max_history_years = p.max_history_years
        # Relatieve tolerantie: 0.15 = de ratio van een kandidaat moet binnen
        # 15% van die van de opleiding liggen, ongeacht of die ratio ~1 of ~15 is.
        self.ratio_tolerance = p.ratio_tolerance
        self.size_factor = p.size_factor
        self.max_group_programmes = p.max_group_programmes
        self.trend_r2_threshold = p.trend_r2_threshold
        self.trend_damping = p.trend_damping
        self.outlier_z_threshold = p.outlier_z_threshold
        self.min_relative_std = p.min_relative_std
        self.mean_alpha = p.mean_alpha
        self.variance_alpha = p.variance_alpha
        self.ratio_floor = p.ratio_floor
        self.pool_shrinkage_k = p.pool_shrinkage_k
        self.min_trusted_confidence = p.min_trusted_confidence
        self.ratio_trend_damping = p.ratio_trend_damping
        # Extra vermenigvuldiger per historisch jaar bovenop de exponentiële
        # recency-decay, voor jaren die als geheel afwijkend bekend zijn
        # (bijv. een coronajaar). Jaar zonder entry = 1.0.
        self.year_weight_overrides: dict = dict(p.year_weight_overrides or {})

    # ------------------------------------------------------------------
    # Recency-gewichten en gewogen statistiek
    # ------------------------------------------------------------------

    def _recency_weights(self, years, alpha):
        years = np.asarray(years, dtype=float)
        if len(years) == 0:
            return np.array([], dtype=float)

        weights = np.exp(-alpha * (np.max(years) - years))
        if self.year_weight_overrides:
            weights = weights * np.array(
                [self.year_weight_overrides.get(int(y), 1.0) for y in years]
            )

        weight_sum = weights.sum()
        if weight_sum <= 0 or not np.isfinite(weight_sum):
            return np.ones(len(years)) / len(years)
        return weights / weight_sum

    @staticmethod
    def _weighted_mean(values, weights):
        return np.sum(
            np.asarray(weights, dtype=float) * np.asarray(values, dtype=float)
        )

    @staticmethod
    def _weighted_std(values, weights):
        """Gewogen populatie-std; gewichten sommeren tot 1."""
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        mean = np.sum(weights * values)
        return np.sqrt(np.sum(weights * (values - mean) ** 2))

    def _log_ratio(self, values):
        """Log van de ratio's, eerst afgekapt op ``ratio_floor``.

        Een echte nul-conversie wordt zo een grote negatieve log in plaats van
        ``-inf``: hij trekt het gemiddelde hard omlaag zonder het op te blazen.
        """
        return np.log(np.maximum(np.asarray(values, dtype=float), self.ratio_floor))

    # ------------------------------------------------------------------
    # Pool-shrinkage
    # ------------------------------------------------------------------

    def _pool_reference_factor(self, programme, own_ratio, own_size, pool):
        """Ratiofactor (zelfde tuple als :meth:`estimate_factor`) uit alleen de
        pool-opleidingen die qua ratio en grootte vergelijkbaar zijn.

        Een ruwe pool kan erg heterogeen zijn (bijv. alle masters met dezelfde
        herkomst); eerst filteren houdt de poolfactor informatief. Geeft
        ``(nan, 0.0, 0.0, nan)`` als er geen bruikbare kandidaat is.
        """
        if (
            pool is None
            or pool.empty
            or programme is None
            or PROGRAMME_COL not in pool.columns
            or INPUT_VALUE_COL not in pool.columns
        ):
            return np.nan, 0.0, 0.0, np.nan

        _, selected_years, selected_counts, selected_inputs = (
            self._select_reference_group(
                programme=programme,
                programme_ratio=own_ratio,
                programme_size=own_size,
                programmes_group=pool[PROGRAMME_COL].to_numpy(dtype=object),
                years_group=pool[YEAR_COL].to_numpy(dtype=float),
                counts_group=pool[STUDENT_COUNT_COL].to_numpy(dtype=float),
                input_values_group=pool[INPUT_VALUE_COL].to_numpy(dtype=float),
            )
        )
        if not selected_years:
            return np.nan, 0.0, 0.0, np.nan

        flat_years = np.concatenate(selected_years)
        flat_counts = np.concatenate(selected_counts)
        flat_inputs = np.concatenate(selected_inputs)

        valid = flat_inputs > 0
        flat_ratios = np.full(len(flat_inputs), np.nan)
        flat_ratios[valid] = flat_counts[valid] / flat_inputs[valid]
        return self.estimate_factor(flat_ratios, flat_years)

    def _shrink_toward_pool(self, own_factor, pool_factor, n_own):
        """Empirical-Bayes-blend van eigen en poolfactor in log-ruimte.

        Eigen gewicht = ``n_own / (n_own + pool_shrinkage_k)``: bij
        ``pool_shrinkage_k`` eigen jaren tellen eigen en pool even zwaar.
        """
        if not np.isfinite(own_factor):
            return pool_factor
        if not np.isfinite(pool_factor) or n_own <= 0:
            return own_factor

        own_weight = n_own / (n_own + self.pool_shrinkage_k)
        log_blend = own_weight * self._log_ratio(own_factor) + (
            1 - own_weight
        ) * self._log_ratio(pool_factor)
        return float(np.exp(log_blend))

    # ------------------------------------------------------------------
    # Count-prior
    # ------------------------------------------------------------------

    def _outlier_z(self, last_value, history):
        mean_hist = np.mean(history)
        std_floor = self.min_relative_std * abs(mean_hist)
        effective_std = max(np.std(history), std_floor, 1e-8)
        return (last_value - mean_hist) / effective_std

    def _trend_strength(self, counts, years):
        """Lineaire trend door (years, counts): ``(slope, R²)``.

        ``(0, 0)`` bij minder dan 3 punten of maar één uniek jaar (een lijn door
        één x-waarde is rang-deficiënt).
        """
        if len(counts) < 3 or len(np.unique(years)) < 2:
            return 0.0, 0.0

        slope, intercept = np.polyfit(years, counts, 1)
        fitted = slope * years + intercept
        ss_res = np.sum((counts - fitted) ** 2)
        ss_tot = np.sum((counts - np.mean(counts)) ** 2)
        if ss_tot <= 0:
            return slope, 0.0
        return slope, max(0.0, min(1.0, 1.0 - ss_res / ss_tot))

    def get_prior(
        self, counts_prog, years, pre_defined_prior=None, has_own_history=True
    ):
        """Count-prior: de verwachte instroom op basis van alleen gerealiseerde aantallen.

        - Geen eigen historie en een vooraf ingestelde prior: die prior.
        - Eén punt: dat punt.
        - Sterke trend (R² ≥ ``trend_r2_threshold``): laatste waarde plus
          ``trend_damping`` × helling.
        - Laatste jaar is een uitschieter t.o.v. eerdere jaren: recency-gewogen
          gemiddelde.
        - Anders: de laatste waarde.
        """
        if (
            not has_own_history
            and pre_defined_prior is not None
            and not pd.isna(pre_defined_prior)
        ):
            return pre_defined_prior

        counts = np.asarray(counts_prog, dtype=float)
        years = np.asarray(years, dtype=float)
        valid = np.isfinite(counts) & np.isfinite(years)
        counts, years = counts[valid], years[valid]
        if len(counts) == 0:
            return np.nan

        order = np.argsort(years)
        counts, years = counts[order], years[order]
        last_value = counts[-1]
        if len(counts) == 1:
            return last_value

        slope, strength = self._trend_strength(counts, years)
        if strength >= self.trend_r2_threshold:
            return last_value + self.trend_damping * slope

        if len(counts) >= 3:
            z = self._outlier_z(last_value, counts[:-1])
            if abs(z) >= self.outlier_z_threshold:
                weights = self._recency_weights(years, self.mean_alpha)
                return self._weighted_mean(counts, weights)

        return last_value

    # ------------------------------------------------------------------
    # Ratiofactor
    # ------------------------------------------------------------------

    def estimate_factor(self, values, years) -> tuple[float, float, float, float]:
        """Schat de ratiofactor uit historische ratio's.

        Returns:
            ``(factor, confidence, stability, cv)``. ``factor`` is het
            recency-gewogen geometrisch gemiddelde (met een kleine trendcorrectie
            bij een sterke trend), ``cv`` de gewogen std van de log-ratio's,
            ``stability = exp(-cv / cv_lambda)`` en ``confidence`` de stabiliteit
            maal ``sqrt(min(n / max_history_years, 1))``.
        """
        values = np.asarray(values, dtype=float)
        years = np.asarray(years)
        if len(values) != len(years):
            raise ValueError(
                f"Factor/jaar-mismatch: {len(values)} waarden vs {len(years)} jaren"
            )

        mask = np.isfinite(values) & np.isfinite(years)
        values, years = values[mask], years[mask]
        if len(values) == 0:
            return np.nan, 0.0, 0.0, np.nan
        if np.all(values == 1):
            return 1.0, 1.0, 1.0, 0.0

        mean_weights = self._recency_weights(years, self.mean_alpha)
        variance_weights = self._recency_weights(years, self.variance_alpha)
        log_values = self._log_ratio(values)

        factor = np.exp(self._weighted_mean(log_values, mean_weights))

        # Het gewogen gemiddelde loopt achter op een consistente meerjarige
        # trend; alleen bij een duidelijke trend een klein deel ervan meenemen.
        slope, strength = self._trend_strength(log_values, years)
        if strength >= self.trend_r2_threshold:
            factor = np.exp(np.log(factor) + self.ratio_trend_damping * slope)

        cv = (
            0.0
            if len(values) == 1
            else self._weighted_std(log_values, variance_weights)
        )
        stability = np.exp(-cv / self.cv_lambda)
        sample_confidence = np.sqrt(min(len(values) / self.max_history_years, 1.0))
        return factor, stability * sample_confidence, stability, cv

    # ------------------------------------------------------------------
    # Puntvoorspelling
    # ------------------------------------------------------------------

    def predict(
        self,
        own: pd.DataFrame,
        current_student_pipeline: float,
        pre_defined_prior: float,
        pool: pd.DataFrame | None = None,
        has_own_history: bool = True,
        programme: object = None,
    ) -> dict:
        """Puntvoorspelling voor één combinatie.

        Args:
            own: Historie van deze combinatie (één rij per jaar met
                ``Collegejaar``, ``Ratio``, ``Vulgraad``, ``Aantal_studenten``).
                Heeft de opleiding geen eigen historie, dan is dit de pool.
            current_student_pipeline: Gewogen vooraanmelders + inschrijvingen in
                de voorspelweek van het voorspeljaar.
            pre_defined_prior: Vooraf ingestelde prior (alleen gebruikt zonder
                eigen historie), of NaN.
            pool: Overige opleidingen in de vergelijkingspool (met
                ``Croho groepeernaam`` en ``Pipeline``), voor de shrinkage.
            has_own_history: ``False`` als ``own`` de pool is.
            programme: Opleidingssleutel, voor de selectie en logging.

        Returns:
            Dict met ``prediction`` (int), ``ratio_factor``, ``confidence``,
            ``stability``, ``baseline`` (pipeline × laatste ratio) en
            ``signal_weight`` (het gewicht dat het ratiosignaal kreeg).
        """
        counts_prog = own[STUDENT_COUNT_COL]
        years = own[YEAR_COL].to_numpy(dtype=float)
        ratios = own[RATIO_COL].to_numpy(dtype=float)
        fill_rates = own[FILL_RATE_COL].to_numpy(dtype=float)

        prior = self.get_prior(counts_prog, years, pre_defined_prior, has_own_history)
        weight = self._weighted_mean(
            fill_rates, self._recency_weights(years, self.mean_alpha)
        )

        R, confidence, stability, _ = self.estimate_factor(ratios, years)
        n_own = int(np.sum(np.isfinite(ratios) & np.isfinite(years)))

        counts_prog_arr = counts_prog.to_numpy(dtype=float)
        own_size = (
            np.nanmedian(counts_prog_arr)
            if np.any(np.isfinite(counts_prog_arr))
            else np.nan
        )

        if has_own_history:
            pool_factor, _, _, _ = self._pool_reference_factor(
                programme, R, own_size, pool
            )
            R = self._shrink_toward_pool(R, pool_factor, n_own)
        else:
            # ``own`` is hier de hele pool: R daaruit is een gemiddelde over veel
            # ongerelateerde opleidingen. Met een ingestelde prior zoeken we
            # opleidingen die lijken op wat die prior voorspelt en nemen hun
            # ratio over.
            if (
                pre_defined_prior is not None
                and np.isfinite(pre_defined_prior)
                and np.isfinite(current_student_pipeline)
                and current_student_pipeline > 0
            ):
                anchor_ratio = pre_defined_prior / current_student_pipeline
                filtered = self._pool_reference_factor(
                    programme, anchor_ratio, pre_defined_prior, pool
                )
                if np.isfinite(filtered[0]):
                    R, confidence, stability, _ = filtered
            else:
                # Geen prior ingesteld: een count-prior over de hele pool is
                # betekenisloos. Gebruik dan alleen het ratiosignaal van de pool.
                prior = np.nan

        # Onder ``min_trusted_confidence`` krijgt het ratiosignaal naar rato
        # minder gewicht: de vulgraad alleen weet niet dat R onbetrouwbaar is.
        confidence_factor = (
            min(1.0, confidence / self.min_trusted_confidence)
            if self.min_trusted_confidence > 0
            else 1.0
        )
        effective_weight = weight * confidence_factor

        if not np.isfinite(R):
            # Geen ratiosignaal (bijv. elk eerder jaar had nog nul aanmelders in
            # deze week): val terug op de prior, of 0 zonder enige historie.
            prediction = prior if np.isfinite(prior) else 0
        elif not np.isfinite(prior):
            prediction = current_student_pipeline * R
            effective_weight = 1.0 if np.isfinite(prediction) else 0.0
        else:
            signal = current_student_pipeline * R
            prediction = effective_weight * signal + (1 - effective_weight) * prior

        baseline = np.nan
        valid = np.isfinite(ratios) & np.isfinite(years)
        if valid.any() and np.isfinite(current_student_pipeline):
            baseline = current_student_pipeline * ratios[valid][np.argmax(years[valid])]

        logger.debug(
            "CWF predict | %s | pipeline=%s prior=%s R=%s conf=%s w=%s -> %s",
            programme,
            current_student_pipeline,
            prior,
            R,
            confidence,
            effective_weight,
            prediction,
        )

        return {
            "prediction": int(np.round(prediction)) if np.isfinite(prediction) else 0,
            "ratio_factor": np.round(R, 2) if np.isfinite(R) else np.nan,
            "confidence": np.round(confidence, 2),
            "stability": np.round(stability, 2),
            "baseline": baseline,
            "signal_weight": (
                float(effective_weight) if np.isfinite(effective_weight) else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Selectie van vergelijkbare opleidingen
    # ------------------------------------------------------------------

    def _select_reference_group(
        self,
        programme,
        programme_ratio,
        programme_size,
        programmes_group,
        years_group,
        counts_group,
        input_values_group,
        relax_if_empty=False,
        min_candidates=0,
    ):
        """Selecteer vergelijkbare opleidingen uit platte pool-observaties.

        Een kandidaat telt mee als zijn ratiofactor binnen ``ratio_tolerance``
        (relatief) van ``programme_ratio`` ligt én zijn mediane grootte binnen
        een factor ``size_factor`` van ``programme_size``. De dichtstbijzijnde
        ``max_group_programmes`` blijven over.

        Args:
            relax_if_empty: Zonder bruikbare doelratio/-grootte (of zonder
                kandidaten) alle pool-opleidingen nemen, gesorteerd op lengte
                van hun historie.
            min_candidates: Vul de groep tot dit aantal aan met de
                eerstvolgende kandidaten op similariteit, zonder harde
                tolerantie.

        Returns:
            Vier lijsten (opleidingen, jaren, aantallen, pipelines), één
            entry per geselecteerde opleiding; leeg zonder kandidaten.
        """
        programmes = np.asarray(programmes_group, dtype=object)
        years = np.asarray(years_group, dtype=float)
        counts = np.asarray(counts_group, dtype=float)
        inputs = np.asarray(input_values_group, dtype=float)

        if not (len(programmes) == len(years) == len(counts) == len(inputs)):
            raise ValueError(
                "Lengtes van de poolinvoer verschillen: "
                f"programmes={len(programmes)}, years={len(years)}, "
                f"counts={len(counts)}, inputs={len(inputs)}"
            )
        if len(programmes) == 0:
            return [], [], [], []

        have_target_stats = (
            np.isfinite(programme_ratio)
            and np.isfinite(programme_size)
            and programme_size > 0
        )
        if not have_target_stats and not relax_if_empty:
            return [], [], [], []

        programme_data: dict = {}
        for p, year, count, input_value in zip(programmes, years, counts, inputs):
            if p == programme:
                continue
            entry = programme_data.setdefault(
                p, {"years": [], "counts": [], "inputs": []}
            )
            entry["years"].append(year)
            entry["counts"].append(count)
            entry["inputs"].append(input_value)

        def _evaluate_candidates(apply_similarity_filter, enforce_tolerance=True):
            found = []
            for candidate_programme, data in programme_data.items():
                c_years = np.asarray(data["years"], dtype=float)
                c_counts = np.asarray(data["counts"], dtype=float)
                c_inputs = np.asarray(data["inputs"], dtype=float)

                valid = (
                    np.isfinite(c_years)
                    & np.isfinite(c_counts)
                    & np.isfinite(c_inputs)
                    & (c_inputs > 0)
                    & (c_counts > 0)
                )
                if not valid.any():
                    continue
                c_years, c_counts, c_inputs = (
                    c_years[valid],
                    c_counts[valid],
                    c_inputs[valid],
                )

                c_ratios = c_counts / c_inputs
                ratio_valid = np.isfinite(c_ratios)
                if not ratio_valid.any():
                    continue

                candidate_ratio, _, _, _ = self.estimate_factor(
                    c_ratios[ratio_valid], c_years[ratio_valid]
                )
                if not np.isfinite(candidate_ratio):
                    continue

                candidate_size = np.nanmedian(c_counts)
                if not (np.isfinite(candidate_size) and candidate_size > 0):
                    continue

                record = {
                    "programme": candidate_programme,
                    "years": c_years,
                    "counts": c_counts,
                    "inputs": c_inputs,
                }

                if not apply_similarity_filter:
                    record["score"] = -len(c_years)
                    found.append(record)
                    continue

                ratio_distance = abs(candidate_ratio - programme_ratio) / max(
                    abs(programme_ratio), 1e-8
                )
                if enforce_tolerance and ratio_distance > self.ratio_tolerance:
                    continue

                size_ratio = candidate_size / programme_size
                if enforce_tolerance and not (
                    1.0 / self.size_factor <= size_ratio <= self.size_factor
                ):
                    continue

                ratio_score = ratio_distance / max(self.ratio_tolerance, 1e-8)
                size_score = abs(np.log(size_ratio)) / np.log(
                    max(self.size_factor, 1.000001)
                )
                record["score"] = ratio_score + size_score
                found.append(record)
            return found

        candidates = (
            _evaluate_candidates(apply_similarity_filter=True)
            if have_target_stats
            else []
        )

        if not candidates and relax_if_empty:
            candidates = _evaluate_candidates(apply_similarity_filter=False)

        if (
            have_target_stats
            and min_candidates > 0
            and len(candidates) < min_candidates
        ):
            ranked = _evaluate_candidates(
                apply_similarity_filter=True, enforce_tolerance=False
            )
            ranked.sort(key=lambda x: x["score"])
            candidates = ranked[:min_candidates]

        if not candidates:
            return [], [], [], []

        candidates.sort(key=lambda x: x["score"])
        candidates = candidates[: self.max_group_programmes]

        return (
            [c["programme"] for c in candidates],
            [c["years"] for c in candidates],
            [c["counts"] for c in candidates],
            [c["inputs"] for c in candidates],
        )


def reliability_label(
    params: CWFParams,
    prediction: float,
    fill_rate: float,
    own_error_count: int,
    own_worst_error: float,
    own_mean_error: float,
    used_pool_fallback: bool,
    exam_type: str | None = None,
    previous_label: str | None = None,
) -> str:
    """Betrouwbaarheidslabel Groen/Oranje/Rood voor één voorspelling.

    - **Rood**: geen eigen historie (pool-fallback), minder dan
      ``reliability_min_own_years`` eigen backtest-fouten, vulgraad onder
      ``reliability_orange_min_fill``, of een grootste eigen fout boven
      ``reliability_red_min_own_error``.
    - **Groen**: vulgraad ≥ ``reliability_green_min_fill``, grootste eigen fout
      ≤ ``reliability_green_max_own_error`` (per examentype te overschrijven) én
      recency-gewogen gemiddelde fout ≤ ``reliability_green_max_own_mean_error``.
    - **Oranje**: al het andere.

    Fouten zijn ``|log(voorspelling / werkelijk)|`` uit leave-one-out backtests
    op eerdere jaren in dezelfde week (0.15 ≈ 16%). ``previous_label`` (het label
    van vorige week) geeft hysterese op de foutdrempels, zodat een label niet
    van week tot week heen en weer springt. De harde Rood-condities en de
    vulgraaddrempels worden niet vastgehouden.
    """
    if not np.isfinite(prediction) or prediction <= 0:
        return LABEL_RED

    fill = float(np.clip(fill_rate, 0.0, 1.0)) if np.isfinite(fill_rate) else 0.0

    if (
        used_pool_fallback
        or int(own_error_count) < params.reliability_min_own_years
        or fill < params.reliability_orange_min_fill
    ):
        return LABEL_RED

    red_max = params.reliability_red_min_own_error
    if previous_label in (LABEL_ORANGE, LABEL_GREEN):
        red_max += params.reliability_red_leave_own_error_add
    if np.isfinite(own_worst_error) and own_worst_error > red_max:
        return LABEL_RED

    green_max = params.reliability_green_max_own_error_by_exam_type.get(
        exam_type, params.reliability_green_max_own_error
    )
    green_mean_max = params.reliability_green_max_own_mean_error
    if previous_label == LABEL_GREEN:
        green_max += params.reliability_green_leave_own_error_add
        green_mean_max += params.reliability_green_leave_own_mean_error_add
    if (
        fill >= params.reliability_green_min_fill
        and np.isfinite(own_worst_error)
        and own_worst_error <= green_max
        and np.isfinite(own_mean_error)
        and own_mean_error <= green_mean_max
    ):
        return LABEL_GREEN

    return LABEL_ORANGE
