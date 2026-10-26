"""CWF-strategie: het confidence-weighted forecast-model als zelfstandig spoor.

Het model zelf staat in :mod:`studentprognose.models.cwf` en
:mod:`studentprognose.models.cwf_interval`; deze module bereidt de data voor
(herkomst *Onbekend* verdelen, weekgrid aanvullen, vulgraad en ratio
berekenen), bouwt per combinatie de trainingshistorie en vergelijkingspool, en
past de nabewerking toe (plafond, minimale intervalbreedte, numerus fixus).

Gebaseerd op het model van Radboud Institutional Research
(MIT-licentie, © 2026 Radboud Institutional Research; zie ``NOTICE``).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from studentprognose.config import get_cpu_count
from studentprognose.data.preprocessing.excluded_data_points import (
    apply_excluded_data_points,
)
from studentprognose.models.cwf import (
    FILL_RATE_COL,
    INPUT_VALUE_COL,
    LABEL_RED,
    RATIO_COL,
    STUDENT_COUNT_COL,
    YEAR_COL,
    ConfidenceWeightedForecast,
    CWFParams,
    reliability_label,
)
from studentprognose.models.cwf_interval import PredictionIntervalForecaster
from studentprognose.strategies.base import PredictionStrategy
from studentprognose.strategies.cumulative import prepare_cumulative_data
from studentprognose.utils.parallel import run_parallel_with_fallback
from studentprognose.utils.weeks import (
    academic_start_week,
    decrement_week,
    week_sort_key,
)

# Outputkolommen van het CWF-model.
CWF_PREDICTION_COL = "Prognose_cwf"
CWF_LOWER_COL = "cwf_ondergrens"
CWF_UPPER_COL = "cwf_bovengrens"
CWF_RELIABILITY_COL = "cwf_betrouwbaarheid"
CWF_FILL_RATE_COL = "cwf_vulgraad"
CWF_RATIO_COL = "cwf_ratiofactor"
CWF_OUTPUT_COLUMNS = [
    CWF_PREDICTION_COL,
    CWF_LOWER_COL,
    CWF_UPPER_COL,
    CWF_RELIABILITY_COL,
    CWF_FILL_RATE_COL,
    CWF_RATIO_COL,
]

KEY_COLS = ["Croho groepeernaam", "Examentype", "Herkomst"]
APPLICANT_COLS = [
    "Gewogen vooraanmelders",
    "Ongewogen vooraanmelders",
    "Aantal aanmelders met 1 aanmelding",
    "Inschrijvingen",
]
KNOWN_ORIGINS = ("NL", "EER", "Niet-EER")
UNKNOWN_ORIGIN = "onbekend"
BACHELOR = "Bachelor"
PRE_MASTER = "Pre-master"


# ---------------------------------------------------------------------------
# Datavoorbereiding
# ---------------------------------------------------------------------------


def redistribute_unknown_origin(df: pd.DataFrame) -> pd.DataFrame:
    """Verdeel aanmeldingen met herkomst *Onbekend* gelijk over NL, EER en Niet-EER.

    Weglaten zou de vraag naar de betreffende opleiding onderschatten. Gewogen
    vooraanmelders zijn al fractioneel en worden door drie gedeeld; de overige
    kolommen zijn koppen en worden met largest-remainder in hele aantallen
    verdeeld, zodat de som exact gelijk blijft.

    Args:
        df: Cumulatieve data na :func:`prepare_cumulative_data`.

    Returns:
        Dezelfde data zonder *Onbekend*-rijen.
    """
    unknown_mask = df["Herkomst"].astype(str).str.strip().str.lower() == UNKNOWN_ORIGIN
    if not unknown_mask.any():
        return df

    unknown = df.loc[unknown_mask]
    n = len(KNOWN_ORIGINS)
    shares = []
    for i, origin in enumerate(KNOWN_ORIGINS):
        share = unknown.copy()
        share["Herkomst"] = origin
        share["Gewogen vooraanmelders"] = share["Gewogen vooraanmelders"] / n
        for col in APPLICANT_COLS[1:]:
            total = np.rint(unknown[col].fillna(0).to_numpy(dtype=float))
            base = np.floor(total / n)
            remainder = np.rint(total - base * n).astype(int)
            share[col] = base + (remainder > i)
        shares.append(share)

    combined = pd.concat([df.loc[~unknown_mask]] + shares, ignore_index=True)
    key_cols = [c for c in combined.columns if c not in APPLICANT_COLS]
    return combined.groupby(key_cols, as_index=False, dropna=False)[
        APPLICANT_COLS
    ].sum()


def _densify_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """Zorg dat elke combinatie in elk jaar een rij heeft voor elke week van dat jaar.

    Een combinatie zonder aanmelders in een week ontbreekt in de telbestanden;
    zonder die rij zou het jaar uit de historie verdwijnen terwijl de
    gerealiseerde instroom wél bekend is (en voor de prior nodig is).
    """
    combos = df[["Collegejaar"] + KEY_COLS].drop_duplicates()
    weeks = df[["Collegejaar", "Weeknummer"]].drop_duplicates()
    skeleton = combos.merge(weeks, on="Collegejaar")
    dense = skeleton.merge(df, on=["Collegejaar", "Weeknummer"] + KEY_COLS, how="left")
    dense[APPLICANT_COLS] = dense[APPLICANT_COLS].fillna(0)
    dense["Faculteit"] = dense.groupby(KEY_COLS)["Faculteit"].transform(
        lambda s: s.ffill().bfill()
    )
    return dense


def _compute_fill_rate(df: pd.DataFrame, final_week: int, alpha: float) -> pd.Series:
    """Vulgraad per rij, zonder look-ahead.

    Per (combinatie, jaar) is de ruwe vulgraad in een week het lopende maximum
    van de gewogen vooraanmelders tot en met die week (in academische
    volgorde) gedeeld door het maximum van dat jaar. De vulgraad die het model
    ziet, is het recency-gewogen gemiddelde (``exp(alpha · (jaar - jaar_ref))``)
    van die ruwe vulgraad in *strikt eerdere* jaren in dezelfde academische
    week. Het eerste jaar van een reeks valt terug op zijn eigen ruwe waarde.
    """
    series_keys = KEY_COLS
    group_keys = ["Collegejaar"] + series_keys
    week_total = df["Gewogen vooraanmelders"]
    max_applicants = week_total.groupby([df[c] for c in group_keys]).transform("max")

    aw = df["Weeknummer"].astype(int).map(lambda w: week_sort_key(w, final_week))
    work = df[group_keys].assign(_aw=aw.to_numpy(), _total=week_total.to_numpy())
    order = work.sort_values(group_keys + ["_aw"]).index
    running_max = (
        work.loc[order].groupby(group_keys)["_total"].cummax().reindex(work.index)
    )
    raw_ratio = (running_max / max_applicants.replace(0, np.nan)).fillna(0).clip(0, 1)

    history = (
        work[series_keys + ["Collegejaar", "_aw"]]
        .assign(_raw=raw_ratio.to_numpy())
        .drop_duplicates(subset=series_keys + ["_aw", "Collegejaar"])
        .sort_values(series_keys + ["_aw", "Collegejaar"])
    )
    grp_cols = series_keys + ["_aw"]
    year_ref = history.groupby(grp_cols, sort=False)["Collegejaar"].transform("max")
    weight = np.exp(alpha * (history["Collegejaar"] - year_ref))
    history = history.assign(_num=weight * history["_raw"], _den=weight)
    grp = history.groupby(grp_cols, sort=False)
    prior_num = grp["_num"].transform(lambda s: s.shift().cumsum())
    prior_den = grp["_den"].transform(lambda s: s.shift().cumsum())
    history = history.assign(_mean=prior_num / prior_den.replace(0, np.nan))

    merged = work[series_keys + ["Collegejaar", "_aw"]].merge(
        history[series_keys + ["_aw", "Collegejaar", "_mean"]],
        on=series_keys + ["_aw", "Collegejaar"],
        how="left",
        validate="m:1",
    )
    weighted_mean = merged["_mean"].to_numpy(dtype=float)
    result = np.where(
        np.isnan(weighted_mean), raw_ratio.to_numpy(dtype=float), weighted_mean
    )
    return pd.Series(np.clip(result, 0.0, 1.0), index=df.index)


def build_cwf_history(
    data_cumulative: pd.DataFrame,
    data_studentcount: pd.DataFrame,
    final_week: int,
    fill_rate_recency_alpha: float,
) -> pd.DataFrame:
    """Bouw de CWF-werkdata: één rij per combinatie × jaar × week.

    Aggregeert over faculteiten (de modelgranulariteit is opleiding × herkomst ×
    examentype, zoals de studentaantallen), vult het weekgrid aan, koppelt de
    gerealiseerde instroom en voegt ``Pipeline`` (gewogen + inschrijvingen),
    ``Ratio`` (instroom / pipeline) en ``Vulgraad`` toe.
    """
    df = data_cumulative.groupby(
        ["Collegejaar", "Weeknummer"] + KEY_COLS, as_index=False
    ).agg({**{c: "sum" for c in APPLICANT_COLS}, "Faculteit": "first"})
    df = _densify_weeks(df)

    counts = (
        data_studentcount.drop(columns="Faculteit", errors="ignore")
        .groupby(["Collegejaar"] + KEY_COLS, as_index=False)[STUDENT_COUNT_COL]
        .sum()
    )
    df = df.merge(counts, on=["Collegejaar"] + KEY_COLS, how="left")

    df[INPUT_VALUE_COL] = df["Gewogen vooraanmelders"] + df["Inschrijvingen"]
    df[RATIO_COL] = df[STUDENT_COUNT_COL] / df[INPUT_VALUE_COL].replace(0, np.nan)
    df[FILL_RATE_COL] = _compute_fill_rate(df, final_week, fill_rate_recency_alpha)
    return df


# ---------------------------------------------------------------------------
# Voorspelling per combinatie
# ---------------------------------------------------------------------------


@dataclass
class _ComboInput:
    programme: object
    exam_type: str
    origin: str
    faculty: object
    own: pd.DataFrame
    pool: pd.DataFrame
    full_own: pd.DataFrame
    pipeline: float
    ceiling: float
    fill_rate: float
    predefined_prior: float
    previous_label: str | None


def _round_half_up(value: float) -> float:
    return float(np.floor(value + 0.5)) if np.isfinite(value) else np.nan


def predict_combo(
    combo: _ComboInput,
    params: CWFParams,
    model: ConfidenceWeightedForecast,
    interval_model: PredictionIntervalForecaster,
) -> dict:
    """Voorspelling, interval en label voor één combinatie.

    Returns:
        Dict met de CWF-outputkolommen.
    """
    has_own_history = not combo.own.empty
    own_for_point = combo.own if has_own_history else combo.pool
    if own_for_point.empty:
        # Geen enkele gerealiseerde instroom om op te kalibreren (bijv. een
        # examentype zonder studentaantallen): geen voorspelling i.p.v. een 0.
        return {
            CWF_PREDICTION_COL: np.nan,
            CWF_LOWER_COL: np.nan,
            CWF_UPPER_COL: np.nan,
            CWF_RELIABILITY_COL: LABEL_RED,
            CWF_FILL_RATE_COL: combo.fill_rate,
            CWF_RATIO_COL: np.nan,
        }

    output = model.predict(
        own=own_for_point,
        current_student_pipeline=combo.pipeline,
        pre_defined_prior=combo.predefined_prior,
        pool=combo.pool,
        has_own_history=has_own_history,
        programme=combo.programme,
    )
    prediction = float(output["prediction"])

    # Het blendgewicht van het interval is de vulgraad, begrensd door het
    # gewicht dat het ratiosignaal in de puntvoorspelling kreeg: het
    # foutinterval meet de ratioschatter en mag niet meer zeggen dan die zelf.
    fill = combo.fill_rate if np.isfinite(combo.fill_rate) else 0.0
    blend_weight = min(fill, output["signal_weight"])
    lower, upper = interval_model.prediction_interval(
        prediction,
        combo.programme,
        combo.own,
        combo.pool,
        combo.full_own,
        confidence_level=params.confidence_level,
        current_fill_rate=blend_weight,
        pre_defined_prior=combo.predefined_prior if not has_own_history else np.nan,
        current_student_pipeline=combo.pipeline,
    )
    if not (np.isfinite(lower) and np.isfinite(upper)):
        lower, upper = interval_model.count_noise_interval(prediction)
    n_own, worst, mean_error = interval_model.own_error_track_record(
        combo.programme, combo.own
    )

    # Plafond: is de cyclus historisch gezien vrijwel voltooid, dan kan de
    # instroom niet meer boven ongewogen aanmelders + inschrijvingen uitkomen.
    if (
        combo.exam_type in params.upper_bound_cap_exam_types
        and np.isfinite(combo.fill_rate)
        and combo.fill_rate >= params.upper_bound_cap_fill_rate
        and np.isfinite(combo.ceiling)
        and np.isfinite(upper)
    ):
        upper = max(prediction, min(upper, combo.ceiling))
        lower = min(lower, upper)

    # Minimale halfbreedte: een interval van één student rond 9 claimt meer dan
    # een paar jaar historie kan weten.
    h = params.interval_min_half_width
    if h > 0 and prediction > 0 and np.isfinite(lower) and np.isfinite(upper):
        lower = min(lower, max(0.0, prediction - h))
        upper = max(upper, prediction + h)

    prediction = _round_half_up(prediction)
    lower = _round_half_up(lower)
    upper = _round_half_up(upper)
    if np.isfinite(lower):
        lower = min(lower, prediction)
    if np.isfinite(upper):
        upper = max(upper, prediction)

    label = reliability_label(
        params,
        prediction,
        combo.fill_rate,
        n_own,
        worst,
        mean_error,
        used_pool_fallback=not has_own_history,
        exam_type=combo.exam_type,
        previous_label=combo.previous_label,
    )

    return {
        CWF_PREDICTION_COL: prediction,
        CWF_LOWER_COL: lower,
        CWF_UPPER_COL: upper,
        CWF_RELIABILITY_COL: label,
        CWF_FILL_RATE_COL: combo.fill_rate,
        CWF_RATIO_COL: output["ratio_factor"],
    }


def _predict_chunk(combos, params):
    model = ConfidenceWeightedForecast(params)
    interval_model = PredictionIntervalForecaster(model, params)
    return [predict_combo(c, params, model, interval_model) for c in combos]


def apply_numerus_fixus_caps(result: pd.DataFrame, numerus_fixus: dict) -> pd.DataFrame:
    """Begrens CWF-voorspelling en bovengrens per numerus-fixusopleiding.

    De voorspellingen van alle herkomsten (Pre-master uitgezonderd) worden
    proportioneel geschaald tot het plafond, met largest-remainder-afronding.
    Daarna worden de marges boven de voorspelling geschaald zodat de som van
    de bovengrenzen het plafond niet overschrijdt; zo blijft
    ``bovengrens ≥ voorspelling`` per rij.
    """
    if not numerus_fixus or result.empty:
        return result

    result = result.copy()
    for programme, limit in numerus_fixus.items():
        if not limit or limit <= 0:
            continue
        mask = (
            (result["Croho groepeernaam"] == programme)
            & (result["Examentype"] != PRE_MASTER)
            & result[CWF_PREDICTION_COL].notna()
        )
        if not mask.any():
            continue

        preds = result.loc[mask, CWF_PREDICTION_COL].astype(float)
        total = preds.sum()
        if total > limit:
            scaled = preds * (limit / total)
            floored = np.floor(scaled)
            remainder = int(limit - floored.sum() + 0.5)
            order = (scaled - floored).sort_values(ascending=False).index
            floored.loc[order[:remainder]] += 1
            result.loc[mask, CWF_PREDICTION_COL] = floored
            preds = floored

        upper = result.loc[mask, CWF_UPPER_COL].astype(float)
        if upper.notna().all() and upper.sum() > limit:
            margins = (upper - preds).clip(lower=0)
            room = max(limit - preds.sum(), 0.0)
            scale = room / margins.sum() if margins.sum() > 0 else 0.0
            new_upper = preds + np.floor(margins * scale)
            result.loc[mask, CWF_UPPER_COL] = new_upper
            result.loc[mask, CWF_LOWER_COL] = np.minimum(
                result.loc[mask, CWF_LOWER_COL].astype(float), new_upper
            )
        result.loc[mask, CWF_LOWER_COL] = np.minimum(
            result.loc[mask, CWF_LOWER_COL].astype(float),
            result.loc[mask, CWF_PREDICTION_COL].astype(float),
        )
    return result


def _warn_combos(labels: list[str], message: str, max_shown: int = 10) -> None:
    """Eén samenvattende waarschuwing voor een lijst combinaties."""
    if not labels:
        return
    shown = "; ".join(labels[:max_shown])
    more = f" (+{len(labels) - max_shown} meer)" if len(labels) > max_shown else ""
    warnings.warn(
        f"CWF: {len(labels)} combinatie(s) {message}: {shown}{more}.",
        UserWarning,
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# Strategie
# ---------------------------------------------------------------------------


class CWFStrategy(PredictionStrategy):
    """Voorspelt de instroom met het confidence-weighted forecast-model.

    Gebruikt de cumulatieve vooraanmelddata en de gerealiseerde eerstejaars-
    aantallen; de individuele aanmelddata is niet nodig.
    """

    def __init__(
        self,
        data_cumulative,
        data_studentcount,
        configuration,
        data_latest,
        ensemble_weights,
        cwd,
        data_option,
        ci_test_n,
    ):
        super().__init__(
            configuration,
            data_latest,
            ensemble_weights,
            data_studentcount,
            cwd,
            data_option,
            ci_test_n,
        )
        self.data_cumulative = data_cumulative
        self.data_studentcount = data_studentcount
        self.params = CWFParams.from_config(configuration)
        self.history: pd.DataFrame | None = None

    def get_dashboard_data(self) -> dict:
        return {
            "data_cumulative": self.data_cumulative,
            "xgboost_curve": None,
        }

    def get_programme_columns_by_track(self) -> dict:
        col = self._programme_column_name()
        if self.data_cumulative is None or col not in self.data_cumulative.columns:
            return {}
        return {"cwf": self.data_cumulative[col]}

    def preprocess(self):
        data = prepare_cumulative_data(self.data_cumulative)
        if self.params.redistribute_unknown_origin:
            data = redistribute_unknown_origin(data)
        self.data_cumulative = data

        studentcount = self.data_studentcount.copy()
        studentcount["Croho groepeernaam"] = studentcount["Croho groepeernaam"].astype(
            data["Croho groepeernaam"].dtype
        )
        self.history = build_cwf_history(
            data,
            studentcount,
            self.final_academic_week,
            self.params.fill_rate_recency_alpha,
        )
        self._studentcount = studentcount
        return self.data_cumulative

    def _previous_labels(self, predict_year: int, predict_week: int) -> dict:
        """Labels van vorige week uit eerdere output, voor de hysterese."""
        if predict_week == academic_start_week(self.final_academic_week):
            return {}
        latest = self.postprocessor.data_latest
        if latest is None or latest.empty or CWF_RELIABILITY_COL not in latest.columns:
            return {}
        rows = latest[
            (latest["Collegejaar"] == predict_year)
            & (latest["Weeknummer"] == decrement_week(predict_week))
            & latest[CWF_RELIABILITY_COL].notna()
        ]
        return {
            tuple(k): v
            for k, v in zip(
                rows[KEY_COLS].itertuples(index=False), rows[CWF_RELIABILITY_COL]
            )
        }

    def _predefined_prior(self, programme, exam_type, origin) -> float:
        value = (
            self.params.predefined_priors.get(str(programme), {})
            .get(exam_type, {})
            .get(origin)
        )
        return np.nan if value is None else float(value)

    def predict_nr_of_students(self, predict_year, predict_week, skip_years=0):
        if skip_years > 0:
            raise ValueError("Het CWF-model ondersteunt geen --skipyears.")

        self.set_year_week(predict_year, predict_week, self.data_cumulative)
        params = self.params
        start_year = params.training_start_year or self.min_training_year

        week_slice = self.history[
            (self.history["Weeknummer"] == predict_week)
            & (self.history["Collegejaar"] >= start_year)
        ]
        if self.excluded_data_points:
            week_slice = apply_excluded_data_points(
                week_slice, self.excluded_data_points, predict_year
            )

        targets = week_slice[week_slice["Collegejaar"] == predict_year]
        if self.programme_filtering != []:
            targets = targets[
                targets["Croho groepeernaam"].isin(self.programme_filtering)
            ]
        if self.herkomst_filtering != []:
            targets = targets[targets["Herkomst"].isin(self.herkomst_filtering)]
        if self.examentype_filtering != []:
            targets = targets[targets["Examentype"].isin(self.examentype_filtering)]
        if targets.empty:
            return None

        history = week_slice[week_slice["Collegejaar"] < predict_year]
        sc = self._studentcount
        sc = sc[sc["Collegejaar"] < predict_year]
        nf_programmes = set(self.numerus_fixus_list)
        previous_labels = self._previous_labels(predict_year, predict_week)

        groups = {k: g for k, g in history.groupby(["Examentype", "Herkomst"])}
        counts_by_combo = {k: g for k, g in sc.groupby(KEY_COLS)}
        empty = history.iloc[0:0]

        combos = []
        no_history: list[str] = []
        pool_only: list[str] = []
        for row in targets.to_dict("records"):
            programme = row["Croho groepeernaam"]
            exam_type = row["Examentype"]
            origin = row["Herkomst"]
            base = groups.get((exam_type, origin), empty)
            # Alleen jaren met bekende instroom tellen als historie.
            has_count = base[STUDENT_COUNT_COL].notna()
            is_own = base["Croho groepeernaam"] == programme
            if exam_type == BACHELOR and programme in nf_programmes:
                pool_mask = pd.Series(False, index=base.index)
            elif exam_type == BACHELOR:
                pool_mask = ~base["Croho groepeernaam"].isin(nf_programmes)
            else:
                pool_mask = pd.Series(True, index=base.index)

            own = base[is_own & has_count]
            pool = base[pool_mask & ~is_own & has_count]
            full_own = counts_by_combo.get((programme, exam_type, origin), sc.iloc[0:0])
            pipeline = float(row[INPUT_VALUE_COL])
            prior = self._predefined_prior(programme, exam_type, origin)
            label = f"{programme} | {origin} | {exam_type}"
            if own.empty and pool.empty:
                no_history.append(label)
            elif own.empty and not np.isfinite(prior) and pipeline >= 3:
                pool_only.append(label)
            combos.append(
                _ComboInput(
                    programme=programme,
                    exam_type=exam_type,
                    origin=origin,
                    faculty=row["Faculteit"],
                    own=own,
                    pool=pool,
                    full_own=full_own[[YEAR_COL, STUDENT_COUNT_COL]],
                    pipeline=pipeline,
                    ceiling=float(
                        row["Ongewogen vooraanmelders"] + row["Inschrijvingen"]
                    ),
                    fill_rate=float(row[FILL_RATE_COL]),
                    predefined_prior=prior,
                    previous_label=previous_labels.get((programme, exam_type, origin)),
                )
            )

        _warn_combos(
            no_history,
            "hebben geen enkele gerealiseerde instroom (ook niet bij vergelijkbare "
            "opleidingen) en krijgen geen CWF-voorspelling",
        )
        _warn_combos(
            pool_only,
            "hebben geen eigen historie en geen ingestelde prior "
            "(model_config.cwf_params.predefined_priors); de voorspelling leunt volledig "
            "op vergelijkbare opleidingen en krijgt label Rood",
        )

        n_jobs = max(1, min(get_cpu_count(self.configuration), len(combos)))
        chunk_size = math.ceil(len(combos) / n_jobs)
        chunks = [combos[i : i + chunk_size] for i in range(0, len(combos), chunk_size)]
        if len(chunks) == 1:
            results = _predict_chunk(chunks[0], params)
        else:
            print("Start parallel predicting (CWF)...")
            results = [
                r
                for chunk_results in run_parallel_with_fallback(
                    (joblib.delayed(_predict_chunk)(chunk, params) for chunk in chunks),
                    n_jobs=n_jobs,
                )
                for r in chunk_results
            ]

        result = pd.DataFrame(
            {
                "Croho groepeernaam": [c.programme for c in combos],
                "Faculteit": [c.faculty for c in combos],
                "Examentype": [c.exam_type for c in combos],
                "Collegejaar": predict_year,
                "Herkomst": [c.origin for c in combos],
                "Weeknummer": predict_week,
            }
        )
        result = pd.concat([result, pd.DataFrame(results)], axis=1)
        result["Croho groepeernaam"] = result["Croho groepeernaam"].astype(
            targets["Croho groepeernaam"].dtype
        )
        return apply_numerus_fixus_caps(result, self.numerus_fixus_list)
