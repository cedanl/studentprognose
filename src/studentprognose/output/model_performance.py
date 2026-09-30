"""Modelperformance per voorspelspoor, klaar voor het dashboard.

Zet de pipeline-output om naar **evaluatie-eenheden** (één rij per collegejaar ×
opleiding × examentype) en vat de fout samen per opleidingsgrootte en per
examentype. De cumulatieve dashboardpagina toont precies deze tabellen.

Keuzes die de getallen bepalen:

- **Alleen de voorspelweek.** De XGBoost-voorspelling staat alleen op de rij van de
  voorspelweek; latere weken zouden de fout verdunnen met makkelijkere
  voorspellingen.
- **Opleidingsniveau.** De pipeline voorspelt per herkomstgroep; beleid kijkt naar
  de opleiding. We tellen de herkomstgroepen daarom eerst op, zodat een opleiding
  van 300 studenten één punt is en niet drie.
- **Gelijke populatie.** Modellen worden alleen vergeleken op eenheden waarvoor
  ze allemaal een voorspelling hebben, anders vergelijk je appels met peren.
- **Naïeve baseline.** "Dit jaar komen er evenveel als vorig jaar." Een model dat
  deze baseline niet verslaat, voegt weinig toe.
- **Numerus fixus telt niet mee**, net als in :func:`evaluate_predictions`: de
  instroom wordt daar door de capaciteit bepaald en niet door de aanmeldingen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACTUAL = "Aantal_studenten"
PROGRAMME = "Croho groepeernaam"
YEAR = "Collegejaar"
EXAM_TYPE = "Examentype"
WEEK = "Weeknummer"

NAIVE = "Naief_vorig_jaar"
NAIVE_LABEL = "Naïef (vorig jaar)"

# Leesbare namen van alle voorspelkolommen. 'SARIMA_cumulative' bevat historisch
# gezien de XGBoost-regressoroutput (#181); 'SARIMA_individual' is de SARIMA-
# extrapolatie van de XGBoost-classificatie per aanmelder.
MODEL_LABELS: dict[str, str] = {
    "Weighted_ensemble_prediction": "Ensemble (gewogen)",
    "Average_ensemble_prediction": "Ensemble (gemiddeld)",
    "Ensemble_prediction": "Ensemble",
    "SARIMA_cumulative": "XGBoost (cumulatief)",
    "SARIMA_individual": "Individueel model",
    "Prognose_ratio": "Ratiomodel",
    NAIVE: NAIVE_LABEL,
}
SHORT_LABELS: dict[str, str] = {
    "Weighted_ensemble_prediction": "Ensemble",
    "Average_ensemble_prediction": "Ensemble gem.",
    "Ensemble_prediction": "Ensemble",
    "SARIMA_cumulative": "XGBoost",
    "SARIMA_individual": "Individueel",
    "Prognose_ratio": "Ratio",
    NAIVE: "Naïef",
}

# Te evalueren kolommen per dashboardpagina, primair model eerst. Voor het
# eindoverzicht is het primaire model de eindprognose (het ensemble); de losse
# sporen staan erbij zodat je ziet of het ensemble iets toevoegt.
TRACK_MODELS: dict[str, list[str]] = {
    "cumulative": ["SARIMA_cumulative", "Prognose_ratio"],
    "individual": ["SARIMA_individual"],
    "final": [
        "Weighted_ensemble_prediction",
        "Ensemble_prediction",
        "SARIMA_cumulative",
        "SARIMA_individual",
        "Prognose_ratio",
    ],
}
# Standaard (en terugwaarts compatibel): het cumulatieve spoor.
MODEL_COLUMNS: dict[str, str] = {m: MODEL_LABELS[m] for m in TRACK_MODELS["cumulative"]}

# Grootteklassen op basis van de werkelijke instroom van de opleiding.
SIZE_EDGES: list[float] = [0, 25, 50, 100, 250, np.inf]
SIZE_LABELS: list[str] = ["< 25", "25–49", "50–99", "100–249", "≥ 250"]

# Een model doet alleen mee als het minstens deze fractie van de eenheden van het
# primaire model dekt; anders zou de gelijke populatie onnodig krimpen.
MIN_COVERAGE = 0.5

WITHIN_THRESHOLD = 0.10

# Betrouwbaarheid per opleiding, op basis van de historische WAPE van het model voor
# die opleiding. Dezelfde grenzen als de MAPE-kleuren in de andere dashboardpagina's
# (groen ≤ 10%, geel ≤ 25%, rood daarboven).
TRUST_HIGH = 0.10
TRUST_MEDIUM = 0.25
# Met minder geëvalueerde jaren is "hoog" niet te onderbouwen: één goed jaar kan toeval zijn.
TRUST_MIN_YEARS_HIGH = 2
TRUST_LEVELS: list[str] = ["hoog", "middel", "laag", "onbekend"]


def size_class(actual: pd.Series) -> pd.Categorical:
    """Deel werkelijke instroom in grootteklassen in (links-gesloten intervallen)."""
    return pd.cut(actual, bins=SIZE_EDGES, labels=SIZE_LABELS, right=False)


def build_evaluation_units(
    data: pd.DataFrame,
    predict_week: int | None,
    data_studentcount: pd.DataFrame | None = None,
    numerus_fixus: dict | list | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Bouw één evaluatie-eenheid per collegejaar × opleiding × examentype.

    Args:
        data: Pipeline-output (``PostProcessor.data``).
        predict_week: De voorspelweek; alleen die rijen dragen de voorspelling.
            Bij ``None`` wordt niet op week gefilterd.
        data_studentcount: Realisaties van alle jaren, nodig voor de naïeve baseline
            (realisatie vorig jaar). Zonder dit ontbreekt de baseline.
        numerus_fixus: Numerus-fixusopleidingen die buiten de evaluatie blijven.
        models: Te evalueren voorspelkolommen; standaard :data:`MODEL_COLUMNS`.

    Returns:
        DataFrame met ``Collegejaar``, ``Croho groepeernaam``, ``Examentype``,
        ``Aantal_studenten``, ``Grootteklasse``, één kolom per aanwezig model en
        ``Naief_vorig_jaar``. Leeg als er niets te
        evalueren valt (bijv. een collegejaar zonder realisatie).
    """
    models = [m for m in (models or list(MODEL_COLUMNS)) if m in data.columns]
    out_cols = [YEAR, PROGRAMME, EXAM_TYPE, ACTUAL, "Grootteklasse", *models, NAIVE]
    if ACTUAL not in data.columns or not models:
        return pd.DataFrame(columns=out_cols)

    frame = _predict_week_rows(data, predict_week, numerus_fixus)
    frame = frame[frame[ACTUAL].notna() & (frame[ACTUAL] > 0)]
    if frame.empty:
        return pd.DataFrame(columns=out_cols)

    units = _programme_totals(frame, models)

    units[NAIVE] = previous_year_actuals(units, data_studentcount)
    units["Grootteklasse"] = size_class(units[ACTUAL])
    units[YEAR] = units[YEAR].astype(int)
    return units[out_cols].reset_index(drop=True)


def _predict_week_rows(
    data: pd.DataFrame, predict_week: int | None, numerus_fixus: dict | list | None
) -> pd.DataFrame:
    """Rijen van de voorspelweek, zonder numerus-fixusopleidingen, met string-sleutel."""
    frame = data
    if predict_week is not None and WEEK in frame.columns:
        frame = frame[frame[WEEK] == predict_week]
    nf = set(
        numerus_fixus.keys() if isinstance(numerus_fixus, dict) else numerus_fixus or []
    )
    if nf:
        frame = frame[~frame[PROGRAMME].astype(str).isin({str(p) for p in nf})]
    return frame.assign(**{PROGRAMME: frame[PROGRAMME].astype(str)})


def _sum_complete(s: pd.Series) -> float:
    # Een opleiding waarvan één herkomstgroep geen voorspelling heeft, zou met een
    # onvolledige som worden vergeleken met de volledige realisatie.
    return float(s.sum()) if s.notna().all() else np.nan


def _programme_totals(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Tel herkomstgroepen op tot één rij per collegejaar × opleiding × examentype."""
    agg = {ACTUAL: lambda s: float(s.sum()) if s.notna().any() else np.nan}
    agg.update({m: _sum_complete for m in models})
    return frame.groupby(
        [YEAR, PROGRAMME, EXAM_TYPE], as_index=False, observed=True
    ).agg(agg)


def current_predictions(
    data: pd.DataFrame,
    prediction_year: int,
    predict_week: int | None,
    numerus_fixus: dict | list | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Prognose per opleiding × examentype voor het voorspeljaar, per model.

    ``Aantal_studenten`` is gevuld zodra de realisatie bekend is (backtest) en
    anders leeg.
    """
    models = [m for m in (models or list(MODEL_COLUMNS)) if m in data.columns]
    cols = [PROGRAMME, EXAM_TYPE, ACTUAL, *models]
    if not models or YEAR not in data.columns:
        return pd.DataFrame(columns=cols)
    frame = _predict_week_rows(data, predict_week, numerus_fixus)
    frame = frame[frame[YEAR] == prediction_year]
    if ACTUAL not in frame.columns:
        frame = frame.assign(**{ACTUAL: np.nan})
    if frame.empty:
        return pd.DataFrame(columns=cols)
    totals = _programme_totals(frame, models)
    totals = totals[totals[models].notna().any(axis=1)]
    return totals[cols].reset_index(drop=True)


def trust_level(wape: float, n_years: int, beats_naive: bool | None = None) -> str:
    """Hoe betrouwbaar is de prognose van een model voor één opleiding?

    - ``hoog``: WAPE ≤ 10%, minstens twee geëvalueerde jaren én niet slechter dan
      de naïeve voorspelling.
    - ``middel``: WAPE ≤ 25% (ook een "hoog" met te weinig jaren of die de naïeve
      voorspelling niet verslaat).
    - ``laag``: WAPE boven 25%.
    - ``onbekend``: geen enkel geëvalueerd jaar.

    ``beats_naive`` is ``None`` als er geen jaar is waarin de naïeve voorspelling
    bestaat; dat telt dan niet tegen het model.
    """
    if n_years == 0 or wape is None or not np.isfinite(wape):
        return "onbekend"
    if wape > TRUST_MEDIUM:
        return "laag"
    if (
        wape <= TRUST_HIGH
        and n_years >= TRUST_MIN_YEARS_HIGH
        and beats_naive is not False
    ):
        return "hoog"
    return "middel"


def programme_summary(units: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Fout per opleiding × examentype over alle jaren in ``units``.

    Anders dan de groepsvergelijking beoordeelt dit elk model op de jaren waarvoor
    het zelf een voorspelling heeft: een opleiding verdwijnt niet uit de tabel omdat
    een ánder model (of de baseline) ontbreekt. De vergelijking met de naïeve
    voorspelling gebruikt alleen jaren waarin beide bestaan.

    Returns:
        Eén rij per opleiding met ``actual_mean`` (over alle geëvalueerde jaren) en
        per model ``n_<model>`` (jaren), ``wape_<model>``, ``bias_<model>`` en
        ``trust_<model>``. De betrouwbaarheid van de naïeve baseline zelf wordt niet
        bepaald.
    """
    base = [PROGRAMME, EXAM_TYPE, "actual_mean"]
    cols = base + [f"{k}_{m}" for m in models for k in ("n", "wape", "bias", "trust")]
    if units.empty or not models:
        return pd.DataFrame(columns=cols)

    rows = []
    for (prog, exam), sub in units.groupby([PROGRAMME, EXAM_TYPE], observed=True):
        actual = sub[ACTUAL].to_numpy(dtype="float64")
        row = {
            PROGRAMME: str(prog),
            EXAM_TYPE: str(exam),
            "actual_mean": float(actual.mean()),
        }
        for m in models:
            has = sub[m].notna()
            met = error_metrics(sub.loc[has, ACTUAL], sub.loc[has, m])
            row[f"n_{m}"] = int(sub.loc[has, YEAR].nunique())
            row[f"wape_{m}"] = met["wape"]
            row[f"bias_{m}"] = met["bias"]
            if m == NAIVE:
                row[f"trust_{m}"] = None
                continue
            beats_naive = None
            if NAIVE in sub.columns:
                both = has & sub[NAIVE].notna()
                if both.any():
                    # Vergelijk op dezelfde jaren, anders is het geen eerlijke vergelijking.
                    a, n = sub.loc[both, ACTUAL], sub.loc[both, NAIVE]
                    beats_naive = (
                        error_metrics(a, sub.loc[both, m])["wape"]
                        <= error_metrics(a, n)["wape"]
                    )
            row[f"trust_{m}"] = trust_level(met["wape"], row[f"n_{m}"], beats_naive)
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def previous_year_actuals(
    units: pd.DataFrame, data_studentcount: pd.DataFrame | None
) -> pd.Series:
    """Realisatie van jaar-1 voor dezelfde opleiding × examentype (de naïeve voorspelling)."""
    if data_studentcount is None or data_studentcount.empty:
        return pd.Series(np.nan, index=units.index)
    needed = {YEAR, PROGRAMME, EXAM_TYPE, ACTUAL}
    if not needed.issubset(data_studentcount.columns):
        return pd.Series(np.nan, index=units.index)

    prev = (
        data_studentcount.assign(
            **{PROGRAMME: data_studentcount[PROGRAMME].astype(str)}
        )
        .groupby([YEAR, PROGRAMME, EXAM_TYPE], as_index=False)[ACTUAL]
        .sum()
    )
    prev[YEAR] = prev[YEAR].astype(int) + 1
    prev = prev[prev[ACTUAL] > 0].rename(columns={ACTUAL: NAIVE})
    merged = (
        units[[YEAR, PROGRAMME, EXAM_TYPE]]
        .astype({YEAR: int})
        .merge(prev, on=[YEAR, PROGRAMME, EXAM_TYPE], how="left")
    )
    return pd.Series(merged[NAIVE].to_numpy(dtype="float64"), index=units.index)


def comparable_models(
    units: pd.DataFrame, models: list[str] | None = None
) -> list[str]:
    """Modellen (incl. baseline) die genoeg dekking hebben voor een eerlijke vergelijking.

    Het eerste aanwezige model uit ``models`` (standaard :data:`MODEL_COLUMNS`) is
    het primaire model en doet altijd mee. Andere kolommen doen mee zodra ze
    minstens :data:`MIN_COVERAGE` van de eenheden van het primaire model dekken.
    """
    present = [
        m for m in [*(models or list(MODEL_COLUMNS)), NAIVE] if m in units.columns
    ]
    present = [m for m in present if units[m].notna().any()]
    if not present:
        return []
    primary = present[0]
    base = units[primary].notna()
    n_base = int(base.sum())
    if n_base == 0:
        return []
    return [primary] + [
        m
        for m in present[1:]
        if (units.loc[base, m].notna().sum() / n_base) >= MIN_COVERAGE
    ]


def common_population(units: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Beperk tot eenheden waarvoor alle ``models`` een voorspelling hebben."""
    if not models:
        return units.iloc[0:0]
    return units[units[models].notna().all(axis=1)]


def error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """WAPE, MAPE, relatieve bias en het aandeel eenheden binnen 10% fout.

    Alle waarden zijn fracties (0.08 == 8%). ``bias`` is (som voorspeld − som
    werkelijk) / som werkelijk: positief betekent dat het model in totaal overschat.
    """
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    n = len(actual)
    if n == 0 or actual.sum() == 0:
        return {
            "n": n,
            "wape": np.nan,
            "mape": np.nan,
            "bias": np.nan,
            "within": np.nan,
        }
    abs_err = np.abs(predicted - actual)
    ape = abs_err / actual
    return {
        "n": n,
        "wape": float(abs_err.sum() / actual.sum()),
        "mape": float(ape.mean()),
        "bias": float((predicted.sum() - actual.sum()) / actual.sum()),
        "within": float((ape <= WITHIN_THRESHOLD).mean()),
    }


def summarise(
    units: pd.DataFrame,
    models: list[str],
    by: str | None = None,
    order: list[str] | None = None,
) -> pd.DataFrame:
    """Vat de fout per model samen, optioneel per groep.

    Args:
        units: Uitvoer van :func:`build_evaluation_units`, al beperkt tot de
            gewenste populatie (zie :func:`common_population`).
        models: Te evalueren kolommen, in weergavevolgorde.
        by: Groepeerkolom (bijv. ``"Grootteklasse"`` of ``"Examentype"``). Bij
            ``None`` één totaalrij per model.
        order: Gewenste volgorde van de groepen; lege groepen worden overgeslagen.

    Returns:
        Long-format DataFrame met ``group``, ``model``, ``n``, ``wape``, ``mape``,
        ``bias`` en ``within``.
    """
    cols = ["group", "model", "n", "wape", "mape", "bias", "within"]
    if units.empty or not models:
        return pd.DataFrame(columns=cols)

    if by is None:
        groups: list[tuple[str, pd.DataFrame]] = [("Totaal", units)]
    else:
        values = (
            order
            if order is not None
            else sorted(units[by].dropna().astype(str).unique())
        )
        groups = [(str(g), units[units[by].astype(str) == str(g)]) for g in values]

    rows = []
    for group, sub in groups:
        if sub.empty:
            continue
        for m in models:
            rows.append(
                {"group": group, "model": m, **error_metrics(sub[ACTUAL], sub[m])}
            )
    return pd.DataFrame(rows, columns=cols)


def model_label(column: str) -> str:
    """Leesbare naam van een modelkolom of de baseline."""
    return MODEL_LABELS.get(column, column)
