"""Cumulatieve dashboardpagina: hoe goed presteert het cumulatieve model?

De pagina is bewust minimaal. Ze beantwoordt drie vragen:

1. Hoe groot is de fout in het algemeen, en verslaat het model de naïeve baseline?
2. Waar zit de fout: bij kleine of grote opleidingen, bij bachelors of masters?
3. Hoe verloopt het aanmeldseizoen van één opleiding, en wat voorspelde het model?

Python rekent alle cijfers uit (:mod:`studentprognose.output.model_performance`)
en schrijft ze als JSON in de pagina. De browser tekent alleen, met de ingebouwde
plotly.js. Zo blijft het bestand klein (één reeks per opleiding in plaats van een
Plotly-trace per opleiding × jaar) en werkt de pagina ook offline, bijvoorbeeld
geopend vanuit een Fabric-lakehouse.
"""

from __future__ import annotations

import html
import json
import math
from importlib import resources

import numpy as np
import pandas as pd

from studentprognose.output import model_performance as mp
from studentprognose.utils.weeks import week_sort_key

ALL_YEARS = "alle"


def _clean(value):
    """Maak waarden JSON-veilig: NaN/inf → None, numpy → Python, floats afgerond."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return None if not math.isfinite(f) else round(f, 4)
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return [_clean(r) for r in frame.to_dict(orient="records")]


def _programme_rows(
    pop: pd.DataFrame, models: list[str], current: pd.DataFrame
) -> list[dict]:
    """Eén tabelrij per opleiding: historische fout, betrouwbaarheid en actuele prognose.

    Opleidingen met een prognose maar zonder geëvalueerd jaar krijgen een rij met
    betrouwbaarheid ``onbekend``, zodat de tabel alle voorspelde opleidingen toont.
    """
    summary = mp.programme_summary(pop, models)
    rows: dict[tuple[str, str], dict] = {}
    for r in summary.to_dict(orient="records"):
        key = (r[mp.PROGRAMME], r[mp.EXAM_TYPE])
        rows[key] = {
            "p": key[0],
            "e": key[1],
            "a": r["actual_mean"],
            "m": {
                m: {
                    "n": int(r[f"n_{m}"]),
                    "w": r[f"wape_{m}"],
                    "b": r[f"bias_{m}"],
                    "t": r[f"trust_{m}"],
                }
                for m in models
            },
        }
    for r in current.to_dict(orient="records"):
        key = (str(r[mp.PROGRAMME]), str(r[mp.EXAM_TYPE]))
        rows.setdefault(
            key,
            {
                "p": key[0],
                "e": key[1],
                "a": None,
                "m": {
                    m: {
                        "n": 0,
                        "w": None,
                        "b": None,
                        "t": None if m == mp.NAIVE else "onbekend",
                    }
                    for m in models
                },
            },
        )
    return list(rows.values())


def _current_payload(current: pd.DataFrame) -> dict:
    """Prognose (en realisatie, indien bekend) van het voorspeljaar per opleiding."""
    models = [m for m in mp.MODEL_COLUMNS if m in current.columns]
    return {
        f"{r[mp.PROGRAMME]}|{r[mp.EXAM_TYPE]}": {
            "a": r[mp.ACTUAL],
            "m": {m: r[m] for m in models},
        }
        for r in current.to_dict(orient="records")
    }


def _performance_payload(units: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Samenvattingen voor 'alle jaren' en per afzonderlijk jaar."""
    models = mp.comparable_models(units)
    pop = mp.common_population(units, models)
    exam_order = sorted(pop[mp.EXAM_TYPE].dropna().astype(str).unique())

    def _block(sub: pd.DataFrame) -> dict:
        return {
            "total": _records(mp.summarise(sub, models)),
            "size": _records(
                mp.summarise(sub, models, by="Grootteklasse", order=mp.SIZE_LABELS)
            ),
            "exam": _records(
                mp.summarise(sub, models, by=mp.EXAM_TYPE, order=exam_order)
            ),
        }

    years = sorted(int(y) for y in pop[mp.YEAR].unique())
    summaries = {ALL_YEARS: _block(pop)}
    # De tabel per opleiding beoordeelt elk model op zijn eigen jaren (zie
    # programme_summary), dus die krijgt alle eenheden i.p.v. de gelijke populatie.
    table_models = [m for m in [*mp.MODEL_COLUMNS, mp.NAIVE] if m in units.columns]
    programmes = {ALL_YEARS: _programme_rows(units, table_models, current)}
    for y in years:
        summaries[str(y)] = _block(pop[pop[mp.YEAR] == y])
        programmes[str(y)] = _programme_rows(
            units[units[mp.YEAR] == y], table_models, current
        )

    unit_rows = []
    for row in pop.to_dict(orient="records"):
        unit_rows.append(
            {
                "y": int(row[mp.YEAR]),
                "p": str(row[mp.PROGRAMME]),
                "e": str(row[mp.EXAM_TYPE]),
                "a": row[mp.ACTUAL],
                "s": str(row["Grootteklasse"]),
                "m": {m: row[m] for m in models},
            }
        )

    return {
        "models": [
            {"key": m, "label": mp.model_label(m), "short": mp.SHORT_LABELS.get(m, m)}
            for m in models
        ],
        "years": years,
        "summaries": summaries,
        "programmes": _clean(programmes),
        "current": _clean(_current_payload(current)),
        "units": _clean(unit_rows),
        "nUnitsAll": len(units),
    }


def _trend_payload(
    data: pd.DataFrame,
    data_cumulative: pd.DataFrame | None,
    data_studentcount: pd.DataFrame | None,
    prediction_year: int,
    predict_week: int | None,
    weeks: list[str],
    final_week: int,
) -> dict:
    """Wekelijkse gewogen vooraanmelders per opleiding per jaar, plus prognose en realisatie."""
    if data_cumulative is None or data_cumulative.empty:
        return {"programmes": []}

    dc = data_cumulative
    pos = {int(w): i for i, w in enumerate(weeks)}
    curves = (
        dc.groupby([mp.PROGRAMME, mp.YEAR, mp.WEEK])["Gewogen vooraanmelders"]
        .sum()
        .reset_index()
    )

    def _first_known(s: pd.Series) -> str:
        return next((v for v in s.dropna().astype(str) if v and v != "Onbekend"), "")

    def _joined(s: pd.Series) -> str:
        return " / ".join(sorted(s.dropna().astype(str).unique()))

    meta = pd.DataFrame(
        index=pd.Index(curves[mp.PROGRAMME].unique(), name=mp.PROGRAMME)
    )
    meta["exam"] = (
        dc.groupby(mp.PROGRAMME)[mp.EXAM_TYPE].agg(_joined)
        if mp.EXAM_TYPE in dc
        else ""
    )
    meta["faculty"] = (
        dc.groupby(mp.PROGRAMME)["Faculteit"].agg(_first_known)
        if "Faculteit" in dc
        else ""
    )
    meta = meta.fillna("")

    # Omvang (voor sortering en de standaardkeuze): recentste bekende eindstand.
    size = (
        curves[curves[mp.YEAR] == curves[mp.YEAR].max()]
        .groupby(mp.PROGRAMME)["Gewogen vooraanmelders"]
        .max()
    )

    actuals: dict[str, dict[str, float]] = {}
    if data_studentcount is not None and {mp.YEAR, mp.PROGRAMME, mp.ACTUAL}.issubset(
        data_studentcount.columns
    ):
        sc = data_studentcount.groupby([mp.PROGRAMME, mp.YEAR])[mp.ACTUAL].sum()
        for (p, y), v in sc.items():
            if v > 0:
                actuals.setdefault(str(p), {})[str(int(y))] = float(v)

    pred_rows = data[data[mp.YEAR] == prediction_year]
    forecast: dict[str, dict[int, float]] = {}
    if "Voorspelde vooraanmelders" in pred_rows.columns and predict_week is not None:
        pw_key = week_sort_key(predict_week, final_week)
        f = pred_rows[pred_rows["Voorspelde vooraanmelders"].notna()]
        f = f[f[mp.WEEK].apply(lambda w: week_sort_key(int(w), final_week) > pw_key)]
        f = f.groupby([mp.PROGRAMME, mp.WEEK])["Voorspelde vooraanmelders"].sum()
        for (p, w), v in f.items():
            forecast.setdefault(str(p), {})[int(w)] = float(v)

    primary = next((m for m in mp.MODEL_COLUMNS if m in data.columns), None)
    predicted: dict[str, float] = {}
    if primary is not None and predict_week is not None:
        pp = pred_rows[pred_rows[mp.WEEK] == predict_week]
        pp = pp.groupby(mp.PROGRAMME)[primary].agg(
            lambda s: s.sum() if s.notna().any() else np.nan
        )
        predicted = {str(p): float(v) for p, v in pp.items() if pd.notna(v)}

    programmes = []
    for prog, sub in curves.groupby(mp.PROGRAMME):
        prog = str(prog)
        series = {}
        for yr, ys in sub.groupby(mp.YEAR):
            arr: list[float | None] = [None] * len(weeks)
            for w, v in zip(ys[mp.WEEK], ys["Gewogen vooraanmelders"]):
                i = pos.get(int(w))
                if i is not None:
                    arr[i] = round(float(v), 1)
            series[str(int(yr))] = arr
        fc = [None] * len(weeks)
        for w, v in forecast.get(prog, {}).items():
            if int(w) in pos:
                fc[pos[int(w)]] = round(v, 1)
        programmes.append(
            {
                "code": prog,
                "exam": meta.at[prog, "exam"] if prog in meta.index else "",
                "faculty": meta.at[prog, "faculty"] if prog in meta.index else "",
                "size": float(size.get(prog, 0.0)),
                "series": series,
                "forecast": fc if any(v is not None for v in fc) else None,
                "predicted": predicted.get(prog),
                "actuals": actuals.get(prog, {}),
            }
        )

    programmes.sort(key=lambda p: -p["size"])
    return {"programmes": _clean(programmes)}


def build_payload(
    data: pd.DataFrame,
    data_cumulative: pd.DataFrame | None,
    data_studentcount: pd.DataFrame | None,
    prediction_year: int,
    predict_week: int | None,
    weeks: list[str],
    final_week: int,
    numerus_fixus: dict | None = None,
) -> dict:
    """Alle data die de cumulatieve pagina nodig heeft, JSON-serialiseerbaar."""
    units = mp.build_evaluation_units(
        data, predict_week, data_studentcount, numerus_fixus
    )
    return {
        "meta": {
            "year": int(prediction_year),
            "week": None if predict_week is None else int(predict_week),
            "finalWeek": int(final_week),
            "weeks": [str(w) for w in weeks],
            "nNumerusFixus": len(numerus_fixus or {}),
            "within": mp.WITHIN_THRESHOLD,
            "sizeLabels": mp.SIZE_LABELS,
            "trust": {
                "high": mp.TRUST_HIGH,
                "medium": mp.TRUST_MEDIUM,
                "minYearsHigh": mp.TRUST_MIN_YEARS_HIGH,
            },
        },
        "performance": _performance_payload(
            units,
            mp.current_predictions(data, prediction_year, predict_week, numerus_fixus),
        ),
        "trend": _trend_payload(
            data,
            data_cumulative,
            data_studentcount,
            prediction_year,
            predict_week,
            weeks,
            final_week,
        ),
    }


def _plotly_js() -> str:
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()


def render_html(payload: dict, nav: list[tuple[str, str, bool]]) -> str:
    """Render de pagina als zelfstandig HTML-document.

    Args:
        payload: Uitvoer van :func:`build_payload`.
        nav: ``(label, href, actief)`` per dashboardpagina die daadwerkelijk bestaat.
    """
    template = (
        resources.files("studentprognose.output")
        .joinpath("templates/cumulative.html")
        .read_text(encoding="utf-8")
    )
    nav_html = "".join(
        f'<a class="nav-link{" is-active" if active else ""}" href="{html.escape(href)}"'
        f"{' aria-current=page' if active else ''}>{html.escape(label)}</a>"
        for label, href, active in nav
    )
    # '</' escapen zodat een waarde de <script>-tag nooit kan sluiten.
    data_json = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace(
        "</", "<\\/"
    )
    return (
        template.replace("/*__PLOTLY__*/", _plotly_js())
        .replace("__NAV__", nav_html)
        .replace("__DATA__", data_json)
    )
