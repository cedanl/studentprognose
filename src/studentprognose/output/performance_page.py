"""Dashboardpagina's rond modelperformance: cumulatief, individueel en eindoverzicht.

Alle drie de pagina's delen één opzet en één template. Ze beantwoorden:

1. Hoe groot is de fout in het algemeen, en verslaat het model de naïeve baseline?
2. Waar zit de fout: bij kleine of grote opleidingen, bij bachelors of masters?
3. Per opleiding: hoe goed was het model, en hoeveel vertrouwen verdient de prognose?
4. (Cumulatief/individueel) Hoe verloopt het seizoen van één opleiding?
5. (Eindoverzicht) Wat is de totale prognose, per herkomst en voor numerus fixus?

Python rekent alle cijfers uit (:mod:`studentprognose.output.model_performance`)
en schrijft ze als JSON in de pagina. De browser tekent alleen, met de ingebouwde
plotly.js. Zo blijft het bestand klein en werkt de pagina ook offline, bijvoorbeeld
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
APPLICANTS = "Gewogen vooraanmelders"
FORECAST = "Voorspelde vooraanmelders"
ENSEMBLE_COLUMNS = (
    "Weighted_ensemble_prediction",
    "Average_ensemble_prediction",
    "Ensemble_prediction",
)

# Paginaspecifieke teksten; de rest van de pagina volgt uit de data.
PAGES: dict[str, dict] = {
    "cumulative": {
        "title": "Cumulatief model",
        "nav": "Cumulatief",
        "question": "Hoe goed voorspelt het cumulatieve model de instroom?",
        "trend": {
            "title": "Verloop per opleiding",
            "yTitle": "Gewogen vooraanmelders",
            "lede": "Gewogen vooraanmelders per week. Grijs zijn eerdere jaren, de donkere lijn is het "
            "voorspeljaar tot de voorspelweek, de blauwe stippellijn de prognose van de "
            "vooraanmelders. Rechts: werkelijke instroom (★) en voorspelde instroom (◆).",
            "current": "t/m voorspelweek",
            "forecast": "Prognose vooraanmelders",
        },
        "backtest": "-d c",
    },
    "individual": {
        "title": "Individueel model",
        "nav": "Individueel",
        "question": "Hoe goed voorspelt het individuele model de instroom?",
        "trend": {
            "title": "Verloop per opleiding",
            "yTitle": "Verwachte inschrijvingen",
            "lede": "Verwacht aantal inschrijvingen per week: de som van de inschrijfkansen die "
            "XGBoost per aanmelder berekent (donkere lijn, t/m de voorspelweek) en de "
            "SARIMA-doortrekking naar het eind van het seizoen (stippellijn). Rechts: "
            "werkelijke instroom in eerdere jaren (★) en de voorspelde instroom (◆).",
            "current": "XGBoost t/m voorspelweek",
            "forecast": "SARIMA-doortrekking",
        },
        "backtest": "-d i",
    },
    "final": {
        "title": "Eindoverzicht",
        "nav": "Eindoverzicht",
        "question": "Wat is de prognose, en hoeveel vertrouwen verdient die?",
        "trend": None,
        "backtest": "-d b",
    },
}


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


def track_models(track: str, data: pd.DataFrame) -> list[str]:
    """Voorspelkolommen die deze pagina evalueert, primair model eerst.

    Voor het eindoverzicht is dat het eerste ensemble dat daadwerkelijk gevuld is
    (dezelfde voorkeursvolgorde als de rest van het dashboard), gevolgd door de
    losse sporen.
    """

    def _filled(c: str) -> bool:
        return c in data.columns and data[c].notna().any()

    if track == "final":
        ensemble = next((c for c in ENSEMBLE_COLUMNS if _filled(c)), None)
        parts = [
            c
            for c in ("SARIMA_cumulative", "SARIMA_individual", "Prognose_ratio")
            if _filled(c)
        ]
        return ([ensemble] if ensemble else []) + parts
    return [c for c in mp.TRACK_MODELS[track] if _filled(c)]


def _model_meta(models: list[str]) -> list[dict]:
    return [
        {"key": m, "label": mp.model_label(m), "short": mp.SHORT_LABELS.get(m, m)}
        for m in models
    ]


def _programme_rows(
    units: pd.DataFrame, models: list[str], current: pd.DataFrame
) -> list[dict]:
    """Eén tabelrij per opleiding: historische fout, betrouwbaarheid en actuele prognose.

    Opleidingen met een prognose maar zonder geëvalueerd jaar krijgen een rij met
    betrouwbaarheid ``onbekend``, zodat de tabel alle voorspelde opleidingen toont.
    """
    summary = mp.programme_summary(units, models)
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
        rows.setdefault(key, {"p": key[0], "e": key[1], "a": None, "m": {}})
    return list(rows.values())


def _current_frame(
    data: pd.DataFrame,
    data_studentcount: pd.DataFrame | None,
    prediction_year: int,
    predict_week: int | None,
    models: list[str],
) -> pd.DataFrame:
    """Prognose per opleiding voor het voorspeljaar, inclusief numerus fixus.

    Numerus-fixusopleidingen tellen niet mee in de evaluatie, maar horen wel in het
    overzicht van prognoses; ze krijgen daar een eigen label.
    """
    cur = (
        mp.current_predictions(data, prediction_year, predict_week, None, models)
        if models
        else pd.DataFrame()
    )
    if cur.empty:
        return pd.DataFrame(
            columns=[
                mp.PROGRAMME,
                mp.EXAM_TYPE,
                mp.ACTUAL,
                *models,
                mp.YEAR,
                mp.NAIVE,
                "Faculteit",
            ]
        )
    cur = cur.assign(**{mp.YEAR: int(prediction_year)})
    cur[mp.NAIVE] = mp.previous_year_actuals(cur, data_studentcount)
    if "Faculteit" in data.columns:
        fac = (
            data.assign(**{mp.PROGRAMME: data[mp.PROGRAMME].astype(str)})
            .groupby(mp.PROGRAMME)["Faculteit"]
            .agg(_first_known)
        )
        cur["Faculteit"] = cur[mp.PROGRAMME].map(fac).fillna("")
    else:
        cur["Faculteit"] = ""
    return cur


def _current_payload(current: pd.DataFrame, models: list[str], nf: set[str]) -> dict:
    """Prognose, realisatie vorig jaar (en dit jaar, indien bekend) per opleiding."""
    return {
        f"{r[mp.PROGRAMME]}|{r[mp.EXAM_TYPE]}": {
            "a": r[mp.ACTUAL],
            "prev": r[mp.NAIVE],
            "f": r["Faculteit"],
            "nf": str(r[mp.PROGRAMME]) in nf,
            "m": {m: r[m] for m in models if m in r},
        }
        for r in current.to_dict(orient="records")
    }


def _performance_payload(
    units: pd.DataFrame, models: list[str], current: pd.DataFrame, nf: set[str]
) -> dict:
    """Samenvattingen voor 'alle jaren' en per afzonderlijk jaar, plus de tabel."""
    chart_models = mp.comparable_models(units, models)
    pop = mp.common_population(units, chart_models)
    exam_order = sorted(pop[mp.EXAM_TYPE].dropna().astype(str).unique())

    def _block(sub: pd.DataFrame) -> dict:
        return {
            "total": _records(mp.summarise(sub, chart_models)),
            "size": _records(
                mp.summarise(
                    sub, chart_models, by="Grootteklasse", order=mp.SIZE_LABELS
                )
            ),
            "exam": _records(
                mp.summarise(sub, chart_models, by=mp.EXAM_TYPE, order=exam_order)
            ),
        }

    years = sorted(int(y) for y in pop[mp.YEAR].unique())
    summaries = {ALL_YEARS: _block(pop)}
    # De tabel per opleiding beoordeelt elk model op zijn eigen jaren (zie
    # programme_summary), dus die krijgt alle eenheden i.p.v. de gelijke populatie.
    table_models = [m for m in [*models, mp.NAIVE] if m in units.columns]
    programmes = {ALL_YEARS: _programme_rows(units, table_models, current)}
    for y in sorted(int(y) for y in units[mp.YEAR].unique()):
        if str(y) not in summaries:
            summaries[str(y)] = _block(pop[pop[mp.YEAR] == y])
        programmes[str(y)] = _programme_rows(
            units[units[mp.YEAR] == y], table_models, current
        )

    unit_rows = [
        {
            "y": int(row[mp.YEAR]),
            "p": str(row[mp.PROGRAMME]),
            "e": str(row[mp.EXAM_TYPE]),
            "a": row[mp.ACTUAL],
            "s": str(row["Grootteklasse"]),
            "m": {m: row[m] for m in chart_models},
        }
        for row in pop.to_dict(orient="records")
    ]

    return {
        "models": _model_meta(chart_models),
        "tableModels": _model_meta([m for m in models if m != mp.NAIVE]),
        "years": years,
        "summaries": summaries,
        "programmes": _clean(programmes),
        "current": _clean(_current_payload(current, models, nf)),
        "units": _clean(unit_rows),
        "nUnitsAll": len(units),
    }


# ── Verloop per opleiding ─────────────────────────────────────────────


def _first_known(s: pd.Series) -> str:
    return next((v for v in s.dropna().astype(str) if v and v != "Onbekend"), "")


def _joined(s: pd.Series) -> str:
    return " / ".join(sorted(s.dropna().astype(str).unique()))


def _actuals_by_programme(
    data_studentcount: pd.DataFrame | None,
) -> dict[str, dict[str, float]]:
    actuals: dict[str, dict[str, float]] = {}
    if data_studentcount is None or not {mp.YEAR, mp.PROGRAMME, mp.ACTUAL}.issubset(
        data_studentcount.columns
    ):
        return actuals
    sc = data_studentcount.groupby([mp.PROGRAMME, mp.YEAR])[mp.ACTUAL].sum()
    for (p, y), v in sc.items():
        if v > 0:
            actuals.setdefault(str(p), {})[str(int(y))] = float(v)
    return actuals


def _forecast_by_programme(
    data: pd.DataFrame, prediction_year: int, predict_week: int | None, final_week: int
) -> dict[str, dict[int, float]]:
    """De doorgetrokken curve na de voorspelweek (SARIMA) per opleiding × week."""
    rows = data[data[mp.YEAR] == prediction_year]
    out: dict[str, dict[int, float]] = {}
    if FORECAST not in rows.columns or predict_week is None:
        return out
    pw_key = week_sort_key(predict_week, final_week)
    f = rows[rows[FORECAST].notna()]
    f = f[f[mp.WEEK].apply(lambda w: week_sort_key(int(w), final_week) > pw_key)]
    for (p, w), v in f.groupby([mp.PROGRAMME, mp.WEEK])[FORECAST].sum().items():
        out.setdefault(str(p), {})[int(w)] = float(v)
    return out


def _predicted_by_programme(
    data: pd.DataFrame,
    prediction_year: int,
    predict_week: int | None,
    primary: str | None,
) -> dict[str, float]:
    if primary is None or predict_week is None or primary not in data.columns:
        return {}
    pp = data[(data[mp.YEAR] == prediction_year) & (data[mp.WEEK] == predict_week)]
    pp = pp.groupby(mp.PROGRAMME)[primary].agg(
        lambda s: s.sum() if s.notna().any() else np.nan
    )
    return {str(p): float(v) for p, v in pp.items() if pd.notna(v)}


def _trend_payload(
    curves: pd.DataFrame,
    value: str,
    source: pd.DataFrame,
    forecast: dict[str, dict[int, float]],
    predicted: dict[str, float],
    actuals: dict[str, dict[str, float]],
    weeks: list[str],
) -> dict:
    """Eén compacte reeks per opleiding × jaar, uitgelijnd op de academische weken."""
    if curves.empty:
        return {"programmes": []}
    pos = {int(w): i for i, w in enumerate(weeks)}

    meta = pd.DataFrame(
        index=pd.Index(curves[mp.PROGRAMME].unique(), name=mp.PROGRAMME)
    )
    meta["exam"] = (
        source.groupby(mp.PROGRAMME)[mp.EXAM_TYPE].agg(_joined)
        if mp.EXAM_TYPE in source
        else ""
    )
    meta["faculty"] = (
        source.groupby(mp.PROGRAMME)["Faculteit"].agg(_first_known)
        if "Faculteit" in source
        else ""
    )
    meta = meta.fillna("")

    # Omvang (voor sortering en de standaardkeuze): recentste bekende eindstand.
    size = (
        curves[curves[mp.YEAR] == curves[mp.YEAR].max()]
        .groupby(mp.PROGRAMME)[value]
        .max()
    )

    programmes = []
    for prog, sub in curves.groupby(mp.PROGRAMME):
        prog = str(prog)
        series = {}
        for yr, ys in sub.groupby(mp.YEAR):
            arr: list[float | None] = [None] * len(weeks)
            for w, v in zip(ys[mp.WEEK], ys[value]):
                i = pos.get(int(w))
                if i is not None:
                    arr[i] = round(float(v), 1)
            series[str(int(yr))] = arr
        fc: list[float | None] = [None] * len(weeks)
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


def _applicant_curves(data_cumulative: pd.DataFrame | None) -> pd.DataFrame:
    if data_cumulative is None or data_cumulative.empty:
        return pd.DataFrame(columns=[mp.PROGRAMME, mp.YEAR, mp.WEEK, APPLICANTS])
    return (
        data_cumulative.groupby([mp.PROGRAMME, mp.YEAR, mp.WEEK])[APPLICANTS]
        .sum()
        .reset_index()
    )


def _individual_curves(xgboost_curve: pd.DataFrame | None) -> pd.DataFrame:
    cols = [mp.PROGRAMME, mp.YEAR, mp.WEEK, "XGBoost_cumulative"]
    if xgboost_curve is None or xgboost_curve.empty:
        return pd.DataFrame(columns=cols)
    xc = xgboost_curve.assign(**{mp.PROGRAMME: xgboost_curve[mp.PROGRAMME].astype(str)})
    return xc.groupby(cols[:3])["XGBoost_cumulative"].sum().reset_index()


# ── Eindoverzicht: prognose ───────────────────────────────────────────


def _forecast_overview(
    data: pd.DataFrame,
    current: pd.DataFrame,
    data_studentcount: pd.DataFrame | None,
    prediction_year: int,
    predict_week: int | None,
    primary: str | None,
    numerus_fixus: dict,
) -> dict | None:
    """Totale prognose, verandering t.o.v. vorig jaar, per herkomst en numerus fixus."""
    if primary is None or current.empty or primary not in current.columns:
        return None
    cur = current[current[primary].notna()]
    both = cur[cur[mp.NAIVE].notna()]
    total = float(cur[primary].sum())
    actual_known = cur[mp.ACTUAL].notna().any()

    herkomst: list[dict] = []
    if "Herkomst" in data.columns and predict_week is not None:
        rows = data[
            (data[mp.YEAR] == prediction_year) & (data[mp.WEEK] == predict_week)
        ]
        prog = rows.groupby("Herkomst")[primary].sum(min_count=1)
        prev = pd.Series(dtype="float64")
        if data_studentcount is not None and "Herkomst" in data_studentcount.columns:
            sc = data_studentcount[data_studentcount[mp.YEAR] == prediction_year - 1]
            codes = set(cur[mp.PROGRAMME].astype(str))
            sc = sc[sc[mp.PROGRAMME].astype(str).isin(codes)]
            prev = sc.groupby("Herkomst")[mp.ACTUAL].sum()
        order = ["NL", "EER", "Niet-EER"]
        for h in sorted(
            prog.index.astype(str),
            key=lambda x: (order.index(x) if x in order else 9, x),
        ):
            herkomst.append({"h": h, "prognose": prog.get(h), "prev": prev.get(h)})

    nf_rows = []
    for code, cap in (numerus_fixus or {}).items():
        sub = cur[cur[mp.PROGRAMME].astype(str) == str(code)]
        if sub.empty:
            continue
        nf_rows.append(
            {
                "p": str(code),
                "cap": float(cap),
                "prognose": float(sub[primary].sum()),
                "prev": float(sub[mp.NAIVE].sum())
                if sub[mp.NAIVE].notna().any()
                else None,
            }
        )

    return _clean(
        {
            "total": total,
            "n": int(len(cur)),
            "prevTotal": float(both[mp.NAIVE].sum()) if not both.empty else None,
            "prognoseComparable": float(both[primary].sum())
            if not both.empty
            else None,
            "nComparable": int(len(both)),
            "actualTotal": float(cur[mp.ACTUAL].sum()) if actual_known else None,
            "herkomst": herkomst,
            "nf": nf_rows,
        }
    )


# ── Samenstellen ──────────────────────────────────────────────────────


def build_payload(
    data: pd.DataFrame,
    data_cumulative: pd.DataFrame | None,
    data_studentcount: pd.DataFrame | None,
    prediction_year: int,
    predict_week: int | None,
    weeks: list[str],
    final_week: int,
    numerus_fixus: dict | None = None,
    track: str = "cumulative",
    xgboost_curve: pd.DataFrame | None = None,
) -> dict:
    """Alle data die een performancepagina nodig heeft, JSON-serialiseerbaar.

    Args:
        track: ``"cumulative"``, ``"individual"`` of ``"final"``.
        xgboost_curve: Geaggregeerde XGBoost-curve van het individuele spoor; nodig
            voor het verloop op de individuele pagina.
    """
    page = PAGES[track]
    numerus_fixus = numerus_fixus or {}
    nf = {str(k) for k in numerus_fixus}
    models = track_models(track, data)
    primary = models[0] if models else None
    if models:
        units = mp.build_evaluation_units(
            data, predict_week, data_studentcount, numerus_fixus, models
        )
    else:  # geen enkele voorspelkolom gevuld: niets te evalueren
        units = pd.DataFrame(
            columns=[
                mp.YEAR,
                mp.PROGRAMME,
                mp.EXAM_TYPE,
                mp.ACTUAL,
                "Grootteklasse",
                mp.NAIVE,
            ]
        )
    current = _current_frame(
        data, data_studentcount, prediction_year, predict_week, models
    )

    if track == "cumulative":
        curves, value, source = (
            _applicant_curves(data_cumulative),
            APPLICANTS,
            data_cumulative,
        )
    elif track == "individual":
        curves, value, source = (
            _individual_curves(xgboost_curve),
            "XGBoost_cumulative",
            xgboost_curve,
        )
    else:
        curves, value, source = pd.DataFrame(), "", None
    trend = (
        _trend_payload(
            curves,
            value,
            source if source is not None else pd.DataFrame(),
            _forecast_by_programme(data, prediction_year, predict_week, final_week),
            _predicted_by_programme(data, prediction_year, predict_week, primary),
            _actuals_by_programme(data_studentcount),
            weeks,
        )
        if page["trend"]
        else {"programmes": []}
    )

    return {
        "meta": {
            "track": track,
            "page": page,
            "year": int(prediction_year),
            "week": None if predict_week is None else int(predict_week),
            "finalWeek": int(final_week),
            "weeks": [str(w) for w in weeks],
            "nNumerusFixus": len(nf),
            "within": mp.WITHIN_THRESHOLD,
            "sizeLabels": mp.SIZE_LABELS,
            "trust": {
                "high": mp.TRUST_HIGH,
                "medium": mp.TRUST_MEDIUM,
                "minYearsHigh": mp.TRUST_MIN_YEARS_HIGH,
            },
        },
        "performance": _performance_payload(units, models, current, nf),
        "forecast": (
            _forecast_overview(
                data,
                current,
                data_studentcount,
                prediction_year,
                predict_week,
                primary,
                numerus_fixus,
            )
            if track == "final"
            else None
        ),
        "trend": trend,
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
        .joinpath("templates/performance.html")
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
        .replace("__TITLE__", html.escape(payload["meta"]["page"]["title"]))
        .replace("__NAV__", nav_html)
        .replace("__DATA__", data_json)
    )
