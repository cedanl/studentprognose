"""Dashboardpagina's rond modelperformance: cumulatief, individueel en eindoverzicht.

Alle drie de pagina's delen één opzet en één template. Ze beantwoorden:

1. Hoe groot is de fout in het algemeen, en verslaat het model de naïeve baseline?
2. Waar zit de fout: bij kleine of grote opleidingen, bij bachelors of masters?
3. Per opleiding: hoe goed was het model, en hoeveel vertrouwen verdient de prognose?
4. (Cumulatief/individueel) Hoe verloopt het seizoen van één opleiding?
5. (Eindoverzicht) Wat is de totale prognose, per herkomst en voor numerus fixus?

Elke pagina kan worden gefilterd op herkomst (NL, EER, Niet-EER). De standaard is
"alle herkomsten", waarin de herkomstgroepen per opleiding zijn opgeteld; per
herkomst worden alle cijfers opnieuw berekend op opleiding × herkomst-niveau.

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
    model_units: dict[str, pd.DataFrame], models: list[str], current: pd.DataFrame
) -> list[dict]:
    """Eén tabelrij per opleiding: historische fout, betrouwbaarheid en actuele prognose.

    Elk model wordt beoordeeld op zijn eigen eenheden (``model_units``): alleen de
    herkomstgroepen die dát model voorspelt tellen mee, ook in de realisatie
    (``a``) en de naïeve WAPE (``wn``) ernaast. De naïeve baseline (``m[NAIVE]``)
    en de gemiddelde instroom van de rij komen van het eerste model met een
    evaluatie.

    Opleidingen met een prognose maar zonder geëvalueerd jaar krijgen een rij met
    betrouwbaarheid ``onbekend``, zodat de tabel alle voorspelde opleidingen toont.
    """
    rows: dict[tuple[str, str], dict] = {}
    for m in models:
        units = model_units.get(m)
        if units is None or units.empty:
            continue
        summary = mp.programme_summary(units, [m, mp.NAIVE])
        for r in summary.to_dict(orient="records"):
            key = (r[mp.PROGRAMME], r[mp.EXAM_TYPE])
            row = rows.setdefault(
                key, {"p": key[0], "e": key[1], "a": r["actual_mean"], "m": {}}
            )
            if r[f"n_{m}"] == 0:
                continue
            row["m"][m] = {
                "n": int(r[f"n_{m}"]),
                "a": r[f"actual_{m}"],
                "w": r[f"wape_{m}"],
                "b": r[f"bias_{m}"],
                "t": r[f"trust_{m}"],
                "wn": r[f"naive_{m}"],
            }
            if mp.NAIVE not in row["m"] and r[f"n_{mp.NAIVE}"] > 0:
                row["m"][mp.NAIVE] = {
                    "n": int(r[f"n_{mp.NAIVE}"]),
                    "w": r[f"wape_{mp.NAIVE}"],
                    "b": r[f"bias_{mp.NAIVE}"],
                    "t": None,
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
    by_origin: bool = False,
) -> pd.DataFrame:
    """Prognose per opleiding voor het voorspeljaar, inclusief numerus fixus.

    Elk model telt alleen de herkomstgroepen die het zelf voorspelt; per model
    staan de bijbehorende realisatie (``_a_<model>``) en realisatie van vorig jaar
    (``_p_<model>``) erbij. ``Aantal_studenten`` en ``Naief_vorig_jaar`` zijn die
    van het eerste model met een prognose.

    Numerus-fixusopleidingen tellen niet mee in de evaluatie, maar horen wel in het
    overzicht van prognoses; ze krijgen daar een eigen label.
    """
    keys = [mp.PROGRAMME, mp.EXAM_TYPE, *([mp.ORIGIN] if by_origin else [])]
    frames = []
    for m in models:
        c = mp.current_predictions(
            data,
            prediction_year,
            predict_week,
            None,
            [m],
            by_origin,
            data_studentcount,
            population=[m],
        )
        if not c.empty:
            frames.append(
                c.rename(columns={mp.ACTUAL: f"_a_{m}", mp.NAIVE: f"_p_{m}"}).set_index(
                    keys
                )
            )
    if not frames:
        return pd.DataFrame(
            columns=[
                *keys,
                mp.ACTUAL,
                *models,
                mp.YEAR,
                mp.NAIVE,
                "Faculteit",
            ]
        )
    cur = pd.concat(frames, axis=1).reset_index()
    present = [m for m in models if m in cur.columns]
    for col, prefix in ((mp.ACTUAL, "_a_"), (mp.NAIVE, "_p_")):
        first = pd.Series(np.nan, index=cur.index)
        for m in present:
            take = first.isna() & cur[m].notna()
            first = first.mask(take, cur[f"{prefix}{m}"])
        cur[col] = first.astype("float64")
    cur = cur.assign(**{mp.YEAR: int(prediction_year)})
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
            # Realisatie en vorig jaar van precies de herkomstgroepen die dit model
            # voorspelt; daartegen worden fout en verschil van zijn prognose gemeten.
            "am": {m: r[f"_a_{m}"] for m in models if f"_a_{m}" in r},
            "pm": {m: r[f"_p_{m}"] for m in models if f"_p_{m}" in r},
        }
        for r in current.to_dict(orient="records")
    }


class _Units:
    """Evaluatie-eenheden per (modellen, populatie), gecachet.

    Welke herkomstgroepen meetellen hangt af van de vraag (modelvergelijking of
    tabel per model), dus de payload vraagt de eenheden meerdere keren op met een
    andere ``population``. ``only(h)`` geeft dezelfde eenheden voor één herkomst.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        predict_week: int | None,
        data_studentcount: pd.DataFrame | None,
        numerus_fixus: dict,
        by_origin: bool = False,
    ):
        self._args = (data, predict_week, data_studentcount, numerus_fixus)
        self._by_origin = by_origin
        self._origin: str | None = None
        self._cache: dict[tuple, pd.DataFrame] = {}

    def only(self, origin: str) -> _Units:
        view = _Units.__new__(_Units)
        view.__dict__.update(self.__dict__)
        view._origin = origin
        return view

    def __call__(self, models: list[str], population: list[str]) -> pd.DataFrame:
        key = (tuple(models), tuple(population))
        if key not in self._cache:
            data, week, sc, nf = self._args
            self._cache[key] = mp.build_evaluation_units(
                data, week, sc, nf, list(models), self._by_origin, list(population)
            )
        units = self._cache[key]
        if self._origin is None:
            return units
        return units[units[mp.ORIGIN] == self._origin].drop(columns=[mp.ORIGIN])

    def skipped(self, population: list[str]) -> pd.Series:
        """Realisatie van de herkomstgroepen die buiten de cijfers vallen."""
        data, week, _, nf = self._args
        rows = mp.unpredicted_rows(data, week, nf, population)
        if rows.empty:
            return pd.Series(dtype="float64")
        if self._origin is not None:
            rows = rows[rows[mp.ORIGIN] == self._origin]
        return rows[mp.ACTUAL]


def _performance_payload(
    units_for: _Units,
    models: list[str],
    current: pd.DataFrame,
    nf: set[str],
    origin_units_for: _Units | None = None,
    origins: list[str] | None = None,
) -> dict:
    """Samenvattingen voor 'alle jaren' en per afzonderlijk jaar, plus de tabel.

    Alleen herkomstgroepen met een voorspelling tellen mee (#299). Welke dat zijn
    hangt af van wat je meet:

    - **Modelvergelijking** (grafieken en kerncijfers): de groepen die álle
      vergeleken modellen voorspellen, zodat elk model op dezelfde studenten
      wordt gemeten.
    - **Tabel per opleiding**: elk model op de groepen die het zelf voorspelt.

    Args:
        units_for: Evaluatie-eenheden op opleidingsniveau.
        origin_units_for: Idem op opleiding × herkomst-niveau. Alleen meegeven voor
            de weergave "alle herkomsten": dan krijgt die ook de fout per herkomst.
        origins: Volgorde van de herkomstgroepen in die uitsplitsing.
    """
    primary_units = units_for(models[:1], models[:1]) if models else pd.DataFrame()
    chart_models = (
        mp.comparable_models(units_for(models, models[:1]), models) if models else []
    )
    units = units_for(chart_models, chart_models) if chart_models else primary_units
    pop = mp.common_population(units, chart_models)
    exam_order = sorted(pop[mp.EXAM_TYPE].dropna().astype(str).unique())
    opop = (
        mp.common_population(origin_units_for(chart_models, chart_models), chart_models)
        if origin_units_for is not None and chart_models
        else None
    )

    def _block(sub: pd.DataFrame, osub: pd.DataFrame | None) -> dict:
        out = {
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
        if osub is not None:
            out["origin"] = _records(
                mp.summarise(osub, chart_models, by=mp.ORIGIN, order=origins)
            )
        return out

    years = sorted(int(y) for y in pop[mp.YEAR].unique())
    summaries = {ALL_YEARS: _block(pop, opop)}
    # De tabel per opleiding beoordeelt elk model op zijn eigen herkomstgroepen en
    # jaren (zie programme_summary), dus die krijgt per model eigen eenheden.
    model_units = {m: units_for([m], [m]) for m in models if m != mp.NAIVE}
    programmes = {ALL_YEARS: _programme_rows(model_units, models, current)}
    table_years = sorted(
        {int(y) for u in model_units.values() for y in u[mp.YEAR].unique()}
    )
    for y in table_years:
        if str(y) not in summaries:
            summaries[str(y)] = _block(
                pop[pop[mp.YEAR] == y],
                None if opop is None else opop[opop[mp.YEAR] == y],
            )
        programmes[str(y)] = _programme_rows(
            {m: u[u[mp.YEAR] == y] for m, u in model_units.items()}, models, current
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
    skipped = units_for.skipped(chart_models or models[:1])

    return {
        "models": _model_meta(chart_models),
        "tableModels": _model_meta([m for m in models if m != mp.NAIVE]),
        "years": years,
        "summaries": summaries,
        "programmes": _clean(programmes),
        "current": _clean(_current_payload(current, models, nf)),
        "units": _clean(unit_rows),
        "nUnitsAll": len(primary_units),
        # Herkomstgroepen met een voorspelling maar zonder realisatie, meegeteld als 0.
        "nUnrealised": int(units[mp.UNREALISED].sum())
        if mp.UNREALISED in units.columns
        else 0,
        # Herkomstgroepen met een realisatie maar zonder voorspelling: buiten de cijfers.
        "nUnpredicted": len(skipped),
        "nUnpredictedStudents": float(skipped.sum()),
    }


def _origin_views(
    origin_units: _Units,
    origin_current: pd.DataFrame,
    models: list[str],
    nf: set[str],
    origins: list[str],
) -> dict[str, dict]:
    """Eén volledige performance-payload per herkomstgroep."""
    views = {}
    for h in origins:
        c = origin_current[origin_current[mp.ORIGIN] == h].drop(columns=[mp.ORIGIN])
        views[h] = _performance_payload(origin_units.only(h), models, c, nf)
    return views


def _first_known(s: pd.Series) -> str:
    return next((v for v in s.dropna().astype(str) if v and v != "Onbekend"), "")


def _joined(s: pd.Series) -> str:
    return " / ".join(sorted(s.dropna().astype(str).unique()))


def _origin_of(frame: pd.DataFrame) -> pd.Series:
    """Herkomst per rij als string; ``""`` als de bron geen herkomst kent."""
    if mp.ORIGIN in frame.columns:
        return frame[mp.ORIGIN].astype(str)
    return pd.Series("", index=frame.index)


def _actuals_by_programme(
    data_studentcount: pd.DataFrame | None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Werkelijke instroom per opleiding → herkomst → jaar."""
    actuals: dict[str, dict[str, dict[str, float]]] = {}
    if data_studentcount is None or not {mp.YEAR, mp.PROGRAMME, mp.ACTUAL}.issubset(
        data_studentcount.columns
    ):
        return actuals
    sc = data_studentcount.assign(_h=_origin_of(data_studentcount))
    sc = sc.groupby([mp.PROGRAMME, "_h", mp.YEAR])[mp.ACTUAL].sum()
    for (p, h, y), v in sc.items():
        if v > 0:
            actuals.setdefault(str(p), {}).setdefault(h, {})[str(int(y))] = float(v)
    return actuals


def _forecast_by_programme(
    data: pd.DataFrame, prediction_year: int, predict_week: int | None, final_week: int
) -> dict[str, dict[str, dict[int, float]]]:
    """De doorgetrokken curve na de voorspelweek (SARIMA) per opleiding × herkomst × week."""
    rows = data[data[mp.YEAR] == prediction_year]
    out: dict[str, dict[str, dict[int, float]]] = {}
    if FORECAST not in rows.columns or predict_week is None:
        return out
    pw_key = week_sort_key(predict_week, final_week)
    f = rows[rows[FORECAST].notna()]
    f = f[f[mp.WEEK].apply(lambda w: week_sort_key(int(w), final_week) > pw_key)]
    f = f.assign(_h=_origin_of(f))
    for (p, h, w), v in (
        f.groupby([mp.PROGRAMME, "_h", mp.WEEK])[FORECAST].sum().items()
    ):
        out.setdefault(str(p), {}).setdefault(h, {})[int(w)] = float(v)
    return out


def _predicted_by_programme(
    current: pd.DataFrame, primary: str | None
) -> dict[str, dict[str, float]]:
    """Voorspelde instroom per opleiding → herkomst, uit de prognose op herkomstniveau."""
    if primary is None or current.empty or primary not in current.columns:
        return {}
    cur = current[current[primary].notna()].assign(_h=_origin_of(current))
    pp = cur.groupby([mp.PROGRAMME, "_h"])[primary].sum()
    out: dict[str, dict[str, float]] = {}
    for (p, h), v in pp.items():
        out.setdefault(str(p), {})[h] = float(v)
    return out


def _trend_payload(
    curves: pd.DataFrame,
    value: str,
    source: pd.DataFrame,
    forecast: dict[str, dict[str, dict[int, float]]],
    predicted: dict[str, dict[str, float]],
    actuals: dict[str, dict[str, dict[str, float]]],
    weeks: list[str],
) -> dict:
    """Eén compacte reeks per opleiding × herkomst × jaar, uitgelijnd op de academische weken.

    De browser telt de herkomstgroepen op voor de weergave "alle herkomsten"; zo
    staat elke reeks maar één keer in de pagina.
    """
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
    latest = curves[curves[mp.YEAR] == curves[mp.YEAR].max()]
    size = latest.groupby([mp.PROGRAMME, mp.WEEK])[value].sum().groupby(level=0).max()

    def _row(ws: pd.Series, vs: pd.Series) -> list[float | None]:
        arr: list[float | None] = [None] * len(weeks)
        for w, v in zip(ws, vs):
            i = pos.get(int(w))
            if i is not None:
                arr[i] = round(float(v), 1)
        return arr

    curves = curves.assign(_h=_origin_of(curves))
    programmes = []
    for prog, sub in curves.groupby(mp.PROGRAMME):
        prog = str(prog)
        series: dict[str, dict[str, list]] = {}
        for (h, yr), ys in sub.groupby(["_h", mp.YEAR]):
            series.setdefault(h, {})[str(int(yr))] = _row(ys[mp.WEEK], ys[value])
        fc: dict[str, list] = {}
        for h, fw in forecast.get(prog, {}).items():
            arr = _row(pd.Series(list(fw)), pd.Series(list(fw.values())))
            if any(v is not None for v in arr):
                fc[h] = arr
        programmes.append(
            {
                "code": prog,
                "exam": meta.at[prog, "exam"] if prog in meta.index else "",
                "faculty": meta.at[prog, "faculty"] if prog in meta.index else "",
                "size": float(size.get(prog, 0.0)),
                "series": series,
                "forecast": fc,
                "predicted": predicted.get(prog, {}),
                "actuals": actuals.get(prog, {}),
            }
        )

    programmes.sort(key=lambda p: -p["size"])
    return {"programmes": _clean(programmes)}


def _applicant_curves(data_cumulative: pd.DataFrame | None) -> pd.DataFrame:
    keys = [mp.PROGRAMME, mp.YEAR, mp.WEEK]
    if data_cumulative is None or data_cumulative.empty:
        return pd.DataFrame(columns=[*keys, APPLICANTS])
    if mp.ORIGIN in data_cumulative.columns:
        keys.append(mp.ORIGIN)
    return data_cumulative.groupby(keys)[APPLICANTS].sum().reset_index()


def _individual_curves(xgboost_curve: pd.DataFrame | None) -> pd.DataFrame:
    keys = [mp.PROGRAMME, mp.YEAR, mp.WEEK]
    if xgboost_curve is None or xgboost_curve.empty:
        return pd.DataFrame(columns=[*keys, "XGBoost_cumulative"])
    xc = xgboost_curve.assign(**{mp.PROGRAMME: xgboost_curve[mp.PROGRAMME].astype(str)})
    if mp.ORIGIN in xc.columns:
        keys.append(mp.ORIGIN)
    return xc.groupby(keys)["XGBoost_cumulative"].sum().reset_index()


# ── Eindoverzicht: prognose ───────────────────────────────────────────


def _forecast_totals(cur: pd.DataFrame, primary: str) -> dict:
    """Kerncijfers van de prognose: totaal, t.o.v. vorig jaar en (indien bekend) werkelijk."""
    cur = cur[cur[primary].notna()]
    both = cur[cur[mp.NAIVE].notna()]
    return {
        "total": float(cur[primary].sum()),
        "n": int(cur[[mp.PROGRAMME, mp.EXAM_TYPE]].drop_duplicates().shape[0]),
        "prevTotal": float(both[mp.NAIVE].sum()) if not both.empty else None,
        "prognoseComparable": float(both[primary].sum()) if not both.empty else None,
        "nComparable": int(
            both[[mp.PROGRAMME, mp.EXAM_TYPE]].drop_duplicates().shape[0]
        ),
        "actualTotal": float(cur[mp.ACTUAL].sum())
        if cur[mp.ACTUAL].notna().any()
        else None,
    }


def _forecast_overview(
    current: pd.DataFrame,
    origin_current: pd.DataFrame | None,
    origins: list[str],
    primary: str | None,
    numerus_fixus: dict,
) -> dict | None:
    """Totale prognose, verandering t.o.v. vorig jaar, per herkomst en numerus fixus.

    De herkomstuitsplitsing komt uit dezelfde prognose op herkomstniveau als het
    herkomstfilter, zodat de staafjes optellen tot de kerncijfers per herkomst.
    """
    if primary is None or current.empty or primary not in current.columns:
        return None
    cur = current[current[primary].notna()]

    herkomst: list[dict] = []
    by_origin: dict[str, dict] = {}
    if origin_current is not None and not origin_current.empty:
        # Vorig jaar van dezelfde (voorspelde) herkomstgroepen als de prognose.
        oc_all = origin_current[origin_current[primary].notna()]
        prev = oc_all.groupby(oc_all[mp.ORIGIN].astype(str))[mp.NAIVE].sum(min_count=1)
        for h in origins:
            oc = origin_current[origin_current[mp.ORIGIN] == h]
            if oc[primary].notna().any():
                by_origin[h] = _forecast_totals(oc, primary)
                herkomst.append(
                    {"h": h, "prognose": by_origin[h]["total"], "prev": prev.get(h)}
                )

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
            **_forecast_totals(cur, primary),
            "herkomst": herkomst,
            "byOrigin": by_origin,
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

    Returns:
        ``performance`` bevat de weergave "alle herkomsten"; ``performanceByOrigin``
        dezelfde opbouw per herkomstgroep (leeg zonder kolom ``Herkomst``).
    """
    page = PAGES[track]
    numerus_fixus = numerus_fixus or {}
    nf = {str(k) for k in numerus_fixus}
    models = track_models(track, data)
    primary = models[0] if models else None
    has_origin = mp.ORIGIN in data.columns

    units = _Units(data, predict_week, data_studentcount, numerus_fixus)
    current = _current_frame(
        data, data_studentcount, prediction_year, predict_week, models
    )
    origin_units = origin_current = None
    origins: list[str] = []
    if has_origin:
        origin_units = _Units(
            data, predict_week, data_studentcount, numerus_fixus, by_origin=True
        )
        origin_current = _current_frame(
            data,
            data_studentcount,
            prediction_year,
            predict_week,
            models,
            by_origin=True,
        )
        origins = mp.origin_order(
            pd.concat(
                [origin_units(models, models[:1])[mp.ORIGIN], origin_current[mp.ORIGIN]]
            )
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
            _predicted_by_programme(
                origin_current if origin_current is not None else current, primary
            ),
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
            "origins": origins,
            "trust": {
                "high": mp.TRUST_HIGH,
                "medium": mp.TRUST_MEDIUM,
                "minYearsHigh": mp.TRUST_MIN_YEARS_HIGH,
            },
        },
        "performance": _performance_payload(
            units, models, current, nf, origin_units, origins
        ),
        "performanceByOrigin": (
            _origin_views(origin_units, origin_current, models, nf, origins)
            if has_origin
            else {}
        ),
        "forecast": (
            _forecast_overview(
                current,
                origin_current,
                origins,
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
