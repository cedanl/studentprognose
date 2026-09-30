"""Interactive Plotly dashboard builder for the studentprognose pipeline.

Generates self-contained HTML pages (no Dash server) under
data/output/visualisations[_ci_test_N{n}]/{individual,cumulative,final}/dashboard.html.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from studentprognose.utils.weeks import (
    DataOption, StudentYearPrediction, week_sort_key, get_all_weeks_ordered,
)
from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK
from studentprognose.output import cumulative_page


# ── Colour palette ──────────────────────────────────────────────────
MODEL_COLOURS = {
    "SARIMA_individual": "#1f77b4",
    "SARIMA_cumulative": "#ff7f0e",
    "Prognose_ratio": "#2ca02c",
    "Ensemble_prediction": "#d62728",
    "Weighted_ensemble_prediction": "#9467bd",
    "Average_ensemble_prediction": "#8c564b",
    "Aantal_studenten": "#333333",
}

DISPLAY_NAMES = {
    "SARIMA_cumulative": "XGBoost (cumulatief)",
    "SARIMA_individual": "SARIMA (individueel)",
    "Prognose_ratio": "Prognose ratio",
    "Weighted_ensemble_prediction": "Ensemble",
    "Aantal_studenten": "Werkelijk",
}

ACTUAL_TREND_COLOUR = "#00bcd4"
CONVERSION_COLOURS = {"predicted": "#4682b4", "actual": "#708090"}
CAPACITY_LINE_COLOUR = "#d62728"

MAPE_GOOD = 0.10
MAPE_WARN = 0.25
DELTA_GOOD = 0.05
DELTA_WARN = 0.15

STATUS_COLOURS = {"good": "#2ca02c", "warn": "#f0ad4e", "bad": "#d62728"}
STATUS_FILLS = {"good": "#d4edda", "warn": "#fff3cd", "bad": "#f8d7da"}

PLOTLY_CONFIG = {"responsive": True, "displayModeBar": False}


def _mape_color(val: float) -> str:
    if val <= MAPE_GOOD:
        return STATUS_COLOURS["good"]
    if val <= MAPE_WARN:
        return STATUS_COLOURS["warn"]
    return STATUS_COLOURS["bad"]


def _mape_row_fill(val: float) -> str:
    if val <= MAPE_GOOD:
        return STATUS_FILLS["good"]
    if val <= MAPE_WARN:
        return STATUS_FILLS["warn"]
    return STATUS_FILLS["bad"]


def _chart_height(n: int, per_item: int = 30, base: int = 200, min_h: int = 350) -> int:
    return max(min_h, n * per_item + base)


def _display(col: str) -> str:
    """Return a human-readable display name for a model column."""
    return DISPLAY_NAMES.get(col, col)


PASTEL_YEAR_COLOURS = [
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]

HERKOMST_COLOURS = {"NL": "#4472C4", "EER": "#ED7D31", "Niet-EER": "#A5A5A5"}


# ── Academic week helpers ───────────────────────────────────────────
def _academic_weeks(final_week: int = FINAL_ACADEMIC_WEEK) -> list[str]:
    """Weekvolgorde voor de week-x-as, afgeleid van de academische jaargrens.

    ``get_all_weeks_ordered`` geeft de seizoensvolgorde (UvA 37→36, legacy 39→38);
    ISO-week 53 (lange ISO-jaren) wordt tussen 52 en 1 ingevoegd — consistent met
    ``week_sort_key`` — zodat Plotly hem op de juiste plek op de x-as zet i.p.v.
    als losse categorie achteraan.
    """
    weeks = get_all_weeks_ordered(final_week)
    i = weeks.index("52")
    return weeks[: i + 1] + ["53"] + weeks[i + 1:]


# ════════════════════════════════════════════════════════════════════
# DashboardBuilder
# ════════════════════════════════════════════════════════════════════


class DashboardBuilder:
    def __init__(
        self,
        data: pd.DataFrame,
        data_option: DataOption,
        numerus_fixus_list: dict,
        student_year_prediction: StudentYearPrediction,
        ci_test_n: int | None,
        cwd: str,
        predict_week: int | None,
        data_cumulative: pd.DataFrame | None,
        data_studentcount: pd.DataFrame | None,
        data_xgboost_curve: pd.DataFrame | None = None,
        xgb_classifier_importance: dict[str, float] | None = None,
        xgb_regressor_importance: dict[str, float] | None = None,
        final_academic_week: int = FINAL_ACADEMIC_WEEK,
    ):
        self.data = data.copy()
        self.data["Weeknummer"] = self.data["Weeknummer"].astype(int)
        # Het dashboard is een presentatielaag: de programmesleutel dient hier als
        # tekstlabel (Plotly-dropdowns, titels) én als filtersleutel binnen het
        # dashboard zelf. We brengen 'm overal naar string zodat numerieke
        # isatcodes (Int64) niet botsen met Plotly's string-only button-labels en
        # de interne filters/numerus_fixus-vergelijkingen op één dtype lopen.
        self.data["Croho groepeernaam"] = self.data["Croho groepeernaam"].astype(str)
        self.data_option = data_option
        self.numerus_fixus_list = {str(k): v for k, v in numerus_fixus_list.items() if v > 0}
        self.student_year_prediction = student_year_prediction
        self.ci_test_n = ci_test_n
        self.cwd = cwd
        self.predict_week = predict_week
        # Academische jaargrens (UvA 36, legacy 38) bepaalt de week-x-as-volgorde
        # en de eindweek-markers; afgeleid i.p.v. hardcoded zodat het dashboard
        # bij de juiste week begint/eindigt.
        self.final_academic_week = final_academic_week
        self.academic_weeks = _academic_weeks(final_academic_week)
        if data_cumulative is not None:
            data_cumulative = data_cumulative.copy()
            data_cumulative["Weeknummer"] = data_cumulative["Weeknummer"].astype(int)
            data_cumulative["Croho groepeernaam"] = data_cumulative["Croho groepeernaam"].astype(str)
        self.data_cumulative = data_cumulative
        if data_studentcount is not None and "Croho groepeernaam" in data_studentcount.columns:
            data_studentcount = data_studentcount.copy()
            data_studentcount["Croho groepeernaam"] = data_studentcount["Croho groepeernaam"].astype(str)
        self.data_studentcount = data_studentcount
        self.data_xgboost_curve = data_xgboost_curve
        self.xgb_classifier_importance = xgb_classifier_importance
        self.xgb_regressor_importance = xgb_regressor_importance

        self.prediction_year = int(self.data["Collegejaar"].max())

        # Deduplicate: replace_latest_data() broadcasts prediction values onto
        # individual student rows. The dashboard only needs one row per group.
        group_cols = ["Croho groepeernaam", "Collegejaar", "Herkomst", "Examentype", "Weeknummer"]
        self.data = self.data.drop_duplicates(subset=group_cols)

        ci_suffix = f"_ci_test_N{ci_test_n}" if ci_test_n is not None else ""
        self.base_dir = os.path.join(cwd, "data", "output", f"visualisations{ci_suffix}")

    # ── Data helpers ─────────────────────────────────────────────────

    def _sort_weeks(self, s: pd.Series) -> pd.Series:
        """Pandas-key voor het sorteren van een Weeknummer-reeks in academische
        volgorde, met de geconfigureerde jaargrens (``final_academic_week``)."""
        return s.apply(lambda w: week_sort_key(int(w), self.final_academic_week))

    def _hist_data(self) -> pd.DataFrame:
        """Return only rows with Collegejaar < prediction_year (actuals only)."""
        return self.data[self.data["Collegejaar"] < self.prediction_year]

    def _get_actual_trend(self, programme: str) -> pd.DataFrame | None:
        """Weighted pre-applicants for *prediction_year*, summed over Herkomst, up to predict_week."""
        if self.data_cumulative is None:
            return None
        dc = self.data_cumulative
        mask = (dc["Croho groepeernaam"] == programme) & (dc["Collegejaar"] == self.prediction_year)
        sub = dc.loc[mask]
        if sub.empty:
            return None
        agg = sub.groupby("Weeknummer")["Gewogen vooraanmelders"].sum().reset_index()
        if self.predict_week is not None:
            pw_key = week_sort_key(self.predict_week)
            agg = agg[agg["Weeknummer"].apply(lambda w: week_sort_key(int(w)) <= pw_key)]
        return agg.sort_values("Weeknummer", key=self._sort_weeks)

    def _best_prediction_col(self) -> str | None:
        """Pick the best available prediction column by priority."""
        for col in (
            "Weighted_ensemble_prediction",
            "Average_ensemble_prediction",
            "Ensemble_prediction",
            "SARIMA_cumulative",
            "SARIMA_individual",
        ):
            if col in self.data.columns and self.data[col].notna().any():
                return col
        return None

    def _hist_accuracy(self) -> pd.DataFrame | None:
        """Per-programme, per-year conversion: week-38 Gewogen vooraanmelders vs Aantal_studenten.

        NOTE: this is NOT model accuracy — it measures the historical relationship
        between weighted pre-applicants and actual enrolments (conversion ratio).
        """
        if self.data_cumulative is None or self.data_studentcount is None:
            return None
        dc = self.data_cumulative
        sc = self.data_studentcount
        hist_years = sorted(y for y in dc["Collegejaar"].unique() if y < self.prediction_year)
        # Keep only the last 5 years for relevance
        hist_years = hist_years[-5:]
        if not hist_years:
            return None

        rows = []
        for yr in hist_years:
            for prog in dc["Croho groepeernaam"].unique():
                w38 = dc[
                    (dc["Croho groepeernaam"] == prog)
                    & (dc["Collegejaar"] == yr)
                    & (dc["Weeknummer"] == 38)
                ]
                predicted = w38["Gewogen vooraanmelders"].sum() if not w38.empty else 0
                act = sc[
                    (sc["Croho groepeernaam"] == prog) & (sc["Collegejaar"] == yr)
                ]
                actual = act["Aantal_studenten"].sum() if not act.empty else 0
                if actual > 0 and predicted > 0:
                    rows.append({
                        "Croho groepeernaam": prog,
                        "Collegejaar": yr,
                        "predicted": predicted,
                        "actual": actual,
                        "mape": abs(predicted - actual) / actual,
                    })
        return pd.DataFrame(rows) if rows else None

    # ── Plotly / HTML helpers ────────────────────────────────────────

    def _add_predict_week_line(self, fig: go.Figure) -> None:
        """Add a red dashed vertical line at predict_week and set academic week order."""
        fig.update_xaxes(categoryorder="array", categoryarray=self.academic_weeks)
        if self.predict_week is None:
            return
        pw_str = str(self.predict_week)
        pw_idx = self.academic_weeks.index(pw_str)
        fig.add_vline(x=pw_idx, line_dash="dash", line_color="red", line_width=1.5)
        fig.add_annotation(
            x=pw_idx, y=1.02, yref="paper", yanchor="bottom",
            text=f"Voorspelweek {self.predict_week}",
            showarrow=False, font=dict(color="red", size=11),
            bgcolor="rgba(255,255,255,0.8)", borderpad=3,
        )

    def _pages(self) -> list[tuple[str, str]]:
        """Dashboardpagina's die :meth:`build_and_save` daadwerkelijk schrijft.

        De navigatie linkt alleen hiernaar, zodat een ``-d c``-run geen dode link
        naar de individuele pagina toont.
        """
        pages = []
        if self._builds_individual():
            pages.append(("Individueel", "../individual/dashboard.html"))
        if self._builds_cumulative():
            pages.append(("Cumulatief", "../cumulative/dashboard.html"))
        pages.append(("Eindoverzicht", "../final/dashboard.html"))
        return pages

    def _builds_individual(self) -> bool:
        return (
            self.data_option in (DataOption.INDIVIDUAL, DataOption.BOTH_DATASETS)
            and "SARIMA_individual" in self.data.columns
        )

    def _builds_cumulative(self) -> bool:
        return self.data_option in (DataOption.CUMULATIVE, DataOption.BOTH_DATASETS) and any(
            c in self.data.columns for c in ("SARIMA_cumulative", "Prognose_ratio")
        )

    def _nav_html(self, active: str) -> str:
        links = []
        for label, href in self._pages():
            cls = "nav-link nav-active" if label == active else "nav-link"
            links.append(f'<a class="{cls}" href="{href}">{label}</a>')
        return '<div class="nav-bar">' + "".join(links) + "</div>"

    def _save_page(
        self,
        subdir: str,
        charts: list[tuple[str, go.Figure]] | list[tuple[str, go.Figure, str]],
        active_label: str,
        kpi_html: str = "",
    ) -> None:
        out_dir = os.path.join(self.base_dir, subdir)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "dashboard.html")

        chart_blocks = []
        for i, entry in enumerate(charts):
            title, fig = entry[0], entry[1]
            desc = entry[2] if len(entry) > 2 else ""
            html = fig.to_html(full_html=False, include_plotlyjs=(i == 0), config=PLOTLY_CONFIG)
            desc_html = f'<p class="chart-desc">{desc}</p>' if desc else ""
            chart_blocks.append(
                f'<details class="chart-section" open>'
                f'<summary class="chart-toggle"><h2>{title}</h2></summary>'
                f'{desc_html}{html}</details>'
            )

        nav = self._nav_html(active_label)
        body = "\n".join(chart_blocks)

        html_doc = f"""<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Studentprognose Dashboard &ndash; {active_label}</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#f4f5f7;color:#222}}
  .page-wrap{{max-width:1400px;margin:0 auto;padding:24px 16px}}
  .nav-bar{{display:flex;gap:8px;margin-bottom:24px}}
  .nav-link{{padding:8px 20px;border-radius:6px;text-decoration:none;color:#555;background:#e8e8e8;font-weight:500}}
  .nav-link:hover{{background:#d0d0d0}}
  .nav-active{{background:#1f77b4!important;color:#fff!important}}
  .kpi-row{{display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap}}
  .kpi-card{{flex:1;min-width:160px;padding:16px 20px;border-radius:8px;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
  .kpi-card .label{{font-size:13px;color:#666;margin-bottom:4px}}
  .kpi-card .value{{font-size:26px;font-weight:700}}
  .kpi-card.good{{border-left:4px solid #2ca02c}}
  .kpi-card.warn{{border-left:4px solid #f0ad4e}}
  .kpi-card.bad{{border-left:4px solid #d62728}}
  .chart-section{{background:#fff;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:20px;margin-bottom:24px}}
  .chart-toggle{{cursor:pointer;list-style:none}}
  .chart-toggle::-webkit-details-marker{{display:none}}
  .chart-toggle h2{{font-size:16px;margin-bottom:4px;color:#333;display:inline}}
  .chart-toggle::before{{content:"▾ ";font-size:14px;color:#999}}
  .chart-section:not([open]) .chart-toggle::before{{content:"▸ "}}
  .chart-desc{{font-size:13px;color:#666;margin-bottom:12px;line-height:1.4}}
  @media(max-width:768px){{
    .page-wrap{{padding:12px 8px}}
    .kpi-row{{flex-direction:column}}
    .kpi-card{{min-width:auto;margin-bottom:8px}}
    .chart-section{{padding:12px}}
    .nav-bar{{flex-wrap:wrap}}
    .nav-link{{padding:6px 12px;font-size:13px}}
  }}
  @media(max-width:480px){{
    .chart-toggle h2{{font-size:14px}}
    .kpi-card .value{{font-size:18px}}
  }}
</style>
</head>
<body>
<div class="page-wrap">
{nav}
{kpi_html}
{body}
</div>
</body>
</html>"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        print(f"  Dashboard saved: {path}")

    # ── KPI cards ────────────────────────────────────────────────────

    def _build_individual_kpi_cards(self) -> str:
        """Six KPI cards tailored to the individual (SARIMA) model."""
        hist = self._hist_data()
        if hist.empty:
            return ""
        mape_col = "MAPE_SARIMA_individual"
        mae_col = "MAE_SARIMA_individual"
        if mape_col not in hist.columns:
            return ""

        if self.predict_week is not None:
            hist = hist[hist["Weeknummer"] == self.predict_week]

        prog_mapes = hist.groupby("Croho groepeernaam")[mape_col].mean().dropna()
        if prog_mapes.empty:
            return ""

        avg_mape = float(prog_mapes.mean())
        median_mape = float(prog_mapes.median())
        below_10 = int((prog_mapes <= MAPE_GOOD).sum())
        total = len(prog_mapes)
        best = prog_mapes.idxmin()
        worst = prog_mapes.idxmax()

        avg_mae = ""
        mae_cls = "good"
        if mae_col in hist.columns:
            prog_maes = hist.groupby("Croho groepeernaam")[mae_col].mean().dropna()
            if not prog_maes.empty:
                m = float(prog_maes.mean())
                avg_mae = f"{m:.1f}"
                mae_cls = "good" if m <= 5 else ("warn" if m <= 15 else "bad")

        avg_cls = "good" if avg_mape <= MAPE_GOOD else ("warn" if avg_mape <= MAPE_WARN else "bad")
        med_cls = "good" if median_mape <= MAPE_GOOD else ("warn" if median_mape <= MAPE_WARN else "bad")
        ratio = below_10 / total if total else 0
        below_cls = "good" if ratio >= 0.70 else ("warn" if ratio >= 0.40 else "bad")

        mae_display = f"{avg_mae} studenten" if avg_mae else "–"

        return f"""<div class="kpi-row">
  <div class="kpi-card {avg_cls}" role="status" aria-label="Gemiddelde MAPE: {avg_mape:.1%}"><div class="label">Gem. MAPE (out-of-sample)</div><div class="value">{avg_mape:.1%}</div></div>
  <div class="kpi-card {med_cls}" role="status" aria-label="Mediaan MAPE: {median_mape:.1%}"><div class="label">Mediaan MAPE</div><div class="value">{median_mape:.1%}</div></div>
  <div class="kpi-card {mae_cls}" role="status" aria-label="Gemiddelde MAE: {mae_display}"><div class="label">Gem. MAE</div><div class="value">{mae_display}</div></div>
  <div class="kpi-card {below_cls}" role="status" aria-label="Opleidingen onder 10% fout: {below_10} van {total}"><div class="label">Opleidingen &lt; 10% fout</div><div class="value">{below_10} van {total}</div></div>
  <div class="kpi-card good" role="status" aria-label="Best presterend: {best}"><div class="label">Best presterend</div><div class="value">{best}<br><span style="font-size:16px">{prog_mapes[best]:.1%}</span></div></div>
  <div class="kpi-card bad" role="status" aria-label="Meest afwijkend: {worst}"><div class="label">Meest afwijkend</div><div class="value">{worst}<br><span style="font-size:16px">{prog_mapes[worst]:.1%}</span></div></div>
</div>"""

    def _model_accuracy_heatmaps(
        self, mape_cols: list[str] | None = None,
    ) -> list[tuple[str, go.Figure]]:
        """Per-model heatmaps: programmes (y) × collegejaar (x), coloured by MAPE.

        Returns one heatmap per MAPE column so each model can be evaluated
        independently. Returns an empty list when no MAPE data is available.
        When *mape_cols* is given, only those columns are used.
        """
        hist = self._hist_data()
        if hist.empty:
            return []

        all_mape_cols = mape_cols or [c for c in hist.columns if c.startswith("MAPE_")]
        all_mape_cols = [c for c in all_mape_cols if c in hist.columns]
        if not all_mape_cols:
            return []
        if hist[all_mape_cols].dropna(how="all").empty:
            return []

        tmp = hist.copy()

        # Point-prediction models (XGBoost, ratio) only predict at predict_week.
        # Using >= would dilute the real error with easier near-deadline weeks
        # from data_latest, inflating apparent accuracy.
        if self.predict_week is not None:
            tmp = tmp[tmp["Weeknummer"] == self.predict_week]

        results: list[tuple[str, go.Figure]] = []

        for col in all_mape_cols:
            if tmp[col].dropna().empty:
                continue

            model_name = col.replace("MAPE_", "")
            pivot = tmp.pivot_table(
                index="Croho groepeernaam", columns="Collegejaar",
                values=col, aggfunc="mean",
            )
            pivot.columns = [str(int(c)) for c in pivot.columns]
            programmes = sorted(pivot.index)
            pivot = pivot.reindex(index=programmes)

            annotations = np.where(
                np.isnan(pivot.values), "",
                np.vectorize(lambda v: f"{v:.0%}")(pivot.values),
            )

            fig = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale=[[0, "#2ca02c"], [0.15, "#8fce8f"], [0.3, "#f0ad4e"], [0.6, "#e74c3c"], [1.0, "#8b0000"]],
                zmin=0, zmax=0.5,
                colorbar=dict(title="MAPE", tickformat=".0%"),
                text=annotations, texttemplate="%{text}", textfont=dict(size=11),
                hovertemplate="Opleiding: %{y}<br>Jaar: %{x}<br>MAPE: %{z:.1%}<extra></extra>",
            ))
            display = _display(model_name)
            fig.update_layout(
                title=f"Modelnauwkeurigheid {display} per opleiding × jaar (MAPE)",
                xaxis_title="Collegejaar", yaxis_title="Opleiding",
                height=_chart_height(len(programmes)),
            )
            results.append((
                f"Modelnauwkeurigheid {display}", fig,
                f"Gemiddelde voorspelfout (MAPE) van {display} per opleiding en collegejaar. "
                "Groen = nauwkeurig, rood = grote afwijking. Vergelijk met de andere modelheatmap om te zien welk model waar beter presteert.",
            ))

        return results

    # ════════════════════════════════════════════════════════════════
    # Shared chart builders
    # ════════════════════════════════════════════════════════════════

    def _model_performance_summary(
        self, mape_cols: list[str], mae_cols: list[str] | None = None,
    ) -> go.Figure | None:
        """Summary table with overall metrics, per-herkomst, per-size, and bias."""
        hist = self._hist_data()
        valid_mape = [c for c in mape_cols if c in hist.columns] if not hist.empty else []
        if hist.empty or not valid_mape:
            ha = self._hist_accuracy()
            if ha is not None and not ha.empty:
                return self._model_performance_summary_from_hist(ha)
            return self._model_performance_summary_from_studentcount()

        if mae_cols is None:
            mae_cols = [c.replace("MAPE_", "MAE_") for c in valid_mape]
        valid_mae = [c for c in mae_cols if c in hist.columns]

        models = [c.replace("MAPE_", "") for c in valid_mape]

        rows_section: list[list[str]] = []
        row_fills: list[str] = []

        def _add_header(label: str) -> None:
            rows_section.append([f"<b>{label}</b>"] + [""] * len(models))
            row_fills.append("#e8edf2")

        def _add_row(label: str, values: list[str], fill: str = "white") -> None:
            rows_section.append([label] + values)
            row_fills.append(fill)

        # ── Overall ──────────────────────────────────────────────
        _add_header("Overzicht")
        for stat_label, agg_fn in [("Gem. MAPE", "mean"), ("Mediaan MAPE", "median")]:
            vals = []
            for mc in valid_mape:
                s = hist[mc].dropna()
                v = float(getattr(s, agg_fn)()) if not s.empty else np.nan
                vals.append(f"{v:.1%}" if pd.notna(v) else "–")
            _add_row(stat_label, vals)

        if valid_mae:
            vals = []
            for mc in valid_mae:
                s = hist[mc].dropna()
                v = float(s.mean()) if not s.empty else np.nan
                vals.append(f"{v:.1f}" if pd.notna(v) else "–")
            _add_row("Gem. MAE (studenten)", vals)

        # Fraction below 10%
        vals = []
        for mc in valid_mape:
            s = hist[mc].dropna()
            if not s.empty:
                prog_mape = hist.groupby("Croho groepeernaam")[mc].mean().dropna()
                below = int((prog_mape <= MAPE_GOOD).sum())
                vals.append(f"{below} / {len(prog_mape)}")
            else:
                vals.append("–")
        _add_row("Opleidingen < 10% fout", vals)

        # ── Per herkomst ─────────────────────────────────────────
        if "Herkomst" in hist.columns:
            _add_header("Per herkomst")
            for herkomst in ["NL", "EER", "Niet-EER"]:
                sub = hist[hist["Herkomst"] == herkomst]
                if sub.empty:
                    continue
                vals = []
                for mc in valid_mape:
                    s = sub[mc].dropna()
                    v = float(s.mean()) if not s.empty else np.nan
                    vals.append(f"{v:.1%}" if pd.notna(v) else "–")
                _add_row(herkomst, vals)

        # ── Per programmagrootte ─────────────────────────────────
        if "Aantal_studenten" in hist.columns:
            _add_header("Per programmagrootte")
            for label, lo, hi in [("Klein (< 20)", 0, 20), ("Groot (≥ 20)", 20, 1e9)]:
                sub = hist[
                    (hist["Aantal_studenten"] >= lo) & (hist["Aantal_studenten"] < hi)
                ]
                if sub.empty:
                    continue
                vals = []
                for mc in valid_mape:
                    s = sub[mc].dropna()
                    v = float(s.mean()) if not s.empty else np.nan
                    vals.append(f"{v:.1%}" if pd.notna(v) else "–")
                _add_row(label, vals)

        # ── Bias ─────────────────────────────────────────────────
        if "Aantal_studenten" in hist.columns:
            _add_header("Bias (voorspelling − werkelijk)")
            for model, mc in zip(models, valid_mape):
                pred_col = model
                if pred_col not in hist.columns:
                    continue
                sub = hist[hist[pred_col].notna() & hist["Aantal_studenten"].notna()].copy()
                if sub.empty:
                    continue
                residual = sub[pred_col] - sub["Aantal_studenten"]
                mean_bias = float(residual.mean())
                over_pct = float((residual > 0).mean())
                vals_row = [f"{mean_bias:+.1f} studenten ({over_pct:.0%} overschatting)"]
                vals_row += [""] * (len(models) - 1)
                _add_row(_display(model), vals_row)

        # ── Naïef baseline ───────────────────────────────────────
        if self.data_studentcount is not None and "Aantal_studenten" in hist.columns:
            _add_header("Vergelijking met naïef model (vorig jaar = voorspelling)")
            prog_years = hist.groupby(["Croho groepeernaam", "Collegejaar"]).first().reset_index()
            sc = self.data_studentcount
            baseline_errors: dict[str, list[float]] = {m: [] for m in models}
            naive_errors: list[float] = []
            for _, row in prog_years.iterrows():
                prog, yr = row["Croho groepeernaam"], row["Collegejaar"]
                actual = sc[
                    (sc["Croho groepeernaam"] == prog) & (sc["Collegejaar"] == yr)
                ]["Aantal_studenten"].sum()
                prev_actual = sc[
                    (sc["Croho groepeernaam"] == prog) & (sc["Collegejaar"] == yr - 1)
                ]["Aantal_studenten"].sum()
                if actual > 0 and prev_actual > 0:
                    naive_errors.append(abs(actual - prev_actual) / actual)
                if actual > 0:
                    for model, mc in zip(models, valid_mape):
                        v = row.get(mc)
                        if pd.notna(v):
                            baseline_errors[model].append(float(v))

            if naive_errors:
                naive_mape = float(np.mean(naive_errors))
                vals = []
                for model in models:
                    if baseline_errors[model]:
                        model_mape = float(np.mean(baseline_errors[model]))
                        diff = model_mape - naive_mape
                        vals.append(f"{model_mape:.1%} vs {naive_mape:.1%} ({diff:+.1%})")
                    else:
                        vals.append("–")
                _add_row("Model vs naïef (MAPE)", vals)

        # ── Build Plotly table ───────────────────────────────────
        header_labels = [""] + [_display(m) for m in models]
        cells_by_col = list(zip(*rows_section)) if rows_section else [[] for _ in header_labels]

        fig = go.Figure(go.Table(
            header=dict(
                values=[f"<b>{h}</b>" for h in header_labels],
                fill_color="#2c3e50", font=dict(color="white", size=13),
                align="center", height=32,
            ),
            cells=dict(
                values=list(cells_by_col),
                fill_color=[row_fills] * len(header_labels),
                align=["left"] + ["center"] * len(models), font=dict(size=12), height=28,
            ),
        ))
        fig.update_layout(height=max(350, 50 + 28 * len(rows_section)), margin=dict(l=10, r=10, t=30, b=10))
        return fig

    def _model_performance_summary_from_hist(self, ha: pd.DataFrame) -> go.Figure:
        """Fallback summary when no MAPE columns exist — uses historical conversion accuracy."""
        prog_mapes = ha.groupby("Croho groepeernaam")["mape"].mean()
        overall_mape = float(prog_mapes.mean())
        median_mape = float(prog_mapes.median())
        below_10 = int((prog_mapes <= MAPE_GOOD).sum())
        total = len(prog_mapes)

        rows = [
            ["<b>Overzicht (conversie-ratio)</b>", ""],
            ["Gem. afwijking", f"{overall_mape:.1%}"],
            ["Mediaan afwijking", f"{median_mape:.1%}"],
            ["Opleidingen < 10% fout", f"{below_10} / {total}"],
        ]

        # Bias from conversion
        ha_copy = ha.copy()
        ha_copy["residual"] = ha_copy["predicted"] - ha_copy["actual"]
        mean_bias = float(ha_copy["residual"].mean())
        over_pct = float((ha_copy["residual"] > 0).mean())
        rows.append(["<b>Bias</b>", ""])
        rows.append(["Gem. afwijking", f"{mean_bias:+.1f} studenten ({over_pct:.0%} overschatting)"])

        fills = ["#e8edf2", "white", "white", "white", "#e8edf2", "white"]
        cells = list(zip(*rows))

        fig = go.Figure(go.Table(
            header=dict(
                values=["<b></b>", "<b>Conversie-ratio</b>"],
                fill_color="#2c3e50", font=dict(color="white", size=13),
                align="center", height=32,
            ),
            cells=dict(
                values=list(cells),
                fill_color=[fills] * 2,
                align=["left", "center"], font=dict(size=12), height=28,
            ),
        ))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
        return fig

    def _model_performance_summary_from_studentcount(self) -> go.Figure | None:
        """Fallback summary using year-over-year student counts as naive baseline."""
        sc = self.data_studentcount
        if sc is None or sc.empty:
            return None

        years = sorted(sc["Collegejaar"].unique())
        if len(years) < 2:
            return None

        rows_data: list[list[str]] = []
        fills: list[str] = []

        rows_data.append(["<b>Naïef baselinemodel (vorig jaar = voorspelling)</b>", ""])
        fills.append("#e8edf2")

        naive_errors = []
        for yr in years[1:]:
            for prog in sc["Croho groepeernaam"].unique():
                actual = sc[(sc["Croho groepeernaam"] == prog) & (sc["Collegejaar"] == yr)]["Aantal_studenten"].sum()
                prev = sc[(sc["Croho groepeernaam"] == prog) & (sc["Collegejaar"] == yr - 1)]["Aantal_studenten"].sum()
                if actual > 0 and prev > 0:
                    naive_errors.append(abs(actual - prev) / actual)

        if not naive_errors:
            return None

        naive_mape = float(np.mean(naive_errors))
        naive_median = float(np.median(naive_errors))
        below_10 = sum(1 for e in naive_errors if e <= MAPE_GOOD)

        rows_data.append(["Gem. MAPE (naïef)", f"{naive_mape:.1%}"])
        fills.append("white")
        rows_data.append(["Mediaan MAPE (naïef)", f"{naive_median:.1%}"])
        fills.append("white")
        rows_data.append(["Meetpunten < 10% fout", f"{below_10} / {len(naive_errors)}"])
        fills.append("white")

        rows_data.append(["<b>Jaarlijkse schommeling</b>", ""])
        fills.append("#e8edf2")
        prog_volatility = sc.groupby("Croho groepeernaam")["Aantal_studenten"].std() / sc.groupby("Croho groepeernaam")["Aantal_studenten"].mean()
        prog_volatility = prog_volatility.dropna().sort_values()
        if not prog_volatility.empty:
            rows_data.append(["Meest stabiel", f"{prog_volatility.index[0]} ({prog_volatility.iloc[0]:.1%} CV)"])
            fills.append("white")
            rows_data.append(["Meest volatiel", f"{prog_volatility.index[-1]} ({prog_volatility.iloc[-1]:.1%} CV)"])
            fills.append("white")

        cells = list(zip(*rows_data))
        fig = go.Figure(go.Table(
            header=dict(
                values=["<b></b>", "<b>Historische stabiliteit</b>"],
                fill_color="#2c3e50", font=dict(color="white", size=13),
                align="center", height=32,
            ),
            cells=dict(
                values=list(cells),
                fill_color=[fills] * 2,
                align=["left", "center"], font=dict(size=12), height=28,
            ),
        ))
        fig.update_layout(height=max(300, 50 + 28 * len(rows_data)), margin=dict(l=10, r=10, t=30, b=10))
        return fig

    def _predicted_vs_actual_scatter(self, pred_col: str) -> go.Figure | None:
        """Scatter: X = predicted, Y = actual. Points coloured by error %."""
        hist = self._hist_data()
        if hist.empty or pred_col not in hist.columns:
            return None

        # Use only predict_week rows (XGBoost/ratio produce one prediction per programme)
        if self.predict_week is not None:
            hist = hist[hist["Weeknummer"] == self.predict_week]

        agg = (
            hist.groupby(["Collegejaar", "Croho groepeernaam"])
            .agg(predicted=(pred_col, "sum"), actual=("Aantal_studenten", "sum"))
            .reset_index()
        )
        agg = agg.dropna(subset=["predicted", "actual"])
        agg = agg[agg["actual"] > 0]
        if agg.empty:
            return None
        agg["error_pct"] = (abs(agg["predicted"] - agg["actual"]) / agg["actual"] * 100).clip(upper=30)
        max_val = max(agg["predicted"].max(), agg["actual"].max()) * 1.05
        sizeref = 2.0 * agg["actual"].max() / (40.0**2)

        fig = go.Figure()

        years = sorted(agg["Collegejaar"].unique())
        year_colours = PASTEL_YEAR_COLOURS[:len(years)]
        for yr, colour in zip(years, year_colours):
            yr_data = agg[agg["Collegejaar"] == yr]
            fig.add_trace(go.Scatter(
                x=yr_data["predicted"], y=yr_data["actual"], mode="markers",
                name=str(int(yr)),
                marker=dict(
                    color=colour,
                    size=yr_data["actual"],
                    sizemode="area",
                    sizeref=sizeref,
                    sizemin=4,
                    line=dict(width=1, color="#333"),
                ),
                text=yr_data.apply(
                    lambda r: (
                        f"{r['Croho groepeernaam']}<br>Jaar: {int(r['Collegejaar'])}"
                        f"<br>Voorspeld: {r['predicted']:.0f}<br>Werkelijk: {r['actual']:.0f}"
                        f"<br>Fout: {r['error_pct']:.1f}%"
                    ), axis=1,
                ),
                hoverinfo="text",
            ))

        fig.update_layout(
            title=f"Voorspeld vs Realisatie ({_display(pred_col)})",
            xaxis_title="Voorspelde inschrijvingen", yaxis_title="Werkelijke inschrijvingen",
            xaxis=dict(range=[0, max_val]),
            yaxis=dict(range=[0, max_val], scaleanchor="x"),
            height=_chart_height(len(agg), per_item=15, min_h=450),
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                      line=dict(dash="dash", color="black", width=2))
        return fig

    def _relative_error_scatter(self, pred_col: str) -> go.Figure | None:
        """Scatter: X = actual enrolment (programme size), Y = MAPE, size = absolute error."""
        hist = self._hist_data()
        if hist.empty or pred_col not in hist.columns:
            return None

        if self.predict_week is not None:
            hist = hist[hist["Weeknummer"] == self.predict_week]

        agg = (
            hist.groupby(["Collegejaar", "Croho groepeernaam"])
            .agg(predicted=(pred_col, "sum"), actual=("Aantal_studenten", "sum"))
            .reset_index()
        )
        agg = agg.dropna(subset=["predicted", "actual"])
        agg = agg[agg["actual"] > 0]
        if agg.empty:
            return None

        agg["mape"] = (abs(agg["predicted"] - agg["actual"]) / agg["actual"] * 100)
        agg["abs_error"] = abs(agg["predicted"] - agg["actual"])

        sizeref = 2.0 * max(agg["abs_error"].max(), 1) / (35.0**2)

        fig = go.Figure()
        years = sorted(agg["Collegejaar"].unique())
        for yr, colour in zip(years, PASTEL_YEAR_COLOURS[:len(years)]):
            yr_data = agg[agg["Collegejaar"] == yr]
            fig.add_trace(go.Scatter(
                x=yr_data["actual"], y=yr_data["mape"], mode="markers",
                name=str(int(yr)),
                marker=dict(
                    color=colour,
                    size=yr_data["abs_error"],
                    sizemode="area",
                    sizeref=sizeref,
                    sizemin=4,
                    line=dict(width=1, color="#333"),
                ),
                text=yr_data.apply(
                    lambda r: (
                        f"{r['Croho groepeernaam']}<br>Jaar: {int(r['Collegejaar'])}"
                        f"<br>Werkelijk: {r['actual']:.0f}<br>Voorspeld: {r['predicted']:.0f}"
                        f"<br>MAPE: {r['mape']:.1f}%<br>Absoluut verschil: {r['abs_error']:.0f}"
                    ), axis=1,
                ),
                hoverinfo="text",
            ))

        max_mape = min(agg["mape"].max() * 1.1, 100)
        for threshold, color, label in [(10, "green", "10%"), (25, "orange", "25%")]:
            if threshold < max_mape:
                fig.add_hline(
                    y=threshold, line_dash="dash", line_color=color, line_width=1.5,
                    annotation_text=label, annotation_position="top right",
                )

        fig.update_layout(
            title=f"Relatieve fout per opleidingsgrootte ({_display(pred_col)})",
            xaxis_title="Werkelijke inschrijvingen (opleidingsgrootte)",
            yaxis_title="MAPE (%)",
            yaxis=dict(range=[0, max_mape]),
            height=_chart_height(len(agg), per_item=15, min_h=450),
        )
        return fig

    def _error_bar_per_year(self, pred_col: str) -> go.Figure | None:
        """Bar chart: MAPE per opleiding with year dropdown.

        XGBoost and ratio produce a single prediction per opleiding per year,
        so a bar chart is the honest visualisation (not a week×opleiding heatmap).
        """
        hist = self._hist_data()
        if hist.empty or pred_col not in hist.columns:
            return None

        # Use predict_week rows only (that's where the actual prediction lives)
        if self.predict_week is not None:
            tmp = hist[hist["Weeknummer"] == self.predict_week].copy()
        else:
            tmp = hist.copy()

        actual = tmp["Aantal_studenten"].replace(0, np.nan)
        tmp["_mape"] = (abs(tmp[pred_col].fillna(0) - tmp["Aantal_studenten"]) / actual)

        years = sorted(tmp["Collegejaar"].unique())
        programmes = sorted(tmp["Croho groepeernaam"].unique())
        if not years or not programmes:
            return None

        fig = go.Figure()
        buttons = []

        for idx, yr in enumerate(years):
            yr_data = (
                tmp[tmp["Collegejaar"] == yr]
                .groupby("Croho groepeernaam")
                .agg(_mape=("_mape", "mean"), _pred=(pred_col, "sum"), _act=("Aantal_studenten", "first"))
                .reindex(programmes)
            )
            mapes = yr_data["_mape"].fillna(0).tolist()
            preds = yr_data["_pred"].fillna(0).tolist()
            acts = yr_data["_act"].fillna(0).tolist()

            colors = [_mape_color(m) for m in mapes]

            visible = idx == len(years) - 1

            fig.add_trace(go.Bar(
                x=programmes, y=mapes,
                marker_color=colors, visible=visible,
                text=[f"{m:.1%}" for m in mapes], textposition="outside",
                hovertemplate=[
                    f"{p}<br>MAPE: {m:.1%}<br>Voorspeld: {pr:.0f}<br>Realisatie: {a:.0f}<extra></extra>"
                    for p, m, pr, a in zip(programmes, mapes, preds, acts)
                ],
            ))

            vis = [False] * len(years)
            vis[idx] = True
            buttons.append(dict(
                label=str(int(yr)), method="update",
                args=[{"visible": vis}, {"title": f"Voorspelfout per opleiding ({_display(pred_col)}) — {int(yr)}"}],
            ))

        default_yr = years[-1]
        fig.update_layout(
            title=f"Voorspelfout per opleiding ({_display(pred_col)}) — {int(default_yr)}",
            xaxis_title="Opleiding", yaxis_title="MAPE",
            yaxis=dict(tickformat=".0%"),
            height=_chart_height(len(programmes), base=200),
            showlegend=False,
            updatemenus=[dict(
                active=len(years) - 1, buttons=buttons,
                x=0, y=1.15, xanchor="left", yanchor="top", direction="down",
            )],
        )
        fig.add_hline(y=MAPE_GOOD, line_dash="dot", line_color="#2ca02c", line_width=1)
        return fig

    # ════════════════════════════════════════════════════════════════
    # PAGE 1 — Individual
    # ════════════════════════════════════════════════════════════════

    def _individual_pipeline_chart(self) -> go.Figure | None:
        """Two-stage ML pipeline: XGBoost aggregated curve + SARIMA extrapolation per programme."""
        if self.data_xgboost_curve is None or self.predict_week is None:
            return None
        pred_col = "SARIMA_individual"
        if pred_col not in self.data.columns:
            return None

        xgb = self.data_xgboost_curve.copy()
        programmes = sorted(xgb["Croho groepeernaam"].unique())
        if not programmes:
            return None

        # SARIMA forecast rows for prediction year (weeks after predict_week)
        sarima = self.data[
            (self.data["Collegejaar"] == self.prediction_year)
            & (self.data["Voorspelde vooraanmelders"].notna())
            & (self.data["Voorspelde vooraanmelders"] > 0)
        ].copy() if "Voorspelde vooraanmelders" in self.data.columns else pd.DataFrame()

        # Actual enrolment from data_studentcount (previous years for context)
        sc = self.data_studentcount
        prev_year = self.prediction_year - 1

        fig = go.Figure()
        prog_traces: dict[str, list[int]] = {}

        for prog in programmes:
            prog_traces[prog] = []

            # ── XGBoost cumulative curve (week 39 → predict_week) ──
            xgb_prog = xgb[xgb["Croho groepeernaam"] == prog].copy()
            if not xgb_prog.empty:
                xgb_agg = (
                    xgb_prog.groupby("Weeknummer")["XGBoost_cumulative"]
                    .sum().reset_index()
                    .sort_values("Weeknummer", key=lambda s: s.apply(week_sort_key))
                )
                idx = len(fig.data)
                fig.add_trace(go.Scatter(
                    x=xgb_agg["Weeknummer"].astype(str),
                    y=xgb_agg["XGBoost_cumulative"],
                    mode="lines+markers", name="XGBoost (geaggregeerd)",
                    line=dict(color="#1f77b4", width=3),
                    marker=dict(size=5),
                    hovertemplate="XGBoost<br>Week %{x}<br>Cumulatief: %{y:.0f}<extra></extra>",
                    visible=False,
                ))
                prog_traces[prog].append(idx)

                # Connection point: last XGBoost value feeds into SARIMA
                xgb_last_week = str(int(xgb_agg["Weeknummer"].iloc[-1]))
                xgb_last_val = xgb_agg["XGBoost_cumulative"].iloc[-1]
            else:
                xgb_last_week = None
                xgb_last_val = None

            # ── SARIMA extrapolation (predict_week → week 38) ──
            if not sarima.empty:
                sar_prog = sarima[sarima["Croho groepeernaam"] == prog].copy()
                if not sar_prog.empty:
                    sar_agg = (
                        sar_prog.groupby("Weeknummer")["Voorspelde vooraanmelders"]
                        .sum().reset_index()
                        .sort_values("Weeknummer", key=lambda s: s.apply(week_sort_key))
                    )
                    # Prepend XGBoost endpoint for visual continuity
                    if xgb_last_week is not None and xgb_last_val is not None:
                        bridge = pd.DataFrame({
                            "Weeknummer": [int(xgb_last_week)],
                            "Voorspelde vooraanmelders": [xgb_last_val],
                        })
                        sar_agg = pd.concat([bridge, sar_agg], ignore_index=True)

                    idx = len(fig.data)
                    fig.add_trace(go.Scatter(
                        x=sar_agg["Weeknummer"].astype(str),
                        y=sar_agg["Voorspelde vooraanmelders"],
                        mode="lines+markers", name="SARIMA (extrapolatie)",
                        line=dict(color="#ff7f0e", width=3, dash="dash"),
                        marker=dict(size=5, symbol="diamond"),
                        hovertemplate="SARIMA<br>Week %{x}<br>Cumulatief: %{y:.0f}<extra></extra>",
                        visible=False,
                    ))
                    prog_traces[prog].append(idx)

            # ── Actual enrolment previous year (reference) ──
            if sc is not None:
                real = sc[
                    (sc["Croho groepeernaam"] == prog) & (sc["Collegejaar"] == prev_year)
                ]["Aantal_studenten"].sum()
                if real > 0:
                    idx = len(fig.data)
                    fig.add_trace(go.Scatter(
                        x=[self.academic_weeks[0], self.academic_weeks[-1]],
                        y=[real, real],
                        mode="lines", name=f"Realisatie {prev_year}",
                        line=dict(color="#333333", dash="dot", width=1.5),
                        hovertemplate=f"Realisatie {prev_year}: {real:.0f}<extra></extra>",
                        visible=False,
                    ))
                    prog_traces[prog].append(idx)

        total_traces = len(fig.data)
        if total_traces == 0:
            return None

        buttons = []
        for prog in programmes:
            vis = [False] * total_traces
            for t_idx in prog_traces.get(prog, []):
                vis[t_idx] = True
            buttons.append(dict(label=prog, method="update", args=[{"visible": vis}]))

        # Activate first programme
        if programmes and prog_traces.get(programmes[0]):
            for t_idx in prog_traces[programmes[0]]:
                fig.data[t_idx].visible = True

        fig.update_layout(
            title=f"ML-pipeline: XGBoost + SARIMA ({self.prediction_year})",
            xaxis_title="Weeknummer (academisch jaar)",
            yaxis_title="Cumulatief voorspeld aantal studenten",
            height=_chart_height(5, min_h=500), hovermode="x unified",
            updatemenus=[dict(
                active=0, buttons=buttons,
                x=0, y=1.15, xanchor="left", yanchor="top", direction="down",
            )],
        )
        self._add_predict_week_line(fig)
        return fig

    def _individual_accuracy_table(self) -> go.Figure | None:
        """Table: opleiding × predicted × actual × delta × MAPE, with year dropdown."""
        hist = self._hist_data()
        pred_col = "SARIMA_individual"
        if hist.empty or pred_col not in hist.columns:
            return None
        if "Aantal_studenten" not in hist.columns:
            return None

        if self.predict_week is not None:
            hist = hist[hist["Weeknummer"] == self.predict_week]

        agg = (
            hist.groupby(["Collegejaar", "Croho groepeernaam"])
            .agg(predicted=(pred_col, "sum"), actual=("Aantal_studenten", "first"))
            .reset_index()
        )
        agg = agg[agg["actual"] > 0].copy()
        if agg.empty:
            return None

        agg["delta"] = agg["predicted"] - agg["actual"]
        agg["mape"] = abs(agg["delta"]) / agg["actual"]

        years = sorted(agg["Collegejaar"].unique())
        fig = go.Figure()
        buttons = []

        for idx, yr in enumerate(years):
            yr_data = agg[agg["Collegejaar"] == yr].sort_values("mape")
            colors = [_mape_row_fill(m) for m in yr_data["mape"]]
            visible = idx == len(years) - 1

            fig.add_trace(go.Table(
                header=dict(
                    values=["Opleiding", "Voorspeld", "Werkelijk", "Verschil", "MAPE"],
                    fill_color="#1f77b4", font=dict(color="white", size=13),
                    align="center",
                ),
                cells=dict(
                    values=[
                        yr_data["Croho groepeernaam"].tolist(),
                        [f"{v:.0f}" for v in yr_data["predicted"]],
                        [f"{v:.0f}" for v in yr_data["actual"]],
                        [f"{v:+.0f}" for v in yr_data["delta"]],
                        [f"{v:.1%}" for v in yr_data["mape"]],
                    ],
                    fill_color=[colors] * 5,
                    align=["left", "center", "center", "center", "center"], font=dict(size=12),
                ),
                visible=visible,
            ))

            vis = [False] * len(years)
            vis[idx] = True
            buttons.append(dict(
                label=str(int(yr)), method="update",
                args=[{"visible": vis}, {"title": f"Voorspelling vs Realisatie — {int(yr)}"}],
            ))

        default_yr = years[-1]
        n_progs = agg["Croho groepeernaam"].nunique()
        fig.update_layout(
            title=f"Voorspelling vs Realisatie — {int(default_yr)}",
            height=_chart_height(n_progs, per_item=35, base=120, min_h=300),
            updatemenus=[dict(
                active=len(years) - 1, buttons=buttons,
                x=0, y=1.15, xanchor="left", yanchor="top", direction="down",
            )],
        )
        return fig

    def _individual_bias_scatter(self) -> go.Figure | None:
        """Scatter: X = programme size, Y = signed residual. Reveals systematic bias."""
        hist = self._hist_data()
        pred_col = "SARIMA_individual"
        if hist.empty or pred_col not in hist.columns:
            return None
        if "Aantal_studenten" not in hist.columns:
            return None

        if self.predict_week is not None:
            hist = hist[hist["Weeknummer"] == self.predict_week]

        agg = (
            hist.groupby(["Collegejaar", "Croho groepeernaam"])
            .agg(predicted=(pred_col, "sum"), actual=("Aantal_studenten", "first"))
            .reset_index()
        )
        agg = agg[(agg["actual"] > 0) & agg["predicted"].notna()].copy()
        if agg.empty:
            return None
        agg["residual"] = agg["predicted"] - agg["actual"]

        fig = go.Figure()
        years = sorted(agg["Collegejaar"].unique())
        for j, yr in enumerate(years):
            yr_data = agg[agg["Collegejaar"] == yr]
            fig.add_trace(go.Scatter(
                x=yr_data["actual"], y=yr_data["residual"],
                mode="markers", name=str(int(yr)),
                marker=dict(
                    color=PASTEL_YEAR_COLOURS[j % len(PASTEL_YEAR_COLOURS)],
                    size=10, line=dict(width=1, color="#333"),
                ),
                customdata=np.column_stack([
                    yr_data["Croho groepeernaam"],
                    yr_data["predicted"].values,
                    yr_data["residual"].values,
                ]),
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    "Voorspeld: %{customdata[1]}<br>"
                    "Werkelijk: %{x:.0f}<br>"
                    "Verschil: %{customdata[2]}"
                    "<extra></extra>"
                ),
            ))

        fig.add_hline(y=0, line_dash="dash", line_color="grey")

        # Linear trend line
        valid = agg.dropna(subset=["actual", "residual"])
        if len(valid) >= 2:
            coeffs = np.polyfit(valid["actual"], valid["residual"], 1)
            x_range = np.array([valid["actual"].min(), valid["actual"].max()])
            fig.add_trace(go.Scatter(
                x=x_range, y=np.polyval(coeffs, x_range),
                mode="lines", name="Trendlijn",
                line=dict(color="rgba(0,0,0,0.4)", dash="dot", width=2),
                showlegend=True,
            ))

        fig.update_layout(
            title="Bias-analyse: over- en onderschatting",
            xaxis_title="Werkelijke inschrijvingen",
            yaxis_title="Voorspeld \u2212 Werkelijk",
            height=_chart_height(len(agg), per_item=15, min_h=450),
        )
        return fig

    def _individual_error_by_herkomst(self) -> go.Figure | None:
        """Bar chart: mean MAPE per herkomst group with error bars."""
        hist = self._hist_data()
        mape_col = "MAPE_SARIMA_individual"
        if hist.empty or mape_col not in hist.columns:
            return None
        if "Herkomst" not in hist.columns:
            return None

        if self.predict_week is not None:
            hist = hist[hist["Weeknummer"] == self.predict_week]

        grouped = hist.groupby("Herkomst")[mape_col].agg(["mean", "std", "count"]).reset_index()
        grouped = grouped.dropna(subset=["mean"])
        if grouped.empty:
            return None

        herkomst_order = list(HERKOMST_COLOURS)
        grouped["_sort"] = grouped["Herkomst"].map(
            {h: i for i, h in enumerate(herkomst_order)}
        )
        grouped = grouped.sort_values("_sort").drop(columns="_sort")

        colors = [HERKOMST_COLOURS.get(h, "#999") for h in grouped["Herkomst"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=grouped["Herkomst"], y=grouped["mean"],
            marker_color=colors,
            error_y=dict(
                type="data",
                array=grouped["std"].fillna(0).tolist(),
                visible=True,
            ),
            text=[
                f"{m:.1%} (n={int(c)})" for m, c in zip(grouped["mean"], grouped["count"])
            ],
            textposition="outside",
            hovertemplate="%{x}<br>Gem. MAPE: %{y:.1%}<extra></extra>",
        ))

        fig.update_layout(
            title="Voorspelfout per herkomstgroep",
            xaxis_title="Herkomst", yaxis_title="Gemiddelde MAPE",
            yaxis=dict(tickformat=".0%"),
            height=_chart_height(3, min_h=380),
            showlegend=False,
        )
        return fig

    def _individual_cockpit(self) -> go.Figure | None:
        """Summary table: prognosis per programme, compared to last year's actual."""
        pred_col = "SARIMA_individual"
        if pred_col not in self.data.columns:
            return None
        pred = self.data[
            (self.data["Collegejaar"] == self.prediction_year)
            & (self.data["Weeknummer"] == self.predict_week)
        ] if self.predict_week is not None else self.data[
            self.data["Collegejaar"] == self.prediction_year
        ]
        if pred.empty or pred[pred_col].dropna().empty:
            return None

        programmes = sorted(pred["Croho groepeernaam"].unique())
        prev_year = self.prediction_year - 1
        sc = self.data_studentcount

        opl, prognose_vals, real_vals, delta_vals = [], [], [], []
        delta_colors = []

        for prog in programmes:
            opl.append(prog)
            prog_pred = pred[pred["Croho groepeernaam"] == prog][pred_col].sum()
            prognose_vals.append(f"{prog_pred:.0f}")

            real = 0
            if sc is not None:
                real = sc[
                    (sc["Croho groepeernaam"] == prog) & (sc["Collegejaar"] == prev_year)
                ]["Aantal_studenten"].sum()
            real_vals.append(f"{real:.0f}" if real > 0 else "\u2013")

            if real > 0:
                delta = (prog_pred - real) / real
                delta_vals.append(f"{delta:+.1%}")
                ad = abs(delta)
                delta_colors.append("#2ca02c" if ad <= 0.05 else ("#f0ad4e" if ad <= 0.15 else "#d62728"))
            else:
                delta_vals.append("\u2013")
                delta_colors.append("#666666")

        # Totaalrij
        total_prog = sum(float(v) for v in prognose_vals)
        total_real = sum(float(v) for v in real_vals if v != "\u2013")
        if total_real > 0:
            total_d = (total_prog - total_real) / total_real
            total_d_str = f"{total_d:+.1%}"
            atd = abs(total_d)
            total_d_color = "#2ca02c" if atd <= 0.05 else ("#f0ad4e" if atd <= 0.15 else "#d62728")
        else:
            total_d_str = "\u2013"
            total_d_color = "#666666"

        opl.append("Totaal")
        prognose_vals.append(f"{total_prog:.0f}")
        real_vals.append(f"{total_real:.0f}" if total_real > 0 else "\u2013")
        delta_vals.append(total_d_str)
        delta_colors.append(total_d_color)

        def _tint(hex_color: str) -> str:
            r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
            return f"rgba({r},{g},{b},0.15)"

        n = len(opl)
        row_bg = ["white"] * (n - 1) + ["#e8e8e8"]
        bold = [11] * (n - 1) + [13]

        fig = go.Figure(go.Table(
            header=dict(
                values=["Opleiding", f"Prognose {self.prediction_year}",
                        f"Realisatie {prev_year}", "\u0394%"],
                fill_color="#1f77b4", font=dict(color="white", size=12), align="center",
            ),
            cells=dict(
                values=[opl, prognose_vals, real_vals, delta_vals],
                fill_color=[
                    row_bg, row_bg, row_bg,
                    [_tint(c) if i < n - 1 else "#e8e8e8" for i, c in enumerate(delta_colors)],
                ],
                font=dict(size=bold), align=["left", "center", "center", "center"],
            ),
        ))
        fig.update_layout(
            title=f"Verwacht aantal studenten per opleiding ({self.prediction_year})",
            height=_chart_height(n, per_item=32, base=120, min_h=300),
        )
        return fig

    def _individual_growth_bars(self) -> go.Figure | None:
        """Horizontal bar chart: predicted enrolment vs previous year per programme."""
        pred_col = "SARIMA_individual"
        if pred_col not in self.data.columns or self.data_studentcount is None:
            return None
        pred = self.data[
            (self.data["Collegejaar"] == self.prediction_year)
            & (self.data["Weeknummer"] == self.predict_week)
        ] if self.predict_week is not None else self.data[
            self.data["Collegejaar"] == self.prediction_year
        ]
        if pred.empty or pred[pred_col].dropna().empty:
            return None

        prev_year = self.prediction_year - 1
        sc = self.data_studentcount
        programmes = sorted(pred["Croho groepeernaam"].unique())

        rows = []
        for prog in programmes:
            prognose = pred[pred["Croho groepeernaam"] == prog][pred_col].sum()
            prev = sc[
                (sc["Croho groepeernaam"] == prog) & (sc["Collegejaar"] == prev_year)
            ]["Aantal_studenten"].sum()
            if prev > 0:
                pct = (prognose - prev) / prev
                rows.append({"prog": prog, "pct": pct, "delta": prognose - prev})

        if not rows:
            return None

        rows.sort(key=lambda r: r["pct"])
        progs = [r["prog"] for r in rows]
        pcts = [max(-2.0, min(2.0, r["pct"])) for r in rows]
        deltas = [r["delta"] for r in rows]
        colors = ["#2ca02c" if p >= 0 else "#d62728" for p in pcts]
        labels = [f"{d:+.0f}" for d in deltas]

        fig = go.Figure(go.Bar(
            y=progs, x=pcts, orientation="h",
            marker_color=colors,
            text=labels, textposition="outside",
            hovertemplate="%{y}<br>%{x:.1%} (%{text})<extra></extra>",
        ))
        fig.add_vline(x=0, line_color="#333", line_width=1)
        fig.update_layout(
            title=f"Verwachte groei/krimp t.o.v. {prev_year}",
            xaxis=dict(title="Verandering", tickformat="+.0%"),
            height=_chart_height(len(progs)),
            margin=dict(l=200),
        )
        return fig

    def _individual_enrollment_history(self) -> go.Figure | None:
        """Grouped bar chart: actual enrolment per programme per year from data_studentcount."""
        if self.data_studentcount is None:
            return None
        sc = self.data_studentcount
        programmes = sorted(self.data["Croho groepeernaam"].unique())
        sc = sc[sc["Croho groepeernaam"].isin(programmes)]
        if sc.empty:
            return None

        # Aggregate per programme per year
        agg = sc.groupby(["Collegejaar", "Croho groepeernaam"])["Aantal_studenten"].sum().reset_index()
        years = sorted(agg["Collegejaar"].unique())
        if not years:
            return None

        fig = go.Figure()
        for j, yr in enumerate(years):
            yr_data = agg[agg["Collegejaar"] == yr].set_index("Croho groepeernaam").reindex(programmes)
            vals = yr_data["Aantal_studenten"].fillna(0).tolist()
            fig.add_trace(go.Bar(
                x=programmes, y=vals, name=str(int(yr)),
                marker_color=PASTEL_YEAR_COLOURS[j % len(PASTEL_YEAR_COLOURS)],
                text=[f"{v:.0f}" for v in vals], textposition="outside",
                hovertemplate="%{x}<br>" + str(int(yr)) + ": %{y:.0f}<extra></extra>",
            ))

        fig.update_layout(
            title="Historische instroom per opleiding",
            xaxis_title="Opleiding", yaxis_title="Inschrijvingen",
            barmode="group",
            height=_chart_height(len(programmes), base=200),
            xaxis=dict(tickangle=-45),
        )
        return fig

    def _individual_herkomst_prediction(self) -> go.Figure | None:
        """Stacked bar: predicted enrolment split by herkomst per programme."""
        pred_col = "SARIMA_individual"
        if pred_col not in self.data.columns:
            return None
        pred = self.data[
            (self.data["Collegejaar"] == self.prediction_year)
            & (self.data["Weeknummer"] == self.predict_week)
        ] if self.predict_week is not None else self.data[
            self.data["Collegejaar"] == self.prediction_year
        ]
        if pred.empty or pred[pred_col].dropna().empty:
            return None

        programmes = sorted(pred["Croho groepeernaam"].unique())

        fig = go.Figure()
        for herkomst in HERKOMST_COLOURS:
            colour = HERKOMST_COLOURS.get(herkomst, "#999")
            vals = []
            for prog in programmes:
                v = pred[
                    (pred["Croho groepeernaam"] == prog) & (pred["Herkomst"] == herkomst)
                ][pred_col].sum()
                vals.append(v if pd.notna(v) else 0)
            fig.add_trace(go.Bar(
                x=programmes, y=vals,
                name=herkomst, marker_color=colour,
                hovertemplate="%{x}<br>" + herkomst + ": %{y:.0f}<extra></extra>",
            ))

        fig.update_layout(
            title=f"Verwachte instroom per herkomst ({self.prediction_year})",
            xaxis_title="Opleiding", yaxis_title="Voorspelde inschrijvingen",
            barmode="stack",
            height=_chart_height(len(programmes), base=200),
            xaxis=dict(tickangle=-45),
        )
        return fig

    def _individual_kpi_cards_current(self) -> str:
        """KPI cards based on current prediction vs last year (always available)."""
        pred_col = "SARIMA_individual"
        if pred_col not in self.data.columns or self.data_studentcount is None:
            return ""
        pred = self.data[
            (self.data["Collegejaar"] == self.prediction_year)
            & (self.data["Weeknummer"] == self.predict_week)
        ] if self.predict_week is not None else self.data[
            self.data["Collegejaar"] == self.prediction_year
        ]
        if pred.empty or pred[pred_col].dropna().empty:
            return ""

        programmes = sorted(pred["Croho groepeernaam"].unique())
        prev_year = self.prediction_year - 1
        sc = self.data_studentcount

        total_prognose = sum(
            pred[pred["Croho groepeernaam"] == p][pred_col].sum() for p in programmes
        )
        total_prev = sum(
            sc[(sc["Croho groepeernaam"] == p) & (sc["Collegejaar"] == prev_year)]["Aantal_studenten"].sum()
            for p in programmes
        )

        # YoY change
        if total_prev > 0:
            yoy = (total_prognose - total_prev) / total_prev
            yoy_cls = "good" if yoy >= 0 else ("warn" if yoy >= -0.05 else "bad")
            yoy_val = f"{yoy:+.1%}"
        else:
            yoy_cls = "warn"
            yoy_val = "\u2013"

        # Biggest grower and biggest decliner
        changes = []
        for p in programmes:
            prog_pred = pred[pred["Croho groepeernaam"] == p][pred_col].sum()
            prog_prev = sc[
                (sc["Croho groepeernaam"] == p) & (sc["Collegejaar"] == prev_year)
            ]["Aantal_studenten"].sum()
            if prog_prev > 0:
                changes.append((p, (prog_pred - prog_prev) / prog_prev))
        if changes:
            changes.sort(key=lambda x: x[1])
            grower = changes[-1]
            decliner = changes[0]
        else:
            grower = ("\u2013", 0)
            decliner = ("\u2013", 0)

        decl_cls = "warn" if decliner[1] < -DELTA_GOOD else "good"
        return f"""<div class="kpi-row">
  <div class="kpi-card good" role="status" aria-label="Totaal verwacht: {total_prognose:.0f} studenten"><div class="label">Totaal verwacht {self.prediction_year}</div><div class="value">{total_prognose:.0f} studenten</div></div>
  <div class="kpi-card {yoy_cls}" role="status" aria-label="Verandering: {yoy_val}"><div class="label">Verandering t.o.v. {prev_year}</div><div class="value">{yoy_val}</div><div class="label">{total_prognose:.0f} vs {total_prev:.0f}</div></div>
  <div class="kpi-card good" role="status" aria-label="Meeste groei: {grower[0]}"><div class="label">Meeste groei</div><div class="value">{grower[0]}<br><span style="font-size:16px">{grower[1]:+.1%}</span></div></div>
  <div class="kpi-card {decl_cls}" role="status" aria-label="Meeste krimp: {decliner[0]}"><div class="label">Meeste krimp</div><div class="value">{decliner[0]}<br><span style="font-size:16px">{decliner[1]:+.1%}</span></div></div>
</div>"""

    def _save_individual(self) -> None:
        pred_col = "SARIMA_individual"
        mape_cols = [c for c in self.data.columns if c.startswith("MAPE_") and "individual" in c]
        charts: list[tuple[str, go.Figure, str]] = []

        # ── Overzicht (altijd beschikbaar) ─────────────────────────
        mae_cols = [c.replace("MAPE_", "MAE_") for c in mape_cols]
        fig = self._model_performance_summary(mape_cols, mae_cols)
        if fig:
            charts.append(("Samenvatting modelprestaties", fig,
                "Overzichtstabel met de belangrijkste evaluatiemetrieken: overall MAPE en MAE, "
                "uitsplitsing naar herkomst en programmagrootte, bias-analyse, "
                "en vergelijking met een naïef baselinemodel (vorig jaar = voorspelling)."))

        fig = self._individual_pipeline_chart()
        if fig:
            charts.append(("ML-pipeline: XGBoost + SARIMA", fig,
                "De twee stappen van het individuele model. "
                "Blauw: XGBoost classificeert per aanmelder en aggregeert tot een cumulatieve curve (t/m voorspelweek). "
                "Oranje: SARIMA extrapoleert de trend naar week 38. "
                "Stippellijn: realisatie vorig jaar als referentie."))

        fig = self._individual_cockpit()
        if fig:
            charts.append(("Verwacht aantal studenten per opleiding", fig,
                "Overzichtstabel: prognose per opleiding op basis van het individuele model, "
                "vergeleken met de realisatie van vorig jaar. Percentage toont de verwachte verandering."))

        fig = self._individual_growth_bars()
        if fig:
            charts.append(("Groei en krimp per opleiding", fig,
                "Verwachte verandering per opleiding ten opzichte van vorig jaar. "
                "Groen = groei, rood = krimp. Het getal toont het absolute verschil in studenten."))

        fig = self._individual_herkomst_prediction()
        if fig:
            charts.append(("Verwachte instroom per herkomst", fig,
                "Verdeling van de voorspelde instroom naar herkomst (NL, EER, niet-EER) per opleiding. "
                "Verschuivingen t.o.v. vorig jaar kunnen wijzen op veranderende wervingspatronen."))

        fig = self._individual_enrollment_history()
        if fig:
            charts.append(("Historische instroom", fig,
                "Werkelijke inschrijvingen per opleiding over de afgelopen jaren. "
                "Biedt context voor de huidige prognose: stijgt of daalt de instroom structureel?"))

        fig = self._feature_importance_chart(
            self.xgb_classifier_importance,
            "Feature importance — XGBoost classifier",
        )
        if fig:
            charts.append(("XGBoost Feature Importance", fig,
                "De belangrijkste kenmerken die het XGBoost-model gebruikt om te voorspellen "
                "of een vooraanmelder zich daadwerkelijk inschrijft. "
                "Categorische variabelen zijn teruggegroepeerd naar de originele kolomnaam."))

        # ── Nauwkeurigheidsanalyse (bij meerdere voorspeljaren) ────
        fig = self._individual_accuracy_table()
        if fig:
            charts.append(("Voorspelling vs Realisatie (tabel)", fig,
                "Voorspeld aantal studenten naast de werkelijke inschrijvingen per opleiding. "
                "Rijkleur: groen &lt; 10% fout, geel 10-25%, rood &gt; 25%. Gebruik de dropdown om per jaar te vergelijken."))

        fig = self._predicted_vs_actual_scatter(pred_col)
        if fig:
            charts.append(("Voorspeld vs Realisatie (scatter)", fig,
                "Elke bol is een opleiding in een bepaald jaar. Bolgrootte toont het werkelijke aantal studenten. "
                "Hoe dichter bij de diagonaal, hoe beter de voorspelling."))

        fig = self._relative_error_scatter(pred_col)
        if fig:
            charts.append(("Relatieve fout per opleidingsgrootte", fig,
                "MAPE (%) uitgezet tegen opleidingsgrootte. Bolgrootte toont het absolute verschil. "
                "Kleine opleidingen linksboven worden relatief slecht voorspeld."))

        charts.extend(self._model_accuracy_heatmaps(mape_cols))

        fig = self._error_bar_per_year(pred_col)
        if fig:
            charts.append(("Voorspelfout per opleiding", fig,
                "MAPE per opleiding per collegejaar. Groen &lt; 10%, geel 10-25%, rood &gt; 25%. "
                "Gebruik de dropdown om jaren te vergelijken en structurele uitschieters te identificeren."))

        fig = self._individual_bias_scatter()
        if fig:
            charts.append(("Bias-analyse", fig,
                "Verschil tussen voorspelling en werkelijkheid uitgezet tegen het werkelijke aantal studenten. "
                "Punten boven de nullijn = overschatting, eronder = onderschatting. "
                "De trendlijn toont of de bias toeneemt bij grotere opleidingen."))

        fig = self._individual_error_by_herkomst()
        if fig:
            charts.append(("Foutanalyse per herkomst", fig,
                "Gemiddelde voorspelfout per herkomstgroep (NL, EER, niet-EER). "
                "Foutbalken tonen de spreiding. Grote verschillen wijzen op ongelijke modelnauwkeurigheid per groep."))

        # KPI: prefer accuracy-based cards, fallback to current-prediction cards
        kpi = self._build_individual_kpi_cards()
        if not kpi:
            kpi = self._individual_kpi_cards_current()
        self._save_page("individual", charts, "Individueel", kpi)

    # ════════════════════════════════════════════════════════════════
    # PAGE 2 — Cumulative
    # ════════════════════════════════════════════════════════════════

    def _save_cumulative(self) -> None:
        """Minimale performancepagina voor het cumulatieve model (zie cumulative_page)."""
        payload = cumulative_page.build_payload(
            data=self.data,
            data_cumulative=self.data_cumulative,
            data_studentcount=self.data_studentcount,
            prediction_year=self.prediction_year,
            predict_week=self.predict_week,
            weeks=self.academic_weeks,
            final_week=self.final_academic_week,
            numerus_fixus=self.numerus_fixus_list,
        )
        nav = [(label, href, label == "Cumulatief") for label, href in self._pages()]
        out_dir = os.path.join(self.base_dir, "cumulative")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "dashboard.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(cumulative_page.render_html(payload, nav))
        print(f"  Dashboard saved: {path}")

    # ════════════════════════════════════════════════════════════════
    # PAGE 3 — Final (Eindoverzicht)
    # ════════════════════════════════════════════════════════════════

    def _cockpit_with_confidence(self) -> go.Figure | None:
        """Summary table: one prognosis number per programme for policymakers."""
        best_col = self._best_prediction_col()
        if best_col is None:
            return None

        pred = self.data[self.data["Collegejaar"] == self.prediction_year]
        if pred.empty:
            return None
        if self.predict_week is not None:
            pred = pred[pred["Weeknummer"] == self.predict_week]

        programmes = sorted(pred["Croho groepeernaam"].unique())
        prev_year = self.prediction_year - 1
        hist = self._hist_data()
        all_mape_cols = [c for c in hist.columns if c.startswith("MAPE_")] if not hist.empty else []

        rows_opl: list[str] = []
        rows_prog: list[str] = []
        rows_real: list[str] = []
        rows_diff: list[str] = []
        rows_conf: list[str] = []
        conf_colors: list[str] = []

        for prog in programmes:
            rows_opl.append(prog)
            prog_pred = pred[pred["Croho groepeernaam"] == prog]

            prognose = prog_pred[best_col].sum()
            rows_prog.append(f"{prognose:.0f}" if pd.notna(prognose) and prognose > 0 else "–")

            realisatie = 0
            if self.data_studentcount is not None:
                sc = self.data_studentcount
                realisatie = sc[
                    (sc["Croho groepeernaam"] == prog) & (sc["Collegejaar"] == prev_year)
                ]["Aantal_studenten"].sum()
            rows_real.append(f"{realisatie:.0f}" if realisatie > 0 else "–")

            if prognose > 0 and realisatie > 0:
                delta = prognose - realisatie
                pct = delta / realisatie
                rows_diff.append(f"{delta:+.0f} ({pct:+.0%})")
            else:
                rows_diff.append("–")

            avg_mape = None
            if not hist.empty and all_mape_cols:
                ph = hist[hist["Croho groepeernaam"] == prog]
                if not ph.empty:
                    vals = ph[all_mape_cols].mean().mean()
                    if pd.notna(vals):
                        avg_mape = float(vals)

            if avg_mape is None and self.data_studentcount is not None and self.data_cumulative is not None:
                dc = self.data_cumulative
                for yr in sorted(dc["Collegejaar"].unique(), reverse=True):
                    if yr >= self.prediction_year:
                        continue
                    w38 = dc[(dc["Croho groepeernaam"] == prog) & (dc["Collegejaar"] == yr) & (dc["Weeknummer"] == 38)]
                    act = self.data_studentcount[
                        (self.data_studentcount["Croho groepeernaam"] == prog)
                        & (self.data_studentcount["Collegejaar"] == yr)
                    ]
                    if not w38.empty and not act.empty:
                        pre = w38["Gewogen vooraanmelders"].sum()
                        stu = act["Aantal_studenten"].sum()
                        if stu > 0:
                            avg_mape = abs(pre - stu) / stu
                    break

            if avg_mape is not None:
                if avg_mape <= MAPE_GOOD:
                    rows_conf.append("Hoog")
                    conf_colors.append("#2ca02c")
                elif avg_mape <= MAPE_WARN:
                    rows_conf.append("Midden")
                    conf_colors.append("#f0ad4e")
                else:
                    rows_conf.append("Laag")
                    conf_colors.append("#d62728")
            else:
                rows_conf.append("–")
                conf_colors.append("#666666")

        # ── Totaalrij ───────────────────────────────────────────────
        rows_opl.append("<b>Totaal</b>")
        total_prog = sum(float(v) for v in rows_prog if v != "–")
        rows_prog.append(f"<b>{total_prog:.0f}</b>")
        total_real = sum(float(v) for v in rows_real if v != "–")
        rows_real.append(f"<b>{total_real:.0f}</b>" if total_real > 0 else "–")
        if total_prog > 0 and total_real > 0:
            delta = total_prog - total_real
            pct = delta / total_real
            rows_diff.append(f"<b>{delta:+.0f} ({pct:+.0%})</b>")
        else:
            rows_diff.append("–")
        rows_conf.append("")
        conf_colors.append("#666666")

        # ── Build table ─────────────────────────────────────────────
        def _tint(hex_color: str) -> str:
            r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
            return f"rgba({r},{g},{b},0.15)"

        n = len(rows_opl)
        row_bg = ["white"] * (n - 1) + ["#e8e8e8"]

        header_vals = ["Opleiding", f"Prognose {self.prediction_year}", f"Realisatie {prev_year}", "Verschil", "Betrouwbaarheid"]
        cell_vals = [rows_opl, rows_prog, rows_real, rows_diff, rows_conf]
        fill_cols = [row_bg, row_bg, row_bg, row_bg,
                     [_tint(c) if i < n - 1 else "#e8e8e8" for i, c in enumerate(conf_colors)]]

        fig = go.Figure(go.Table(
            header=dict(
                values=[f"<b>{h}</b>" for h in header_vals],
                fill_color="#1f77b4", font=dict(color="white", size=12), align="center",
            ),
            cells=dict(
                values=cell_vals,
                fill_color=fill_cols,
                font=dict(size=[11] * (n - 1) + [13]),
                align=["left"] + ["center"] * (len(header_vals) - 1),
            ),
        ))
        fig.update_layout(
            title=f"Verwacht aantal studenten per opleiding ({self.prediction_year})",
            height=_chart_height(n),
        )
        return fig

    def _origin_stacked_bar(self) -> go.Figure | None:
        """Stacked bars per programme: NL / EER / Niet-EER, current vs previous year."""
        if self.data_cumulative is None:
            return None

        dc = self.data_cumulative
        prev_year = self.prediction_year - 1
        programmes = sorted(dc["Croho groepeernaam"].unique())

        # Current year: latest available week per programme × herkomst
        cur = dc[dc["Collegejaar"] == self.prediction_year]
        if self.predict_week is not None:
            pw_key = week_sort_key(self.predict_week)
            cur = cur[cur["Weeknummer"].apply(lambda w: week_sort_key(int(w)) <= pw_key)]
        # Take the value at the maximum week per group
        cur_max = (
            cur.sort_values("Weeknummer", key=self._sort_weeks)
            .groupby(["Croho groepeernaam", "Herkomst"])["Gewogen vooraanmelders"]
            .last().reset_index()
        )

        prev = dc[dc["Collegejaar"] == prev_year]
        prev_max = (
            prev.sort_values("Weeknummer", key=self._sort_weeks)
            .groupby(["Croho groepeernaam", "Herkomst"])["Gewogen vooraanmelders"]
            .last().reset_index()
        )

        fig = go.Figure()
        for herkomst in HERKOMST_COLOURS:
            colour = HERKOMST_COLOURS.get(herkomst, "#999")

            vals = [
                cur_max.loc[
                    (cur_max["Croho groepeernaam"] == p) & (cur_max["Herkomst"] == herkomst),
                    "Gewogen vooraanmelders",
                ].sum()
                for p in programmes
            ]
            fig.add_trace(go.Bar(
                x=programmes, y=vals,
                name=f"{herkomst} ({self.prediction_year})",
                marker_color=colour,
                hovertemplate=f"{herkomst} ({self.prediction_year})<br>" + "%{x}<br>%{y:.0f} vooraanmelders<extra></extra>",
            ))

            vals_prev = [
                prev_max.loc[
                    (prev_max["Croho groepeernaam"] == p) & (prev_max["Herkomst"] == herkomst),
                    "Gewogen vooraanmelders",
                ].sum()
                for p in programmes
            ]
            fig.add_trace(go.Bar(
                x=programmes, y=vals_prev,
                name=f"{herkomst} ({prev_year})",
                marker_color=colour, opacity=0.5,
                hovertemplate=f"{herkomst} ({prev_year})<br>" + "%{x}<br>%{y:.0f} vooraanmelders<extra></extra>",
            ))

        fig.update_layout(
            title="Herkomstverdeling per opleiding",
            barmode="stack", xaxis_title="Opleiding",
            yaxis_title="Gewogen vooraanmelders", height=_chart_height(len(programmes), base=200),
            xaxis=dict(tickangle=-45),
        )
        return fig

    def _numerus_fixus_bars(self) -> go.Figure | None:
        """Horizontal grouped bars: current stand + prognosis, with capacity line."""
        if not self.numerus_fixus_list:
            return None
        best_col = self._best_prediction_col()
        if best_col is None:
            return None

        pred = self.data[self.data["Collegejaar"] == self.prediction_year]
        progs, prog_vals, curr_vals, caps = [], [], [], []
        for prog, cap in self.numerus_fixus_list.items():
            sub = pred[pred["Croho groepeernaam"] == prog]
            if sub.empty:
                continue
            progs.append(prog)
            prog_vals.append(sub[best_col].sum())
            caps.append(cap)

            # Current stand: Gewogen vooraanmelders up to predict_week
            current = 0
            if self.data_cumulative is not None and self.predict_week is not None:
                dc = self.data_cumulative
                pw_key = week_sort_key(self.predict_week)
                mask = (
                    (dc["Croho groepeernaam"] == prog)
                    & (dc["Collegejaar"] == self.prediction_year)
                    & (dc["Weeknummer"].apply(lambda w: week_sort_key(int(w)) <= pw_key))
                )
                sub_dc = dc.loc[mask]
                if not sub_dc.empty:
                    current = sub_dc.groupby("Weeknummer")["Gewogen vooraanmelders"].sum().max()
            curr_vals.append(current)

        if not progs:
            return None

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=progs, x=curr_vals, orientation="h",
            name=f"Huidige stand (week {self.predict_week})",
            marker_color="#aec7e8",
        ))
        fig.add_trace(go.Bar(
            y=progs, x=prog_vals, orientation="h",
            name="Prognose (eindejaar)", marker_color="#1f77b4",
        ))
        for i, cap in enumerate(caps):
            fig.add_shape(
                type="line", x0=cap, x1=cap, y0=i - 0.45, y1=i + 0.45,
                line=dict(color=CAPACITY_LINE_COLOUR, dash="dash", width=2),
            )
            fig.add_annotation(
                x=cap, y=i, text=f"Cap: {cap}", showarrow=False,
                xanchor="left", xshift=5, font=dict(size=10, color=CAPACITY_LINE_COLOUR),
            )
        fig.update_layout(
            title="Numerus Fixus Voortgang",
            xaxis_title="Aantal studenten",
            barmode="group",
            height=_chart_height(len(progs), per_item=70, min_h=300),
        )
        return fig

    def _feature_importance_chart(
        self, importance: dict[str, float] | None, title: str, top_n: int = 15,
    ) -> go.Figure | None:
        if not importance:
            return None

        sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        sorted_feats.reverse()
        features = [f[0] for f in sorted_feats]
        values = [f[1] for f in sorted_feats]
        total = sum(values)
        pcts = [v / total if total > 0 else 0 for v in values]

        fig = go.Figure(go.Bar(
            y=features, x=values, orientation="h",
            marker=dict(
                color=values,
                colorscale=[[0, "#aec7e8"], [1, "#1f77b4"]],
            ),
            text=[f"{p:.1%}" for p in pcts], textposition="outside",
            hovertemplate="%{y}<br>Importance: %{x:.4f} (%{text})<extra></extra>",
        ))
        fig.update_layout(
            title=title,
            xaxis=dict(title="Feature Importance"),
            height=_chart_height(len(features), per_item=25),
            margin=dict(l=250),
        )
        return fig

    def _pipeline_banner(self) -> str:
        labels = {
            DataOption.INDIVIDUAL: ("Individueel", "#1f77b4"),
            DataOption.CUMULATIVE: ("Cumulatief", "#ff7f0e"),
            DataOption.BOTH_DATASETS: ("Beide (ensemble)", "#2ca02c"),
        }
        label, color = labels.get(self.data_option, ("Onbekend", "#999"))
        pw_text = f" | Voorspelweek: {self.predict_week}" if self.predict_week else ""
        n_prog = len(self.data["Croho groepeernaam"].unique())
        return (
            f'<div style="background:{color};color:#fff;padding:14px 24px;'
            f'border-radius:8px;margin-bottom:20px;font-size:18px;font-weight:700">'
            f'Pipeline: {label} | Prognose: {self.prediction_year} | {n_prog} opleidingen{pw_text}</div>'
        )

    def _final_kpi_cards(self) -> str:
        """KPI cards for the final dashboard: totals, year-on-year change, NF overshoot."""
        best_col = self._best_prediction_col()
        if best_col is None:
            return ""

        pred = self.data[self.data["Collegejaar"] == self.prediction_year]
        if pred.empty:
            return ""

        prev_year = self.prediction_year - 1
        programmes = pred["Croho groepeernaam"].unique()

        # Total prognosis
        total_prognose = sum(
            pred[pred["Croho groepeernaam"] == p][best_col].sum() for p in programmes
        )

        # Total previous year
        total_prev = 0
        if self.data_studentcount is not None:
            sc = self.data_studentcount
            for p in programmes:
                total_prev += sc[
                    (sc["Croho groepeernaam"] == p) & (sc["Collegejaar"] == prev_year)
                ]["Aantal_studenten"].sum()

        # Year-on-year change
        if total_prev > 0:
            yoy_pct = (total_prognose - total_prev) / total_prev
            yoy_cls = "good" if yoy_pct >= 0 else ("warn" if yoy_pct >= -0.05 else "bad")
            yoy_val = f"{yoy_pct:+.1%}"
            yoy_sub = f"{total_prognose:.0f} vs {total_prev:.0f}"
        else:
            yoy_cls = "warn"
            yoy_val = "–"
            yoy_sub = ""

        # NF overshoot count
        nf_over = 0
        nf_total = len(self.numerus_fixus_list)
        for prog, cap in self.numerus_fixus_list.items():
            sub = pred[pred["Croho groepeernaam"] == prog]
            if not sub.empty and sub[best_col].sum() > cap:
                nf_over += 1
        nf_cls = "good" if nf_over == 0 else ("warn" if nf_over <= 1 else "bad")

        cards = f"""<div class="kpi-row">
  <div class="kpi-card good" role="status" aria-label="Totaal verwachte eerstejaars: {total_prognose:.0f} studenten"><div class="label">Totaal verwachte eerstejaars {self.prediction_year}</div><div class="value">{total_prognose:.0f} studenten</div></div>
  <div class="kpi-card {yoy_cls}" role="status" aria-label="Verandering: {yoy_val}"><div class="label">Verandering t.o.v. {prev_year}</div><div class="value">{yoy_val}</div><div class="label">{yoy_sub}</div></div>"""

        if nf_total > 0:
            cards += f"""
  <div class="kpi-card {nf_cls}" role="status" aria-label="Numerus fixus boven capaciteit: {nf_over} van {nf_total}"><div class="label">Numerus fixus boven capaciteit</div><div class="value">{nf_over} van {nf_total}</div></div>"""

        cards += "\n</div>"
        return cards

    def _growth_bar_chart(self) -> go.Figure | None:
        """Horizontal bar chart: growth/decline per programme vs previous year."""
        best_col = self._best_prediction_col()
        if best_col is None:
            return None

        pred = self.data[self.data["Collegejaar"] == self.prediction_year]
        if pred.empty:
            return None
        if self.data_studentcount is None:
            return None

        prev_year = self.prediction_year - 1
        sc = self.data_studentcount
        programmes = sorted(pred["Croho groepeernaam"].unique())

        rows = []
        for prog in programmes:
            prognose = pred[pred["Croho groepeernaam"] == prog][best_col].sum()
            prev = sc[
                (sc["Croho groepeernaam"] == prog) & (sc["Collegejaar"] == prev_year)
            ]["Aantal_studenten"].sum()
            if prev > 0:
                pct = (prognose - prev) / prev
                rows.append({"prog": prog, "pct": pct, "delta": prognose - prev})

        if not rows:
            return None

        rows.sort(key=lambda r: r["pct"])
        progs = [r["prog"] for r in rows]
        pcts = [max(-2.0, min(2.0, r["pct"])) for r in rows]
        deltas = [r["delta"] for r in rows]
        colors = ["#2ca02c" if p >= 0 else "#d62728" for p in pcts]
        labels = [f"{d:+.0f}" for d in deltas]

        fig = go.Figure(go.Bar(
            y=progs, x=pcts, orientation="h",
            marker_color=colors,
            text=labels, textposition="outside",
            hovertemplate="%{y}<br>%{x:.1%} (%{text})<extra></extra>",
        ))
        fig.add_vline(x=0, line_color="#333", line_width=1)
        fig.update_layout(
            title=f"Verwachte groei/krimp t.o.v. {prev_year}",
            xaxis=dict(title="Verandering", tickformat="+.0%"),
            height=_chart_height(len(progs)),
            margin=dict(l=200),
        )
        return fig

    def _save_final(self) -> None:
        charts: list[tuple[str, go.Figure, str]] = []

        fig = self._cockpit_with_confidence()
        if fig:
            charts.append(("Verwacht aantal studenten per opleiding", fig,
                "Overzichtstabel met de prognose, realisatie vorig jaar, verschil en betrouwbaarheid per opleiding. "
                "Lage betrouwbaarheid betekent dat de modellen onderling sterk afwijken."))

        fig = self._growth_bar_chart()
        if fig:
            charts.append(("Groei en krimp per opleiding", fig,
                "Verwachte verandering per opleiding t.o.v. vorig jaar. "
                "Groen = groei, rood = krimp. Het getal toont het absolute verschil in studenten."))

        fig = self._origin_stacked_bar()
        if fig:
            charts.append(("Herkomstverdeling", fig,
                "Samenstelling van vooraanmelders naar herkomst (NL, EER, niet-EER) per opleiding. "
                "Verschuivingen t.o.v. vorig jaar kunnen wijzen op veranderende wervingspatronen."))

        fig = self._numerus_fixus_bars()
        if fig:
            charts.append(("Numerus Fixus Voortgang", fig,
                "Voorspeld aantal studenten voor numerus-fixusopleidingen afgezet tegen de capaciteitsgrens. "
                "Balk voorbij de rode lijn = verwachte overschrijding, mogelijk wachtlijst."))

        kpi = self._pipeline_banner() + self._final_kpi_cards()
        self._save_page("final", charts, "Eindoverzicht", kpi)

    # ════════════════════════════════════════════════════════════════
    # Entry point
    # ════════════════════════════════════════════════════════════════

    def build_and_save(self) -> None:
        """Generate all applicable dashboard pages."""
        print("Generating dashboards...")

        if self._builds_individual():
            self._save_individual()

        if self._builds_cumulative():
            self._save_cumulative()

        self._save_final()
        print("Dashboards done.")
