import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.utils.weeks import DataOption, StudentYearPrediction, get_all_weeks_ordered


_COLORS = {
    "SARIMA_individual": "#1f77b4",
    "SARIMA_cumulative": "#ff7f0e",
    "Prognose_ratio": "#2ca02c",
    "Ensemble_prediction": "#d62728",
    "Weighted_ensemble_prediction": "#9467bd",
    "Average_ensemble_prediction": "#8c564b",
    "Aantal_studenten": "#333333",
}

_ACADEMIC_WEEKS = get_all_weeks_ordered()


def _week_sort_key(week):
    w = int(week)
    return w - 39 if w >= 39 else w + 13


def _add_predict_week_line(fig, predict_week):
    """Add a vertical red dashed line at the prediction week and set full academic year x-axis."""
    fig.update_xaxes(categoryorder="array", categoryarray=_ACADEMIC_WEEKS)
    if predict_week is not None:
        pw_str = str(predict_week)
        if pw_str in _ACADEMIC_WEEKS:
            # Plotly interprets numeric-looking strings as axis positions,
            # so we must pass the actual category index for the vline and annotation.
            pw_idx = _ACADEMIC_WEEKS.index(pw_str)
            fig.add_vline(
                x=pw_idx, line_dash="dash", line_color="red", line_width=2,
            )
            fig.add_annotation(
                x=pw_idx, y=1, yref="paper",
                text=f"Week {predict_week}", showarrow=False,
                font=dict(color="red", size=11),
                xanchor="left", yanchor="bottom",
            )
    return fig


def _build_page(title, subtitle, figures):
    """Build a self-contained HTML page from a list of (section_title, fig) tuples."""
    plotly_js_included = False
    divs = []
    for section_title, fig in figures:
        include = not plotly_js_included
        div = fig.to_html(full_html=False, include_plotlyjs=include)
        if include:
            plotly_js_included = True
        divs.append(f'<h3>{section_title}</h3>\n{div}')

    return f"""<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
  h2 {{ margin-top: 0; }}
  h3 {{ color: #333; border-bottom: 1px solid #ddd; padding-bottom: 6px; }}
  .header {{ margin-bottom: 20px; }}
  .meta {{ color: #666; font-size: 14px; }}
</style>
</head>
<body>
<div class="header">
  <h2>{title}</h2>
  <p class="meta">{subtitle}</p>
</div>
{''.join(divs)}
</body>
</html>"""


def _save_page(path, title, subtitle, figures):
    """Build and write an HTML page. Returns path or None if no figures."""
    figs = [(t, f) for t, f in figures if f is not None]
    if not figs:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    html = _build_page(title, subtitle, figs)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Dashboard saved: {path}")
    return path


class DashboardBuilder:
    """Generates separate dashboards per audience into visualisations/ subdirectories.

    Structure:
        data/output/visualisations/individual/  — SARIMA individual performance + reasoning
        data/output/visualisations/cumulative/  — SARIMA cumulative + ratio performance + reasoning
        data/output/visualisations/final/       — Aggregated predictions + confidence for policy makers
    """

    def __init__(self, data, data_option, numerus_fixus_list,
                 student_year_prediction, ci_test_n, cwd, predict_week=None,
                 data_cumulative=None, data_studentcount=None):
        self.data = data.copy()
        self.data_option = data_option
        self.numerus_fixus_list = {k: v for k, v in numerus_fixus_list.items() if v > 0}
        self.student_year_prediction = student_year_prediction
        self.ci_test_n = ci_test_n
        self.cwd = cwd
        self.predict_week = predict_week
        self.data_cumulative = data_cumulative
        self.data_studentcount = data_studentcount

        self.prediction_year = int(self.data["Collegejaar"].max())
        self.pred_cols = [c for c in self.data.columns if c in _COLORS and c != "Aantal_studenten"]
        self.mape_cols = [c for c in self.data.columns if c.startswith("MAPE_")]
        self.has_actuals = "Aantal_studenten" in self.data.columns

    def _vis_dir(self, subdir):
        ci_suffix = f"_ci_test_N{self.ci_test_n}" if self.ci_test_n is not None else ""
        return os.path.join(self.cwd, "data", "output", f"visualisations{ci_suffix}", subdir)

    def _subtitle(self):
        year_label = {
            StudentYearPrediction.FIRST_YEARS: "Eerstejaars",
            StudentYearPrediction.HIGHER_YEARS: "Hogerejaars",
            StudentYearPrediction.VOLUME: "Volume",
        }[self.student_year_prediction]
        return f"{year_label} &mdash; Collegejaar {self.prediction_year}"

    def _get_latest_week(self):
        pred_data = self.data[self.data["Collegejaar"] == self.prediction_year]
        if pred_data.empty:
            return None
        return int(pred_data["Weeknummer"].max())

    def _hist_data(self):
        return self.data[self.data["Collegejaar"] < self.prediction_year]

    def _get_actual_trend(self, programme=None):
        """Get actual pre-registration data for weeks 39→predict_week from data_cumulative.

        Returns a DataFrame with columns: Weeknummer, week_str, Gewogen vooraanmelders
        aggregated (summed) across Herkomst for the given programme.
        If programme is None, returns all programmes (with Croho groepeernaam column).
        """
        if self.data_cumulative is None or self.predict_week is None:
            return pd.DataFrame()

        dc = self.data_cumulative
        if "Gewogen vooraanmelders" not in dc.columns:
            return pd.DataFrame()
        pred_year = self.prediction_year
        pw_sort = _week_sort_key(self.predict_week)

        actual = dc[dc["Collegejaar"] == pred_year].copy()
        if programme is not None:
            actual = actual[actual["Croho groepeernaam"] == programme]
        if actual.empty:
            return pd.DataFrame()

        # Keep weeks up to and including predict_week in academic order
        actual = actual[actual["Weeknummer"].map(_week_sort_key) <= pw_sort]
        if actual.empty:
            return pd.DataFrame()

        # Aggregate: sum across Herkomst per week (and per programme if no filter)
        group_cols = ["Weeknummer"]
        if programme is None:
            group_cols.insert(0, "Croho groepeernaam")

        agg = actual.groupby(group_cols)["Gewogen vooraanmelders"].sum().reset_index()
        agg["week_str"] = agg["Weeknummer"].astype(str)
        agg = agg.sort_values("Weeknummer", key=lambda s: s.map(_week_sort_key))
        return agg

    # ── Public entry point ───────────────────────────────────────────

    def build_and_save(self):
        print("Generating dashboards...")
        if self.data_option in (DataOption.INDIVIDUAL, DataOption.BOTH_DATASETS):
            self._save_individual()
        if self.data_option in (DataOption.CUMULATIVE, DataOption.BOTH_DATASETS):
            self._save_cumulative()
        self._save_final()

    # ── Individual dashboard ─────────────────────────────────────────

    def _save_individual(self):
        col = "SARIMA_individual"
        if col not in self.data.columns:
            return
        mape_col = f"MAPE_{col}"
        _save_page(
            os.path.join(self._vis_dir("individual"), "dashboard.html"),
            "Individueel Model — Performance & Reasoning",
            self._subtitle(),
            [
                ("MAPE per week", self._error_progression([mape_col])),
                ("Voorspelling vs Realisatie", self._prediction_vs_actual(col)),
                ("Fout-heatmap", self._error_heatmap([mape_col], f"MAPE {col}")),
                ("Reasoning: voorspelstabiliteit",
                 self._prediction_stability(col)),
            ],
        )

    # ── Cumulative dashboard ─────────────────────────────────────────

    def _save_cumulative(self):
        cols = [c for c in ["SARIMA_cumulative", "Prognose_ratio"] if c in self.data.columns]
        if not cols:
            return
        mape_cols = [f"MAPE_{c}" for c in cols if f"MAPE_{c}" in self.data.columns]
        _save_page(
            os.path.join(self._vis_dir("cumulative"), "dashboard.html"),
            "Cumulatief Model — Performance & Reasoning",
            self._subtitle(),
            [
                ("Verloop per opleiding (alle jaren)", self._yearly_trend()),
                ("MAPE per week", self._error_progression(mape_cols)),
                ("Fout-heatmap", self._error_heatmap(mape_cols, "Gem. MAPE cumulatieve modellen")),
                ("Reasoning: modelconsensus",
                 self._model_consensus(cols)),
            ],
        )

    # ── Final dashboard (policy makers) ──────────────────────────────

    def _save_final(self):
        _save_page(
            os.path.join(self._vis_dir("final"), "dashboard.html"),
            "Prognose Overzicht — Eindconclusie",
            self._subtitle(),
            [
                ("Verwacht aantal studenten per opleiding",
                 self._cockpit_with_confidence()),
                ("Prognose vs Vorig Jaar", self._prognose_vs_vorig_jaar()),
                ("Herkomstverdeling", self._origin_stacked_bar()),
                ("Numerus Fixus Voortgang", self._numerus_fixus_bars()),
            ],
        )

    # ══════════════════════════════════════════════════════════════════
    #  REUSABLE CHART BUILDERS
    # ══════════════════════════════════════════════════════════════════

    # ── Yearly trend (year-over-year comparison per programme) ───────

    def _yearly_trend(self):
        """Show Gewogen vooraanmelders for each year per programme.

        All years show the full available data from data_cumulative.
        The prediction year is highlighted in bold; the vline marks
        where the prediction was made from.  After the vline, a dashed
        red prognose line shows the SARIMA multi-week forecast
        (Voorspelde vooraanmelders), which is at the same scale as
        Gewogen vooraanmelders.
        """
        if self.data_cumulative is None or "Gewogen vooraanmelders" not in self.data_cumulative.columns:
            return None

        dc = self.data_cumulative
        col_name = "Croho groepeernaam"
        if col_name not in dc.columns and "Groepeernaam Croho" in dc.columns:
            dc = dc.rename(columns={"Groepeernaam Croho": col_name})

        programmes = sorted(dc[col_name].unique())
        if not programmes:
            return None

        years = sorted(dc["Collegejaar"].unique())
        year_palette = [
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
            "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
        ]

        has_forecast = "Voorspelde vooraanmelders" in self.data.columns

        fig = go.Figure()
        buttons = []

        for pi, prog in enumerate(programmes):
            traces_this = []

            for yi, year in enumerate(years):
                year_data = dc[
                    (dc[col_name] == prog) & (dc["Collegejaar"] == year)
                ]
                agg = year_data.groupby("Weeknummer")["Gewogen vooraanmelders"].sum().reset_index()
                agg["week_str"] = agg["Weeknummer"].astype(str)
                agg = agg.sort_values("Weeknummer", key=lambda s: s.map(_week_sort_key))

                is_pred_year = year == self.prediction_year
                color = "#1f77b4" if is_pred_year else year_palette[yi % len(year_palette)]
                width = 3 if is_pred_year else 1.5

                fig.add_trace(go.Scatter(
                    x=agg["week_str"], y=agg["Gewogen vooraanmelders"],
                    name=str(year), mode="lines+markers",
                    line=dict(color=color, width=width),
                    marker=dict(size=3 if not is_pred_year else 5),
                    visible=pi == 0,
                    legendgroup=str(year), showlegend=pi == 0,
                ))
                traces_this.append(len(fig.data) - 1)

            # Add SARIMA multi-week forecast (Voorspelde vooraanmelders)
            if has_forecast:
                pred_data = self.data[
                    (self.data["Croho groepeernaam"] == prog)
                    & (self.data["Collegejaar"] == self.prediction_year)
                ]
                if not pred_data.empty:
                    pred_agg = pred_data.groupby("Weeknummer")["Voorspelde vooraanmelders"] \
                        .sum().reset_index()
                    pred_agg = pred_agg[pred_agg["Voorspelde vooraanmelders"] > 0]
                    pred_agg["week_str"] = pred_agg["Weeknummer"].astype(str)
                    pred_agg = pred_agg.sort_values(
                        "Weeknummer", key=lambda s: s.map(_week_sort_key),
                    )

                    # Prepend the last actual point so the lines connect
                    if self.predict_week is not None and not pred_agg.empty:
                        pw_sort = _week_sort_key(self.predict_week)
                        actual_at_pw = dc[
                            (dc[col_name] == prog)
                            & (dc["Collegejaar"] == self.prediction_year)
                            & (dc["Weeknummer"].map(_week_sort_key) == pw_sort)
                        ]
                        if not actual_at_pw.empty:
                            bridge = pd.DataFrame({
                                "Weeknummer": [self.predict_week],
                                "Voorspelde vooraanmelders": [
                                    actual_at_pw["Gewogen vooraanmelders"].sum()
                                ],
                                "week_str": [str(self.predict_week)],
                            })
                            pred_agg = pd.concat([bridge, pred_agg], ignore_index=True)

                    if not pred_agg.empty:
                        fig.add_trace(go.Scatter(
                            x=pred_agg["week_str"],
                            y=pred_agg["Voorspelde vooraanmelders"],
                            name=f"Prognose {self.prediction_year}",
                            mode="lines+markers",
                            line=dict(color="#d62728", width=3, dash="dash"),
                            marker=dict(size=5),
                            visible=pi == 0,
                            legendgroup="prognose", showlegend=pi == 0,
                        ))
                        traces_this.append(len(fig.data) - 1)

            # Build visibility toggle
            vis = [False] * len(fig.data)
            for idx in traces_this:
                vis[idx] = True
            buttons.append(dict(label=prog, method="update", args=[{"visible": vis}]))

        fig.update_layout(
            xaxis=dict(title="Week"),
            yaxis=dict(title="Gewogen vooraanmelders (cumulatief)"),
            updatemenus=[dict(buttons=buttons, direction="down",
                              x=0, xanchor="left", y=1.15, yanchor="top", showactive=True)],
            margin=dict(l=60, r=20, t=60, b=40), height=500,
        )
        return _add_predict_week_line(fig, self.predict_week)

    # ── Error progression (MAPE over weeks) ──────────────────────────

    def _error_progression(self, mape_cols):
        mape_cols = [c for c in mape_cols if c in self.data.columns]
        if not mape_cols:
            return None
        hist = self._hist_data()
        if hist.empty:
            return None

        fig = go.Figure()
        for col in mape_cols:
            model_name = col.replace("MAPE_", "")
            agg = hist.groupby("Weeknummer")[col].mean().reset_index()
            agg["week_str"] = agg["Weeknummer"].astype(str)
            agg = agg.sort_values("Weeknummer", key=lambda s: s.map(_week_sort_key))
            agg[col] = agg[col].clip(upper=2.0)
            fig.add_trace(go.Scatter(
                x=agg["week_str"], y=agg[col],
                name=model_name, mode="lines+markers",
                line=dict(color=_COLORS.get(model_name), width=2),
            ))
        fig.update_layout(
            xaxis=dict(title="Week"),
            yaxis=dict(title="MAPE", tickformat=".0%"),
            margin=dict(l=60, r=20, t=40, b=40), height=400,
        )
        return _add_predict_week_line(fig, self.predict_week)

    # ── Prediction vs actual (single model, dropdown per opleiding) ──

    def _prediction_vs_actual(self, pred_col):
        if pred_col not in self.data.columns:
            return None

        programmes = sorted(self.data["Croho groepeernaam"].unique())
        if not programmes:
            return None

        fig = go.Figure()
        buttons = []
        traces_per = 3  # actual trend, prediction, realisatie
        for i, prog in enumerate(programmes):
            # Actual pre-registration data (weeks 39 → predict_week)
            actual = self._get_actual_trend(prog)
            fig.add_trace(go.Scatter(
                x=actual.get("week_str", []), y=actual.get("Gewogen vooraanmelders", []),
                name="Vooraanmeldingen (actueel)", mode="lines+markers",
                line=dict(color="#17becf", width=2),
                visible=i == 0, showlegend=i == 0,
            ))

            # Model prediction (predict_week → 38)
            pdata = self.data[
                (self.data["Croho groepeernaam"] == prog)
                & (self.data["Collegejaar"] == self.prediction_year)
            ]
            pred_agg = pdata.groupby("Weeknummer")[pred_col].sum().reset_index()
            pred_agg["ws"] = pred_agg["Weeknummer"].astype(str)
            pred_agg = pred_agg.sort_values("Weeknummer", key=lambda s: s.map(_week_sort_key))
            fig.add_trace(go.Scatter(
                x=pred_agg["ws"], y=pred_agg[pred_col], name=pred_col,
                line=dict(color=_COLORS.get(pred_col, "#1f77b4"), width=2),
                visible=i == 0, showlegend=i == 0,
            ))

            # Realisatie (if available)
            if self.has_actuals:
                act_agg = pdata.groupby("Weeknummer")["Aantal_studenten"].sum().reset_index()
                act_agg["ws"] = act_agg["Weeknummer"].astype(str)
                act_agg = act_agg.sort_values("Weeknummer", key=lambda s: s.map(_week_sort_key))
            else:
                act_agg = pd.DataFrame(columns=["ws", "Aantal_studenten"])
            fig.add_trace(go.Scatter(
                x=act_agg["ws"], y=act_agg.get("Aantal_studenten", []),
                name="Realisatie", line=dict(color="#333", width=2, dash="dash"),
                visible=i == 0, showlegend=i == 0,
            ))

            vis = [False] * (len(programmes) * traces_per)
            for j in range(traces_per):
                vis[i * traces_per + j] = True
            buttons.append(dict(label=prog, method="update", args=[{"visible": vis}]))

        fig.update_layout(
            xaxis=dict(title="Week"),
            yaxis=dict(title="Aantal studenten"),
            updatemenus=[dict(buttons=buttons, direction="down",
                              x=0, xanchor="left", y=1.15, yanchor="top", showactive=True)],
            margin=dict(l=60, r=20, t=60, b=40), height=450,
        )
        return _add_predict_week_line(fig, self.predict_week)

    # ── Model spread (multiple models + actuals, dropdown) ───────────

    def _model_spread(self, pred_cols):
        pred_cols = [c for c in pred_cols if c in self.data.columns]
        if not pred_cols:
            return None
        all_cols = pred_cols + (["Aantal_studenten"] if self.has_actuals else [])
        programmes = sorted(self.data["Croho groepeernaam"].unique())
        if not programmes:
            return None

        fig = go.Figure()
        buttons = []
        n = len(all_cols) + 1  # +1 for actual trend trace
        for i, prog in enumerate(programmes):
            # Actual pre-registration trend (weeks 39 → predict_week)
            actual = self._get_actual_trend(prog)
            fig.add_trace(go.Scatter(
                x=actual.get("week_str", []), y=actual.get("Gewogen vooraanmelders", []),
                name="Vooraanmeldingen (actueel)", mode="lines+markers",
                line=dict(color="#17becf", width=2),
                visible=i == 0, legendgroup="actual", showlegend=i == 0,
            ))

            # Model predictions (predict_week → 38)
            pdata = self.data[
                (self.data["Croho groepeernaam"] == prog)
                & (self.data["Collegejaar"] == self.prediction_year)
            ]
            for col in all_cols:
                agg = pdata.groupby("Weeknummer")[col].sum().reset_index()
                agg["ws"] = agg["Weeknummer"].astype(str)
                agg = agg.sort_values("Weeknummer", key=lambda s: s.map(_week_sort_key))
                is_actual = col == "Aantal_studenten"
                fig.add_trace(go.Scatter(
                    x=agg["ws"], y=agg[col], name=col, mode="lines+markers",
                    line=dict(color=_COLORS.get(col, "#999"),
                              width=3 if is_actual else 2,
                              dash="dash" if is_actual else "solid"),
                    visible=i == 0, legendgroup=col, showlegend=i == 0,
                ))
            vis = [False] * (len(programmes) * n)
            for j in range(n):
                vis[i * n + j] = True
            buttons.append(dict(label=prog, method="update", args=[{"visible": vis}]))

        fig.update_layout(
            xaxis=dict(title="Week"),
            yaxis=dict(title="Aantal studenten"),
            updatemenus=[dict(buttons=buttons, direction="down",
                              x=0, xanchor="left", y=1.15, yanchor="top", showactive=True)],
            margin=dict(l=60, r=20, t=60, b=40), height=450,
        )
        return _add_predict_week_line(fig, self.predict_week)

    # ── Error heatmap ────────────────────────────────────────────────

    def _error_heatmap(self, mape_cols, title_text):
        mape_cols = [c for c in mape_cols if c in self.data.columns]
        if not mape_cols:
            return None
        hist = self._hist_data()
        if hist.empty:
            return None

        hist = hist.copy()
        hist["_mape"] = hist[mape_cols].mean(axis=1)
        pivot = hist.pivot_table(values="_mape", index="Croho groepeernaam",
                                 columns="Weeknummer", aggfunc="mean")
        week_cols = [int(w) for w in _ACADEMIC_WEEKS if int(w) in pivot.columns]
        pivot = pivot.reindex(columns=week_cols).clip(upper=2.0)

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values, x=[str(w) for w in pivot.columns], y=list(pivot.index),
            colorscale=[[0, "#d4edda"], [0.15, "#fff3cd"], [0.5, "#f8d7da"], [1, "#721c24"]],
            colorbar=dict(title="MAPE", tickformat=".0%"), zmin=0, zmax=1.0,
        ))
        fig.update_layout(
            title=title_text,
            xaxis=dict(title="Week"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=200, r=20, t=60, b=40),
            height=max(300, 40 + 25 * len(pivot.index)),
        )
        return _add_predict_week_line(fig, self.predict_week)

    # ── Reasoning: prediction stability (week-over-week change) ──────

    def _prediction_stability(self, pred_col):
        """Shows how much the prediction changes from week to week.
        Large swings indicate the model is still uncertain."""
        if pred_col not in self.data.columns:
            return None

        cur = self.data[self.data["Collegejaar"] == self.prediction_year]
        if cur.empty:
            return None

        programmes = sorted(cur["Croho groepeernaam"].unique())
        fig = go.Figure()
        buttons = []

        for i, prog in enumerate(programmes):
            pdata = cur[cur["Croho groepeernaam"] == prog]
            agg = pdata.groupby("Weeknummer")[pred_col].sum().reset_index()
            agg = agg.sort_values("Weeknummer", key=lambda s: s.map(_week_sort_key))
            agg["delta"] = agg[pred_col].diff()
            agg["delta_pct"] = agg["delta"] / agg[pred_col].shift().replace(0, np.nan)
            agg["ws"] = agg["Weeknummer"].astype(str)

            colors = ["#d4edda" if abs(d) <= 0.05
                       else "#fff3cd" if abs(d) <= 0.15
                       else "#f8d7da"
                       for d in agg["delta_pct"].fillna(0)]

            fig.add_trace(go.Bar(
                x=agg["ws"], y=agg["delta"],
                marker_color=colors,
                text=[f"{d:+.0%}" if pd.notna(d) else "" for d in agg["delta_pct"]],
                textposition="auto",
                visible=i == 0,
            ))
            vis = [False] * len(programmes)
            vis[i] = True
            buttons.append(dict(label=prog, method="update", args=[{"visible": vis}]))

        fig.update_layout(
            title="Week-over-week verandering van voorspelling",
            xaxis=dict(title="Week"),
            yaxis=dict(title="Δ studenten"),
            updatemenus=[dict(buttons=buttons, direction="down",
                              x=0, xanchor="left", y=1.15, yanchor="top", showactive=True)],
            margin=dict(l=60, r=20, t=80, b=40), height=400,
        )
        return _add_predict_week_line(fig, self.predict_week)

    # ── Reasoning: model consensus (spread between models) ───────────

    def _model_consensus(self, pred_cols):
        """Shows agreement/disagreement between models.
        Wide bands = models disagree = less confidence."""
        pred_cols = [c for c in pred_cols if c in self.data.columns]
        if len(pred_cols) < 2:
            return None

        cur = self.data[self.data["Collegejaar"] == self.prediction_year]
        if cur.empty:
            return None

        programmes = sorted(cur["Croho groepeernaam"].unique())
        fig = go.Figure()
        buttons = []
        traces_per = 3  # min line, max line, fill between

        for i, prog in enumerate(programmes):
            pdata = cur[cur["Croho groepeernaam"] == prog]
            agg = pdata.groupby("Weeknummer")[pred_cols].sum().reset_index()
            agg = agg.sort_values("Weeknummer", key=lambda s: s.map(_week_sort_key))
            agg["ws"] = agg["Weeknummer"].astype(str)
            agg["_min"] = agg[pred_cols].min(axis=1)
            agg["_max"] = agg[pred_cols].max(axis=1)
            agg["_mean"] = agg[pred_cols].mean(axis=1)

            fig.add_trace(go.Scatter(
                x=agg["ws"], y=agg["_max"], mode="lines",
                line=dict(width=0), showlegend=False, visible=i == 0,
            ))
            fig.add_trace(go.Scatter(
                x=agg["ws"], y=agg["_min"], mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor="rgba(255, 127, 14, 0.25)",
                name="Spreiding tussen modellen", showlegend=i == 0,
                visible=i == 0,
            ))
            fig.add_trace(go.Scatter(
                x=agg["ws"], y=agg["_mean"], mode="lines+markers",
                line=dict(color="#ff7f0e", width=2),
                name="Gemiddelde", showlegend=i == 0, visible=i == 0,
            ))

            vis = [False] * (len(programmes) * traces_per)
            for j in range(traces_per):
                vis[i * traces_per + j] = True
            buttons.append(dict(label=prog, method="update", args=[{"visible": vis}]))

        fig.update_layout(
            title="Modelconsensus — smalle band = modellen eens = meer zekerheid",
            xaxis=dict(title="Week"),
            yaxis=dict(title="Aantal studenten"),
            updatemenus=[dict(buttons=buttons, direction="down",
                              x=0, xanchor="left", y=1.15, yanchor="top", showactive=True)],
            margin=dict(l=60, r=20, t=80, b=40), height=450,
        )
        return _add_predict_week_line(fig, self.predict_week)

    # ══════════════════════════════════════════════════════════════════
    #  FINAL DASHBOARD CHARTS (policy makers)
    # ══════════════════════════════════════════════════════════════════

    def _best_pred_col(self):
        for col in ["Weighted_ensemble_prediction", "Average_ensemble_prediction",
                     "Ensemble_prediction", "SARIMA_cumulative", "SARIMA_individual"]:
            if col in self.data.columns:
                return col
        return self.pred_cols[0] if self.pred_cols else None

    def _confidence_score(self, programme):
        """Derive confidence from historical prediction accuracy for this programme.

        Compares week-38 Gewogen vooraanmelders (from data_cumulative) against
        actual Aantal_studenten (from data_studentcount) for historical years.
        Returns (label, color_hex, avg_mape_or_nan) tuple.
        """
        # Try approach 1: historical MAPE from output data
        hist = self._hist_data()
        if not hist.empty and self.mape_cols:
            prog_hist = hist[hist["Croho groepeernaam"] == programme]
            if not prog_hist.empty:
                avg_mape = prog_hist[self.mape_cols].mean().mean()
                if pd.notna(avg_mape):
                    return self._mape_to_label(avg_mape)

        # Try approach 2: compare historical week-38 vooraanmeldingen vs actual enrollment
        if self.data_cumulative is not None and self.data_studentcount is not None:
            col_name = "Croho groepeernaam"
            dc = self.data_cumulative
            sc = self.data_studentcount

            # data_cumulative may use "Groepeernaam Croho" instead of "Croho groepeernaam"
            if col_name not in dc.columns and "Groepeernaam Croho" in dc.columns:
                col_name_dc = "Groepeernaam Croho"
            else:
                col_name_dc = col_name

            if "Gewogen vooraanmelders" in dc.columns and col_name_dc in dc.columns:
                wk38 = dc[
                    (dc["Weeknummer"] == 38) & (dc[col_name_dc] == programme)
                ].groupby("Collegejaar")["Gewogen vooraanmelders"].sum()

                actuals = sc[
                    sc["Croho groepeernaam"] == programme
                ].groupby("Collegejaar")["Aantal_studenten"].sum()

                common_years = wk38.index.intersection(actuals.index)
                if len(common_years) >= 1:
                    mapes = []
                    for yr in common_years:
                        pred_val = wk38[yr]
                        act_val = actuals[yr]
                        if act_val > 0:
                            mapes.append(abs(pred_val - act_val) / act_val)
                    if mapes:
                        return self._mape_to_label(np.mean(mapes))

        return "Onbekend", "#e0e0e0", np.nan

    @staticmethod
    def _mape_to_label(avg_mape):
        if avg_mape <= 0.10:
            return "Hoog", "#d4edda", avg_mape
        if avg_mape <= 0.25:
            return "Midden", "#fff3cd", avg_mape
        return "Laag", "#f8d7da", avg_mape

    # ── Cockpit with confidence ──────────────────────────────────────

    def _cockpit_with_confidence(self):
        best = self._best_pred_col()
        if best is None:
            return None
        latest_week = self._get_latest_week()
        if latest_week is None:
            return None

        cur = self.data[
            (self.data["Collegejaar"] == self.prediction_year)
            & (self.data["Weeknummer"] == latest_week)
        ].groupby("Croho groepeernaam")[best].sum()

        prev_data = self.data[self.data["Collegejaar"] == self.prediction_year - 1]
        if self.has_actuals and not prev_data.empty:
            prev = prev_data.groupby("Croho groepeernaam")["Aantal_studenten"].max()
        else:
            prev = pd.Series(dtype=float)

        # Collect all available prediction columns for the "bronmodellen" column
        model_info = {}
        for prog in cur.index:
            sources = []
            prog_row = self.data[
                (self.data["Croho groepeernaam"] == prog)
                & (self.data["Collegejaar"] == self.prediction_year)
                & (self.data["Weeknummer"] == latest_week)
            ]
            for col in self.pred_cols:
                val = prog_row[col].sum()
                if val > 0:
                    sources.append(f"{col}: {round(val)}")
            model_info[prog] = "<br>".join(sources) if sources else "—"

        programmes = sorted(cur.index)
        predicted = [round(cur.get(p, 0)) for p in programmes]
        actual_prev = [round(prev.get(p, 0)) for p in programmes]

        delta_pct, delta_colors = [], []
        for pred, act in zip(predicted, actual_prev):
            if act and act > 0:
                d = (pred - act) / act * 100
                delta_pct.append(f"{d:+.1f}%")
                delta_colors.append("#d4edda" if abs(d) <= 5 else "#fff3cd" if abs(d) <= 15 else "#f8d7da")
            else:
                delta_pct.append("n.v.t.")
                delta_colors.append("#ffffff")

        conf_labels, conf_colors = [], []
        for p in programmes:
            label, color, mape_val = self._confidence_score(p)
            suffix = f" ({mape_val:.0%})" if pd.notna(mape_val) else ""
            conf_labels.append(f"{label}{suffix}")
            conf_colors.append(color)

        sources = [model_info.get(p, "—") for p in programmes]

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=["Opleiding", f"Prognose {self.prediction_year}",
                         f"Realisatie {self.prediction_year - 1}", "Δ%",
                         "Betrouwbaarheid", "Bronmodellen"],
                fill_color="#4472C4", font=dict(color="white", size=13), align="left",
            ),
            cells=dict(
                values=[programmes, predicted, actual_prev, delta_pct,
                        conf_labels, sources],
                fill_color=[
                    ["#f9f9f9"] * len(programmes),
                    ["#f9f9f9"] * len(programmes),
                    ["#f9f9f9"] * len(programmes),
                    delta_colors,
                    conf_colors,
                    ["#f9f9f9"] * len(programmes),
                ],
                align="left", font=dict(size=12), height=28,
            ),
        )])
        fig.update_layout(
            title=f"Overzicht week {latest_week} — Betrouwbaarheid op basis van historische nauwkeurigheid"
                  f"<br><sub>Hoog (&le;10% afwijking) | Midden (10-25%) | Laag (&gt;25%)</sub>",
            margin=dict(l=20, r=20, t=80, b=20),
            height=max(350, 80 + 28 * len(programmes)),
        )
        return fig

    # ── Prognose vs vorig jaar ───────────────────────────────────────

    def _prognose_vs_vorig_jaar(self):
        best = self._best_pred_col()
        if best is None:
            return None
        programmes = sorted(self.data["Croho groepeernaam"].unique())
        if not programmes:
            return None

        fig = go.Figure()
        buttons = []
        traces_per = 3  # actual trend, prediction, previous year
        for i, prog in enumerate(programmes):
            # Actual pre-registration data (weeks 39 → predict_week)
            actual = self._get_actual_trend(prog)
            fig.add_trace(go.Scatter(
                x=actual.get("week_str", []), y=actual.get("Gewogen vooraanmelders", []),
                name="Vooraanmeldingen (actueel)", mode="lines+markers",
                line=dict(color="#17becf", width=2),
                visible=i == 0, showlegend=i == 0,
            ))

            # Model prediction (predict_week → 38)
            pdata = self.data[self.data["Croho groepeernaam"] == prog]
            cur = pdata[pdata["Collegejaar"] == self.prediction_year]
            cur_agg = cur.groupby("Weeknummer")[best].sum().reset_index()
            cur_agg["ws"] = cur_agg["Weeknummer"].astype(str)
            cur_agg = cur_agg.sort_values("Weeknummer", key=lambda s: s.map(_week_sort_key))
            fig.add_trace(go.Scatter(
                x=cur_agg["ws"], y=cur_agg[best],
                name=f"Prognose {self.prediction_year}",
                line=dict(color="#1f77b4", width=2), visible=i == 0,
                showlegend=i == 0,
            ))

            # Previous year realisatie
            prev = pdata[pdata["Collegejaar"] == self.prediction_year - 1]
            if self.has_actuals and not prev.empty:
                prev_agg = prev.groupby("Weeknummer")["Aantal_studenten"].sum().reset_index()
                prev_agg["ws"] = prev_agg["Weeknummer"].astype(str)
                prev_agg = prev_agg.sort_values("Weeknummer", key=lambda s: s.map(_week_sort_key))
            else:
                prev_agg = pd.DataFrame(columns=["ws", "Aantal_studenten"])
            fig.add_trace(go.Scatter(
                x=prev_agg["ws"], y=prev_agg.get("Aantal_studenten", []),
                name=f"Realisatie {self.prediction_year - 1}",
                line=dict(color="#333", width=2, dash="dash"), visible=i == 0,
                showlegend=i == 0,
            ))

            vis = [False] * (len(programmes) * traces_per)
            for j in range(traces_per):
                vis[i * traces_per + j] = True
            buttons.append(dict(label=prog, method="update", args=[{"visible": vis}]))

        fig.update_layout(
            xaxis=dict(title="Week"),
            yaxis=dict(title="Aantal studenten"),
            updatemenus=[dict(buttons=buttons, direction="down",
                              x=0, xanchor="left", y=1.15, yanchor="top", showactive=True)],
            margin=dict(l=60, r=20, t=60, b=40), height=450,
        )
        return _add_predict_week_line(fig, self.predict_week)

    # ── Herkomstverdeling ────────────────────────────────────────────

    def _origin_stacked_bar(self):
        best = self._best_pred_col()
        if best is None or "Herkomst" not in self.data.columns:
            return None
        latest_week = self._get_latest_week()

        cur = self.data[
            (self.data["Collegejaar"] == self.prediction_year)
            & (self.data["Weeknummer"] == latest_week)
        ]
        cur_agg = cur.groupby(["Croho groepeernaam", "Herkomst"])[best].sum().reset_index()

        prev = self.data[self.data["Collegejaar"] == self.prediction_year - 1]
        if self.has_actuals and not prev.empty:
            prev_agg = prev.groupby(["Croho groepeernaam", "Herkomst"])["Aantal_studenten"].max().reset_index()
        else:
            prev_agg = pd.DataFrame(columns=["Croho groepeernaam", "Herkomst", "Aantal_studenten"])

        programmes = sorted(cur_agg["Croho groepeernaam"].unique())
        herkomst_colors = {"NL": "#4472C4", "EER": "#ED7D31", "Niet-EER": "#A5A5A5"}

        fig = go.Figure()
        for herkomst, color in herkomst_colors.items():
            h_cur = cur_agg[cur_agg["Herkomst"] == herkomst].set_index("Croho groepeernaam")
            vals_cur = [round(h_cur.loc[p, best]) if p in h_cur.index else 0 for p in programmes]

            h_prev = prev_agg[prev_agg["Herkomst"] == herkomst].set_index("Croho groepeernaam")
            vals_prev = [round(h_prev.loc[p, "Aantal_studenten"]) if p in h_prev.index else 0 for p in programmes]

            x_cur = [f"{p} ({self.prediction_year})" for p in programmes]
            x_prev = [f"{p} ({self.prediction_year - 1})" for p in programmes]

            fig.add_trace(go.Bar(x=x_cur, y=vals_cur, name=herkomst,
                                 marker_color=color, legendgroup=herkomst))
            fig.add_trace(go.Bar(x=x_prev, y=vals_prev, name=herkomst,
                                 marker_color=color, legendgroup=herkomst,
                                 showlegend=False, opacity=0.5))

        fig.update_layout(
            barmode="stack",
            title="Herkomstverdeling: huidig (vol) vs vorig jaar (transparant)",
            xaxis=dict(title="Opleiding", tickangle=-45),
            yaxis=dict(title="Aantal studenten"),
            margin=dict(l=60, r=20, t=60, b=120), height=500,
        )
        return fig

    # ── Numerus fixus voortgang ──────────────────────────────────────

    def _numerus_fixus_bars(self):
        best = self._best_pred_col()
        if not self.numerus_fixus_list or best is None:
            return None

        nf_progs = [p for p in self.numerus_fixus_list if p in self.data["Croho groepeernaam"].values]
        if not nf_progs:
            return None

        latest_week = self._get_latest_week()
        cur = self.data[
            (self.data["Collegejaar"] == self.prediction_year)
            & (self.data["Weeknummer"] == latest_week)
        ]

        progs, preds, caps = [], [], []
        for p in sorted(nf_progs):
            pred = cur[cur["Croho groepeernaam"] == p][best].sum()
            progs.append(p)
            preds.append(round(pred))
            caps.append(self.numerus_fixus_list[p])

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=progs, x=preds, orientation="h",
            name="Prognose", marker_color="#4472C4",
            text=[str(v) for v in preds], textposition="auto",
        ))
        for i, cap in enumerate(caps):
            fig.add_shape(type="line", x0=cap, x1=cap, y0=i - 0.4, y1=i + 0.4,
                          line=dict(color="red", width=3, dash="dash"))

        fig.update_layout(
            title=f"Numerus Fixus — week {latest_week}",
            xaxis=dict(title="Aantal studenten"),
            margin=dict(l=200, r=40, t=60, b=40),
            height=max(250, 60 + 50 * len(progs)), showlegend=False,
        )
        return fig
