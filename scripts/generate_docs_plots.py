"""Generate interactive Plotly plots for the methodology documentation pages.

Produces self-contained HTML files in docs/assets/plots/ that are embedded
via <iframe> in the MkDocs Markdown pages.  Uses synthetic demo data only.

Usage:
    uv run python scripts/generate_docs_plots.py
"""

import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import beta as beta_dist

SEED = 42
rng = np.random.default_rng(SEED)

WEEKS_ORDERED = list(range(39, 53)) + list(range(1, 39))
ACADEMIC_WEEKS = [str(w) for w in WEEKS_ORDERED]

MODEL_COLOURS = {
    "SARIMA_individual": "#1f77b4",
    "SARIMA_cumulative": "#ff7f0e",
    "Prognose_ratio": "#2ca02c",
    "Ensemble": "#d62728",
    "Werkelijk": "#333333",
}

PASTEL_YEAR_COLOURS = [
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]

PLOTLY_CONFIG = {"responsive": True, "displayModeBar": True, "displaylogo": False}

DEMO_PROGRAMMES = [
    ("B Psychologie", "B", 620, 120, 55),
    ("B Kunstmatige Intelligentie", "B", 450, 180, 90),
    ("M Cognitive Neuroscience", "M", 120, 80, 45),
]


# ---------------------------------------------------------------------------
# Synthetic data generation (reuses logic from generate_demo_data.py)
# ---------------------------------------------------------------------------


def _cumulative_curve(final_count: int, prog_type: str) -> np.ndarray:
    """Generate a realistic cumulative S-curve over 52 academic weeks."""
    n = 52
    a, b = (3.0, 1.8) if prog_type == "B" else (2.2, 2.0)

    x = np.linspace(0, 1, n)
    year_shift = rng.uniform(-0.02, 0.02)
    x_shifted = np.clip(x + year_shift, 0, 1)
    cdf = beta_dist.cdf(x_shifted, a, b)
    cdf = cdf / cdf[-1]

    noise = rng.normal(0, 0.005, n)
    cdf = np.maximum.accumulate(cdf + noise)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])

    values = np.round(cdf * final_count).astype(int)
    values = np.maximum.accumulate(values)
    values[-1] = final_count

    if final_count > 50 and values[0] == 0:
        values[0] = rng.integers(1, max(3, final_count // 100))
        values = np.maximum.accumulate(values)

    return values


def _add_year_trend(base: int, year: int) -> int:
    trend = 1.0 + 0.02 * (year - 2020) + rng.normal(0, 0.05)
    return max(5, int(round(base * trend)))


ALL_YEARS = list(range(2016, 2025))
RECENT_YEARS = [2022, 2023, 2024]


def _generate_demo_curves() -> dict:
    """Generate cumulative S-curves for 3 programmes x 3 herkomsten x multiple years.

    B Psychologie gets curves for ALL_YEARS (2016-2024, needed for the yearly
    trend chart).  Other programmes only get RECENT_YEARS (2022-2024).
    """
    curves: dict[tuple[str, str, int], np.ndarray] = {}

    for prog_name, prog_type, base_nl, base_eer, base_neer in DEMO_PROGRAMMES:
        years = ALL_YEARS if prog_name == "B Psychologie" else RECENT_YEARS
        for herkomst, base in [("NL", base_nl), ("EER", base_eer), ("Niet-EER", base_neer)]:
            for year in years:
                final = _add_year_trend(base, year)
                curves[(prog_name, herkomst, year)] = _cumulative_curve(final, prog_type)

    return curves


def _generate_student_counts(curves: dict) -> dict:
    """Generate synthetic student counts (yield from pre-applicants)."""
    counts: dict[tuple[str, str, int], int] = {}
    for (prog_name, herkomst, year), curve in curves.items():
        final = int(curve[-1])
        yield_rate = {"NL": 0.65, "EER": 0.45, "Niet-EER": 0.35}.get(herkomst, 0.5)
        yield_rate += rng.uniform(-0.05, 0.05)
        counts[(prog_name, herkomst, year)] = max(1, int(round(final * yield_rate)))
    return counts


# ---------------------------------------------------------------------------
# HTML wrapper with dark-mode support
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: white;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #2e303e; }}
  }}
</style>
</head>
<body>
{plotly_div}
<script>
window.addEventListener("message", function(event) {{
  if (event.data && event.data.type === "theme") {{
    var dark = event.data.scheme === "slate";
    var plotDiv = document.querySelector(".js-plotly-plot");
    if (plotDiv) {{
      Plotly.relayout(plotDiv, {{
        "paper_bgcolor": dark ? "#2e303e" : "white",
        "plot_bgcolor": dark ? "#2e303e" : "white",
        "font.color": dark ? "#ccc" : "#333",
        "xaxis.gridcolor": dark ? "#444" : "#eee",
        "yaxis.gridcolor": dark ? "#444" : "#eee",
        "xaxis.linecolor": dark ? "#555" : "#ccc",
        "yaxis.linecolor": dark ? "#555" : "#ccc",
        "legend.font.color": dark ? "#ccc" : "#333",
      }});
    }}
  }}
}});
</script>
</body>
</html>
"""


def _wrap_html(fig: go.Figure, filename: str, output_dir: str) -> None:
    """Save a Plotly figure as a self-contained HTML file with dark-mode support."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#333", size=13),
        margin=dict(l=60, r=30, t=60, b=60),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )

    plotly_div = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config=PLOTLY_CONFIG,
    )

    html = _HTML_TEMPLATE.format(plotly_div=plotly_div)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  {filename}")


# ---------------------------------------------------------------------------
# Plot 1a & 1b: Pipeline visualizations — methodologie/index.md
# ---------------------------------------------------------------------------

HERKOMST_COLOURS = {"NL": "#4472C4", "EER": "#ED7D31", "Niet-EER": "#A5A5A5"}


def plot_pipeline_cumulative(curves: dict, student_counts: dict) -> go.Figure:
    """Cumulatief spoor: telbestanden → SARIMA → XGBoost → studenten (vertical)."""
    prog, herkomst, year = "B Psychologie", "NL", 2024
    pw = 16
    pw_idx = WEEKS_ORDERED.index(pw)
    curve = curves[(prog, herkomst, year)]
    students = student_counts[(prog, herkomst, year)]
    known = curve[: pw_idx + 1]

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.35, 0.40, 0.25],
        vertical_spacing=0.18,
        subplot_titles=[
            "① Input — wekelijkse telbestanden",
            "② SARIMA — extrapolatie naar week 38",
            "③ XGBoost regressor — vooraanmelders → studenten",
        ],
    )

    # ── Row 1: input data (known cumulative curve) ───────────────────
    fig.add_trace(go.Scatter(
        x=ACADEMIC_WEEKS[: pw_idx + 1], y=known.tolist(),
        mode="lines+markers", line=dict(color="#1f77b4", width=2.5),
        marker=dict(size=4), showlegend=False,
        hovertemplate="Week %{x}: %{y} vooraanmelders<extra></extra>",
    ), row=1, col=1)

    n_hint = min(4, 52 - pw_idx - 1)
    hint_weeks = ACADEMIC_WEEKS[pw_idx + 1: pw_idx + 1 + n_hint]
    fig.add_trace(go.Scatter(
        x=hint_weeks, y=[float(known[-1])] * n_hint,
        mode="markers",
        marker=dict(size=7, color="rgba(31,119,180,0.15)", symbol="circle-open"),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_annotation(
        x=hint_weeks[-1] if hint_weeks else ACADEMIC_WEEKS[pw_idx],
        y=float(known[-1]), xref="x", yref="y",
        text="?", font=dict(size=22, color="#bbb"), showarrow=False,
    )

    # ── Row 2: SARIMA extrapolation ──────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ACADEMIC_WEEKS[: pw_idx + 1], y=known.tolist(),
        mode="lines", line=dict(color="#1f77b4", width=2.5),
        name="Bekend", showlegend=True,
    ), row=2, col=1)

    sarima = curve[pw_idx:].astype(float).copy()
    n_fc = len(sarima)
    sarima += np.linspace(0, 0.06, n_fc) * sarima[-1] + rng.normal(0, 3, n_fc)
    sarima = np.maximum.accumulate(np.maximum(sarima, float(known[-1])))

    fig.add_trace(go.Scatter(
        x=ACADEMIC_WEEKS[pw_idx:], y=sarima.tolist(),
        mode="lines", line=dict(color="#d62728", width=2.5, dash="dash"),
        name="SARIMA prognose", showlegend=True,
    ), row=2, col=1)

    hist_finals = [float(curves[(prog, herkomst, yr)][-1]) for yr in ALL_YEARS if yr < year]
    avg_std = float(np.std(hist_finals)) if len(hist_finals) > 1 else sarima[-1] * 0.05
    total_steps = max(n_fc - 1, 1)
    band = np.array([avg_std * np.sqrt(i / total_steps) for i in range(n_fc)])
    fig.add_trace(go.Scatter(
        x=ACADEMIC_WEEKS[pw_idx:], y=(sarima + band).tolist(),
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=ACADEMIC_WEEKS[pw_idx:], y=np.maximum(sarima - band, 0).tolist(),
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(214,39,40,0.12)",
        name="Onzekerheidsmarge", showlegend=True,
    ), row=2, col=1)


    # ── Row 3: XGBoost output ────────────────────────────────────────
    sarima_38 = int(sarima[-1])
    pred_students = int(students * rng.uniform(0.95, 1.05))
    labels_bar = [
        "Vooraanmelders (SARIMA wk 38)",
        "Voorspelde studenten (XGBoost)",
        "Werkelijke studenten",
    ]
    values_bar = [sarima_38, pred_students, students]
    colors_bar = ["rgba(214,39,40,0.35)", "#2ca02c", "rgba(85,85,85,0.6)"]

    fig.add_trace(go.Bar(
        x=labels_bar, y=values_bar, marker=dict(color=colors_bar),
        showlegend=False,
        text=[str(v) for v in values_bar], textposition="outside",
        textfont=dict(size=12),
        hovertemplate="%{x}: %{y}<extra></extra>",
    ), row=3, col=1)

    # ── Transition annotations in the gaps ───────────────────────────
    fig.add_annotation(
        x=0.5, y=0.686, xref="paper", yref="paper",
        text="<span style='font-size:22px'>▼</span><br><b>SARIMA extrapoleert de curve naar week 38</b>",
        font=dict(size=15, color="#666"), showarrow=False,
    )
    fig.add_annotation(
        x=0.5, y=0.25, xref="paper", yref="paper",
        text="<span style='font-size:22px'>▼</span><br><b>XGBoost converteert vooraanmelders → studenten</b>",
        font=dict(size=15, color="#666"), showarrow=False,
    )

    # ── Axes ─────────────────────────────────────────────────────────
    for row in [1, 2]:
        fig.update_xaxes(
            categoryorder="array", categoryarray=ACADEMIC_WEEKS,
            tickvals=[str(w) for w in range(1, 53, 4)], tickangle=-45,
            row=row, col=1,
        )
    fig.update_yaxes(title_text="Vooraanmelders", row=1, col=1)
    fig.update_yaxes(title_text="Vooraanmelders", row=2, col=1)
    fig.update_yaxes(title_text="Aantal", row=3, col=1)

    fig.update_layout(
        height=1000,
        margin=dict(t=60, b=40, l=70, r=30),
        legend=dict(
            orientation="h", yanchor="top", y=-0.03,
            xanchor="center", x=0.5, font=dict(size=11),
        ),
        title=dict(
            text="<b>Cumulatief spoor</b> — B Psychologie (NL, demodata)",
            font=dict(size=16), x=0.5, xanchor="center",
        ),
    )
    return fig


def plot_pipeline_individual(curves: dict, student_counts: dict) -> go.Figure:
    """Individueel spoor: per-student records → XGBoost classifier → ΣP = cohort."""
    prog, year = "B Psychologie", 2024
    herkomsten = ["NL", "EER", "Niet-EER"]
    total_applicants = sum(int(curves[(prog, h, year)][-1]) for h in herkomsten)
    total_students = sum(student_counts[(prog, h, year)] for h in herkomsten)
    n_not = total_applicants - total_students

    p_enrolled = np.clip(rng.beta(6, 2, size=total_students), 0.01, 0.99)
    p_not = np.clip(rng.beta(2, 5, size=n_not), 0.01, 0.99)

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.28, 0.42, 0.30],
        vertical_spacing=0.18,
        subplot_titles=[
            "① Input — per-student vooraanmeldingen",
            "② XGBoost classifier — P(inschrijving) per student",
            "③ Aggregatie — ΣP = verwacht cohort",
        ],
    )

    # ── Row 1: waffle chart of applicants by herkomst ────────────────
    dots_per_student = 5
    dot_colors: list[str] = []
    for h in herkomsten:
        n_dots = int(curves[(prog, h, year)][-1]) // dots_per_student
        dot_colors.extend([HERKOMST_COLOURS[h]] * n_dots)
    dot_arr = np.array(dot_colors)
    rng.shuffle(dot_arr)
    dot_colors = dot_arr.tolist()
    n_grid = len(dot_colors)
    grid_cols = 30
    gx = [i % grid_cols for i in range(n_grid)]
    gy = [i // grid_cols for i in range(n_grid)]

    fig.add_trace(go.Scatter(
        x=gx, y=gy, mode="markers",
        marker=dict(size=8, color=dot_colors, line=dict(width=0.5, color="white")),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    for h in herkomsten:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=8, color=HERKOMST_COLOURS[h]),
            name=h, legendgroup=f"h_{h}",
        ), row=1, col=1)

    fig.add_annotation(
        x=grid_cols / 2, y=-1.2, xref="x", yref="y",
        text=f"<i>{total_applicants} vooraanmelders · elke stip ≈ {dots_per_student} studenten</i>",
        font=dict(size=11, color="#999"), showarrow=False,
    )
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)

    # ── Row 2: probability histogram ─────────────────────────────────
    fig.add_trace(go.Histogram(
        x=p_enrolled.tolist(), nbinsx=30,
        marker_color="rgba(31,119,180,0.7)",
        name="Schrijft zich in", legendgroup="enrolled",
    ), row=2, col=1)
    fig.add_trace(go.Histogram(
        x=p_not.tolist(), nbinsx=30,
        marker_color="rgba(200,200,200,0.7)",
        name="Schrijft zich niet in", legendgroup="not_enrolled",
    ), row=2, col=1)

    fig.add_shape(
        type="line", x0=0.5, x1=0.5, y0=0, y1=1,
        yref="y2 domain", xref="x2",
        line=dict(dash="dash", color="#999", width=1),
    )
    fig.add_annotation(
        x=0.52, y=0.95, yref="y2 domain", xref="x2",
        text="P = 0.5", font=dict(size=10, color="#999"),
        showarrow=False,
    )
    fig.update_xaxes(title_text="P(inschrijving)", row=2, col=1)
    fig.update_yaxes(title_text="Aantal studenten", row=2, col=1)

    # ── Row 3: aggregation bars ──────────────────────────────────────
    sum_p = int(round(float(np.sum(p_enrolled)) + float(np.sum(p_not))))
    fig.add_trace(go.Bar(
        x=["Verwacht (ΣP)", "Werkelijk"],
        y=[sum_p, total_students],
        marker=dict(color=["#1f77b4", "rgba(85,85,85,0.6)"]),
        showlegend=False,
        text=[str(sum_p), str(total_students)], textposition="outside",
        textfont=dict(size=14),
        hovertemplate="%{x}: %{y}<extra></extra>",
        width=0.5,
    ), row=3, col=1)
    fig.update_yaxes(title_text="Aantal studenten", row=3, col=1)

    # ── Transition annotations ───────────────────────────────────────
    fig.add_annotation(
        x=0.5, y=0.731, xref="paper", yref="paper",
        text="<span style='font-size:22px'>▼</span><br><b>XGBoost voorspelt inschrijfkans per student</b>",
        font=dict(size=15, color="#666"), showarrow=False,
    )
    fig.add_annotation(
        x=0.5, y=0.282, xref="paper", yref="paper",
        text="<span style='font-size:22px'>▼</span><br><b>Sommeer kansen → verwacht cohortaantal</b>",
        font=dict(size=15, color="#666"), showarrow=False,
    )

    fig.update_layout(
        height=1000,
        margin=dict(t=60, b=40, l=70, r=30),
        barmode="overlay",
        legend=dict(
            orientation="h", yanchor="top", y=-0.03,
            xanchor="center", x=0.5, font=dict(size=11),
        ),
        title=dict(
            text="<b>Individueel spoor</b> — B Psychologie (demodata)",
            font=dict(size=16), x=0.5, xanchor="center",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Plot 2: SARIMA extrapolation — methodologie/sarima.md
# ---------------------------------------------------------------------------


def plot_sarima_extrapolation(curves: dict, student_counts: dict) -> go.Figure:
    """Yearly trend chart matching the dashboard's 'Verloop per opleiding — alle jaren'.

    Shows B Psychologie (NL) only, no dropdown.  Historical years as thin pastel
    lines (legendonly), prediction year split into 'actueel' + 'na voorspelweek',
    SARIMA prognose (red dashed) with uncertainty band, realisation stars, and
    predicted enrolment diamond.
    """
    fig = go.Figure()
    prog = "B Psychologie"
    herkomst = "NL"
    prediction_year = 2024
    predict_week = 16
    predict_week_idx = WEEKS_ORDERED.index(predict_week)

    hist_years = [y for y in ALL_YEARS if y < prediction_year]

    # ── Historical year lines (thin pastels, default legendonly) ──
    for j, yr in enumerate(hist_years):
        curve = curves[(prog, herkomst, yr)]
        fig.add_trace(
            go.Scatter(
                x=ACADEMIC_WEEKS,
                y=curve.tolist(),
                mode="lines",
                name=str(yr),
                line=dict(color=PASTEL_YEAR_COLOURS[j % len(PASTEL_YEAR_COLOURS)], width=1),
                visible="legendonly",
                legendgroup=str(yr),
                hovertemplate=f"Week %{{x}}: %{{y}} vooraanmelders<extra>{yr}</extra>",
            )
        )

    # ── Prediction year: actueel (up to predict week, thick blue + markers) ──
    curve_pred = curves[(prog, herkomst, prediction_year)]
    known = curve_pred[: predict_week_idx + 1]
    after = curve_pred[predict_week_idx:]

    fig.add_trace(
        go.Scatter(
            x=ACADEMIC_WEEKS[: predict_week_idx + 1],
            y=known.tolist(),
            mode="lines+markers",
            name=f"{prediction_year} (actueel)",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=4),
            hovertemplate=f"Week %{{x}}: %{{y}} vooraanmelders<extra>{prediction_year} (actueel)</extra>",
        )
    )

    # ── Na voorspelweek (thin dotted blue, legendonly) ──
    fig.add_trace(
        go.Scatter(
            x=ACADEMIC_WEEKS[predict_week_idx:],
            y=after.tolist(),
            mode="lines",
            name=f"{prediction_year} (na voorspelweek)",
            line=dict(color="#1f77b4", width=1.5, dash="dot"),
            visible="legendonly",
            hovertemplate=f"Week %{{x}}: %{{y}} vooraanmelders<extra>{prediction_year} (na voorspelweek)</extra>",
        )
    )

    # ── SARIMA prognose (red dashed) ──
    # Simulate a SARIMA forecast: use the actual curve with slight upward bias and noise
    sarima_forecast = curve_pred[predict_week_idx:].astype(float).copy()
    n_forecast = len(sarima_forecast)
    drift = np.linspace(0, 0.06, n_forecast) * sarima_forecast[-1]
    sarima_forecast = sarima_forecast + drift + rng.normal(0, 3, n_forecast)
    sarima_forecast = np.maximum.accumulate(np.maximum(sarima_forecast, known[-1]))

    fig.add_trace(
        go.Scatter(
            x=ACADEMIC_WEEKS[predict_week_idx:],
            y=sarima_forecast.tolist(),
            mode="lines",
            name="SARIMA prognose",
            line=dict(color="#d62728", dash="dash", width=3),
            hovertemplate="Week %{x}: %{y:.0f} vooraanmelders<extra>SARIMA prognose</extra>",
        )
    )

    # ── Uncertainty band (widens with forecast horizon) ──
    hist_finals = np.array([float(curves[(prog, herkomst, yr)][-1]) for yr in hist_years])
    avg_std = float(np.std(hist_finals)) if len(hist_finals) > 1 else sarima_forecast[-1] * 0.05
    total_steps = max(n_forecast - 1, 1)
    band_std = np.array([avg_std * np.sqrt(i / total_steps) for i in range(n_forecast)])
    upper = sarima_forecast + band_std
    lower = np.maximum(sarima_forecast - band_std, 0)

    fig.add_trace(
        go.Scatter(
            x=ACADEMIC_WEEKS[predict_week_idx:],
            y=upper.tolist(),
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hovertemplate="Bovenmarge: %{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ACADEMIC_WEEKS[predict_week_idx:],
            y=lower.tolist(),
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(214, 39, 40, 0.12)",
            name="Onzekerheidsmarge",
            hovertemplate="Ondermarge: %{y:.0f}<extra></extra>",
        )
    )

    # ── Actuele inschrijvingen stars (week 38, one per year) ──
    first_star = True
    for yr in hist_years + [prediction_year]:
        students = student_counts.get((prog, herkomst, yr), 0)
        if students > 0:
            is_pred_yr = yr == prediction_year
            fig.add_trace(
                go.Scatter(
                    x=["38"],
                    y=[students],
                    mode="markers+text",
                    name="Actuele inschrijvingen" if first_star else f"Realisatie {yr}",
                    marker=dict(
                        symbol="star",
                        size=14,
                        color="#d62728" if is_pred_yr else "#333333",
                    ),
                    text=[str(yr)],
                    textposition="middle right",
                    textfont=dict(size=9),
                    visible="legendonly",
                    showlegend=first_star,
                    legendgroup="actuele_inschrijvingen",
                    hovertemplate=f"Realisatie {yr}: %{{y}}<extra></extra>",
                )
            )
            first_star = False

    # ── Voorspelde inschrijvingen diamond (week 38) ──
    pred_students = student_counts.get((prog, herkomst, prediction_year), 0)
    if pred_students > 0:
        predicted_val = int(pred_students * rng.uniform(0.95, 1.05))
        fig.add_trace(
            go.Scatter(
                x=["38"],
                y=[predicted_val],
                mode="markers+text",
                name="Voorspelde inschrijvingen",
                marker=dict(symbol="diamond", size=12, color="#2ca02c"),
                text=[str(prediction_year)],
                textposition="middle right",
                textfont=dict(size=9),
                hovertemplate=f"Voorspeld {prediction_year}: %{{y}}<extra></extra>",
            )
        )

    # ── Predict week vertical line ──
    pw_str = str(predict_week)
    pw_cat_idx = ACADEMIC_WEEKS.index(pw_str)
    fig.add_vline(x=pw_cat_idx, line_dash="dash", line_color="red", line_width=1.5)
    fig.add_annotation(
        x=pw_cat_idx,
        y=1.02,
        yref="paper",
        yanchor="bottom",
        text=f"Voorspelweek {predict_week}",
        showarrow=False,
        font=dict(color="red", size=11),
        bgcolor="rgba(255,255,255,0.8)",
        borderpad=3,
    )

    # ── Layout ──
    tick_indices = list(range(0, 52, 4))
    tickvals = [ACADEMIC_WEEKS[i] for i in tick_indices]

    fig.update_layout(
        title=dict(
            text="Verloop per opleiding — alle jaren (demodata)",
            font=dict(size=15),
        ),
        xaxis=dict(
            title="Weeknummer",
            categoryorder="array",
            categoryarray=ACADEMIC_WEEKS,
            tickvals=tickvals,
            tickangle=-45,
        ),
        yaxis=dict(title="Gewogen vooraanmelders"),
        legend=dict(
            orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02,
            font=dict(size=11), tracegroupgap=2,
        ),
        height=550,
        margin=dict(r=180),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Plot 3 & 4: Feature importance — methodologie/xgboost.md
# ---------------------------------------------------------------------------


def plot_feature_importance() -> tuple[go.Figure, go.Figure]:
    classifier_importance = {
        "Croho groepeernaam": 0.18,
        "Weeknummer": 0.14,
        "Type vooropleiding": 0.11,
        "Faculteit": 0.09,
        "Herkomst": 0.08,
        "Nationaliteit": 0.07,
        "Geslacht": 0.05,
        "Collegejaar": 0.05,
        "EER": 0.04,
        "School eerste vooropleiding": 0.04,
        "BBC ontvangen": 0.03,
        "Studieadres postcode": 0.03,
    }

    regressor_importance = {
        "Week 12": 0.22,
        "Week 11": 0.15,
        "Week 10": 0.10,
        "Gewogen_acceleration": 0.08,
        "exclusivity_ratio": 0.07,
        "Croho groepeernaam": 0.06,
        "Gewogen_t-2": 0.06,
        "Week 9": 0.05,
        "Herkomst": 0.04,
        "Faculteit": 0.04,
        "Collegejaar": 0.03,
        "Examentype": 0.03,
    }

    def _make_importance_chart(data: dict, title: str) -> go.Figure:
        sorted_items = sorted(data.items(), key=lambda x: x[1])
        features = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        fig = go.Figure(
            go.Bar(
                x=values,
                y=features,
                orientation="h",
                marker=dict(
                    color=values,
                    colorscale=[[0, "#aec7e8"], [1, "#1f77b4"]],
                ),
                text=[f"{v:.0%}" for v in values],
                textposition="outside",
                hovertemplate="%{y}: %{x:.1%}<extra></extra>",
            )
        )
        fig.update_layout(
            title=dict(text=title, font=dict(size=15)),
            xaxis=dict(
                title="Relatieve importance",
                tickformat=".0%",
                range=[0, max(values) * 1.25],
            ),
            yaxis=dict(title=""),
            height=420,
            margin=dict(l=185, r=50, t=60, b=50),
        )
        return fig

    fig_cls = _make_importance_chart(
        classifier_importance,
        "Feature importance — XGBoost classifier · individueel spoor (demodata)",
    )
    fig_reg = _make_importance_chart(
        regressor_importance,
        "Feature importance — XGBoost regressor · cumulatief spoor (demodata)",
    )
    return fig_cls, fig_reg


# ---------------------------------------------------------------------------
# Plot 5: Historical ratios — methodologie/ratio-model.md
# ---------------------------------------------------------------------------


def plot_historical_ratios(curves: dict, student_counts: dict) -> go.Figure:
    """Ratio per week for one programme across multiple years.

    Shows only the second half of the academic year (week 1-38) where the ratio
    is meaningful — early weeks have too few applicants for a stable ratio.
    """
    fig = go.Figure()
    prog = "B Psychologie"
    herkomst = "NL"

    # Only show weeks 1-38 (index 14 onward in ACADEMIC_WEEKS)
    start_idx = WEEKS_ORDERED.index(1)
    display_weeks = ACADEMIC_WEEKS[start_idx:]

    for i, year in enumerate(RECENT_YEARS):
        curve = curves[(prog, herkomst, year)]
        students = student_counts[(prog, herkomst, year)]

        ratios = curve[start_idx:] / students

        fig.add_trace(
            go.Scatter(
                x=display_weeks,
                y=ratios.tolist(),
                mode="lines",
                name=str(year),
                line=dict(color=PASTEL_YEAR_COLOURS[i], width=2.5),
                hovertemplate="Week %{x}: ratio %{y:.2f}<extra>" + str(year) + "</extra>",
            )
        )

    # 3-year average ratio line
    avg_ratios = np.zeros(len(display_weeks))
    for i, year in enumerate(RECENT_YEARS):
        curve = curves[(prog, herkomst, year)]
        students = student_counts[(prog, herkomst, year)]
        avg_ratios += curve[start_idx:] / students
    avg_ratios /= 3

    fig.add_trace(
        go.Scatter(
            x=display_weeks,
            y=avg_ratios.tolist(),
            mode="lines",
            name="Gemiddelde (3 jaar)",
            line=dict(color="#333", width=2, dash="dash"),
            hovertemplate="Week %{x}: gem. ratio %{y:.2f}<extra>3-jaars gemiddelde</extra>",
        )
    )

    fig.add_hline(
        y=1.0,
        line_dash="dot",
        line_color="#bbb",
        annotation_text="Ratio = 1",
        annotation_position="bottom right",
        annotation_font_color="#999",
    )

    fig.update_layout(
        title=dict(
            text=f"{prog} (NL) — Ratio vooraanmelders / inschrijvingen (demodata)",
            font=dict(size=15),
        ),
        xaxis=dict(
            title="Week in het jaar",
            categoryorder="array",
            categoryarray=display_weeks,
            tickvals=[str(w) for w in range(1, 39, 4)],
            tickangle=-45,
        ),
        yaxis=dict(
            title="Ratio (vooraanmelders / inschrijvingen)",
            range=[0, max(float(avg_ratios.max()) * 1.1, 2.5)],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Plot 6: Ensemble comparison — methodologie/ensemble.md
# ---------------------------------------------------------------------------


def plot_ensemble_comparison(curves: dict, student_counts: dict) -> go.Figure:
    """Bar chart comparing individual model predictions vs ensemble."""
    fig = go.Figure()
    predict_week_idx = WEEKS_ORDERED.index(16)
    progs = [p[0] for p in DEMO_PROGRAMMES]

    sarima_ind, sarima_cum, ratio_pred, ensemble_pred, actuals = [], [], [], [], []

    for prog_name, prog_type, *_ in DEMO_PROGRAMMES:
        curve = curves[(prog_name, "NL", 2024)]
        students = student_counts[(prog_name, "NL", 2024)]

        si = int(students * rng.uniform(0.90, 1.15))
        sc = int(students * rng.uniform(0.85, 1.10))

        ratio_at_week = curve[predict_week_idx] / students if students > 0 else 1.0
        rp = int(curve[predict_week_idx] / max(ratio_at_week, 0.1))

        ens = int(0.3 * si + 0.3 * sc + 0.2 * rp + 0.2 * students)

        sarima_ind.append(si)
        sarima_cum.append(sc)
        ratio_pred.append(rp)
        ensemble_pred.append(ens)
        actuals.append(students)

    for name, values, color in [
        ("SARIMA (individueel)", sarima_ind, MODEL_COLOURS["SARIMA_individual"]),
        ("XGBoost (cumulatief)", sarima_cum, MODEL_COLOURS["SARIMA_cumulative"]),
        ("Ratio-model", ratio_pred, MODEL_COLOURS["Prognose_ratio"]),
        ("Ensemble", ensemble_pred, MODEL_COLOURS["Ensemble"]),
    ]:
        fig.add_trace(
            go.Bar(
                name=name,
                x=progs,
                y=values,
                marker_color=color,
                hovertemplate="%{x}<br>" + name + ": %{y}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            name="Werkelijk",
            x=progs,
            y=actuals,
            mode="markers",
            marker=dict(
                color=MODEL_COLOURS["Werkelijk"],
                size=14,
                symbol="diamond",
                line=dict(width=2, color="white"),
            ),
            hovertemplate="%{x}<br>Werkelijk: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Modelvoorspellingen vs werkelijk — week 16, NL (demodata)",
            font=dict(size=15),
        ),
        xaxis=dict(title=""),
        yaxis=dict(title="Aantal inschrijvingen"),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs",
        "assets",
        "plots",
    )
    os.makedirs(output_dir, exist_ok=True)

    print("Generating documentation plots...")
    curves = _generate_demo_curves()
    student_counts = _generate_student_counts(curves)

    _wrap_html(
        plot_pipeline_cumulative(curves, student_counts),
        "pipeline_cumulative.html",
        output_dir,
    )
    _wrap_html(
        plot_pipeline_individual(curves, student_counts),
        "pipeline_individual.html",
        output_dir,
    )
    _wrap_html(
        plot_sarima_extrapolation(curves, student_counts),
        "sarima_extrapolation.html",
        output_dir,
    )

    fig_cls, fig_reg = plot_feature_importance()
    _wrap_html(fig_cls, "xgb_classifier_importance.html", output_dir)
    _wrap_html(fig_reg, "xgb_regressor_importance.html", output_dir)

    _wrap_html(
        plot_historical_ratios(curves, student_counts),
        "ratio_historical.html",
        output_dir,
    )
    _wrap_html(
        plot_ensemble_comparison(curves, student_counts),
        "ensemble_comparison.html",
        output_dir,
    )

    print(f"\nGenerated 7 plots in {output_dir}")


if __name__ == "__main__":
    main()
