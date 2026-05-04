"""Generate HTML benchmark reports for cumulative and individual spoor."""

import json
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CSS = """\
:root {
    --clr-primary: #1a5276;
    --clr-accent: #2980b9;
    --clr-bg: #f8f9fa;
    --clr-card: #ffffff;
    --clr-border: #dee2e6;
    --clr-best: #d4edda;
    --clr-text: #212529;
    --clr-muted: #6c757d;
    --radius: 8px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--clr-bg);
    color: var(--clr-text);
    line-height: 1.6;
    padding: 2rem;
}
.container { max-width: 1200px; margin: 0 auto; }
header {
    background: var(--clr-primary);
    color: white;
    padding: 2rem;
    border-radius: var(--radius);
    margin-bottom: 2rem;
}
header h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
header p { opacity: 0.9; font-size: 0.95rem; }
.meta-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}
.meta-card {
    background: var(--clr-card);
    border: 1px solid var(--clr-border);
    border-radius: var(--radius);
    padding: 1.2rem;
    text-align: center;
}
.meta-card .number {
    font-size: 2rem;
    font-weight: 700;
    color: var(--clr-accent);
}
.meta-card .label {
    font-size: 0.85rem;
    color: var(--clr-muted);
    margin-top: 0.3rem;
}
section {
    background: var(--clr-card);
    border: 1px solid var(--clr-border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
h2 {
    font-size: 1.3rem;
    color: var(--clr-primary);
    margin-bottom: 0.3rem;
}
.section-desc {
    color: var(--clr-muted);
    font-size: 0.9rem;
    margin-bottom: 1rem;
}
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    margin-top: 0.5rem;
}
th {
    background: var(--clr-primary);
    color: white;
    padding: 0.6rem 0.8rem;
    text-align: left;
    font-weight: 600;
}
td {
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid var(--clr-border);
}
tr:hover { background: #f1f3f5; }
.best-row { background: var(--clr-best) !important; }
.best-row:hover { background: #c3e6cb !important; }
.chart-container {
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.chart-box {
    flex: 1;
    min-width: 300px;
    height: 300px;
    position: relative;
}
canvas { width: 100% !important; height: 100% !important; }
.tab-nav {
    display: flex;
    gap: 0;
    border-bottom: 2px solid var(--clr-border);
    margin-bottom: 1rem;
}
.tab-btn {
    padding: 0.5rem 1.2rem;
    border: none;
    background: none;
    cursor: pointer;
    font-size: 0.9rem;
    color: var(--clr-muted);
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    transition: all 0.2s;
}
.tab-btn.active {
    color: var(--clr-primary);
    border-bottom-color: var(--clr-accent);
    font-weight: 600;
}
.tab-content { display: none; }
.tab-content.active { display: block; }
footer {
    text-align: center;
    color: var(--clr-muted);
    font-size: 0.8rem;
    margin-top: 2rem;
    padding: 1rem;
}
.roc-wrapper {
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.roc-main {
    flex: 2;
    min-width: 400px;
    position: relative;
}
.roc-main canvas {
    width: 100% !important;
    aspect-ratio: 1 / 1;
    max-height: 480px;
}
.roc-sidebar {
    flex: 1;
    min-width: 220px;
}
.roc-legend {
    list-style: none;
    padding: 0;
    margin: 0;
}
.roc-legend li {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0.7rem;
    border-radius: 6px;
    margin-bottom: 0.3rem;
    font-size: 0.9rem;
    cursor: pointer;
    transition: background 0.15s;
    user-select: none;
}
.roc-legend li:hover { background: #f1f3f5; }
.roc-legend li.dimmed { opacity: 0.35; }
.roc-legend .dot {
    width: 14px; height: 14px;
    border-radius: 50%;
    flex-shrink: 0;
}
.roc-legend .model-name { font-weight: 600; }
.roc-legend .auc-val {
    margin-left: auto;
    font-variant-numeric: tabular-nums;
    color: var(--clr-muted);
    font-size: 0.85rem;
}
.roc-year-btns {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 1rem;
}
.roc-year-btn {
    padding: 0.3rem 0.7rem;
    border: 1px solid var(--clr-border);
    border-radius: 16px;
    background: var(--clr-card);
    cursor: pointer;
    font-size: 0.8rem;
    color: var(--clr-muted);
    transition: all 0.15s;
}
.roc-year-btn:hover { border-color: var(--clr-accent); color: var(--clr-text); }
.roc-year-btn.active {
    background: var(--clr-accent);
    color: white;
    border-color: var(--clr-accent);
}
.roc-help {
    margin-top: 1rem;
    padding: 0.8rem 1rem;
    background: #eef6fc;
    border-radius: 6px;
    font-size: 0.82rem;
    color: #2c3e50;
    line-height: 1.5;
}
.roc-help strong { color: var(--clr-primary); }
@media (max-width: 768px) {
    body { padding: 1rem; }
    .chart-container { flex-direction: column; }
    .roc-wrapper { flex-direction: column; }
    table { font-size: 0.8rem; }
}"""

_TAB_JS = """\
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const section = btn.closest('section');
        section.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        section.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        section.querySelector('#' + btn.dataset.tab).classList.add('active');
    });
});"""

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "gradient_boosting": "Gradient Boosting",
    "extra_trees": "Extra Trees",
}

COLORS_JS = """\
const COLORS = {
    sarima: '#2980b9',
    ets: '#e67e22',
    theta: '#27ae60',
    xgboost: '#2980b9',
    ridge: '#e67e22',
    random_forest: '#27ae60',
    logistic_regression: '#8e44ad',
    gradient_boosting: '#d35400',
    extra_trees: '#16a085',
};"""


def _fmt(val: float, decimals: int = 4) -> str:
    return f"{val:.{decimals}f}"


def _pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def _label(model: str) -> str:
    return MODEL_LABELS.get(model, model.upper())


def _write(path: str, html: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Benchmark rapport: {path}")


# ---------------------------------------------------------------------------
# Shared table builders (timeseries)
# ---------------------------------------------------------------------------

def _ts_models(ts: pd.DataFrame) -> list[str]:
    order = ["sarima", "ets", "theta"]
    return [m for m in order if m in ts["model"].unique()]


def _build_ts_overall(ts: pd.DataFrame) -> str:
    models = _ts_models(ts)
    agg = (
        ts.groupby("model")[["mape", "mae", "rmse", "train_time_s"]]
        .agg(["mean", "median"])
        .round(4)
    )
    best = ts.groupby("model")["mape"].mean().idxmin()
    rows = ""
    for m in models:
        r = agg.loc[m]
        cls = ' class="best-row"' if m == best else ""
        rows += f"""<tr{cls}>
            <td><strong>{m.upper()}</strong></td>
            <td>{_pct(r[('mape','mean')])}</td><td>{_pct(r[('mape','median')])}</td>
            <td>{_fmt(r[('mae','mean')], 2)}</td><td>{_fmt(r[('mae','median')], 2)}</td>
            <td>{_fmt(r[('rmse','mean')], 2)}</td><td>{_fmt(r[('rmse','median')], 2)}</td>
            <td>{_fmt(r[('train_time_s','mean')], 3)}s</td>
        </tr>"""
    return rows


def _build_ts_by_group(ts: pd.DataFrame, group_col: str) -> str:
    models = _ts_models(ts)
    agg = ts.groupby(["model", group_col])[["mape", "mae", "rmse"]].mean().round(4)
    groups = sorted(ts[group_col].unique())
    rows = ""
    for m in models:
        for i, g in enumerate(groups):
            r = agg.loc[(m, g)]
            rowspan = f' rowspan="{len(groups)}"' if i == 0 else ""
            mcell = f"<td{rowspan}><strong>{m.upper()}</strong></td>" if i == 0 else ""
            rows += f"""<tr>{mcell}<td>{g}</td>
                <td>{_pct(r['mape'])}</td><td>{_fmt(r['mae'], 2)}</td><td>{_fmt(r['rmse'], 2)}</td>
            </tr>"""
    return rows


def _chart_ts_overall_json(ts: pd.DataFrame) -> str:
    models = _ts_models(ts)
    agg = ts.groupby("model")["mape"].mean().round(4)
    return json.dumps({m: float(agg.loc[m]) * 100 for m in models})


def _chart_ts_by_year_json(ts: pd.DataFrame) -> str:
    models = _ts_models(ts)
    agg = ts.groupby(["model", "test_year"])["mape"].mean().round(4)
    years = sorted(ts["test_year"].unique())
    data = {m: [float(agg.loc[(m, y)]) * 100 for y in years] for m in models}
    return json.dumps({"labels": [str(y) for y in years], "datasets": data})


def _ts_section(ts: pd.DataFrame, prefix: str, step_label: str) -> str:
    has_herkomst = "herkomst" in ts.columns
    herkomst_tab_btn = (
        f'<button class="tab-btn" data-tab="{prefix}-herkomst">Per herkomst</button>'
        if has_herkomst else ""
    )
    herkomst_tab_content = ""
    if has_herkomst:
        herkomst_tab_content = f"""
    <div id="{prefix}-herkomst" class="tab-content">
        <table>
            <thead><tr><th>Model</th><th>Herkomst</th><th>MAPE</th><th>MAE</th><th>RMSE</th></tr></thead>
            <tbody>{_build_ts_by_group(ts, "herkomst")}</tbody>
        </table>
    </div>"""

    return f"""
<section>
    <h2>{step_label} — Tijdreeksmodellen</h2>
    <p class="section-desc">
        Per opleiding × herkomst × examentype wordt een tijdreeksmodel gefit om
        het aantal inschrijvingen te voorspellen. Getest via expanding-window cross-validatie.
    </p>
    <div class="tab-nav">
        <button class="tab-btn active" data-tab="{prefix}-overall">Overzicht</button>
        <button class="tab-btn" data-tab="{prefix}-examentype">Per examentype</button>
        <button class="tab-btn" data-tab="{prefix}-year">Per testjaar</button>
        {herkomst_tab_btn}
    </div>
    <div id="{prefix}-overall" class="tab-content active">
        <table>
            <thead><tr>
                <th>Model</th>
                <th>MAPE (gem.)</th><th>MAPE (med.)</th>
                <th>MAE (gem.)</th><th>MAE (med.)</th>
                <th>RMSE (gem.)</th><th>RMSE (med.)</th>
                <th>Traintijd</th>
            </tr></thead>
            <tbody>{_build_ts_overall(ts)}</tbody>
        </table>
        <div class="chart-container">
            <div class="chart-box"><canvas id="chart-{prefix}-bar"></canvas></div>
            <div class="chart-box"><canvas id="chart-{prefix}-line"></canvas></div>
        </div>
    </div>
    <div id="{prefix}-examentype" class="tab-content">
        <table>
            <thead><tr><th>Model</th><th>Examentype</th><th>MAPE</th><th>MAE</th><th>RMSE</th></tr></thead>
            <tbody>{_build_ts_by_group(ts, "examentype")}</tbody>
        </table>
    </div>
    <div id="{prefix}-year" class="tab-content">
        <table>
            <thead><tr><th>Model</th><th>Testjaar</th><th>MAPE</th><th>MAE</th><th>RMSE</th></tr></thead>
            <tbody>{_build_ts_by_group(ts, "test_year")}</tbody>
        </table>
    </div>
    {herkomst_tab_content}
</section>"""


def _ts_chart_js(ts: pd.DataFrame, prefix: str) -> str:
    models = _ts_models(ts)
    labels_js = json.dumps([m.upper() for m in models])
    keys_js = json.dumps(models)
    return f"""
const {prefix}Overall = {_chart_ts_overall_json(ts)};
const {prefix}ByYear = {_chart_ts_by_year_json(ts)};
new Chart(document.getElementById('chart-{prefix}-bar'), {{
    type: 'bar',
    data: {{
        labels: {labels_js},
        datasets: [{{
            label: 'MAPE (%)',
            data: {json.dumps(keys_js)}.map(k => {prefix}Overall[k]),
            backgroundColor: {json.dumps(keys_js)}.map(k => COLORS[k]),
            borderRadius: 4,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ title: {{ display: true, text: 'Gemiddelde MAPE per model' }}, legend: {{ display: false }} }},
        scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'MAPE (%)' }} }} }}
    }}
}});
new Chart(document.getElementById('chart-{prefix}-line'), {{
    type: 'line',
    data: {{
        labels: {prefix}ByYear.labels,
        datasets: Object.entries({prefix}ByYear.datasets).map(([name, vals]) => ({{
            label: name.toUpperCase(),
            data: vals,
            borderColor: COLORS[name],
            backgroundColor: COLORS[name] + '20',
            tension: 0.3, fill: false, pointRadius: 4,
        }}))
    }},
    options: {{
        responsive: true,
        plugins: {{ title: {{ display: true, text: 'MAPE per testjaar' }} }},
        scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'MAPE (%)' }} }} }}
    }}
}});"""


# ---------------------------------------------------------------------------
# Shared table builders (regressor)
# ---------------------------------------------------------------------------

def _reg_models(reg: pd.DataFrame) -> list[str]:
    order = ["xgboost", "ridge", "random_forest", "gradient_boosting", "extra_trees"]
    present = reg["model"].unique()
    return [m for m in order if m in present] + [m for m in present if m not in order]


def _build_reg_overall(reg: pd.DataFrame) -> str:
    models = _reg_models(reg)
    agg = reg.groupby("model")[["mape", "mae", "rmse", "train_time_s"]].mean().round(4)
    best = reg.groupby("model")["mape"].mean().idxmin()
    rows = ""
    for m in models:
        r = agg.loc[m]
        cls = ' class="best-row"' if m == best else ""
        rows += f"""<tr{cls}>
            <td><strong>{_label(m)}</strong></td>
            <td>{_pct(r['mape'])}</td><td>{_fmt(r['mae'], 2)}</td><td>{_fmt(r['rmse'], 2)}</td>
            <td>{_fmt(r['train_time_s'], 3)}s</td>
        </tr>"""
    return rows


def _build_reg_by_group(reg: pd.DataFrame, group_col: str) -> str:
    models = _reg_models(reg)
    agg = reg.groupby(["model", group_col])[["mape", "mae", "rmse"]].mean().round(4)
    groups = sorted(reg[group_col].unique())
    rows = ""
    for m in models:
        for i, g in enumerate(groups):
            r = agg.loc[(m, g)]
            rowspan = f' rowspan="{len(groups)}"' if i == 0 else ""
            mcell = f"<td{rowspan}><strong>{_label(m)}</strong></td>" if i == 0 else ""
            rows += f"""<tr>{mcell}<td>{g}</td>
                <td>{_pct(r['mape'])}</td><td>{_fmt(r['mae'], 2)}</td><td>{_fmt(r['rmse'], 2)}</td>
            </tr>"""
    return rows


def _chart_reg_overall_json(reg: pd.DataFrame) -> str:
    models = _reg_models(reg)
    agg = reg.groupby("model")["mape"].mean().round(4)
    return json.dumps({m: float(agg.loc[m]) * 100 for m in models})


def _chart_reg_by_year_json(reg: pd.DataFrame) -> str:
    models = _reg_models(reg)
    agg = reg.groupby(["model", "test_year"])["mape"].mean().round(4)
    years = sorted(reg["test_year"].unique())
    data = {m: [float(agg.loc[(m, y)]) * 100 for y in years] for m in models}
    return json.dumps({"labels": [str(y) for y in years], "datasets": data})


def _reg_section(reg: pd.DataFrame, prefix: str, step_label: str) -> str:
    return f"""
<section>
    <h2>{step_label} — Regressiemodellen</h2>
    <p class="section-desc">
        Op geaggregeerd niveau wordt een regressiemodel getraind op cumulatieve weekcijfers
        om het eindtotaal te voorspellen. Getest via rolling-origin cross-validatie.
    </p>
    <div class="tab-nav">
        <button class="tab-btn active" data-tab="{prefix}-overall">Overzicht</button>
        <button class="tab-btn" data-tab="{prefix}-examentype">Per examentype</button>
        <button class="tab-btn" data-tab="{prefix}-year">Per testjaar</button>
    </div>
    <div id="{prefix}-overall" class="tab-content active">
        <table>
            <thead><tr><th>Model</th><th>MAPE</th><th>MAE</th><th>RMSE</th><th>Traintijd</th></tr></thead>
            <tbody>{_build_reg_overall(reg)}</tbody>
        </table>
        <div class="chart-container">
            <div class="chart-box"><canvas id="chart-{prefix}-bar"></canvas></div>
            <div class="chart-box"><canvas id="chart-{prefix}-line"></canvas></div>
        </div>
    </div>
    <div id="{prefix}-examentype" class="tab-content">
        <table>
            <thead><tr><th>Model</th><th>Examentype</th><th>MAPE</th><th>MAE</th><th>RMSE</th></tr></thead>
            <tbody>{_build_reg_by_group(reg, "examentype")}</tbody>
        </table>
    </div>
    <div id="{prefix}-year" class="tab-content">
        <table>
            <thead><tr><th>Model</th><th>Testjaar</th><th>MAPE</th><th>MAE</th><th>RMSE</th></tr></thead>
            <tbody>{_build_reg_by_group(reg, "test_year")}</tbody>
        </table>
    </div>
</section>"""


def _reg_chart_js(reg: pd.DataFrame, prefix: str) -> str:
    models = _reg_models(reg)
    labels_js = json.dumps([_label(m) for m in models])
    keys_js = json.dumps(models)
    label_map_js = json.dumps({m: _label(m) for m in models})
    return f"""
const {prefix}Overall = {_chart_reg_overall_json(reg)};
const {prefix}ByYear = {_chart_reg_by_year_json(reg)};
new Chart(document.getElementById('chart-{prefix}-bar'), {{
    type: 'bar',
    data: {{
        labels: {labels_js},
        datasets: [{{
            label: 'MAPE (%)',
            data: {keys_js}.map(k => {prefix}Overall[k]),
            backgroundColor: {keys_js}.map(k => COLORS[k]),
            borderRadius: 4,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ title: {{ display: true, text: 'Gemiddelde MAPE per model' }}, legend: {{ display: false }} }},
        scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'MAPE (%)' }} }} }}
    }}
}});
new Chart(document.getElementById('chart-{prefix}-line'), {{
    type: 'line',
    data: {{
        labels: {prefix}ByYear.labels,
        datasets: Object.entries({prefix}ByYear.datasets).map(([name, vals]) => ({{
            label: {label_map_js}[name] || name,
            data: vals,
            borderColor: COLORS[name],
            backgroundColor: COLORS[name] + '20',
            tension: 0.3, fill: false, pointRadius: 4,
        }}))
    }},
    options: {{
        responsive: true,
        plugins: {{ title: {{ display: true, text: 'MAPE per testjaar' }} }},
        scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'MAPE (%)' }} }} }}
    }}
}});"""


# ---------------------------------------------------------------------------
# Shared table builders (classifier)
# ---------------------------------------------------------------------------

def _clf_models(clf: pd.DataFrame) -> list[str]:
    order = ["xgboost", "random_forest", "logistic_regression", "gradient_boosting", "extra_trees"]
    present = clf["model"].unique()
    return [m for m in order if m in present] + [m for m in present if m not in order]


def _build_clf_overall(clf: pd.DataFrame) -> str:
    models = _clf_models(clf)
    agg = (
        clf.groupby("model")[["accuracy", "auc_roc", "f1", "aggregate_mae", "train_time_s"]]
        .mean()
        .round(4)
    )
    best = clf.groupby("model")["auc_roc"].mean().idxmax()
    rows = ""
    for m in models:
        r = agg.loc[m]
        cls = ' class="best-row"' if m == best else ""
        rows += f"""<tr{cls}>
            <td><strong>{_label(m)}</strong></td>
            <td>{_pct(r['accuracy'])}</td>
            <td>{_fmt(r['auc_roc'], 4)}</td>
            <td>{_fmt(r['f1'], 4)}</td>
            <td>{_fmt(r['aggregate_mae'], 2)}</td>
            <td>{_fmt(r['train_time_s'], 3)}s</td>
        </tr>"""
    return rows


def _build_clf_by_year(clf: pd.DataFrame) -> str:
    models = _clf_models(clf)
    agg = (
        clf.groupby(["model", "test_year"])[["accuracy", "auc_roc", "f1", "aggregate_mae"]]
        .mean()
        .round(4)
    )
    years = sorted(clf["test_year"].unique())
    rows = ""
    for m in models:
        for i, y in enumerate(years):
            r = agg.loc[(m, y)]
            rowspan = f' rowspan="{len(years)}"' if i == 0 else ""
            mcell = f"<td{rowspan}><strong>{_label(m)}</strong></td>" if i == 0 else ""
            rows += f"""<tr>{mcell}<td>{y}</td>
                <td>{_pct(r['accuracy'])}</td><td>{_fmt(r['auc_roc'], 4)}</td>
                <td>{_fmt(r['f1'], 4)}</td><td>{_fmt(r['aggregate_mae'], 2)}</td>
            </tr>"""
    return rows


def _chart_clf_overall_json(clf: pd.DataFrame) -> str:
    models = _clf_models(clf)
    agg = clf.groupby("model")["auc_roc"].mean().round(4)
    return json.dumps({m: float(agg.loc[m]) for m in models})


def _chart_clf_by_year_json(clf: pd.DataFrame) -> str:
    models = _clf_models(clf)
    agg = clf.groupby(["model", "test_year"])["auc_roc"].mean().round(4)
    years = sorted(clf["test_year"].unique())
    data = {m: [float(agg.loc[(m, y)]) for y in years] for m in models}
    return json.dumps({"labels": [str(y) for y in years], "datasets": data})


def _clf_section(clf: pd.DataFrame) -> str:
    return f"""
<section>
    <h2>Stap 1 — Classificatiemodellen</h2>
    <p class="section-desc">
        Per aanmelder wordt een classificatiemodel getraind om te voorspellen of iemand
        zich daadwerkelijk inschrijft. Getest via expanding-window cross-validatie.
    </p>
    <div class="tab-nav">
        <button class="tab-btn active" data-tab="clf-overall">Overzicht</button>
        <button class="tab-btn" data-tab="clf-year">Per testjaar</button>
    </div>
    <div id="clf-overall" class="tab-content active">
        <table>
            <thead><tr>
                <th>Model</th><th>Accuracy</th><th>AUC-ROC</th><th>F1</th>
                <th>Agg. MAE</th><th>Traintijd</th>
            </tr></thead>
            <tbody>{_build_clf_overall(clf)}</tbody>
        </table>
        <div class="chart-container">
            <div class="chart-box"><canvas id="chart-clf-bar"></canvas></div>
            <div class="chart-box"><canvas id="chart-clf-line"></canvas></div>
        </div>
    </div>
    <div id="clf-year" class="tab-content">
        <table>
            <thead><tr>
                <th>Model</th><th>Testjaar</th><th>Accuracy</th><th>AUC-ROC</th>
                <th>F1</th><th>Agg. MAE</th>
            </tr></thead>
            <tbody>{_build_clf_by_year(clf)}</tbody>
        </table>
    </div>
</section>"""


def _clf_chart_js(clf: pd.DataFrame) -> str:
    models = _clf_models(clf)
    labels_js = json.dumps([_label(m) for m in models])
    keys_js = json.dumps(models)
    label_map_js = json.dumps({m: _label(m) for m in models})
    return f"""
const clfOverall = {_chart_clf_overall_json(clf)};
const clfByYear = {_chart_clf_by_year_json(clf)};
new Chart(document.getElementById('chart-clf-bar'), {{
    type: 'bar',
    data: {{
        labels: {labels_js},
        datasets: [{{
            label: 'AUC-ROC',
            data: {keys_js}.map(k => clfOverall[k]),
            backgroundColor: {keys_js}.map(k => COLORS[k]),
            borderRadius: 4,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ title: {{ display: true, text: 'Gemiddelde AUC-ROC per model' }}, legend: {{ display: false }} }},
        scales: {{ y: {{ beginAtZero: true, max: 1, title: {{ display: true, text: 'AUC-ROC' }} }} }}
    }}
}});
new Chart(document.getElementById('chart-clf-line'), {{
    type: 'line',
    data: {{
        labels: clfByYear.labels,
        datasets: Object.entries(clfByYear.datasets).map(([name, vals]) => ({{
            label: {label_map_js}[name] || name,
            data: vals,
            borderColor: COLORS[name],
            backgroundColor: COLORS[name] + '20',
            tension: 0.3, fill: false, pointRadius: 4,
        }}))
    }},
    options: {{
        responsive: true,
        plugins: {{ title: {{ display: true, text: 'AUC-ROC per testjaar' }} }},
        scales: {{ y: {{ min: 0.5, max: 1, title: {{ display: true, text: 'AUC-ROC' }} }} }}
    }}
}});"""


# ---------------------------------------------------------------------------
# ROC curve helpers
# ---------------------------------------------------------------------------

N_ROC_POINTS = 200


def _interpolate_roc(fpr: list, tpr: list) -> tuple[list, list]:
    """Interpolate a single ROC curve to N_ROC_POINTS evenly-spaced FPR values."""
    base_fpr = np.linspace(0, 1, N_ROC_POINTS)
    interp_tpr = np.interp(base_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    return base_fpr.tolist(), interp_tpr.tolist()


def _compute_mean_roc(curves: list[dict]) -> dict:
    """Compute mean ROC curve + std band from a list of per-fold curves."""
    if not curves:
        return {"fpr": [], "tpr": [], "tpr_upper": [], "tpr_lower": [], "auc": 0}

    base_fpr = np.linspace(0, 1, N_ROC_POINTS)
    all_tpr = []
    for c in curves:
        interp_tpr = np.interp(base_fpr, c["fpr"], c["tpr"])
        interp_tpr[0] = 0.0
        all_tpr.append(interp_tpr)

    tpr_matrix = np.array(all_tpr)
    mean_tpr = tpr_matrix.mean(axis=0)
    std_tpr = tpr_matrix.std(axis=0)
    mean_tpr[-1] = 1.0

    auc_val = float(np.trapezoid(mean_tpr, base_fpr))

    return {
        "fpr": np.round(base_fpr, 6).tolist(),
        "tpr": np.round(mean_tpr, 6).tolist(),
        "tpr_upper": np.round(np.minimum(mean_tpr + std_tpr, 1), 6).tolist(),
        "tpr_lower": np.round(np.maximum(mean_tpr - std_tpr, 0), 6).tolist(),
        "auc": round(auc_val, 4),
    }


def _prepare_roc_data(
    roc_curves: dict[str, list[dict]],
) -> dict:
    """Build the full ROC data structure for the JS chart.

    Returns dict with:
        models: {model_name: {mean: {...}, per_year: {year: {fpr, tpr}}}}
        years: sorted list of all years
    """
    models = {}
    all_years = set()

    for model_name, curves in roc_curves.items():
        mean = _compute_mean_roc(curves)
        per_year = {}
        for c in curves:
            year = str(c["test_year"])
            all_years.add(year)
            fpr_i, tpr_i = _interpolate_roc(c["fpr"], c["tpr"])
            auc_i = float(np.trapezoid(tpr_i, fpr_i))
            per_year[year] = {
                "fpr": [round(v, 6) for v in fpr_i],
                "tpr": [round(v, 6) for v in tpr_i],
                "auc": round(auc_i, 4),
            }
        models[model_name] = {"mean": mean, "per_year": per_year}

    return {"models": models, "years": sorted(all_years)}


def _roc_section(roc_curves: dict[str, list[dict]]) -> str:
    """Build the ROC curve HTML section."""
    if not roc_curves or all(len(v) == 0 for v in roc_curves.values()):
        return ""

    models = list(roc_curves.keys())
    model_labels_html = ""
    for m in models:
        model_labels_html += f"""
            <li data-model="{m}">
                <span class="dot" style="background: {_roc_color(m)}"></span>
                <span class="model-name">{_label(m)}</span>
                <span class="auc-val" id="auc-{m}"></span>
            </li>"""

    return f"""
<section>
    <h2>ROC-curves — Classificatiemodellen</h2>
    <p class="section-desc">
        Receiver Operating Characteristic curve per model. De curve toont de
        trade-off tussen True Positive Rate en False Positive Rate bij
        verschillende drempelwaarden.
    </p>
    <div class="roc-wrapper">
        <div class="roc-main">
            <canvas id="chart-roc"></canvas>
        </div>
        <div class="roc-sidebar">
            <ul class="roc-legend" id="roc-legend">
                {model_labels_html}
            </ul>
            <div class="roc-year-btns" id="roc-year-btns">
                <button class="roc-year-btn active" data-year="mean">Gemiddeld</button>
            </div>
            <div class="roc-help">
                <strong>Hoe lees je dit?</strong><br>
                Hoe verder de curve naar linksboven buigt, hoe beter het model onderscheid maakt
                tussen inschrijvers en niet-inschrijvers. De stippellijn is een random classifier
                (AUC = 0.50). Het gekleurde vlak toont de spreiding over test-jaren.
            </div>
        </div>
    </div>
</section>"""


def _roc_color(model: str) -> str:
    colors = {
        "xgboost": "#2980b9",
        "random_forest": "#27ae60",
        "logistic_regression": "#8e44ad",
        "gradient_boosting": "#d35400",
        "extra_trees": "#16a085",
    }
    return colors.get(model, "#e67e22")


def _roc_chart_js(roc_curves: dict[str, list[dict]]) -> str:
    """Generate the Chart.js code for interactive ROC curves."""
    if not roc_curves or all(len(v) == 0 for v in roc_curves.values()):
        return ""

    roc_data = _prepare_roc_data(roc_curves)
    models = list(roc_data["models"].keys())

    color_map = {m: _roc_color(m) for m in models}
    label_map = {m: _label(m) for m in models}

    return f"""
const ROC_DATA = {json.dumps(roc_data)};
const ROC_COLORS = {json.dumps(color_map)};
const LABEL_MAP = {json.dumps(label_map)};

(function() {{
    const ctx = document.getElementById('chart-roc').getContext('2d');
    let currentView = 'mean';
    const hiddenModels = new Set();

    function hexToRgba(hex, alpha) {{
        const r = parseInt(hex.slice(1,3), 16);
        const g = parseInt(hex.slice(3,5), 16);
        const b = parseInt(hex.slice(5,7), 16);
        return `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
    }}

    function buildDatasets() {{
        const datasets = [];

        // Diagonal reference line
        datasets.push({{
            label: 'Random (AUC = 0.50)',
            data: [{{x:0,y:0}},{{x:1,y:1}}],
            showLine: true,
            borderColor: '#bdc3c8',
            borderDash: [6, 4],
            borderWidth: 1.5,
            pointRadius: 0,
            pointHitRadius: 0,
            order: 100,
        }});

        const modelKeys = Object.keys(ROC_DATA.models);

        for (const model of modelKeys) {{
            const info = ROC_DATA.models[model];
            const color = ROC_COLORS[model];
            const hidden = hiddenModels.has(model);

            if (currentView === 'mean') {{
                // Confidence band (upper)
                const bandUpper = info.mean.fpr.map((x, i) => ({{x, y: info.mean.tpr_upper[i]}}));
                datasets.push({{
                    label: '_band_upper_' + model,
                    data: bandUpper,
                    showLine: true,
                    borderColor: 'transparent',
                    backgroundColor: hexToRgba(color, 0.10),
                    fill: '+1',
                    pointRadius: 0,
                    pointHitRadius: 0,
                    hidden,
                    order: 20,
                }});
                // Confidence band (lower)
                const bandLower = info.mean.fpr.map((x, i) => ({{x, y: info.mean.tpr_lower[i]}}));
                datasets.push({{
                    label: '_band_lower_' + model,
                    data: bandLower,
                    showLine: true,
                    borderColor: 'transparent',
                    backgroundColor: 'transparent',
                    fill: false,
                    pointRadius: 0,
                    pointHitRadius: 0,
                    hidden,
                    order: 21,
                }});
                // Mean line
                const meanPts = info.mean.fpr.map((x, i) => ({{x, y: info.mean.tpr[i]}}));
                datasets.push({{
                    label: LABEL_MAP[model] + ' (AUC = ' + info.mean.auc.toFixed(4) + ')',
                    data: meanPts,
                    showLine: true,
                    borderColor: color,
                    borderWidth: 2.5,
                    backgroundColor: hexToRgba(color, 0.05),
                    fill: false,
                    pointRadius: 0,
                    pointHitRadius: 6,
                    tension: 0.1,
                    hidden,
                    order: 10,
                }});
            }} else {{
                // Single year
                const yearData = info.per_year[currentView];
                if (yearData) {{
                    const pts = yearData.fpr.map((x, i) => ({{x, y: yearData.tpr[i]}}));
                    datasets.push({{
                        label: LABEL_MAP[model] + ' (AUC = ' + yearData.auc.toFixed(4) + ')',
                        data: pts,
                        showLine: true,
                        borderColor: color,
                        borderWidth: 2.5,
                        fill: false,
                        pointRadius: 0,
                        pointHitRadius: 6,
                        tension: 0.1,
                        hidden,
                        order: 10,
                    }});
                }}
            }}
        }}
        return datasets;
    }}

    function updateAucLabels() {{
        const modelKeys = Object.keys(ROC_DATA.models);
        for (const model of modelKeys) {{
            const el = document.getElementById('auc-' + model);
            if (!el) continue;
            const info = ROC_DATA.models[model];
            let auc;
            if (currentView === 'mean') {{
                auc = info.mean.auc;
            }} else {{
                const yd = info.per_year[currentView];
                auc = yd ? yd.auc : null;
            }}
            el.textContent = auc !== null ? 'AUC ' + auc.toFixed(4) : '';
        }}
    }}

    const rocChart = new Chart(ctx, {{
        type: 'scatter',
        data: {{ datasets: buildDatasets() }},
        options: {{
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 1,
            animation: {{ duration: 350, easing: 'easeOutQuart' }},
            interaction: {{
                mode: 'nearest',
                axis: 'x',
                intersect: false,
            }},
            plugins: {{
                legend: {{ display: false }},
                title: {{ display: false }},
                tooltip: {{
                    backgroundColor: 'rgba(26, 82, 118, 0.92)',
                    titleFont: {{ weight: '600' }},
                    bodyFont: {{ size: 12 }},
                    padding: 10,
                    cornerRadius: 6,
                    callbacks: {{
                        label: function(ctx) {{
                            if (ctx.dataset.label.startsWith('_')) return null;
                            return ctx.dataset.label.split(' (')[0] +
                                ': FPR=' + ctx.parsed.x.toFixed(3) +
                                ', TPR=' + ctx.parsed.y.toFixed(3);
                        }}
                    }},
                    filter: function(item) {{
                        return !item.dataset.label.startsWith('_');
                    }}
                }}
            }},
            scales: {{
                x: {{
                    type: 'linear',
                    min: 0, max: 1,
                    title: {{ display: true, text: 'False Positive Rate', font: {{ weight: '600' }} }},
                    grid: {{ color: '#eee' }},
                    ticks: {{ stepSize: 0.2 }}
                }},
                y: {{
                    type: 'linear',
                    min: 0, max: 1,
                    title: {{ display: true, text: 'True Positive Rate', font: {{ weight: '600' }} }},
                    grid: {{ color: '#eee' }},
                    ticks: {{ stepSize: 0.2 }}
                }}
            }}
        }}
    }});

    updateAucLabels();

    // Year buttons
    const yearBtnsContainer = document.getElementById('roc-year-btns');
    ROC_DATA.years.forEach(year => {{
        const btn = document.createElement('button');
        btn.className = 'roc-year-btn';
        btn.dataset.year = year;
        btn.textContent = year;
        yearBtnsContainer.appendChild(btn);
    }});

    yearBtnsContainer.addEventListener('click', function(e) {{
        const btn = e.target.closest('.roc-year-btn');
        if (!btn) return;
        yearBtnsContainer.querySelectorAll('.roc-year-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentView = btn.dataset.year;
        rocChart.data.datasets = buildDatasets();
        rocChart.update();
        updateAucLabels();
    }});

    // Legend toggle
    document.getElementById('roc-legend').addEventListener('click', function(e) {{
        const li = e.target.closest('li');
        if (!li) return;
        const model = li.dataset.model;
        if (hiddenModels.has(model)) {{
            hiddenModels.delete(model);
            li.classList.remove('dimmed');
        }} else {{
            hiddenModels.add(model);
            li.classList.add('dimmed');
        }}
        rocChart.data.datasets = buildDatasets();
        rocChart.update();
    }});
}})();"""


# ---------------------------------------------------------------------------
# Metrics explanation (shared)
# ---------------------------------------------------------------------------

_METRICS_SECTION = """
<section>
    <h2>Toelichting metrieken</h2>
    <table>
        <thead><tr><th>Metriek</th><th>Beschrijving</th><th>Interpretatie</th></tr></thead>
        <tbody>
            <tr><td><strong>MAPE</strong></td><td>Mean Absolute Percentage Error</td><td>Gemiddelde procentuele afwijking — hoe lager, hoe beter. Onafhankelijk van schaal.</td></tr>
            <tr><td><strong>MAE</strong></td><td>Mean Absolute Error</td><td>Gemiddelde absolute afwijking in aantallen studenten.</td></tr>
            <tr><td><strong>RMSE</strong></td><td>Root Mean Squared Error</td><td>Wortel van gemiddelde kwadratische fout — straft grote uitschieters zwaarder af.</td></tr>
            <tr><td><strong>Accuracy</strong></td><td>Fractie correcte voorspellingen</td><td>Aandeel aanmelders dat correct als inschrijver/niet-inschrijver wordt voorspeld.</td></tr>
            <tr><td><strong>AUC-ROC</strong></td><td>Area Under ROC Curve</td><td>Onderscheidend vermogen — 1.0 is perfect, 0.5 is random.</td></tr>
            <tr><td><strong>F1</strong></td><td>Harmonisch gemiddelde precision/recall</td><td>Balans tussen correcte positieven en volledigheid.</td></tr>
            <tr><td><strong>Agg. MAE</strong></td><td>Aggregate MAE</td><td>MAE op geaggregeerd niveau (per opleiding × herkomst).</td></tr>
        </tbody>
    </table>
</section>"""


# ---------------------------------------------------------------------------
# HTML wrappers
# ---------------------------------------------------------------------------

def _html_wrap(title: str, subtitle: str, meta_cards: str, body: str,
               chart_js: str, footer_text: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — Studentprognose</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
<header>
    <h1>{title}</h1>
    <p>{subtitle}</p>
</header>
<div class="meta-cards">{meta_cards}</div>
{body}
{_METRICS_SECTION}
<footer>{footer_text}</footer>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<script>
{_TAB_JS}
{COLORS_JS}
{chart_js}
</script>
</body>
</html>"""


def _meta_card(number: str, label: str) -> str:
    return f"""<div class="meta-card">
        <div class="number">{number}</div>
        <div class="label">{label}</div>
    </div>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_cumulative_report(
    ts_df: pd.DataFrame,
    reg_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Generate benchmark_cumulative.html with Stap 1 (timeseries) + Stap 2 (regressor)."""
    n_prog = ts_df["programme"].nunique()
    n_bach = ts_df[ts_df["examentype"] == "Bachelor"]["programme"].nunique()
    n_master = ts_df[ts_df["examentype"] == "Master"]["programme"].nunique()
    ts_years = sorted(ts_df["test_year"].unique())
    reg_years = sorted(reg_df["test_year"].unique())

    meta = (
        _meta_card(str(n_prog), f"Opleidingen ({n_bach} Ba + {n_master} Ma)")
        + _meta_card(
            f"{len(_ts_models(ts_df))} + {len(_reg_models(reg_df))}",
            "Modellen (tijdreeks + regressie)",
        )
        + _meta_card(str(len(ts_years)), f"Test-jaren ({ts_years[0]}–{ts_years[-1]})")
    )

    body = _ts_section(ts_df, "ts", "Stap 1") + _reg_section(reg_df, "reg", "Stap 2")
    chart_js = _ts_chart_js(ts_df, "ts") + _reg_chart_js(reg_df, "reg")
    footer = (
        f"Gegenereerd door studentprognose benchmark &middot; "
        f"Data: {len(ts_df)} tijdreeksmetingen, {len(reg_df)} regressiemetingen"
    )

    html = _html_wrap(
        "Benchmark — Cumulatief spoor",
        "Modelvergelijking Stap 1 (tijdreeks) en Stap 2 (regressie) voor het cumulatieve spoor",
        meta, body, chart_js, footer,
    )
    _write(os.path.join(output_dir, "benchmark", "benchmark_cumulative.html"), html)


def generate_individual_report(
    clf_df: pd.DataFrame,
    output_dir: str,
    roc_curves: dict[str, list[dict]] | None = None,
) -> None:
    """Generate benchmark_individual.html with Stap 1 (classifier) + ROC curves."""
    clf_years = sorted(clf_df["test_year"].unique())

    meta = (
        _meta_card(str(len(_clf_models(clf_df))), "Classificatiemodellen")
        + _meta_card(str(len(clf_years)), f"Test-jaren ({clf_years[0]}–{clf_years[-1]})")
        + _meta_card(
            str(int(clf_df["n_test"].mean())),
            "Gem. testgrootte (aanmelders)",
        )
    )

    body = _clf_section(clf_df)
    if roc_curves:
        body += _roc_section(roc_curves)

    chart_js = _clf_chart_js(clf_df)
    if roc_curves:
        chart_js += _roc_chart_js(roc_curves)

    footer = (
        f"Gegenereerd door studentprognose benchmark &middot; "
        f"Data: {len(clf_df)} classificatiemetingen"
    )

    html = _html_wrap(
        "Benchmark — Individueel spoor",
        "Modelvergelijking Stap 1 (classificatie) voor het individuele spoor",
        meta, body, chart_js, footer,
    )
    _write(os.path.join(output_dir, "benchmark", "benchmark_individual.html"), html)
