"""Concept: API & Integraties — REST API documentatie en connector-overzicht.

Interactieve mockup. Alle data is illustratief.
"""

from __future__ import annotations

from nicegui import ui

from gui import nav, theme
from gui.components.layout import page_shell
from gui.components.states import concept_banner, section_title

_ACCENT = theme.ACCENT
_GREEN = theme.POSITIVE
_MUTED = theme.MUTED

_BASE_URL = "https://api.studentprognose.nl/v1"

_ENDPOINTS = [
    {
        "method": "GET",
        "path": "/forecast",
        "summary": "Haal de huidige prognose op voor één opleiding en week.",
        "params": [
            ("croho", "string", "CROHO-code (bijv. 34397)", True),
            ("week",  "integer", "ISO-weeknummer (1–53)", True),
            ("year",  "integer", "Studiejaar (bijv. 2025)", True),
            ("model", "string", "ensemble | sarima | xgboost (default: ensemble)", False),
        ],
        "response": '''{
  "programme": "Technische Informatica",
  "croho": "34397",
  "week": 14,
  "year": 2025,
  "forecast": 312,
  "confidence_interval": [287, 338],
  "model": "ensemble",
  "generated_at": "2025-04-07T08:00:00Z"
}''',
    },
    {
        "method": "GET",
        "path": "/actuals",
        "summary": "Haal historische werkelijke inschrijvingen op.",
        "params": [
            ("croho", "string", "CROHO-code", True),
            ("from",  "string", "Startweek ISO (bijv. 2025-W01)", True),
            ("to",    "string", "Eindweek ISO (bijv. 2025-W38)", True),
        ],
        "response": '''{
  "programme": "Technische Informatica",
  "croho": "34397",
  "actuals": [
    {"week": 1, "year": 2025, "count": 12},
    {"week": 2, "year": 2025, "count": 28},
    ...
  ]
}''',
    },
    {
        "method": "GET",
        "path": "/benchmark",
        "summary": "Vergelijk met geanonimiseerde peers (minimaal 5 instellingen).",
        "params": [
            ("croho", "string", "CROHO-code", True),
            ("week",  "integer", "ISO-weeknummer", True),
            ("year",  "integer", "Studiejaar", True),
        ],
        "response": '''{
  "croho": "34397",
  "week": 14,
  "your_pct": 87,
  "median_pct": 94,
  "top25_pct": 112,
  "n_peers": 7,
  "trend": "stabiel"
}''',
    },
    {
        "method": "POST",
        "path": "/scenarios",
        "summary": "Sla een scenario op en bereken de impact.",
        "params": [
            ("croho",        "string",  "CROHO-code", True),
            ("base_week",    "integer", "Referentieweek", True),
            ("adjustments",  "object",  "Parameteraanpassingen (zie schema)", True),
        ],
        "response": '''{
  "scenario_id": "sc_a3f8b2",
  "name": "LinkedIn campagne",
  "baseline": 312,
  "forecast": 330,
  "delta": 18,
  "delta_pct": 5.8,
  "share_url": "https://app.studentprognose.nl/s/sc_a3f8b2"
}''',
    },
]

_CODE_EXAMPLES = {
    "Python": """\
import requests

API_KEY = "sp_live_••••••••••••"
BASE    = "https://api.studentprognose.nl/v1"

resp = requests.get(
    f"{BASE}/forecast",
    params={"croho": "34397", "week": 14, "year": 2025},
    headers={"Authorization": f"Bearer {API_KEY}"},
)
data = resp.json()
print(data["forecast"])  # → 312""",

    "R": """\
library(httr2)

api_key <- "sp_live_••••••••••••"
base    <- "https://api.studentprognose.nl/v1"

resp <- request(paste0(base, "/forecast")) |>
  req_headers(Authorization = paste("Bearer", api_key)) |>
  req_url_query(croho = "34397", week = 14, year = 2025) |>
  req_perform()

data <- resp_body_json(resp)
cat(data$forecast)  # → 312""",

    "Power BI (M)": """\
let
    ApiKey  = "sp_live_••••••••••••",
    BaseUrl = "https://api.studentprognose.nl/v1/forecast",
    Params  = "?croho=34397&week=14&year=2025",
    Source  = Json.Document(
        Web.Contents(BaseUrl & Params, [
            Headers = [Authorization = "Bearer " & ApiKey]
        ])
    )
in
    Source""",
}

_CONNECTORS = [
    ("Microsoft Power BI", "power_bi", "Officiële certified connector — download via AppSource.", True),
    ("Tableau", "table_chart", "Web Data Connector beschikbaar via Tableau Exchange.", True),
    ("Qlik Sense", "bar_chart", "REST connector via Qlik Web Connectors.", False),
    ("Python (pandas)", "code", "Pip-installeerbaar: pip install studentprognose-sdk", True),
    ("R (tidyverse)", "code", "CRAN-package: install.packages('studentprognose')", False),
]


def _code_block(code: str) -> str:
    escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<pre style="background:#1e1e2e;color:#cdd6f4;border-radius:8px;'
        f'padding:16px;font-size:12px;line-height:1.6;overflow-x:auto;margin:0;">'
        f'<code>{escaped}</code></pre>'
    )


def _json_block(code: str) -> str:
    escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<pre style="background:#f6f8fa;color:#24292f;border-radius:6px;'
        f'padding:12px;font-size:11px;line-height:1.5;overflow-x:auto;margin:0;">'
        f'<code>{escaped}</code></pre>'
    )


def create() -> None:
    nav.register_route("/api")

    @ui.page("/api")
    def api_page() -> None:
        with page_shell(active="/api", title="API & integraties", show_stepper=False):
            concept_banner()
            section_title(
                "API & integraties",
                f"Embed prognoses in Power BI, Tableau of je eigen systemen via REST.",
            )
            _ApiView()


class _ApiView:
    def __init__(self) -> None:
        self._build()

    def _build(self) -> None:
        # ── API-sleutel ─────────────────────────────────────────────────────
        with ui.card().classes("w-full").style(f"border-left:4px solid {_ACCENT};"):
            with ui.row().classes("items-center gap-3 no-wrap w-full"):
                ui.icon("vpn_key").classes("text-2xl flex-none").style(f"color:{_ACCENT}")
                with ui.column().classes("gap-0 flex-1"):
                    ui.label("Jouw API-sleutel").classes("font-medium")
                    ui.label("sp_live_••••••••••••••••••••••••••••••").classes(
                        "text-sm font-mono opacity-60"
                    )
                ui.button("Kopieer", icon="content_copy").props("outline dense color=accent")
                ui.button("Vernieuwen", icon="refresh").props("flat dense color=grey-7")
            ui.label(
                f"Basis-URL: {_BASE_URL}  ·  Authenticatie: Bearer token  ·  Rate limit: 1000 req/uur"
            ).classes("text-xs opacity-50 mt-1 font-mono")

        # ── Endpoints ────────────────────────────────────────────────────────
        ui.label("Endpoints").classes("font-medium mt-2")
        for ep in _ENDPOINTS:
            method_color = {"GET": "#2196F3", "POST": _GREEN, "DELETE": theme.NEGATIVE}.get(
                ep["method"], _MUTED
            )
            with ui.expansion(
                f'{ep["method"]}  {ep["path"]}',
            ).classes("w-full").style(
                f"border:1px solid #e8e8e8;border-radius:8px;margin-bottom:8px;"
            ):
                ui.label(ep["summary"]).classes("text-sm opacity-70 mb-3")

                # Parameters
                ui.label("Parameters").classes("text-xs font-semibold uppercase opacity-50 mb-1")
                with ui.element("div").style(
                    "border:1px solid #efefef;border-radius:6px;overflow:hidden;"
                ):
                    for i, (name, type_, desc, required) in enumerate(ep["params"]):
                        bg = "#fafafa" if i % 2 == 0 else "#fff"
                        with ui.row().classes("items-start gap-3 px-3 py-2 no-wrap").style(
                            f"background:{bg};"
                        ):
                            ui.label(name).classes("text-xs font-mono font-semibold w-28 flex-none")
                            ui.badge(type_).props("color=grey-3 text-color=grey-8 dense outline").classes("flex-none mt-0.5")
                            if required:
                                ui.badge("vereist").props("color=orange-2 text-color=orange-9 dense outline").classes("flex-none mt-0.5")
                            ui.label(desc).classes("text-xs opacity-60 flex-1")

                # Response
                ui.label("Voorbeeldrespons").classes(
                    "text-xs font-semibold uppercase opacity-50 mt-3 mb-1"
                )
                ui.html(_json_block(ep["response"]))

                # Curl
                params_str = "&".join(
                    f"{p[0]}=..." for p in ep["params"] if p[3]
                )
                curl = (
                    f'curl -H "Authorization: Bearer sp_live_••••" \\\n'
                    f'  "{_BASE_URL}{ep["path"]}?{params_str}"'
                )
                ui.label("cURL").classes("text-xs font-semibold uppercase opacity-50 mt-2 mb-1")
                ui.html(_code_block(curl))

        # ── Code-voorbeelden ─────────────────────────────────────────────────
        ui.label("Code-voorbeelden").classes("font-medium mt-2")
        with ui.tabs() as tabs:
            for lang in _CODE_EXAMPLES:
                ui.tab(lang)
        with ui.tab_panels(tabs, value=list(_CODE_EXAMPLES.keys())[0]).classes("w-full"):
            for lang, code in _CODE_EXAMPLES.items():
                with ui.tab_panel(lang):
                    ui.html(_code_block(code))

        # ── Connectoren ──────────────────────────────────────────────────────
        ui.label("Certified connectoren").classes("font-medium mt-2")
        with ui.row().classes("w-full gap-3 flex-wrap"):
            for name, icon, desc, available in _CONNECTORS:
                color = _ACCENT if available else "#bbb"
                with ui.card().classes("flex-none").style(
                    f"width:200px;border-top:3px solid {color};"
                ):
                    with ui.row().classes("items-center gap-2 mb-1"):
                        ui.icon(icon).style(f"color:{color}").classes("text-xl")
                        ui.label(name).classes("text-sm font-medium flex-1")
                        if available:
                            ui.icon("check_circle").style(f"color:{_GREEN}").classes("text-sm")
                    ui.label(desc).classes("text-xs opacity-60 leading-snug")
                    ui.button(
                        "Downloaden" if available else "Binnenkort",
                        icon="download" if available else "schedule",
                    ).props(
                        f"flat dense {'color=accent' if available else 'color=grey-5 disable'}"
                    ).classes("mt-1 -ml-2")
