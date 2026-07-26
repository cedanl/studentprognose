"""Concept: Verklaarbaar AI — waarom voorspelt het model wat het voorspelt?

Interactieve mockup. Alle data is illustratief.
"""

from __future__ import annotations

from nicegui import ui

from gui import nav, theme
from gui.components.layout import page_shell
from gui.components.states import concept_banner, section_title

_ACCENT = theme.ACCENT
_GREEN = theme.POSITIVE
_WARNING = theme.WARNING
_RED = theme.NEGATIVE
_MUTED = theme.MUTED
_INFO = theme.INFO

_PROGRAMMES = [
    "Technische Informatica (Bachelor · WO)",
    "Bedrijfskunde (Bachelor · WO)",
    "Psychologie (Bachelor · WO)",
    "Werktuigbouwkunde (Bachelor · WO)",
]

# Mock uitlegdata per opleiding
_EXPLAIN: dict[str, dict] = {
    _PROGRAMMES[0]: {
        "forecast": 312,
        "ci_low": 287,
        "ci_high": 338,
        "confidence": "hoog",
        "confidence_pct": 85,
        "factors": [
            {
                "direction": "up",
                "delta": 34,
                "title": "Aanmeldingen 12% hoger dan vorig jaar",
                "body": "In week 14 liggen de cumulatieve aanmeldingen 12% boven hetzelfde "
                        "moment in 2024. Historisch correleert dit sterk met eindinstroom "
                        "(correlatie r=0.91 over 5 jaar).",
                "weight": 0.41,
            },
            {
                "direction": "up",
                "delta": 8,
                "title": "Hoger EER-aandeel (23% vs historisch 18%)",
                "body": "EER-studenten schrijven zich definitief in met een conversieratio "
                        "van 68% vs 54% voor NL. Het gestegen aandeel verhoogt de "
                        "verwachte eindinstroom.",
                "weight": 0.18,
            },
            {
                "direction": "down",
                "delta": -22,
                "title": "Hogere uitval herinschrijvingsfase",
                "body": "De uitval in de herinschrijvingsfase ligt 8% boven het 3-jaarsgemiddelde. "
                        "Mogelijk seizoenseffect — bij 3 van 5 vergelijkbare jaren herstelde "
                        "dit in week 16–18.",
                "weight": 0.28,
            },
            {
                "direction": "neutral",
                "delta": 0,
                "title": "Geen open dag effect (week 12)",
                "body": "De open dag in week 12 heeft statistisch geen significante invloed "
                        "op de week-14-prognose (p=0.34). Effecten van open dagen zijn "
                        "meetbaar pas na 3–4 weken.",
                "weight": 0.13,
            },
        ],
        "historical": [
            (2024, 298, 301, 1.0),
            (2023, 315, 308, -2.2),
            (2022, 287, 292, 1.7),
            (2021, 274, 271, -1.1),
            (2020, 261, 265, 1.5),
        ],
        "similar_year": 2022,
        "similar_note": "2022 had vergelijkbaar aanmeldpatroon in week 14 (afwijking <3%). Eindinstroom dat jaar: 292.",
    },
    _PROGRAMMES[1]: {
        "forecast": 487,
        "ci_low": 451,
        "ci_high": 524,
        "confidence": "hoog",
        "confidence_pct": 88,
        "factors": [
            {"direction": "up", "delta": 52, "title": "Sterke groei NL-aanmeldingen (+18%)", "body": "Hogere instroom vanuit VWO. Mogelijke driver: recent mediamoment.", "weight": 0.45},
            {"direction": "down", "delta": -18, "title": "EER-aanmeldingen -6% t.o.v. 2024", "body": "Lichte daling EER-aanmeldingen, mogelijke Brexit-naschok in NL-aantrekkelijkheid.", "weight": 0.25},
            {"direction": "up", "delta": 12, "title": "Hogere conversie herinschrijvingen", "body": "Herinschrijvingsconversie 4% boven historisch gemiddelde.", "weight": 0.30},
        ],
        "historical": [
            (2024, 471, 475, 0.8),
            (2023, 452, 448, -0.9),
            (2022, 438, 443, 1.1),
        ],
        "similar_year": 2024,
        "similar_note": "2024 had vergelijkbaar patroon. Eindinstroom: 475.",
    },
    _PROGRAMMES[2]: {
        "forecast": 643,
        "ci_low": 588,
        "ci_high": 698,
        "confidence": "matig",
        "confidence_pct": 62,
        "factors": [
            {"direction": "down", "delta": -58, "title": "Aanmeldingen significant lager dan 2024 (-14%)", "body": "Groot afwijkend signaal — mogelijke oorzaak: nieuw concurrerend programma geopend bij naburige instelling.", "weight": 0.52},
            {"direction": "up", "delta": 28, "title": "Hoge conversieratio historisch voor deze opleiding", "body": "Psychologie heeft structureel hoge conversie (72%). Dempend effect op negatieve aanmeldgroei.", "weight": 0.33},
            {"direction": "neutral", "delta": 0, "title": "Onzekerheid hoog — beperkte historische data", "body": "Het model heeft slechts 3 volledige jaarcycli als trainingsdata. Prognose is minder betrouwbaar.", "weight": 0.15},
        ],
        "historical": [
            (2024, 671, 668, -0.4),
            (2023, 648, 655, 1.1),
            (2022, 612, 609, -0.5),
        ],
        "similar_year": None,
        "similar_note": "Geen sterk vergelijkbaar historisch jaar gevonden. Interpreteer met voorzichtigheid.",
    },
    _PROGRAMMES[3]: {
        "forecast": 201,
        "ci_low": 186,
        "ci_high": 217,
        "confidence": "hoog",
        "confidence_pct": 91,
        "factors": [
            {"direction": "up", "delta": 18, "title": "Aanmeldingen 9% boven historisch gemiddelde", "body": "Stabiele stijging, consistent met meerjaarse trend in technische bacheloropleidingen.", "weight": 0.48},
            {"direction": "down", "delta": -8, "title": "Lichte daling Niet-EER aanmeldingen", "body": "Niet-EER daalt 4% t.o.v. 2024. Impact beperkt door lage historische conversieratio (41%).", "weight": 0.22},
            {"direction": "up", "delta": 4, "title": "Positief open dag effect (week 11)", "body": "Significante aanmeldpiek na open dag in week 11 (p=0.02). Effect houdt ~3 weken aan.", "weight": 0.30},
        ],
        "historical": [
            (2024, 194, 196, 1.0),
            (2023, 187, 183, -2.1),
            (2022, 178, 181, 1.7),
        ],
        "similar_year": 2024,
        "similar_note": "2024 vergelijkbaar patroon. Eindinstroom: 196.",
    },
}


def _confidence_gauge_html(pct: int, label: str) -> str:
    color = _GREEN if pct >= 80 else (_WARNING if pct >= 60 else _RED)
    circumference = 2 * 3.14159 * 36
    dash = circumference * pct / 100
    return (
        f'<div style="display:flex;flex-direction:column;align-items:center;gap:4px;">'
        f'<svg width="90" height="90" viewBox="0 0 90 90">'
        f'<circle cx="45" cy="45" r="36" fill="none" stroke="#f0f0f0" stroke-width="8"/>'
        f'<circle cx="45" cy="45" r="36" fill="none" stroke="{color}" stroke-width="8" '
        f'stroke-dasharray="{dash:.1f} {circumference:.1f}" '
        f'stroke-dashoffset="{circumference / 4:.1f}" stroke-linecap="round"/>'
        f'<text x="45" y="41" text-anchor="middle" font-size="16" font-weight="700" fill="{color}">{pct}%</text>'
        f'<text x="45" y="56" text-anchor="middle" font-size="9" fill="#999">zekerheid</text>'
        f'</svg>'
        f'<div style="font-size:11px;color:{color};font-weight:600;">{label}</div>'
        f'</div>'
    )


def _factor_bar_html(weight: float, color: str) -> str:
    w = weight * 100
    return (
        f'<div style="width:100%;background:#f0f0f0;border-radius:100px;height:5px;margin-top:4px;">'
        f'<div style="width:{w:.0f}%;background:{color};height:100%;border-radius:100px;"></div>'
        f'</div>'
    )


def _history_html(rows: list[tuple]) -> str:
    header = (
        f'<tr style="background:#fafafa;">'
        f'<th style="padding:6px 10px;text-align:left;font-size:11px;color:#999;font-weight:500;">Jaar</th>'
        f'<th style="padding:6px 10px;text-align:right;font-size:11px;color:#999;font-weight:500;">Model (w14)</th>'
        f'<th style="padding:6px 10px;text-align:right;font-size:11px;color:#999;font-weight:500;">Werkelijk</th>'
        f'<th style="padding:6px 10px;text-align:right;font-size:11px;color:#999;font-weight:500;">Afwijking</th>'
        f'</tr>'
    )
    body = ""
    for year, model, actual, pct in rows:
        err_color = _GREEN if abs(pct) <= 2 else (_WARNING if abs(pct) <= 5 else _RED)
        sign = "+" if pct > 0 else ""
        body += (
            f'<tr style="border-top:1px solid #f0f0f0;">'
            f'<td style="padding:7px 10px;font-size:12px;">{year}</td>'
            f'<td style="padding:7px 10px;text-align:right;font-size:12px;">{model}</td>'
            f'<td style="padding:7px 10px;text-align:right;font-size:12px;font-weight:600;">{actual}</td>'
            f'<td style="padding:7px 10px;text-align:right;font-size:12px;color:{err_color};font-weight:500;">{sign}{pct:.1f}%</td>'
            f'</tr>'
        )
    return (
        f'<div style="border:1px solid #efefef;border-radius:8px;overflow:hidden;">'
        f'<table style="width:100%;border-collapse:collapse;">{header}{body}</table>'
        f'</div>'
    )


def create() -> None:
    nav.register_route("/explainability")

    @ui.page("/explainability")
    def explainability_page() -> None:
        with page_shell(active="/explainability", title="Verklaarbaar AI", show_stepper=False):
            concept_banner()
            section_title(
                "Verklaarbaar AI",
                "Begrijp waarom het model de prognose geeft die het geeft — in begrijpelijke taal.",
            )
            _ExplainView()


class _ExplainView:
    def __init__(self) -> None:
        self._programme = _PROGRAMMES[0]
        self._build()

    def _build(self) -> None:
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-3 no-wrap w-full"):
                ui.icon("psychology").classes("text-2xl flex-none").style(f"color:{_ACCENT}")
                ui.select(
                    _PROGRAMMES,
                    value=self._programme,
                    label="Opleiding",
                    on_change=lambda e: self._on_programme(e.value),
                ).props("outlined dense").classes("flex-1")
                ui.label("Week 14 · 2025").classes("text-xs opacity-50 flex-none")

        self._content = ui.column().classes("w-full gap-4")
        self._render()

    def _on_programme(self, value: str) -> None:
        self._programme = value
        self._content.clear()
        with self._content:
            self._render_inner()

    def _render(self) -> None:
        with self._content:
            self._render_inner()

    def _render_inner(self) -> None:
        d = _EXPLAIN[self._programme]
        forecast = d["forecast"]

        # ── Kernvraag + zekerheid ────────────────────────────────────────────
        with ui.card().classes("w-full").style(f"border-left:4px solid {_ACCENT};"):
            with ui.row().classes("items-center gap-6 no-wrap"):
                with ui.column().classes("gap-1 flex-1"):
                    ui.label(
                        f"Waarom {forecast} studenten?"
                    ).classes("text-2xl font-bold")
                    ui.label(
                        f"Betrouwbaarheidsinterval: {d['ci_low']} – {d['ci_high']} studenten  "
                        f"·  Zekerheid: {d['confidence']}"
                    ).classes("text-sm opacity-60")
                    if d["similar_year"]:
                        with ui.row().classes("items-center gap-1 mt-1"):
                            ui.icon("history").classes("text-sm").style(f"color:{_INFO}")
                            ui.label(d["similar_note"]).classes("text-xs").style(
                                f"color:{_INFO}"
                            )
                    else:
                        with ui.row().classes("items-center gap-1 mt-1"):
                            ui.icon("warning").classes("text-sm").style(f"color:{_WARNING}")
                            ui.label(d["similar_note"]).classes("text-xs").style(
                                f"color:{_WARNING}"
                            )
                ui.html(_confidence_gauge_html(d["confidence_pct"], d["confidence"]))

        # ── Factoren ────────────────────────────────────────────────────────
        ui.label("Wat drijft deze prognose?").classes("font-medium")
        for f in d["factors"]:
            direction = f["direction"]
            delta = f["delta"]
            color = _GREEN if direction == "up" else (_RED if direction == "down" else _MUTED)
            icon = "arrow_upward" if direction == "up" else ("arrow_downward" if direction == "down" else "remove")
            sign = "+" if delta > 0 else ""

            with ui.card().classes("w-full").style(f"border-left:3px solid {color};"):
                with ui.row().classes("items-start gap-3 no-wrap"):
                    with ui.element("div").style(
                        f"width:36px;height:36px;border-radius:50%;"
                        f"background:{color}18;display:flex;align-items:center;"
                        f"justify-content:center;flex-shrink:0;"
                    ):
                        ui.icon(icon).style(f"color:{color}").classes("text-lg")
                    with ui.column().classes("gap-1 flex-1"):
                        with ui.row().classes("items-center gap-2 no-wrap"):
                            ui.label(f["title"]).classes("font-medium text-sm flex-1")
                            if delta != 0:
                                ui.label(f"{sign}{delta} studenten").classes(
                                    "text-sm font-bold flex-none"
                                ).style(f"color:{color};")
                        ui.label(f["body"]).classes("text-xs opacity-70 leading-relaxed")
                        with ui.row().classes("items-center gap-2 no-wrap w-full"):
                            ui.label(f"Gewicht: {f['weight']:.0%}").classes(
                                "text-xs opacity-40 flex-none"
                            )
                            ui.html(_factor_bar_html(f["weight"], color)).style("flex:1;")

        # ── Historische validatie ────────────────────────────────────────────
        ui.label("Historische nauwkeurigheid").classes("font-medium mt-1")
        ui.label(
            "Hoe goed heeft het model het gedaan op dit prognose-moment in eerdere jaren?"
        ).classes("text-xs opacity-50 -mt-2")
        ui.html(_history_html(d["historical"]))

        # ── Exporteer uitleg ─────────────────────────────────────────────────
        with ui.row().classes("gap-2 mt-1"):
            ui.button("Exporteer uitleg (PDF)", icon="picture_as_pdf").props(
                "outline color=accent"
            )
            ui.button("Kopieer samenvatting", icon="content_copy").props("flat color=grey-7")
            ui.button("Audit trail bekijken", icon="history").props("flat color=grey-7")
