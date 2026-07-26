"""Concept: Peer Benchmark — geanonimiseerde vergelijking met vergelijkbare instellingen.

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

_PROGRAMMES = [
    "Technische Informatica (Bachelor · WO)",
    "Bedrijfskunde (Bachelor · WO)",
    "Psychologie (Bachelor · WO)",
    "Werktuigbouwkunde (Bachelor · WO)",
]

# Mock data: (jij_pct, mediaan_pct, top25_pct, n_peers, trend_richting)
_MOCK: dict[str, tuple] = {
    _PROGRAMMES[0]: (87, 94, 112, 7, "stabiel"),
    _PROGRAMMES[1]: (103, 98, 118, 11, "stijgend"),
    _PROGRAMMES[2]: (79, 91, 109, 9, "dalend"),
    _PROGRAMMES[3]: (95, 93, 115, 6, "stabiel"),
}

# Week-over-week mock (week 8 t/m 14)
_WEEKS = [8, 9, 10, 11, 12, 13, 14]
_WEEK_DATA: dict[str, dict] = {
    _PROGRAMMES[0]: {
        "jij":    [62, 68, 73, 77, 81, 84, 87],
        "mediaan":[65, 71, 76, 80, 85, 90, 94],
        "top25":  [78, 85, 90, 95, 101, 107, 112],
    },
    _PROGRAMMES[1]: {
        "jij":    [88, 91, 94, 97, 100, 101, 103],
        "mediaan":[84, 87, 90, 93, 95, 97, 98],
        "top25":  [100, 103, 107, 110, 113, 116, 118],
    },
    _PROGRAMMES[2]: {
        "jij":    [72, 74, 75, 76, 77, 78, 79],
        "mediaan":[80, 82, 84, 86, 88, 89, 91],
        "top25":  [96, 98, 101, 103, 105, 107, 109],
    },
    _PROGRAMMES[3]: {
        "jij":    [85, 87, 89, 91, 92, 94, 95],
        "mediaan":[82, 84, 87, 89, 90, 92, 93],
        "top25":  [100, 103, 106, 109, 111, 113, 115],
    },
}

_INSIGHTS: dict[str, list[tuple[str, str, str]]] = {
    _PROGRAMMES[0]: [
        ("warning", "Jij zit 7% onder mediaan in week 14.",
         "Bij 4 van 7 peers is dit patroon hersteld via extra open dagen in week 16–18."),
        ("info", "EER-aandeel jij: 23% vs mediaan 18%.",
         "EER-studenten schrijven zich historisch vaker definitief in — positief signaal."),
        ("check_circle", "NL-conversie vergelijkbaar met peers.",
         "Geen actie nodig op NL-instroom."),
    ],
    _PROGRAMMES[1]: [
        ("check_circle", "Jij zit 5% boven mediaan — uitstekend.",
         "Huidige tempo volhouden. Open dag bereik is waarschijnlijk de driver."),
        ("info", "Top 25% zit 15% boven jou.",
         "Verschil verklaard door vroege matching-activiteiten (week 4–6)."),
    ],
    _PROGRAMMES[2]: [
        ("error", "Jij zit 12% onder mediaan — actie aanbevolen.",
         "Historisch patroon: achterstand in week 14 wordt zelden hersteld zonder interventie."),
        ("warning", "Dalende trend afgelopen 3 weken.",
         "Controleer of er een externe factor speelt (concurrentieopleiding, media-aandacht)."),
    ],
    _PROGRAMMES[3]: [
        ("check_circle", "Jij loopt vrijwel gelijk met mediaan.", ""),
        ("info", "Top 25% loopt 20% voor — mogelijk door vroegere deadlines.",
         "Overweeg aanmelddeadline 2 weken naar voren te verschuiven."),
    ],
}


def _gauge_html(jij: int, mediaan: int, top25: int) -> str:
    def bar(label: str, pct: int, color: str, bold: bool = False) -> str:
        w = min(pct, 130) / 130 * 100
        fw = "700" if bold else "400"
        return (
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
            f'<div style="width:120px;font-size:12px;color:#555;font-weight:{fw};flex-shrink:0;">{label}</div>'
            f'<div style="flex:1;background:#f0f0f0;border-radius:100px;height:10px;position:relative;">'
            f'<div style="width:{w:.1f}%;height:100%;background:{color};border-radius:100px;"></div>'
            f'</div>'
            f'<div style="width:48px;text-align:right;font-size:13px;font-weight:{fw};color:{color};">{pct}%</div>'
            f'</div>'
        )

    jij_color = _GREEN if jij >= mediaan else (_WARNING if jij >= mediaan * 0.9 else _RED)

    return (
        f'<div style="background:#fafafa;border:1px solid #efefef;'
        f'border-radius:10px;padding:16px 20px;">'
        f'<div style="font-size:11px;color:#999;margin-bottom:12px;font-weight:500;">'
        f'% van historisch doel — week 14 · 2025</div>'
        + bar("Jouw instelling", jij, jij_color, bold=True)
        + bar("Mediaan 7 peers", mediaan, "#9e9e9e")
        + bar("Top 25%", top25, _ACCENT)
        + f'</div>'
    )


def _sparkline_html(weeks: list[int], data: dict) -> str:
    w, h = 520, 110
    margin = 20
    x_step = (w - margin * 2) / (len(weeks) - 1)
    y_min, y_max = 50, 130

    def y(val: float) -> float:
        return h - margin - (val - y_min) / (y_max - y_min) * (h - margin * 2)

    def polyline(vals: list[int], color: str, dash: str = "") -> str:
        pts = " ".join(
            f"{margin + i * x_step:.1f},{y(v):.1f}" for i, v in enumerate(vals)
        )
        dash_attr = f'stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linecap="round" {dash_attr}/>'
        )

    lines = (
        polyline(data["top25"], _ACCENT, "5,3")
        + polyline(data["mediaan"], "#bbb")
        + polyline(data["jij"], _GREEN if data["jij"][-1] >= data["mediaan"][-1] else _RED)
    )
    x_labels = "".join(
        f'<text x="{margin + i * x_step:.1f}" y="{h - 2}" '
        f'text-anchor="middle" font-size="9" fill="#bbb">W{w_}</text>'
        for i, w_ in enumerate(weeks)
    )
    legend = (
        f'<line x1="0" y1="0" x2="18" y2="0" stroke="{_GREEN if data["jij"][-1] >= data["mediaan"][-1] else _RED}" stroke-width="2"/>'
        f'<text x="22" y="4" font-size="9" fill="#666">Jouw instelling</text>'
        f'<line x1="110" y1="0" x2="128" y2="0" stroke="#bbb" stroke-width="2"/>'
        f'<text x="132" y="4" font-size="9" fill="#666">Mediaan peers</text>'
        f'<line x1="220" y1="0" x2="238" y2="0" stroke="{_ACCENT}" stroke-width="2" stroke-dasharray="5,3"/>'
        f'<text x="242" y="4" font-size="9" fill="#666">Top 25%</text>'
    )

    return (
        f'<div style="background:#fafafa;border:1px solid #efefef;'
        f'border-radius:10px;padding:16px;">'
        f'<div style="font-size:11px;color:#999;margin-bottom:8px;font-weight:500;">'
        f'% van historisch doel — week 8 t/m 14</div>'
        f'<svg width="100%" viewBox="0 0 {w} {h}" style="overflow:visible;">'
        f'{lines}{x_labels}'
        f'<g transform="translate(10, 8)">{legend}</g>'
        f'</svg>'
        f'<div style="font-size:10px;color:#bbb;margin-top:6px;">'
        f'7 geanonimiseerde vergelijkbare WO-instellingen · zelfde CROHO-cluster · '
        f'>200 aanmeldingen/jr · k-anonimiteit gegarandeerd</div>'
        f'</div>'
    )


def create() -> None:
    nav.register_route("/peer-benchmark")

    @ui.page("/peer-benchmark")
    def peer_benchmark_page() -> None:
        with page_shell(active="/peer-benchmark", title="Peer benchmark", show_stepper=False):
            concept_banner()
            section_title(
                "Peer benchmark",
                "Vergelijk jouw instroom met geanonimiseerde vergelijkbare instellingen.",
            )
            _PeerBenchmarkView()


class _PeerBenchmarkView:
    def __init__(self) -> None:
        self._programme = _PROGRAMMES[0]
        self._build()

    def _build(self) -> None:
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-3 no-wrap w-full"):
                ui.icon("leaderboard").classes("text-2xl flex-none").style(
                    f"color:{_ACCENT}"
                )
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
        p = self._programme
        jij, mediaan, top25, n_peers, trend = _MOCK[p]
        jij_color = _GREEN if jij >= mediaan else (_WARNING if jij >= mediaan * 0.9 else _RED)
        trend_icon = {"stijgend": "trending_up", "dalend": "trending_down", "stabiel": "trending_flat"}[trend]
        trend_color = {"stijgend": _GREEN, "dalend": _RED, "stabiel": _MUTED}[trend]

        # Toplijn: 3 stat-kaarten
        with ui.row().classes("w-full gap-3"):
            for label, val, color, suffix in [
                ("Jouw positie", f"{jij}%", jij_color, "van doel"),
                ("Mediaan peers", f"{mediaan}%", _MUTED, f"{n_peers} instellingen"),
                ("Top 25%", f"{top25}%", _ACCENT, "benchmark"),
            ]:
                with ui.card().classes("flex-1").style(f"border-top: 3px solid {color};"):
                    ui.label(label).classes("text-xs opacity-50 uppercase font-medium")
                    ui.label(val).classes("text-3xl font-bold").style(f"color:{color};")
                    ui.label(suffix).classes("text-xs opacity-50 mt-1")

        # Trend indicator
        with ui.row().classes("items-center gap-2"):
            ui.icon(trend_icon).style(f"color:{trend_color}").classes("text-xl")
            ui.label(f"Trend afgelopen 4 weken: {trend}").classes("text-sm").style(
                f"color:{trend_color}"
            )

        # Gauge
        ui.html(_gauge_html(jij, mediaan, top25))

        # Sparkline
        ui.html(_sparkline_html(_WEEKS, _WEEK_DATA[p]))

        # Inzichten
        ui.label("Inzichten & aanbevelingen").classes("font-medium mt-1")
        for icon, title, body in _INSIGHTS.get(p, []):
            color_map = {
                "check_circle": (_GREEN, "#f0faf5"),
                "warning": (_WARNING, "#fff8e1"),
                "error": (_RED, "#fff5f5"),
                "info": (theme.INFO, "#e8f4fb"),
            }
            c, bg = color_map.get(icon, (_MUTED, "#fafafa"))
            with ui.card().classes("w-full").style(f"background:{bg};border-left:3px solid {c};"):
                with ui.row().classes("items-start gap-2 no-wrap"):
                    ui.icon(icon).style(f"color:{c}").classes("text-lg flex-none mt-0.5")
                    with ui.column().classes("gap-0"):
                        ui.label(title).classes("text-sm font-medium")
                        if body:
                            ui.label(body).classes("text-xs opacity-70 leading-relaxed")
