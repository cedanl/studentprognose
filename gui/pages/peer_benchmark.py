"""Concept: Peer Benchmark — directe vergelijking met een gekozen instelling.

Stroom:
  1. Kies een vergelijkende instelling (bijv. Leiden Universiteit)
  2. Kies de opleiding om mee te vergelijken
  3. Bekijk de zij-aan-zij vergelijking

Interactieve mockup. Alle data is illustratief.
"""

from __future__ import annotations

from nicegui import ui

from gui import nav, theme
from gui.components.layout import page_shell
from gui.components.states import concept_banner, section_title

_ACCENT  = theme.ACCENT
_GREEN   = theme.POSITIVE
_WARNING = theme.WARNING
_RED     = theme.NEGATIVE
_INFO    = theme.INFO
_MUTED   = theme.MUTED

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

# "Eigen" opleidingen — de instelling die de tool gebruikt
_OWN_PROGRAMMES = [
    "Technische Informatica (Bachelor · WO)",
    "Bedrijfskunde (Bachelor · WO)",
    "Psychologie (Bachelor · WO)",
    "Werktuigbouwkunde (Bachelor · WO)",
]

# Weken 8 t/m 14 — huidig aanmeldseizoen (cumulatief %)
_WEEKS = [8, 9, 10, 11, 12, 13, 14]

_PEER_INSTELLINGEN = [
    "Leiden Universiteit",
    "VU Amsterdam",
    "Radboud Universiteit",
    "Universiteit Utrecht",
    "TU Delft",
    "Universiteit van Amsterdam",
]

# Vergelijkbare opleidingen per peer-instelling
_PEER_OPLEIDINGEN: dict[str, list[str]] = {
    "Leiden Universiteit": [
        "Informatica (Bachelor · WO)",
        "Economie en Bedrijfskunde (Bachelor · WO)",
        "Psychologie (Bachelor · WO)",
    ],
    "VU Amsterdam": [
        "Computer Science (Bachelor · WO)",
        "Bedrijfskunde (Bachelor · WO)",
        "Psychologie (Bachelor · WO)",
        "Werktuigbouwkunde (Bachelor · WO)",
    ],
    "Radboud Universiteit": [
        "Informatica (Bachelor · WO)",
        "Bedrijfskunde (Bachelor · WO)",
        "Psychologie (Bachelor · WO)",
        "Technische Wetenschappen (Bachelor · WO)",
    ],
    "Universiteit Utrecht": [
        "Informatica (Bachelor · WO)",
        "Economie (Bachelor · WO)",
        "Psychologie (Bachelor · WO)",
        "Mechanical Engineering (Bachelor · WO)",
    ],
    "TU Delft": [
        "Computer Science (Bachelor · WO)",
        "Mechanical Engineering (Bachelor · WO)",
        "Electrical Engineering (Bachelor · WO)",
    ],
    "Universiteit van Amsterdam": [
        "Informatica (Bachelor · WO)",
        "Bedrijfskunde (Bachelor · WO)",
        "Psychologie (Bachelor · WO)",
        "Werktuigbouwkunde (Bachelor · WO)",
    ],
}

# Eigen opleiding-data: wekelijkse % van historisch doel
_OWN_DATA: dict[str, dict] = {
    _OWN_PROGRAMMES[0]: {
        "pct_series": [62, 68, 73, 77, 81, 84, 87],
        "huidig_pct": 87,
        "prognose": 312,
        "vorig_jaar_pct": 96,
    },
    _OWN_PROGRAMMES[1]: {
        "pct_series": [88, 91, 94, 97, 100, 101, 103],
        "huidig_pct": 103,
        "prognose": 487,
        "vorig_jaar_pct": 101,
    },
    _OWN_PROGRAMMES[2]: {
        "pct_series": [72, 74, 75, 76, 77, 78, 79],
        "huidig_pct": 79,
        "prognose": 643,
        "vorig_jaar_pct": 91,
    },
    _OWN_PROGRAMMES[3]: {
        "pct_series": [85, 87, 89, 91, 92, 94, 95],
        "huidig_pct": 95,
        "prognose": 201,
        "vorig_jaar_pct": 93,
    },
}

# Peer-vergelijkingsdata: (instelling, opleiding) → dict
# Sleutels die niet bestaan krijgen een deterministische fallback via _peer_data()
_PEER_DATA_EXACT: dict[tuple[str, str], dict] = {
    ("Leiden Universiteit", "Informatica (Bachelor · WO)"): {
        "pct_series": [65, 71, 76, 80, 85, 90, 94],
        "huidig_pct": 94,
        "prognose": 287,
        "trend": "stijgend",
        "note": "Leiden groeide de afgelopen 4 weken sneller dan het sectorgemiddelde.",
    },
    ("Leiden Universiteit", "Economie en Bedrijfskunde (Bachelor · WO)"): {
        "pct_series": [84, 87, 90, 93, 95, 97, 98],
        "huidig_pct": 98,
        "prognose": 521,
        "trend": "stabiel",
        "note": "Vergelijkbaar tempo; Leiden heeft historisch een hogere eindconversie.",
    },
    ("Leiden Universiteit", "Psychologie (Bachelor · WO)"): {
        "pct_series": [80, 82, 84, 86, 88, 89, 91],
        "huidig_pct": 91,
        "prognose": 704,
        "trend": "stabiel",
        "note": "Leiden loopt 12 procentpunt voor — patroon consistent met 2023.",
    },
    ("VU Amsterdam", "Computer Science (Bachelor · WO)"): {
        "pct_series": [70, 75, 79, 83, 87, 90, 93],
        "huidig_pct": 93,
        "prognose": 241,
        "trend": "stijgend",
        "note": "VU laat sterke groei zien na een open dag in week 11.",
    },
    ("VU Amsterdam", "Bedrijfskunde (Bachelor · WO)"): {
        "pct_series": [90, 93, 95, 97, 99, 101, 102],
        "huidig_pct": 102,
        "prognose": 498,
        "trend": "stabiel",
        "note": "VU en uw instelling lopen nagenoeg gelijk; VU heeft iets hogere instroom.",
    },
    ("VU Amsterdam", "Psychologie (Bachelor · WO)"): {
        "pct_series": [75, 78, 81, 83, 85, 87, 89],
        "huidig_pct": 89,
        "prognose": 688,
        "trend": "stabiel",
        "note": "Vergelijkbaar aanmeldtempo; uw achterstand is kleiner dan in 2024.",
    },
    ("TU Delft", "Computer Science (Bachelor · WO)"): {
        "pct_series": [78, 83, 87, 91, 95, 99, 104],
        "huidig_pct": 104,
        "prognose": 389,
        "trend": "stijgend",
        "note": "TU Delft loopt significant voor — hogere naamsbekendheid in technisch segment.",
    },
    ("TU Delft", "Mechanical Engineering (Bachelor · WO)"): {
        "pct_series": [88, 91, 94, 96, 98, 100, 102],
        "huidig_pct": 102,
        "prognose": 312,
        "trend": "stabiel",
        "note": "TU Delft en uw instelling nagenoeg gelijk; kleine voorsprong TU Delft.",
    },
}


def _peer_data(instelling: str, opleiding: str) -> dict:
    key = (instelling, opleiding)
    if key in _PEER_DATA_EXACT:
        return _PEER_DATA_EXACT[key]
    # Deterministische fallback op basis van namen
    h = (hash(instelling + opleiding) % 30) + 70  # 70–99
    base = max(60, h - 5)
    series = [max(50, base - 20 + i * 4) for i in range(len(_WEEKS))]
    return {
        "pct_series": series,
        "huidig_pct": series[-1],
        "prognose": 200 + (hash(instelling) % 400),
        "trend": ["stabiel", "stijgend", "dalend"][hash(opleiding) % 3],
        "note": f"Vergelijkingsdata voor {instelling} — {opleiding} is illustratief.",
    }


# ---------------------------------------------------------------------------
# SVG-visualisaties
# ---------------------------------------------------------------------------

def _comparison_bars_html(own_pct: int, peer_pct: int, peer_label: str) -> str:
    own_color  = _GREEN if own_pct >= peer_pct else (_WARNING if own_pct >= peer_pct * 0.9 else _RED)
    peer_color = _ACCENT

    def bar(label: str, pct: int, color: str, bold: bool = False) -> str:
        w = min(pct, 130) / 130 * 100
        fw = "700" if bold else "500"
        return (
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
            f'<div style="width:140px;font-size:12px;color:#555;font-weight:{fw};'
            f'flex-shrink:0;line-height:1.3;">{label}</div>'
            f'<div style="flex:1;background:#f0f0f0;border-radius:100px;height:12px;'
            f'position:relative;">'
            f'<div style="width:{w:.1f}%;height:100%;background:{color};'
            f'border-radius:100px;transition:width 0.3s;"></div>'
            f'</div>'
            f'<div style="width:48px;text-align:right;font-size:13px;'
            f'font-weight:{fw};color:{color};">{pct}%</div>'
            f'</div>'
        )

    delta     = own_pct - peer_pct
    delta_s   = f"+{delta}" if delta >= 0 else str(delta)
    delta_col = _GREEN if delta >= 0 else _RED
    delta_txt = "boven peer" if delta >= 0 else "onder peer"

    return (
        '<div style="background:#fafafa;border:1px solid #efefef;'
        'border-radius:10px;padding:16px 20px;">'
        '<div style="font-size:11px;color:#999;margin-bottom:14px;font-weight:500;">'
        '% van historisch doel — week 14 · 2025</div>'
        + bar("Uw instelling", own_pct, own_color, bold=True)
        + bar(peer_label, peer_pct, peer_color)
        + f'<div style="font-size:11px;color:{delta_col};font-weight:600;'
        f'margin-top:8px;padding-top:8px;border-top:1px solid #f0f0f0;">'
        f'{delta_s} procentpunt {delta_txt}</div>'
        f'</div>'
    )


def _sparkline_html(
    weeks: list[int],
    own_series: list[int],
    peer_series: list[int],
    peer_label: str,
) -> str:
    W, H = 540, 130
    m = 22

    y_min = max(40, min(own_series + peer_series) - 8)
    y_max = max(own_series + peer_series) + 12
    x_step = (W - m * 2) / (len(weeks) - 1)

    own_color  = _GREEN if own_series[-1] >= peer_series[-1] else _RED
    peer_color = _ACCENT

    def y(v: float) -> float:
        return H - m - (v - y_min) / (y_max - y_min) * (H - m * 2)

    def polyline(vals: list[int], color: str, bold: bool = False) -> str:
        pts = " ".join(
            f"{m + i * x_step:.1f},{y(v):.1f}" for i, v in enumerate(vals)
        )
        sw = "2.5" if bold else "1.8"
        return (
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    # Eindpunt-dots
    ex = m + (len(weeks) - 1) * x_step
    own_dot  = f'<circle cx="{ex:.1f}" cy="{y(own_series[-1]):.1f}" r="4" fill="{own_color}" stroke="white" stroke-width="1.5"/>'
    peer_dot = f'<circle cx="{ex:.1f}" cy="{y(peer_series[-1]):.1f}" r="4" fill="{peer_color}" stroke="white" stroke-width="1.5"/>'

    x_labels = "".join(
        f'<text x="{m + i * x_step:.1f}" y="{H - 4}" text-anchor="middle" '
        f'font-size="9" fill="#bbb">W{w}</text>'
        for i, w in enumerate(weeks)
    )

    # Legenda
    leg_items = [
        (own_color,  "Uw instelling", False),
        (peer_color, peer_label, True),
    ]
    leg = ""
    lx = m
    for color, label, dashed in leg_items:
        da = 'stroke-dasharray="5,3"' if dashed else ""
        leg += (
            f'<line x1="{lx}" y1="0" x2="{lx + 16}" y2="0" stroke="{color}" '
            f'stroke-width="2" {da}/>'
            f'<text x="{lx + 20}" y="4" font-size="9" fill="#666">{label}</text>'
        )
        lx += len(label) * 6 + 34

    grid = ""
    for tick in range(int(y_min // 10) * 10, int(y_max) + 10, 10):
        if tick < y_min or tick > y_max:
            continue
        yy = y(tick)
        grid += (
            f'<line x1="{m}" y1="{yy:.1f}" x2="{W - m}" y2="{yy:.1f}" '
            f'stroke="#f0f0f0" stroke-width="1"/>'
            f'<text x="{m - 4}" y="{yy + 3:.1f}" text-anchor="end" '
            f'font-size="8" fill="#ddd">{tick}%</text>'
        )

    return (
        f'<div style="background:#fafafa;border:1px solid #efefef;'
        f'border-radius:10px;padding:16px;">'
        f'<div style="font-size:11px;color:#999;margin-bottom:10px;font-weight:500;">'
        f'% van historisch doel — week 8 t/m 14</div>'
        f'<svg width="100%" viewBox="0 0 {W} {H}" style="overflow:visible;display:block;">'
        f'{grid}'
        f'{polyline(peer_series, peer_color, bold=False)}'
        f'{polyline(own_series, own_color, bold=True)}'
        f'{own_dot}{peer_dot}'
        f'{x_labels}'
        f'<g transform="translate(0,8)">{leg}</g>'
        f'</svg>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Pagina
# ---------------------------------------------------------------------------

def create() -> None:
    nav.register_route("/peer-benchmark")

    @ui.page("/peer-benchmark")
    def peer_benchmark_page() -> None:
        with page_shell(active="/peer-benchmark", title="Peer benchmark", show_stepper=False):
            concept_banner()
            section_title(
                "Peer benchmark",
                "Vergelijk uw aanmeldingen direct met een andere instelling.",
            )
            _PeerBenchmarkView()


class _PeerBenchmarkView:
    def __init__(self) -> None:
        self._own_programme: str     = _OWN_PROGRAMMES[0]
        self._instelling: str | None = None
        self._peer_opleiding: str | None = None
        self._build()

    # ── Build ────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        # ── Stap 1 + 2: selectoren ──────────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Stap 1 — uw eigen opleiding").classes(
                "text-xs font-semibold uppercase opacity-50 mb-1"
            )
            with ui.row().classes("items-center gap-3 no-wrap w-full mb-4"):
                ui.icon("school").classes("text-2xl flex-none").style(f"color:{_ACCENT}")
                self._own_sel = ui.select(
                    _OWN_PROGRAMMES,
                    value=self._own_programme,
                    label="Uw opleiding",
                    on_change=lambda e: self._on_own(e.value),
                ).props("outlined dense").classes("flex-1")

            ui.separator()

            ui.label("Stap 2 — vergelijk met instelling").classes(
                "text-xs font-semibold uppercase opacity-50 mt-3 mb-1"
            )
            with ui.row().classes("items-center gap-3 no-wrap w-full mb-3"):
                ui.icon("account_balance").classes("text-2xl flex-none").style(
                    f"color:{_INFO}"
                )
                self._inst_sel = ui.select(
                    _PEER_INSTELLINGEN,
                    value=None,
                    label="Kies een instelling…",
                    on_change=lambda e: self._on_instelling(e.value),
                ).props("outlined dense").classes("flex-1")

            # Opleiding-selector (peer) — aanvankelijk uitgeschakeld
            with ui.row().classes("items-center gap-3 no-wrap w-full"):
                ui.icon("compare").classes("text-2xl flex-none opacity-30").style(
                    f"color:{_INFO}"
                )
                self._opl_sel = ui.select(
                    [],
                    value=None,
                    label="Kies daarna een opleiding…",
                    on_change=lambda e: self._on_peer_opleiding(e.value),
                ).props("outlined dense disable").classes("flex-1")

        # ── Resultaatgebied ──────────────────────────────────────────────────
        self._results = ui.column().classes("w-full gap-4")
        self._show_empty_state()

    # ── Selectors ────────────────────────────────────────────────────────────

    def _on_own(self, val: str) -> None:
        self._own_programme = val
        if self._instelling and self._peer_opleiding:
            self._render_results()

    def _on_instelling(self, val: str | None) -> None:
        self._instelling     = val
        self._peer_opleiding = None

        if val:
            opleidingen = _PEER_OPLEIDINGEN.get(val, [])
            self._opl_sel.options = opleidingen
            self._opl_sel.value   = None
            self._opl_sel.props(remove="disable")
        else:
            self._opl_sel.options = []
            self._opl_sel.value   = None
            self._opl_sel.props("disable")

        self._results.clear()
        with self._results:
            if val:
                self._show_instelling_state(val)
            else:
                self._show_empty_state()

    def _on_peer_opleiding(self, val: str | None) -> None:
        self._peer_opleiding = val
        self._results.clear()
        with self._results:
            if val:
                self._render_results()
            elif self._instelling:
                self._show_instelling_state(self._instelling)

    # ── Tussenstaten ─────────────────────────────────────────────────────────

    def _show_empty_state(self) -> None:
        with ui.card().classes("w-full").style(
            "border:2px dashed #e8e8e8;background:#fafafa;"
            "display:flex;align-items:center;justify-content:center;min-height:160px;"
        ):
            with ui.column().classes("items-center gap-2"):
                ui.icon("account_balance").classes("text-5xl opacity-20")
                ui.label("Kies eerst een instelling om de vergelijking te starten.").classes(
                    "text-sm opacity-40 text-center"
                )

    def _show_instelling_state(self, instelling: str) -> None:
        opleidingen = _PEER_OPLEIDINGEN.get(instelling, [])
        with ui.card().classes("w-full").style(
            "border:2px dashed #e8e8e8;background:#fafafa;"
        ):
            with ui.column().classes("items-center gap-3 py-6"):
                ui.icon("compare").classes("text-4xl").style(f"color:{_INFO};opacity:0.5;")
                ui.label(f"{instelling} geselecteerd").classes("text-sm font-semibold opacity-60")
                ui.label(
                    f"Kies nu een van de {len(opleidingen)} beschikbare opleidingen "
                    f"om de vergelijking te zien."
                ).classes("text-xs opacity-40 text-center")
                with ui.row().classes("gap-2 flex-wrap justify-center"):
                    for opl in opleidingen:
                        ui.chip(
                            opl,
                            on_click=lambda e, o=opl: self._pick_opleiding(o),
                        ).props("outline color=accent clickable")

    def _pick_opleiding(self, opleiding: str) -> None:
        self._opl_sel.value = opleiding
        self._on_peer_opleiding(opleiding)

    # ── Resultaten ────────────────────────────────────────────────────────────

    def _render_results(self) -> None:
        own  = _OWN_DATA[self._own_programme]
        peer = _peer_data(self._instelling, self._peer_opleiding)

        own_pct  = own["huidig_pct"]
        peer_pct = peer["huidig_pct"]
        delta    = own_pct - peer_pct
        delta_s  = f"+{delta}" if delta >= 0 else str(delta)

        own_color  = _GREEN if delta >= 0 else _RED
        trend_icon = {"stijgend": "trending_up", "dalend": "trending_down", "stabiel": "trending_flat"}.get(
            peer["trend"], "trending_flat"
        )
        trend_col  = {"stijgend": _GREEN, "dalend": _RED, "stabiel": _MUTED}.get(
            peer["trend"], _MUTED
        )

        # ── Header: wie staat voor? ──────────────────────────────────────────
        with ui.card().classes("w-full").style(
            f"border-left:4px solid {own_color};background:#fafafa;"
        ):
            with ui.row().classes("items-center gap-4 no-wrap"):
                ui.icon(
                    "arrow_upward" if delta > 0 else ("arrow_downward" if delta < 0 else "remove")
                ).style(f"color:{own_color};font-size:32px;").classes("flex-none")
                with ui.column().classes("gap-0 flex-1"):
                    if delta > 0:
                        kopje = f"Uw instelling loopt {abs(delta)} procentpunt voor"
                    elif delta < 0:
                        kopje = f"Uw instelling loopt {abs(delta)} procentpunt achter"
                    else:
                        kopje = "Uw instelling en peer lopen exact gelijk"
                    ui.label(kopje).classes("text-base font-semibold")
                    ui.label(
                        f"{self._own_programme}  ↔  {self._peer_opleiding} @ {self._instelling}"
                    ).classes("text-xs opacity-50 mt-0.5")
                with ui.row().classes("items-center gap-1 flex-none"):
                    ui.icon(trend_icon).style(f"color:{trend_col}").classes("text-lg")
                    ui.label(f"Trend peer: {peer['trend']}").classes("text-xs").style(
                        f"color:{trend_col};"
                    )

        # ── Drie stat-kaarten ────────────────────────────────────────────────
        with ui.row().classes("w-full gap-3"):
            for label, val, color, sub in [
                ("Uw instelling",   f"{own_pct}%",  own_color,  "% van historisch doel (W14)"),
                ("Gat (procentpunt)", f"{delta_s}pp", own_color, "uw positie t.o.v. peer"),
                (self._instelling,  f"{peer_pct}%", _ACCENT,  "% van historisch doel (W14)"),
            ]:
                with ui.card().classes("flex-1").style(f"border-top:3px solid {color};"):
                    ui.label(label).classes("text-xs opacity-50 font-medium uppercase tracking-wide")
                    ui.label(val).classes("text-3xl font-bold mt-1").style(f"color:{color};")
                    ui.label(sub).classes("text-xs opacity-40 mt-0.5")

        # ── Staafdiagram ─────────────────────────────────────────────────────
        peer_short = self._instelling.split()[0]  # "Leiden", "VU", etc.
        ui.html(_comparison_bars_html(own_pct, peer_pct, peer_short))

        # ── Sparkline ────────────────────────────────────────────────────────
        ui.html(_sparkline_html(_WEEKS, own["pct_series"], peer["pct_series"], peer_short))

        # ── Inzicht ──────────────────────────────────────────────────────────
        with ui.card().classes("w-full").style(
            f"border-left:3px solid {_INFO};background:#f4f7ff;"
        ):
            with ui.row().classes("items-start gap-3 no-wrap"):
                ui.icon("lightbulb").style(f"color:{_INFO}").classes("text-xl flex-none mt-0.5")
                with ui.column().classes("gap-1"):
                    ui.label("Inzicht").classes("font-medium text-sm")
                    ui.label(peer["note"]).classes("text-xs opacity-70 leading-relaxed")
                    ui.label(
                        "Let op: vergelijking is op % van historisch doel, niet op absolute aantallen. "
                        "Instellingen met een hoger instroomdoel tellen niet zwaarder mee."
                    ).classes("text-xs opacity-40 italic mt-1")

        # ── Opleiding wisselen ────────────────────────────────────────────────
        andere = [o for o in _PEER_OPLEIDINGEN.get(self._instelling, []) if o != self._peer_opleiding]
        if andere:
            with ui.row().classes("items-center gap-2 flex-wrap"):
                ui.label("Vergelijk ook met:").classes("text-xs opacity-50 flex-none")
                for opl in andere:
                    ui.chip(
                        opl,
                        on_click=lambda e, o=opl: self._pick_opleiding(o),
                    ).props("outline color=grey-6 clickable dense")
