"""Concept: Wat-als analyse — bandbreedte, historisch jaar en doelvergelijking.

Drie eerlijke scenario-typen die het model werkelijk kan onderbouwen:
  1. Bandbreedte  — pessimistisch (p25) / basis (p50) / optimistisch (p75)
  2. Historisch jaar als scenario — hoe zou dit jaar eruitzien als het 20XX volgt?
  3. Doelvergelijking — we willen X studenten; hoe groot is het gat?

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

_PROGRAMMES = [
    "Technische Informatica (Bachelor · WO)",
    "Bedrijfskunde (Bachelor · WO)",
    "Psychologie (Bachelor · WO)",
    "Werktuigbouwkunde (Bachelor · WO)",
]

_WEEKS = list(range(14, 39))  # weken 14..38


def _s_curve(start: int, end: int, n: int, steepness: float = 0.25) -> list[int]:
    import math
    result = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 1.0
        s  = 1 / (1 + math.exp(-8 * (t - steepness)))
        s0 = 1 / (1 + math.exp(-8 * (0 - steepness)))
        s1 = 1 / (1 + math.exp(-8 * (1 - steepness)))
        norm = (s - s0) / (s1 - s0)
        result.append(round(start + norm * (end - start)))
    return result


_DATA: dict[str, dict] = {
    _PROGRAMMES[0]: {
        "huidig_week": 14, "huidig_count": 187,
        "basis": _s_curve(187, 312, len(_WEEKS), 0.20),
        "p25":   _s_curve(187, 287, len(_WEEKS), 0.20),
        "p75":   _s_curve(187, 338, len(_WEEKS), 0.20),
        "historisch": {
            2022: _s_curve(181, 292, len(_WEEKS), 0.18),
            2023: _s_curve(190, 315, len(_WEEKS), 0.22),
            2024: _s_curve(183, 301, len(_WEEKS), 0.19),
        },
        "instelling_doel": 300,
    },
    _PROGRAMMES[1]: {
        "huidig_week": 14, "huidig_count": 291,
        "basis": _s_curve(291, 487, len(_WEEKS), 0.20),
        "p25":   _s_curve(291, 451, len(_WEEKS), 0.20),
        "p75":   _s_curve(291, 524, len(_WEEKS), 0.20),
        "historisch": {
            2022: _s_curve(275, 438, len(_WEEKS), 0.18),
            2023: _s_curve(283, 452, len(_WEEKS), 0.21),
            2024: _s_curve(288, 471, len(_WEEKS), 0.20),
        },
        "instelling_doel": 480,
    },
    _PROGRAMMES[2]: {
        "huidig_week": 14, "huidig_count": 398,
        "basis": _s_curve(398, 643, len(_WEEKS), 0.20),
        "p25":   _s_curve(398, 588, len(_WEEKS), 0.20),
        "p75":   _s_curve(398, 698, len(_WEEKS), 0.20),
        "historisch": {
            2022: _s_curve(385, 612, len(_WEEKS), 0.19),
            2023: _s_curve(402, 648, len(_WEEKS), 0.21),
            2024: _s_curve(415, 671, len(_WEEKS), 0.20),
        },
        "instelling_doel": 630,
    },
    _PROGRAMMES[3]: {
        "huidig_week": 14, "huidig_count": 118,
        "basis": _s_curve(118, 201, len(_WEEKS), 0.20),
        "p25":   _s_curve(118, 186, len(_WEEKS), 0.20),
        "p75":   _s_curve(118, 217, len(_WEEKS), 0.20),
        "historisch": {
            2022: _s_curve(112, 178, len(_WEEKS), 0.18),
            2023: _s_curve(115, 187, len(_WEEKS), 0.20),
            2024: _s_curve(119, 194, len(_WEEKS), 0.19),
        },
        "instelling_doel": 200,
    },
}

# ---------------------------------------------------------------------------
# SVG grafiek
# ---------------------------------------------------------------------------

def _chart_html(
    weeks: list[int],
    basis: list[int],
    p25: list[int],
    p75: list[int],
    hist_year: int | None,
    hist_vals: list[int] | None,
    target: int | None,
    huidig_count: int,
) -> str:
    W, H = 580, 210
    ml, mr, mt, mb = 40, 8, 16, 26

    chart_w = W - ml - mr
    chart_h = H - mt - mb

    all_vals = p25 + p75 + (hist_vals or []) + ([target] if target else [])
    y_min = max(0, min(all_vals) - 20)
    y_max = max(all_vals) + 30

    def px(i: int) -> float:
        return ml + i / (len(weeks) - 1) * chart_w

    def py(v: float) -> float:
        return mt + (1 - (v - y_min) / (y_max - y_min)) * chart_h

    def pts(vals: list[int]) -> str:
        return " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))

    # Schaduwband
    band_top = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(p75))
    band_bot = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in reversed(list(enumerate(p25))))
    band = f'<polygon points="{band_top} {band_bot}" fill="{_ACCENT}" opacity="0.09"/>'

    # Gridlijnen + y-labels
    grid = ""
    step = 50
    for tick in range(int(y_min // step) * step, int(y_max) + step, step):
        if tick < y_min or tick > y_max:
            continue
        yy = py(tick)
        grid += (
            f'<line x1="{ml}" y1="{yy:.1f}" x2="{W - mr}" y2="{yy:.1f}" '
            f'stroke="#eee" stroke-width="1"/>'
            f'<text x="{ml - 4}" y="{yy + 3.5:.1f}" text-anchor="end" '
            f'font-size="9" fill="#ccc">{tick}</text>'
        )

    # X-labels
    x_labels = ""
    for i, w in enumerate(weeks):
        if w % 4 == 0 or w == weeks[0] or w == weeks[-1]:
            x_labels += (
                f'<text x="{px(i):.1f}" y="{H - 5}" text-anchor="middle" '
                f'font-size="9" fill="#bbb">W{w}</text>'
            )

    # Doellijn (label binnen de chart, right-aligned)
    target_svg = ""
    if target and y_min <= target <= y_max:
        yy = py(target)
        lx = W - mr - 4
        target_svg = (
            f'<line x1="{ml}" y1="{yy:.1f}" x2="{W - mr}" y2="{yy:.1f}" '
            f'stroke="{_INFO}" stroke-width="1.5" stroke-dasharray="6,3"/>'
            f'<text x="{lx}" y="{yy - 4:.1f}" text-anchor="end" '
            f'font-size="8.5" fill="{_INFO}" font-weight="600">Doel {target}</text>'
        )

    # Historisch
    hist_svg = ""
    if hist_vals and hist_year:
        hist_svg = (
            f'<polyline points="{pts(hist_vals)}" fill="none" stroke="#bbb" '
            f'stroke-width="1.5" stroke-dasharray="5,3"/>'
        )

    # Bandbreedte-lijnen
    p25_svg = (
        f'<polyline points="{pts(p25)}" fill="none" stroke="{_RED}" '
        f'stroke-width="1.2" stroke-dasharray="4,3" opacity="0.65"/>'
    )
    p75_svg = (
        f'<polyline points="{pts(p75)}" fill="none" stroke="{_GREEN}" '
        f'stroke-width="1.2" stroke-dasharray="4,3" opacity="0.65"/>'
    )

    # Basislijn
    basis_svg = (
        f'<polyline points="{pts(basis)}" fill="none" stroke="{_ACCENT}" '
        f'stroke-width="2.5" stroke-linecap="round"/>'
    )

    # Nu-markering
    now_x, now_y = px(0), py(huidig_count)
    now_svg = (
        f'<line x1="{now_x:.1f}" y1="{mt}" x2="{now_x:.1f}" y2="{H - mb}" '
        f'stroke="#e0e0e0" stroke-width="1" stroke-dasharray="3,2"/>'
        f'<circle cx="{now_x:.1f}" cy="{now_y:.1f}" r="4" fill="{_ACCENT}" '
        f'stroke="white" stroke-width="1.5"/>'
        f'<text x="{now_x + 5:.1f}" y="{mt + 11}" font-size="8.5" fill="#aaa">Nu (W14)</text>'
    )

    return (
        f'<div style="background:#fafafa;border:1px solid #efefef;'
        f'border-radius:10px;padding:14px 10px 6px 8px;overflow:hidden;">'
        f'<svg width="100%" viewBox="0 0 {W} {H}" style="overflow:visible;display:block;">'
        f'{grid}{band}{target_svg}{hist_svg}{p25_svg}{p75_svg}{basis_svg}{now_svg}'
        f'{x_labels}'
        f'</svg>'
        f'</div>'
    )


def _chart_legend_html(hist_year: int | None, target: int | None) -> str:
    items = [
        (_ACCENT, "Basisprognose", False),
        (_GREEN,  "Optimistisch (P75)", True),
        (_RED,    "Pessimistisch (P25)", True),
    ]
    if hist_year:
        items.append(("#bbb", f"Historisch {hist_year}", True))
    if target:
        items.append((_INFO, f"Doel ({target})", True))

    parts = []
    for color, label, dashed in items:
        line_style = (
            f"width:18px;height:0;border-top:{'2px dashed' if dashed else '2.5px solid'} {color};"
            f"display:inline-block;"
        )
        parts.append(
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'font-size:11px;color:#666;white-space:nowrap;">'
            f'<span style="{line_style}"></span>{label}</span>'
        )

    return (
        '<div style="display:flex;flex-wrap:wrap;gap:8px 14px;'
        'padding:4px 2px;">' + "".join(parts) + "</div>"
    )


# ---------------------------------------------------------------------------
# Pagina
# ---------------------------------------------------------------------------

def create() -> None:
    nav.register_route("/scenarios")

    @ui.page("/scenarios")
    def scenarios_page() -> None:
        with page_shell(active="/scenarios", title="Wat-als analyse", show_stepper=False):
            concept_banner()
            section_title(
                "Wat-als analyse",
                "Drie eerlijke scenario-typen die het model werkelijk kan onderbouwen.",
            )
            _ScenariosView()


class _ScenariosView:
    def __init__(self) -> None:
        self._programme  = _PROGRAMMES[0]
        self._hist_year: int | None = None
        self._target: int | None    = None
        self._build()

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build(self) -> None:
        # Intro
        ui.label(
            "Prognoses zijn geen enkelvoudige getallen — ze hebben marges. "
            "Dit paneel maakt die marges zichtbaar."
        ).classes("text-sm opacity-55 leading-relaxed")

        # Opleiding-selector
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-3 no-wrap w-full"):
                ui.icon("school").classes("text-2xl flex-none").style(f"color:{_ACCENT}")
                ui.select(
                    _PROGRAMMES,
                    value=self._programme,
                    label="Opleiding",
                    on_change=lambda e: self._on_programme(e.value),
                ).props("outlined dense").classes("flex-1")
                ui.label("Prognose week 38 · 2025").classes("text-xs opacity-40 flex-none")

        self._main = ui.column().classes("w-full gap-4")
        self._render()

    def _on_programme(self, val: str) -> None:
        self._programme = val
        self._hist_year = None
        self._target    = None
        self._main.clear()
        with self._main:
            self._render()

    def _render(self) -> None:
        d         = _DATA[self._programme]
        p25_end   = d["p25"][-1]
        basis_end = d["basis"][-1]
        p75_end   = d["p75"][-1]

        # ── Scenario 1: Bandbreedte ──────────────────────────────────────────
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2 mb-1"):
                ui.badge("1").props("color=accent").classes("font-bold text-sm")
                ui.label("Bandbreedte — onzekerheid in kaart").classes("font-semibold")
            ui.label(
                "Het model berekent een betrouwbaarheidsinterval op basis van historische variatie "
                "in het aanmeldpatroon. Dit is geen fout — het is eerlijkheid over wat het model "
                "wél en niet kan weten."
            ).classes("text-xs opacity-60 leading-relaxed mb-3")

            with ui.row().classes("w-full gap-3"):
                for label, val, color, desc in [
                    ("Pessimistisch (P25)", p25_end, _RED,
                     "Slechte week-curve en laag conversieratio — treedt op in ~25% van vergelijkbare jaren"),
                    ("Basisprognose (P50)",  basis_end, _ACCENT,
                     "Huidig aanmeldtempo ongewijzigd voortgezet — meest waarschijnlijk scenario"),
                    ("Optimistisch (P75)",   p75_end, _GREEN,
                     "Bovengemiddelde conversie in resterende weken — treedt op in ~25% van jaren"),
                ]:
                    with ui.card().classes("flex-1").style(f"border-top:3px solid {color};"):
                        ui.label(label).classes("text-xs opacity-50 font-medium uppercase tracking-wide")
                        ui.label(str(val)).classes("text-3xl font-bold mt-1").style(f"color:{color};")
                        ui.label("studenten").classes("text-xs opacity-40")
                        ui.separator().classes("my-2")
                        ui.label(desc).classes("text-xs opacity-55 leading-snug")

        # ── Scenario 2: Historisch jaar ──────────────────────────────────────
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2 mb-1"):
                ui.badge("2").props("color=accent").classes("font-bold text-sm")
                ui.label("Historisch jaar als scenario").classes("font-semibold")
            ui.label(
                "Het model vergelijkt het huidige aanmeldpatroon (week 1–14) met eerdere jaren. "
                "Door een historisch jaar als referentie te kiezen, zie je hoe de resterende "
                "curve eruit zou zien als dit jaar dat patroon volgt. "
                "Let op: dit modelleert patroonovereenkomst, geen oorzaak-gevolg."
            ).classes("text-xs opacity-60 leading-relaxed mb-2")

            with ui.row().classes("items-center gap-3 no-wrap flex-wrap"):
                ui.label("Vergelijk met:").classes("text-sm opacity-60 flex-none")
                self._hist_btns: dict[int, ui.button] = {}
                for yr in sorted(d["historisch"].keys()):
                    btn = (
                        ui.button(str(yr), on_click=lambda e, y=yr: self._on_hist(y))
                        .props("outline dense color=grey-7")
                    )
                    self._hist_btns[yr] = btn
                ui.button("Wis", icon="close", on_click=lambda: self._on_hist(None)).props(
                    "flat dense color=grey-5"
                ).classes("ml-1")

            self._hist_result = ui.column().classes("w-full mt-2")
            self._hist_placeholder = ui.label(
                "Kies een jaar hierboven om de vergelijking te zien."
            ).classes("text-xs opacity-40 italic mt-1")

        # ── Scenario 3: Doelvergelijking ─────────────────────────────────────
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2 mb-1"):
                ui.badge("3").props("color=accent").classes("font-bold text-sm")
                ui.label("Doelvergelijking — liggen we op schema?").classes("font-semibold")
            ui.label(
                "Vul het instellingsdoel in. Het model toont welke scenario's dit doel halen "
                "en schat de kans op basis van historische variatie in vergelijkbare jaren."
            ).classes("text-xs opacity-60 leading-relaxed mb-2")

            with ui.row().classes("items-center gap-3 no-wrap"):
                self._target_input = (
                    ui.number(
                        "Instellingsdoel (studenten)",
                        value=d["instelling_doel"],
                        min=0,
                        max=2000,
                        step=5,
                    )
                    .props("outlined dense")
                    .classes("w-56")
                )
                ui.button("Bereken", icon="calculate", on_click=self._calc_target).props(
                    "unelevated color=accent"
                )

            self._target_result = ui.column().classes("w-full mt-2")
            self._target_placeholder = ui.label(
                "Vul een doel in en klik 'Bereken' om de analyse te zien."
            ).classes("text-xs opacity-40 italic mt-1")

        # ── Gecombineerde grafiek ────────────────────────────────────────────
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2 mb-1"):
                ui.icon("show_chart").style(f"color:{_ACCENT}").classes("text-xl")
                ui.label("Gecombineerd overzicht — week 14 t/m 38").classes("font-medium")
            self._chart_slot  = ui.column().classes("w-full")
            self._legend_slot = ui.column().classes("w-full")
            self._redraw_chart()

    # ── Historisch jaar ───────────────────────────────────────────────────────

    def _on_hist(self, year: int | None) -> None:
        self._hist_year = year
        for yr, btn in self._hist_btns.items():
            if yr == year:
                btn.props("unelevated color=accent")
            else:
                btn.props("outline color=grey-7")

        self._hist_result.clear()
        self._hist_placeholder.set_visibility(year is None)

        if year is not None:
            d         = _DATA[self._programme]
            hist_end  = d["historisch"][year][-1]
            basis_end = d["basis"][-1]
            p25_end   = d["p25"][-1]
            p75_end   = d["p75"][-1]
            delta     = hist_end - basis_end
            sign      = "+" if delta >= 0 else ""
            color     = _GREEN if delta >= 0 else _RED

            if hist_end < p25_end:
                band_note, band_icon, band_col = "onder pessimistisch scenario", "warning",      _RED
            elif hist_end > p75_end:
                band_note, band_icon, band_col = "boven optimistisch scenario", "info",          _GREEN
            else:
                band_note, band_icon, band_col = "binnen verwachte bandbreedte", "check_circle", _GREEN

            h_vals = d["historisch"][year]
            h_now  = h_vals[0]
            b_now  = d["basis"][0]
            d_now  = h_now - b_now
            d_sign = "+" if d_now >= 0 else ""

            with self._hist_result:
                with ui.row().classes("w-full gap-3 items-stretch"):
                    with ui.card().classes("flex-1").style(f"border-top:3px solid {color};"):
                        ui.label(f"Eindinstroom als dit jaar op {year} lijkt:").classes(
                            "text-xs opacity-50 font-medium uppercase"
                        )
                        with ui.row().classes("items-baseline gap-2 mt-1"):
                            ui.label(str(hist_end)).classes("text-3xl font-bold")
                            ui.label(f"{sign}{delta} t.o.v. basisprognose").classes(
                                "text-sm font-medium"
                            ).style(f"color:{color};")
                        ui.label("studenten bij week 38").classes("text-xs opacity-40")

                    with ui.card().classes("flex-none").style(
                        f"border-top:3px solid {band_col};min-width:190px;"
                    ):
                        with ui.row().classes("items-center gap-2"):
                            ui.icon(band_icon).style(f"color:{band_col}").classes("text-xl flex-none")
                            with ui.column().classes("gap-0"):
                                ui.label("Positie t.o.v. bandbreedte").classes("text-xs opacity-50")
                                ui.label(band_note).classes("text-sm font-semibold").style(
                                    f"color:{band_col};"
                                )

                    with ui.card().classes("flex-none").style("min-width:180px;"):
                        ui.label(f"Aanmeldingen W14 in {year}:").classes("text-xs opacity-50")
                        with ui.row().classes("items-baseline gap-1 mt-1"):
                            ui.label(str(h_now)).classes("text-xl font-bold")
                            ui.label(f"({d_sign}{d_now} vs huidig)").classes("text-xs opacity-60")
                        ui.label("maat voor patroonovereenkomst").classes("text-xs opacity-40")

        self._redraw_chart()

    # ── Doelvergelijking ──────────────────────────────────────────────────────

    def _calc_target(self) -> None:
        d = _DATA[self._programme]
        try:
            target = int(self._target_input.value)
        except (TypeError, ValueError):
            target = d["instelling_doel"]
        self._target = target

        basis_end = d["basis"][-1]
        p25_end   = d["p25"][-1]
        p75_end   = d["p75"][-1]
        gap       = target - basis_end
        gap_sign  = "+" if gap >= 0 else ""

        hits_p25   = p25_end   >= target
        hits_basis = basis_end >= target
        hits_p75   = p75_end   >= target

        if hits_p25:
            kans, kans_color, kans_label = 92, _GREEN,   "zeer waarschijnlijk (>90%)"
        elif hits_basis:
            kans, kans_color, kans_label = 58, _GREEN,   "waarschijnlijk (~60%)"
        elif hits_p75:
            kans, kans_color, kans_label = 28, _WARNING, "mogelijk bij meevaller (~30%)"
        else:
            kans, kans_color, kans_label = 5,  _RED,     "onwaarschijnlijk (<10%)"

        self._target_result.clear()
        self._target_placeholder.set_visibility(False)

        with self._target_result:
            with ui.row().classes("w-full gap-3"):
                for scenario, val, hits in [
                    ("Pessimistisch (P25)", p25_end,  hits_p25),
                    ("Basis (P50)",         basis_end, hits_basis),
                    ("Optimistisch (P75)",  p75_end,  hits_p75),
                ]:
                    icon  = "check_circle" if hits else "cancel"
                    color = _GREEN if hits else _RED
                    with ui.card().classes("flex-1").style(f"border-top:3px solid {color};"):
                        ui.label(scenario).classes("text-xs opacity-50 font-medium uppercase")
                        ui.label(str(val)).classes("text-2xl font-bold mt-1").style(f"color:{color};")
                        with ui.row().classes("items-center gap-1 mt-1"):
                            ui.icon(icon).style(f"color:{color}").classes("text-sm")
                            ui.label("haalt doel" if hits else "haalt niet").classes(
                                "text-xs font-medium"
                            ).style(f"color:{color};")

            with ui.row().classes("items-center gap-4 no-wrap").style(
                "background:#f9f9f9;border-radius:8px;padding:12px 16px;"
            ):
                with ui.column().classes("gap-0 flex-1"):
                    ui.label(
                        f"Doel: {target}   ·   Basisprognose: {basis_end}   ·   "
                        f"Gat: {gap_sign}{gap} studenten"
                    ).classes("text-sm font-semibold")
                    ui.label(f"Kans op het halen van het doel: {kans_label}").classes(
                        "text-xs opacity-70 mt-0.5"
                    )
                    ui.label(
                        "Op basis van historische variatie in vergelijkbare jaren — illustratief."
                    ).classes("text-xs opacity-35 italic mt-0.5")
                ui.label(f"~{kans}%").classes("text-3xl font-bold flex-none").style(
                    f"color:{kans_color};"
                )

        self._redraw_chart()

    # ── Grafiek ───────────────────────────────────────────────────────────────

    def _redraw_chart(self) -> None:
        d = _DATA[self._programme]
        hist_vals = d["historisch"].get(self._hist_year) if self._hist_year else None
        self._chart_slot.clear()
        self._legend_slot.clear()
        with self._chart_slot:
            ui.html(_chart_html(
                weeks        = _WEEKS,
                basis        = d["basis"],
                p25          = d["p25"],
                p75          = d["p75"],
                hist_year    = self._hist_year,
                hist_vals    = hist_vals,
                target       = self._target,
                huidig_count = d["huidig_count"],
            ))
        with self._legend_slot:
            ui.html(_chart_legend_html(self._hist_year, self._target))
