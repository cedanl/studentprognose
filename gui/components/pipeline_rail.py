"""Pipeline-rail: de ML-pipeline als klikbare navigatie-hub.

Rendert de vijf hoofdfasen van de voorspelpipeline als een horizontale keten van
knooppunten, met daaronder de drie parallelle modellen die samenkomen in het
ensemble. De status per knoop (afgerond / nu / te doen / vergrendeld)
weerspiegelt de echte projectstatus (:data:`gui.state.STATE`), zodat de
startpagina meteen toont waar je bent en wat de logische volgende stap is. Elk
knooppunt navigeert naar de bijbehorende pagina.

Dit is de primaire navigatie-hub van de app (de landingspagina ``/``); de
zijbalk in :mod:`gui.components.layout` blijft als snelnavigatie op de diepere
pagina's bestaan. Zie ``gui/DESIGN.md`` voor de stijlgids.

De flow volgt :mod:`studentprognose.main`::

    ETL/validatie → laden → preprocessing → [SARIMA ‖ XGBoost ‖ ratio]
    → ensemble → postprocessing → dashboard
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from nicegui import ui

from gui import theme
from gui.state import STATE


@dataclass(frozen=True)
class RailStage:
    """Eén gebruikersgerichte pipeline-fase in de rail.

    Attributes:
        route: URL-pad van de bijbehorende pagina (bijv. ``/config``).
        num: Volgnummer in de flow (0 voor losse tools zoals benchmark).
        label: Korte weergavenaam.
        icon: Material-icoon.
        blurb: Eén zin die de ML-betekenis van de fase uitlegt.
    """

    route: str
    num: int
    label: str
    icon: str
    blurb: str


#: De vijf hoofdfasen, in pipeline-volgorde. Routes komen overeen met de
#: geregistreerde pagina's in :mod:`gui.app`.
STAGES: list[RailStage] = [
    RailStage(
        "/wizard",
        1,
        "Project",
        "folder_open",
        "Kies of maak een projectmap met data en configuratie.",
    ),
    RailStage(
        "/config",
        2,
        "Configuratie",
        "tune",
        "Stel model-, kolom- en ensemble-instellingen in.",
    ),
    RailStage(
        "/filtering",
        3,
        "Filteren",
        "filter_alt",
        "Scoop op opleiding, herkomst en examentype.",
    ),
    RailStage(
        "/run",
        4,
        "Modelleren",
        "play_arrow",
        "Draai ETL, train de modellen en combineer tot het ensemble.",
    ),
    RailStage(
        "/output",
        5,
        "Resultaten",
        "insights",
        "Bekijk de prognose, dashboards en foutmaten.",
    ),
]

#: De drie parallelle modellen die samenkomen in het ensemble (kern van stap 4).
MODELS: list[tuple[str, str, str]] = [
    ("SARIMA", "timeline", "Tijdreeks — seizoenspatroon per week."),
    ("XGBoost", "account_tree", "Machine learning — individueel + cumulatief."),
    ("Ratio", "percent", "Historische conversie vooraanmelding → inschrijving."),
]

#: Losse tool buiten de lineaire flow.
BENCHMARK = RailStage(
    "/benchmark",
    0,
    "Benchmark & tune",
    "science",
    "Vergelijk alternatieve modellen en tune hyperparameters.",
)

#: status → (kleur, statusicoon). Kleur + icoon + tekst dragen samen de betekenis
#: (nooit kleur alleen), zodat de status ook zonder kleurwaarneming leesbaar is.
_TINT = {
    "done": (theme.POSITIVE, "check_circle"),
    "current": (theme.ACCENT, "radio_button_checked"),
    "todo": (theme.MUTED, "radio_button_unchecked"),
    "locked": (theme.MUTED, "lock"),
}

#: Tooltip op vergrendelde knooppunten (identiek aan de zijbalk-logica).
_LOCK_REASON = "Kies eerst een project (stap 1)."


def _has_output() -> bool:
    """True als er al voorspeloutput op schijf staat in het actieve project."""
    output_dir = STATE.output_dir
    if not output_dir or not os.path.isdir(output_dir):
        return False
    return any(files for _root, _dirs, files in os.walk(output_dir))


def compute_states() -> dict[str, str]:
    """Bepaal de status per fase op basis van de echte projectstatus.

    * **Geen project** → stap 1 (Project) is 'nu', de rest is vergrendeld: zonder
      project heeft geen enkele latere stap zin, precies zoals de zijbalk dat toont.
    * **Project, nog geen output** → de voorbereidingsstappen zijn afgerond en
      Modelleren is de aanbevolen volgende stap ('nu').
    * **Output aanwezig** → alles is afgerond en Resultaten is 'nu' — de plek om
      de prognose te bekijken.

    Returns:
        Dict van route → status (``done`` | ``current`` | ``todo`` | ``locked``).
    """
    if not STATE.is_initialised:
        return {
            "/wizard": "current",
            "/config": "locked",
            "/filtering": "locked",
            "/run": "locked",
            "/output": "locked",
            "/benchmark": "locked",
        }
    if _has_output():
        return {
            "/wizard": "done",
            "/config": "done",
            "/filtering": "done",
            "/run": "done",
            "/output": "current",
            "/benchmark": "todo",
        }
    return {
        "/wizard": "done",
        "/config": "done",
        "/filtering": "done",
        "/run": "current",
        "/output": "todo",
        "/benchmark": "todo",
    }


def render(*, on_nav: Callable[[str], None] | None = None) -> None:
    """Render de pipeline-rail-hub.

    Args:
        on_nav: Navigatie-callback die een route ontvangt. Standaard
            :func:`nicegui.ui.navigate.to` — meegeven is vooral handig voor tests.
    """
    navigate = on_nav or ui.navigate.to
    states = compute_states()

    # De horizontale rail met knooppunten en connectoren.
    with ui.row().classes("w-full items-stretch no-wrap gap-0 overflow-x-auto"):
        for i, st in enumerate(STAGES):
            _node(st, states[st.route], navigate)
            if i < len(STAGES) - 1:
                nxt = states[STAGES[i + 1].route]
                _connector(nxt in ("done", "current"))

    _model_card()
    _benchmark_track(states["/benchmark"], navigate)


def _node(st: RailStage, state: str, navigate: Callable[[str], None]) -> None:
    """Eén klikbaar pipeline-knooppunt als kaart."""
    color, sicon = _TINT[state]
    is_current = state == "current"
    locked = state == "locked"

    classes = "min-w-[168px] flex-1 transition-all"
    classes += " cursor-not-allowed" if locked else " cursor-pointer hover:shadow-md"

    border = theme.ACCENT if is_current else "#e0e0e0"
    style = f"border: 1px solid {border}"
    if is_current:
        style += f"; background: {theme.ACCENT}0f"
    if locked:
        style += "; opacity: 0.55"

    card = ui.card().classes(classes).style(style)
    if not locked:
        card.on("click", lambda r=st.route: navigate(r))
    with card:
        with ui.row().classes("items-center justify-between w-full no-wrap"):
            with (
                ui.element("div")
                .classes("flex items-center justify-center rounded-full")
                .style(f"width: 32px; height: 32px; background: {color}1a")
            ):
                ui.icon(st.icon).style(f"color: {color}")
            ui.icon(sicon).classes("text-sm").style(f"color: {color}")
        ui.label(f"Stap {st.num}").classes("text-xs opacity-50 mt-1")
        ui.label(st.label).classes("text-base font-medium leading-tight")
        ui.label(st.blurb).classes("text-xs opacity-60 leading-snug")
    if locked:
        card.tooltip(_LOCK_REASON)


def _connector(reached: bool) -> None:
    """Pijl tussen twee knooppunten; gekleurd als de flow er al is geweest."""
    col = theme.ACCENT if reached else "#cfcfcf"
    with ui.column().classes("justify-center self-center px-1"):
        ui.icon("east").style(f"color: {col}")


def _model_card() -> None:
    """Toon de drie parallelle modellen die samenkomen in het ensemble (stap 4)."""
    with (
        ui.card()
        .classes("w-full")
        .style(f"border: 1px dashed {theme.SECONDARY}66; background: {theme.SECONDARY}0a")
    ):
        with ui.row().classes("items-center gap-2"):
            ui.icon("play_arrow").style(f"color: {theme.ACCENT}")
            ui.label("Binnen stap 4 · Modelleren").classes("font-medium")
        with ui.row().classes("w-full items-stretch no-wrap gap-3 mt-1"):
            with ui.column().classes("gap-2 justify-center"):
                for name, icon, blurb in MODELS:
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        with (
                            ui.element("div")
                            .classes("flex items-center justify-center rounded")
                            .style(f"width: 28px; height: 28px; background: {theme.SECONDARY}1a")
                        ):
                            ui.icon(icon).style(f"color: {theme.SECONDARY}")
                        with ui.column().classes("gap-0"):
                            ui.label(name).classes("text-sm font-medium")
                            ui.label(blurb).classes("text-xs opacity-60")
            # Samenvoeg-beugel: de drie modellen bundelen tot één stroom.
            ui.element("div").classes("self-stretch my-2").style(
                f"width: 14px; border: 2px solid {theme.SECONDARY}55; "
                "border-left: none; border-radius: 0 12px 12px 0"
            )
            ui.icon("chevron_right").classes("text-2xl opacity-40 self-center")
            with (
                ui.column()
                .classes("items-center gap-1 p-3 rounded self-center")
                .style(f"background: {theme.ACCENT}1a; border: 1px solid {theme.ACCENT}")
            ):
                ui.icon("hub").classes("text-2xl").style(f"color: {theme.ACCENT}")
                ui.label("Ensemble").classes("text-sm font-medium")
                ui.label("gewogen combinatie").classes("text-xs opacity-60")


def _benchmark_track(state: str, navigate: Callable[[str], None]) -> None:
    """Benchmark als zijspoor onder de rail."""
    locked = state == "locked"
    with ui.row().classes("items-center gap-2 mt-1").style(
        "opacity: 0.55" if locked else ""
    ):
        ui.icon("alt_route").style(f"color: {theme.MUTED}")
        ui.label("Zijspoor:").classes("text-sm")
        btn = (
            ui.button(BENCHMARK.label, icon=BENCHMARK.icon)
            .props("flat dense no-caps")
            .classes("text-sm")
        )
        if locked:
            btn.props("disable")
        else:
            btn.on("click", lambda: navigate(BENCHMARK.route))
    if locked:
        btn.tooltip(_LOCK_REASON)
