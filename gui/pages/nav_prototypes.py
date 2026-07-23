"""Vier navigatie-alternatieven (``/nav1``–``/nav4``).

Elk prototype is een zelfstandige paginaschil die een ander navigatieparadigma
demonstreert, allemaal gemodelleerd op de echte ML-pipeline-flow uit
:mod:`studentprognose.main`:

    ETL/validatie → laden → preprocessing → [SARIMA ‖ XGBoost ‖ ratio]
    → ensemble → postprocessing → dashboard

De vier alternatieven:

* **/nav1 — Pipeline-rail**: de pipeline-DAG zélf is de navigatie. Horizontale
  keten van stap-knooppunten; het modelleer-knooppunt toont de drie parallelle
  modellen die samenkomen in het ensemble.
* **/nav2 — Gefaseerde zijbalk**: inklapbare zijbalk gegroepeerd per pipeline-fase
  (Voorbereiden · Voorspellen · Analyseren) met voortgang per groep.
* **/nav3 — Tabbladen + commandopalet**: horizontale tabbladen in de header plus
  een ⌘K-commandopalet voor sneltoetsgebruikers.
* **/nav4 — Missiecontrole-hub**: kaart-dashboard dat de hele pipeline in één
  overzicht toont, elke fase met live status en een directe actie.

Dit bestand is een ontwerpspeeltuin: het deelt de huisstijl-tokens uit
:mod:`gui.theme` maar bouwt zijn eigen navigatie-chrome (dat is juist het punt).
"""

from __future__ import annotations

from dataclasses import dataclass

from nicegui import ui

from gui import theme

# --- Pipeline-model (gedeeld door alle vier de prototypes) --------------------


@dataclass(frozen=True)
class Stage:
    """Eén gebruikersgerichte pipeline-fase.

    Attributes:
        route: Bestemming (hier symbolisch — de prototypes navigeren onderling).
        num: Volgnummer in de flow.
        label: Korte weergavenaam.
        icon: Material-icoon.
        blurb: Eén zin die de ML-betekenis van de fase uitlegt.
        state: ``done`` | ``current`` | ``todo`` — demostatus voor de indicatoren.
    """

    route: str
    num: int
    label: str
    icon: str
    blurb: str
    state: str = "todo"


#: De vijf hoofdfasen, in pipeline-volgorde. ``current`` staat bewust op stap 3
#: zodat de prototypes een realistische half-afgeronde flow tonen.
STAGES: list[Stage] = [
    Stage(
        "/wizard",
        1,
        "Project",
        "folder_open",
        "Kies of maak een projectmap met data en configuratie.",
        "done",
    ),
    Stage(
        "/config",
        2,
        "Configuratie",
        "tune",
        "Stel model-, kolom- en ensemble-instellingen in.",
        "done",
    ),
    Stage(
        "/filtering",
        3,
        "Filteren",
        "filter_alt",
        "Scoop op opleiding, herkomst en examentype.",
        "current",
    ),
    Stage(
        "/run",
        4,
        "Modelleren",
        "play_arrow",
        "Draai ETL, train de modellen en combineer tot het ensemble.",
        "todo",
    ),
    Stage(
        "/output",
        5,
        "Resultaten",
        "insights",
        "Bekijk de prognose, dashboards en foutmaten.",
        "todo",
    ),
]

#: De drie parallelle modellen die samenkomen in het ensemble (kern van stap 4).
MODELS: list[tuple[str, str, str]] = [
    ("SARIMA", "timeline", "Tijdreeks — seizoenspatroon per week."),
    ("XGBoost", "account_tree", "Machine learning — individueel + cumulatief."),
    ("Ratio", "percent", "Historische conversie vooraanmelding → inschrijving."),
]

#: Losse tool buiten de lineaire flow.
BENCHMARK = Stage(
    "/benchmark",
    0,
    "Benchmark & tune",
    "science",
    "Vergelijk alternatieve modellen en tune hyperparameters.",
    "todo",
)

#: Kaart-achtergrondtint per status.
_STATE_TINT = {
    "done": (theme.POSITIVE, "check_circle"),
    "current": (theme.ACCENT, "radio_button_checked"),
    "todo": (theme.MUTED, "radio_button_unchecked"),
}


# --- Gedeelde chrome ----------------------------------------------------------


def _appbar(subtitle: str) -> None:
    """Zwarte CEDA/Npuls app-bar met oranje accentlijn (identiek aan de echte app)."""
    with (
        ui.header()
        .classes("items-center justify-between q-px-md")
        .style(f"background: {theme.PRIMARY}; border-bottom: 3px solid {theme.ACCENT}")
    ):
        with ui.row().classes("items-center gap-3 no-wrap"):
            ui.image("/gui-assets/npuls-logo-white.svg").classes("w-8 h-8")
            ui.label("Studentprognose").classes("text-lg font-medium text-white")
        ui.label(subtitle).classes("text-sm text-white opacity-70")


def _switcher(active: str) -> None:
    """Zwevende pillen om tussen de vier prototypes te wisselen (evaluatiehulp)."""
    labels = {
        "/nav1": "1 · Rail",
        "/nav2": "2 · Zijbalk",
        "/nav3": "3 · Tabs+⌘K",
        "/nav4": "4 · Hub",
    }
    with (
        ui.row()
        .classes("fixed z-50 gap-1 p-1 rounded-full shadow-lg items-center no-wrap")
        .style(
            "bottom: 20px; left: 50%; transform: translateX(-50%); "
            f"background: {theme.PRIMARY}"
        )
    ):
        ui.label("Alternatief").classes("text-xs text-white opacity-40 pl-2 pr-1")
        for route, label in labels.items():
            is_active = route == active
            btn = (
                ui.button(label, on_click=lambda r=route: ui.navigate.to(r))
                .props("flat dense no-caps")
                .classes("rounded-full px-3 text-xs !text-white")
            )
            if is_active:
                btn.style(f"background: {theme.ACCENT}")
            else:
                btn.classes("opacity-60")


# ============================================================================
# /nav1 — Pipeline-rail: de pipeline-DAG is de navigatie
# ============================================================================


def _nav1() -> None:
    ui.colors(**theme.QUASAR_COLORS)
    _appbar("Alternatief 1 — Pipeline-rail")
    _switcher("/nav1")

    with ui.column().classes("w-full max-w-6xl mx-auto p-6 gap-6"):
        ui.label("Volg de voorspelling door de pipeline").classes("text-2xl font-bold")
        ui.label(
            "De navigatie is de pipeline zelf. Elke stap voedt de volgende; "
            "het modelleer-knooppunt splitst in drie modellen die samenkomen "
            "in het ensemble."
        ).classes("text-sm opacity-70 -mt-2")

        # De horizontale rail met knooppunten en connectoren.
        with ui.row().classes(
            "w-full items-stretch no-wrap gap-0 mt-2 overflow-x-auto"
        ):
            for i, st in enumerate(STAGES):
                _nav1_node(st)
                if i < len(STAGES) - 1:
                    _nav1_connector(STAGES[i + 1].state != "todo")

        # Onder het modelleer-knooppunt: de drie parallelle modellen → ensemble.
        with (
            ui.card()
            .classes("w-full mt-2")
            .style(
                f"border: 1px dashed {theme.SECONDARY}66; background: {theme.SECONDARY}0a"
            )
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
                                .style(
                                    "width: 28px; height: 28px; "
                                    f"background: {theme.SECONDARY}1a"
                                )
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
                    .style(
                        f"background: {theme.ACCENT}1a; "
                        f"border: 1px solid {theme.ACCENT}"
                    )
                ):
                    ui.icon("hub").classes("text-2xl").style(f"color: {theme.ACCENT}")
                    ui.label("Ensemble").classes("text-sm font-medium")
                    ui.label("gewogen combinatie").classes("text-xs opacity-60")

        # Detour: benchmark als aftakking.
        with ui.row().classes("items-center gap-2 opacity-70 mt-1"):
            ui.icon("alt_route").style(f"color: {theme.MUTED}")
            ui.label("Zijspoor:").classes("text-sm")
            ui.button(BENCHMARK.label, icon=BENCHMARK.icon).props(
                "flat dense no-caps"
            ).classes("text-sm")


def _nav1_node(st: Stage) -> None:
    """Eén klikbaar pipeline-knooppunt als kaart."""
    color, sicon = _STATE_TINT[st.state]
    is_current = st.state == "current"
    card = (
        ui.card()
        .classes("min-w-[168px] flex-1 cursor-pointer transition-all")
        .style(
            "border: 1px solid "
            + (f"{theme.ACCENT}" if is_current else "#e0e0e0")
            + (f"; background: {theme.ACCENT}0f" if is_current else "")
        )
    )
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


def _nav1_connector(reached: bool) -> None:
    """Pijl tussen twee knooppunten; gekleurd als de flow er al is geweest."""
    col = theme.ACCENT if reached else "#cfcfcf"
    with ui.column().classes("justify-center self-center px-1"):
        ui.icon("east").style(f"color: {col}")


# ============================================================================
# /nav2 — Gefaseerde zijbalk
# ============================================================================

_NAV2_GROUPS = [
    ("Voorbereiden", "edit_note", STAGES[0:3]),
    ("Voorspellen", "play_circle", STAGES[3:4]),
    ("Analyseren", "query_stats", STAGES[4:5] + [BENCHMARK]),
]


def _nav2() -> None:
    ui.colors(**theme.QUASAR_COLORS)
    _appbar("Alternatief 2 — Gefaseerde zijbalk")
    _switcher("/nav2")

    with (
        ui.left_drawer(fixed=False).classes("bg-grey-1 gap-2 p-2").style("width: 280px")
    ):
        ui.label("PIPELINE").classes("text-xs uppercase opacity-50 px-2 pt-1")
        for title, gicon, stages in _NAV2_GROUPS:
            done = sum(1 for s in stages if s.state == "done")
            total = len(stages)
            with ui.expansion(value=True).classes("w-full").props("dense") as exp:
                with exp.add_slot("header"):
                    with ui.column().classes("w-full gap-1"):
                        with ui.row().classes("items-center gap-2 w-full no-wrap"):
                            ui.icon(gicon).style(f"color: {theme.PRIMARY}")
                            ui.label(title).classes("text-sm font-medium")
                            ui.space()
                            ui.label(f"{done}/{total}").classes("text-xs opacity-50")
                        # Dunne voortgangsbalk: aandeel afgeronde stappen in de fase.
                        ui.linear_progress(
                            value=done / total, show_value=False
                        ).props("rounded size=4px").classes("w-full").style(
                            f"color: {theme.POSITIVE}"
                        )
                for st in stages:
                    _nav2_item(st)


def _nav2_item(st: Stage) -> None:
    color, sicon = _STATE_TINT[st.state]
    is_current = st.state == "current"
    with (
        ui.row()
        .classes(
            "items-center gap-2 w-full no-wrap rounded px-2 py-2 "
            "cursor-pointer hover:bg-grey-2"
        )
        .style(
            (f"background: {theme.ACCENT}14; border-left: 3px solid {theme.ACCENT}")
            if is_current
            else ""
        )
    ):
        ui.icon(sicon).classes("text-sm").style(f"color: {color}")
        ui.icon(st.icon).style(f"color: {theme.ACCENT if is_current else theme.INK}")
        with ui.column().classes("gap-0"):
            ui.label(st.label).classes(
                "text-sm " + ("font-medium" if is_current else "")
            )
            ui.label(st.blurb).classes("text-xs opacity-50 leading-tight")


def _nav2_content() -> None:
    with ui.column().classes("w-full max-w-4xl mx-auto p-6 gap-4"):
        ui.label("Gefaseerde zijbalk").classes("text-2xl font-bold")
        ui.label(
            "De acht bestemmingen zijn gegroepeerd in de drie pipeline-fasen. "
            "Elke groep toont zijn eigen voortgang, zodat je ziet waar je bent "
            "zonder de flow uit het oog te verliezen."
        ).classes("text-sm opacity-70 -mt-2")
        with ui.card().classes("w-full"):
            ui.label("Huidige stap · Filteren").classes("text-lg font-medium")
            ui.label(
                "Scoop op opleiding, herkomst en examentype voordat je "
                "de modellen draait."
            ).classes("text-sm opacity-70")


# ============================================================================
# /nav3 — Tabbladen + commandopalet
# ============================================================================


def _nav3() -> None:
    ui.colors(**theme.QUASAR_COLORS)

    palette = _nav3_palette()

    with (
        ui.header()
        .classes("items-center justify-between q-px-md")
        .style(f"background: {theme.PRIMARY}; border-bottom: 3px solid {theme.ACCENT}")
    ):
        with ui.row().classes("items-center gap-3 no-wrap"):
            ui.image("/gui-assets/npuls-logo-white.svg").classes("w-8 h-8")
            ui.label("Studentprognose").classes("text-lg font-medium text-white")
        # Commando-trigger midden-rechts.
        with (
            ui.row()
            .classes("items-center gap-2 rounded px-3 py-1 cursor-pointer")
            .style("background: rgba(255,255,255,0.12)")
            .on("click", palette.open)
        ):
            ui.icon("search").classes("text-white text-sm")
            ui.label("Zoek of spring…").classes("text-white text-sm opacity-80")
            ui.label("⌘K").classes("text-white text-xs opacity-50 border rounded px-1")

    # Tabbladen-rij onder de header.
    with (
        ui.row()
        .classes("w-full items-center q-px-md no-wrap overflow-x-auto")
        .style("background: white; border-bottom: 1px solid #e8e8e8")
    ):
        for st in [*STAGES, BENCHMARK]:
            _nav3_tab(st)

    with ui.column().classes("w-full max-w-4xl mx-auto p-6 gap-4"):
        ui.label("Tabbladen + commandopalet").classes("text-2xl font-bold")
        ui.label(
            "Vaste tabbladen bovenaan voor snel wisselen; ⌘K opent een "
            "commandopalet waarmee ervaren gebruikers zonder muis naar elke "
            "stap of actie springen."
        ).classes("text-sm opacity-70 -mt-2")
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("filter_alt").style(f"color: {theme.ACCENT}")
                ui.label("Filteren").classes("text-lg font-medium")
            ui.label(
                "Actief tabblad. Druk op ⌘K om te springen of een actie uit te voeren."
            ).classes("text-sm opacity-70")
        ui.button("Open commandopalet (⌘K)", icon="bolt", on_click=palette.open).props(
            "outline no-caps"
        )

    _switcher("/nav3")

    ui.keyboard(
        on_key=lambda e: (
            palette.open()
            if e.key == "k"
            and (e.modifiers.ctrl or e.modifiers.meta)
            and e.action.keydown
            else None
        )
    )


def _nav3_tab(st: Stage) -> None:
    is_current = st.state == "current"
    is_done = st.state == "done"
    with (
        ui.row()
        .classes("items-center gap-1 no-wrap px-3 py-3 cursor-pointer")
        .style(
            "border-bottom: 2px solid "
            + (theme.ACCENT if is_current else "transparent")
        )
    ):
        # Afgeronde stappen krijgen een groen vinkje; dat maakt de
        # pipeline-voortgang leesbaar zonder de tabbladen te verzwaren.
        if is_done:
            ui.icon("check_circle").classes("text-sm").style(
                f"color: {theme.POSITIVE}"
            )
        else:
            ui.icon(st.icon).classes("text-sm").style(
                f"color: {theme.ACCENT if is_current else theme.MUTED}"
            )
        ui.label(st.label).classes(
            "text-sm "
            + ("font-medium" if is_current else "opacity-70")
        )


def _nav3_palette() -> ui.dialog:
    with ui.dialog() as dialog, ui.card().classes("w-[520px] p-0"):
        with ui.row().classes("items-center gap-2 w-full no-wrap px-3 py-2 border-b"):
            ui.icon("search").classes("opacity-50")
            ui.input(placeholder="Typ om te filteren…").props(
                "borderless autofocus"
            ).classes("flex-1")
            ui.label("esc").classes("text-xs opacity-40 border rounded px-1")
        with ui.column().classes("w-full gap-0 p-1"):
            ui.label("Ga naar").classes("text-xs uppercase opacity-40 px-2 pt-1")
            for st in [*STAGES, BENCHMARK]:
                with (
                    ui.row()
                    .classes(
                        "items-center gap-3 w-full no-wrap px-2 py-2 "
                        "rounded cursor-pointer hover:bg-grey-2"
                    )
                    .on("click", dialog.close)
                ):
                    ui.icon(st.icon).style(f"color: {theme.PRIMARY}")
                    ui.label(st.label).classes("text-sm")
                    ui.space()
                    if st.num:
                        ui.label(f"stap {st.num}").classes("text-xs opacity-40")

            ui.separator().classes("my-1")
            ui.label("Acties").classes("text-xs uppercase opacity-40 px-2 pt-1")
            # Het palet is niet alleen navigatie: sneltoetsgebruikers voeren hier
            # ook de kernacties van de pipeline uit zonder de muis.
            actions = [
                ("Draai volledige pipeline", "rocket_launch", "⌘⏎"),
                ("Open laatste dashboard", "insights", None),
                ("Start benchmark", "science", None),
            ]
            for label, icon, key in actions:
                with (
                    ui.row()
                    .classes(
                        "items-center gap-3 w-full no-wrap px-2 py-2 "
                        "rounded cursor-pointer hover:bg-grey-2"
                    )
                    .on("click", dialog.close)
                ):
                    ui.icon(icon).style(f"color: {theme.ACCENT}")
                    ui.label(label).classes("text-sm")
                    ui.space()
                    if key:
                        ui.label(key).classes(
                            "text-xs opacity-40 border rounded px-1"
                        )
    return dialog


# ============================================================================
# /nav4 — Missiecontrole-hub
# ============================================================================


def _nav4() -> None:
    ui.colors(**theme.QUASAR_COLORS)
    _appbar("Alternatief 4 — Missiecontrole-hub")
    _switcher("/nav4")

    with ui.column().classes("w-full max-w-6xl mx-auto p-6 gap-5"):
        with ui.row().classes("items-end justify-between w-full"):
            with ui.column().classes("gap-0"):
                ui.label("Missiecontrole").classes("text-2xl font-bold")
                ui.label(
                    "De hele pipeline in één overzicht — spring direct naar elke fase."
                ).classes("text-sm opacity-70")
            ui.button("Draai volledige pipeline", icon="rocket_launch").props(
                "unelevated no-caps"
            )

        # KPI-strip.
        with ui.row().classes("w-full gap-3 no-wrap"):
            _nav4_kpi("Project", "sp-demo-x9f2", "folder", theme.PRIMARY)
            _nav4_kpi("Laatste run", "week 6 · 2024", "schedule", theme.SECONDARY)
            _nav4_kpi("Ensemble-fout (MAPE)", "4,2%", "track_changes", theme.POSITIVE)
            _nav4_kpi("Voortgang", "2 / 5 fasen", "donut_large", theme.ACCENT)

        # Fase-kaarten in pipeline-volgorde.
        with ui.grid(columns=3).classes("w-full gap-4"):
            for st in STAGES:
                _nav4_card(st)
            _nav4_card(BENCHMARK, tool=True)


def _nav4_kpi(label: str, value: str, icon: str, color: str) -> None:
    with ui.card().classes("flex-1"):
        with ui.row().classes("items-center gap-2 no-wrap"):
            with (
                ui.element("div")
                .classes("flex items-center justify-center rounded")
                .style(f"width: 36px; height: 36px; background: {color}1a")
            ):
                ui.icon(icon).style(f"color: {color}")
            with ui.column().classes("gap-0"):
                ui.label(label).classes("text-xs opacity-60")
                ui.label(value).classes("text-base font-medium leading-tight")


def _nav4_card(st: Stage, *, tool: bool = False) -> None:
    color, sicon = _STATE_TINT[st.state]
    labels = {"done": "Afgerond", "current": "Nu bezig", "todo": "Te doen"}
    with (
        ui.card()
        .classes("w-full cursor-pointer transition-all hover:shadow-md")
        .style(
            "border: 1px solid "
            + (f"{theme.ACCENT}" if st.state == "current" else "#e8e8e8")
        )
    ):
        with ui.row().classes("items-center justify-between w-full no-wrap"):
            with (
                ui.element("div")
                .classes("flex items-center justify-center rounded-lg")
                .style(f"width: 44px; height: 44px; background: {color}1a")
            ):
                ui.icon(st.icon).classes("text-2xl").style(f"color: {color}")
            if tool:
                ui.badge("Tool", color="secondary").classes("text-xs")
            else:
                with ui.row().classes("items-center gap-1 no-wrap"):
                    ui.icon(sicon).classes("text-sm").style(f"color: {color}")
                    ui.label(labels[st.state]).classes("text-xs").style(
                        f"color: {color}"
                    )
        ui.label((f"Stap {st.num} · " if st.num else "") + st.label).classes(
            "text-base font-medium mt-1"
        )
        ui.label(st.blurb).classes("text-xs opacity-60 leading-snug")
        # Actie-affordance: maakt duidelijk dat de kaart klikbaar is.
        action = (
            "Nu bezig — verdergaan"
            if st.state == "current"
            else "Openen"
            if not tool
            else "Vergelijken"
        )
        ui.separator().classes("my-1")
        with ui.row().classes("items-center gap-1 no-wrap").style(
            f"color: {theme.ACCENT if st.state == 'current' else theme.PRIMARY}"
        ):
            ui.label(action).classes("text-xs font-medium")
            ui.icon("arrow_forward").classes("text-sm")


# --- Registratie --------------------------------------------------------------


def create() -> None:
    """Registreer de vier prototype-routes ``/nav1``–``/nav4``."""
    ui.page("/nav1")(_nav1)
    ui.page("/nav2")(lambda: (_nav2(), _nav2_content()))
    ui.page("/nav3")(_nav3)
    ui.page("/nav4")(_nav4)
