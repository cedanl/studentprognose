"""Methodologie-pagina: het interactieve drie-sporen-schema.

Bereikbaar via de "Methodologie"-knop op de startpagina. Toont de flow van de
drie voorspelsporen; hover geeft korte uitleg, klik op een spoor opent de
uitgebreide methodologie-documentatie.
"""

from __future__ import annotations

from nicegui import ui

from gui import nav, theme, viz
from gui.components.layout import page_shell
from gui.components.states import section_title

_TRACK_DETAILS: list[tuple[str, str, str | None, str, str]] = [
    (
        "show_chart",
        "Cumulatief",
        None,
        "Analyseert de wekelijkse aanmeldcurve en extrapoleert naar de verwachte "
        "eindinstroom.",
        "SARIMA modelleert het meerjarige seizoenspatroon in de aanmeldingen; "
        "XGBoost verfijnt de prognose daarna op basis van opleiding-specifieke "
        "kenmerken. Robuust voor opleidingen met ≥ 3 jaar aanmeldhistorie en een "
        "stabiel aanmeldpatroon. Let op bij nieuwe programma's of jaren met een "
        "structuurbreuk — het model heeft dan te weinig historische basis.",
    ),
    (
        "person_search",
        "Individueel",
        None,
        "Schat per aanmelding de kans dat iemand zich daadwerkelijk inschrijft, "
        "en telt die kansen op tot een totaalprognose.",
        "Een XGBoost-classifier weegt kenmerken als herkomst, aanmeldmoment, "
        "opleidingstype en aanmeldhistorie. Naast het totaal geeft dit spoor "
        "inzicht in de verwachte samenstelling van de instroom — handig voor "
        "numerus-fixus-berekeningen en wervingsanalyse. Minder betrouwbaar vroeg "
        "in het seizoen, wanneer het aantal aanmeldingen nog klein is.",
    ),
    (
        "merge_type",
        "Beide",
        "aanbevolen",
        "Combineert cumulatief en individueel in één ensemble-prognose — de "
        "standaardkeuze voor de meeste instellingen.",
        "Twee onafhankelijke modellen compenseren elkaars zwakten: het cumulatieve "
        "spoor is robuust bij weinig aanmeldingen en stabiele patronen; het "
        "individuele spoor reageert sneller op plotselinge verschuivingen in de "
        "samenstelling. Het gewogen gemiddelde is aanpasbaar via de configuratie. "
        "Kies dit spoor tenzij je bewust één model wil isoleren.",
    ),
]


def create() -> None:
    """Registreer de route ``/methodologie``."""
    nav.register_route("/methodologie")

    @ui.page("/methodologie")
    def methodology_page() -> None:
        A = theme.ACCENT
        with page_shell(active="/methodologie", title="Methodologie", show_stepper=False):

            # ── Drie voorspelsporen — tekstuele uitleg ────────────────────────
            section_title(
                "Drie voorspelsporen",
                "Kies hoe de tool instroom voorspelt — of combineer alles in één "
                "ensemble.",
            )
            with ui.card().classes("w-full"):
                with ui.column().classes("w-full gap-0"):
                    for i, (icon, name, badge, lead, body) in enumerate(_TRACK_DETAILS):
                        sep = i < len(_TRACK_DETAILS) - 1
                        with ui.element("div").style(
                            "padding: 16px 18px;"
                            + ("border-bottom: 1px solid #f3f3f3;" if sep else "")
                        ):
                            with ui.row().classes("items-start gap-3 no-wrap"):
                                with ui.element("div").style(
                                    f"width:36px;height:36px;border-radius:9px;"
                                    f"background:{A}12;display:flex;"
                                    "align-items:center;justify-content:center;"
                                    "flex-shrink:0;margin-top:2px;"
                                ):
                                    ui.icon(icon).classes("text-lg").style(
                                        f"color:{A}"
                                    )
                                with ui.column().classes("gap-1"):
                                    with ui.row().classes("items-center gap-2 no-wrap"):
                                        ui.label(name).classes("text-sm font-bold")
                                        if badge:
                                            ui.badge(badge, color="accent").props(
                                                "outline"
                                            )
                                    ui.label(lead).classes(
                                        "text-sm font-medium"
                                    ).style("color:#333; line-height:1.5;")
                                    ui.label(body).classes(
                                        "text-xs leading-relaxed"
                                    ).style("color:#666; margin-top:4px;")

            # ── Visueel sporen-diagram ────────────────────────────────────────
            section_title(
                "Stroomdiagram",
                "Hover over een spoor voor uitleg, klik erop voor de uitgebreide "
                "documentatie.",
            )
            with ui.card().classes("w-full"):
                ui.html(viz.flow_svg()).classes("w-full")
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("Hover over een spoor voor uitleg.").classes(
                        "text-xs opacity-60"
                    )
                    ui.link("Meer over de methodologie →", viz.DOCS).props(
                        "target=_blank"
                    ).classes("text-sm").style(f"color: {theme.ACCENT}")

            with ui.row().classes("gap-2 flex-wrap"):
                ui.button(
                    "Ga naar documentatie",
                    icon="open_in_new",
                    on_click=lambda: ui.navigate.to(
                        "https://cedanl.github.io/studentprognose"
                    ),
                ).props("outline color=accent")
                ui.button(
                    "Naar Uitvoeren",
                    icon="play_arrow",
                    on_click=lambda: ui.navigate.to("/run"),
                ).props("unelevated")
