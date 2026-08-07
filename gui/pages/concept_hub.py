"""Concept-hub — overzicht van alle concept-features in preview.

Bereikbaar via /concept. Dient als landingspagina voor de concept-sectie
in de zijbalk.
"""

from __future__ import annotations

from nicegui import ui

from gui import nav, theme
from gui.components.layout import page_shell
from gui.components.states import concept_banner

_ACCENT = theme.ACCENT

_FEATURES = [
    {
        "route": "/scenarios",
        "icon": "analytics",
        "title": "Wat-als analyse",
        "body": (
            "Bekijk de modelonzekerheid via pessimistische en optimistische scenario's, "
            "vergelijk met een historisch jaar en toets of uw instellingsdoel haalbaar is."
        ),
    },
    {
        "route": "/peer-benchmark",
        "icon": "leaderboard",
        "title": "Peer benchmark",
        "body": (
            "Vergelijk uw aanmeldingen direct met een gekozen instelling en opleiding. "
            "Zie wie voor- of achterloopt en hoe het patroon zich de afgelopen weken ontwikkelt."
        ),
    },
    {
        "route": "/api",
        "icon": "api",
        "title": "API & integraties",
        "body": (
            "Embed prognoses in Power BI, Tableau of eigen systemen via een REST API. "
            "Inclusief codevoorbeelden voor Python, R en Power BI (M)."
        ),
    },
    {
        "route": "/explainability",
        "icon": "psychology",
        "title": "Verklaarbaar AI",
        "body": (
            "Begrijp waarom het model de prognose geeft die het geeft. "
            "Factorgewichten, historische nauwkeurigheid en een vertrouwensmeter per opleiding."
        ),
    },
]


def create() -> None:
    nav.register_route("/concept")

    @ui.page("/concept")
    def concept_hub_page() -> None:
        with page_shell(active="/concept", title="Concept-features", show_stepper=False):
            concept_banner()

            ui.label("Concept-features — preview").classes("text-lg font-medium mt-2")
            ui.label(
                "Deze functies zijn interactieve mockups voor evaluatie en feedback. "
                "Alle data is illustratief. Geef feedback via de knop linksonder."
            ).classes("text-sm opacity-55 -mt-1 leading-relaxed")

            with ui.row().classes("w-full gap-4 flex-wrap mt-2"):
                for feat in _FEATURES:
                    with (
                        ui.card()
                        .classes("flex-1 cursor-pointer hover:shadow-md transition-shadow")
                        .style("min-width:220px;border-top:3px solid #E53935;")
                        .on("click", lambda r=feat["route"]: ui.navigate.to(r))
                    ):
                        with ui.row().classes("items-center gap-2 mb-2"):
                            ui.icon(feat["icon"]).classes("text-2xl").style("color:#E53935;")
                            ui.label(feat["title"]).classes("font-semibold text-sm")
                        ui.label(feat["body"]).classes("text-xs opacity-60 leading-relaxed")
                        with ui.row().classes("items-center gap-1 mt-3"):
                            ui.label("Bekijken").classes("text-xs font-medium").style(
                                "color:#E53935;"
                            )
                            ui.icon("arrow_forward").classes("text-sm").style("color:#E53935;")
