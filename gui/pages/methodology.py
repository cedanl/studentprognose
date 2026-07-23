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


def create() -> None:
    """Registreer de route ``/methodologie``."""
    nav.register_route("/methodologie")

    @ui.page("/methodologie")
    def methodology_page() -> None:
        with page_shell(active="/methodologie", title="Methodologie", show_stepper=False):
            section_title(
                "Methodologie — drie voorspelsporen",
                "Zo voorspelt de tool studentinstroom. Hover over een spoor voor "
                "uitleg, klik erop voor de uitgebreide documentatie.",
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

            ui.button(
                "Terug naar start", icon="arrow_back", on_click=lambda: ui.navigate.to("/")
            ).props("flat")
