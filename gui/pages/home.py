"""Startpagina van de GUI.

Referentie-implementatie van het design system (#272): welkomst, projectstatus
en — bij een leeg project — een leegsituatie met call-to-action naar de eerste
wizard-stap.
"""

from __future__ import annotations

from nicegui import ui

from gui import nav
from gui.components.layout import page_shell
from gui.components.states import empty_state, section_title, status_badge
from gui.state import STATE


def create() -> None:
    """Registreer de route ``/``."""
    nav.register_route("/")

    @ui.page("/")
    def home_page() -> None:
        with page_shell(active="/", title="Start", show_stepper=False):
            ui.label("Studentprognose").classes("text-3xl font-bold")
            ui.label(
                "Grafische interface rond de studentprognose-pipeline. "
                "Zet een project op, stel de configuratie in en draai voorspellingen."
            ).classes("text-base opacity-70")

            if STATE.is_initialised:
                with ui.card().classes("w-full"):
                    with ui.row().classes("items-center justify-between w-full"):
                        section_title("Actief project")
                        status_badge("ready")
                    ui.label(STATE.project_dir).classes("text-sm opacity-70")
            else:
                with ui.card().classes("w-full"):
                    empty_state(
                        icon="rocket_launch",
                        title="Nog geen project",
                        message="Begin met het opzetten van een projectmap met "
                        "configuratie en de juiste mappenstructuur.",
                        action_label="Project opzetten",
                        on_action=lambda: ui.navigate.to("/wizard"),
                    )
