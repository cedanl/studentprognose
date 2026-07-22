"""Startpagina van de GUI.

Toont een welkomstscherm en de status van het actieve project. De volledige
onboarding (leegsituaties, call-to-action) en navigatie naar de andere pagina's
volgen in #272 en de feature-issues.
"""

from __future__ import annotations

from nicegui import ui

from gui.components.layout import page_shell
from gui.state import STATE


def create() -> None:
    """Registreer de route ``/``."""

    @ui.page("/")
    def home_page() -> None:
        with page_shell(active="/", title="Start"):
            ui.label("Studentprognose").classes("text-3xl font-bold")
            ui.label(
                "Grafische interface rond de studentprognose-pipeline. "
                "Zet een project op, stel de configuratie in en draai voorspellingen."
            ).classes("text-base opacity-70")

            with ui.card().classes("w-full"):
                if STATE.is_initialised:
                    ui.label(f"Actief project: {STATE.project_dir}").classes(
                        "text-positive"
                    )
                else:
                    ui.label("Nog geen project gekozen.").classes("opacity-70")
                    ui.label(
                        "In een volgende stap kun je een project aanmaken of openen."
                    ).classes("text-sm opacity-60")
