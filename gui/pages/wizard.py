"""Init-wizard (#266): zet een nieuw project op via `studentprognose init`.

Drie sub-stappen: (1) projectmap kiezen, (2) bevestigen wat aangemaakt wordt,
(3) aanmaken + optioneel demodata downloaden. Streamt de CLI-uitvoer live.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from nicegui import ui

from gui import demodata, nav
from gui.components.file_picker import DirectoryPicker
from gui.components.layout import page_shell
from gui.components.log_stream import ProcessPanel
from gui.components.states import info_banner, section_title
from gui.state import STATE

#: Mappen die `studentprognose init` aanmaakt (voor de bevestigingsstap).
_CREATED_DIRS = [
    "configuration/filtering/",
    "data/input/",
    "data/input_raw/telbestanden/",
    "data/output/",
]


def create() -> None:
    """Registreer de route ``/wizard``."""
    nav.register_route("/wizard")

    @ui.page("/wizard")
    def wizard_page() -> None:
        with page_shell(active="/wizard", title="Project opzetten"):
            section_title(
                "Nieuw project",
                "Zet een projectmap op met configuratie en de juiste structuur.",
            )
            _WizardView()


class _WizardView:
    """Houdt de wizard-state en rendert de drie stappen."""

    def __init__(self) -> None:
        self._project_dir = os.getcwd()
        self._picker = DirectoryPicker(on_select=self._on_dir_selected)
        self._build()

    def _build(self) -> None:
        with ui.stepper().props("vertical").classes("w-full") as self._stepper:
            # --- Stap 1: map kiezen ------------------------------------------
            with ui.step("Projectmap kiezen"):
                ui.label(
                    "Kies de map waarin het project wordt aangemaakt. "
                    "Een bestaande configuratie wordt niet overschreven."
                ).classes("text-sm opacity-70")
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    self._path_input = (
                        ui.input("Projectmap", value=self._project_dir)
                        .props("outlined dense")
                        .classes("grow")
                    )
                    ui.button(
                        "Bladeren",
                        icon="folder_open",
                        on_click=lambda: self._picker.open(self._path_input.value),
                    ).props("outline")
                with ui.stepper_navigation():
                    ui.button("Volgende", on_click=self._goto_confirm)

            # --- Stap 2: bevestigen ------------------------------------------
            with ui.step("Bevestigen"):
                self._conflict_slot = ui.column().classes("w-full")
                ui.label("De volgende structuur wordt aangemaakt:").classes(
                    "text-sm opacity-70"
                )
                self._confirm_path = ui.label().classes("text-sm font-mono")
                with ui.column().classes("gap-0 ml-2"):
                    for d in _CREATED_DIRS:
                        with ui.row().classes("items-center gap-1"):
                            ui.icon("folder").classes("text-amber-8 text-sm")
                            ui.label(d).classes("text-sm font-mono")
                    with ui.row().classes("items-center gap-1"):
                        ui.icon("description").classes("opacity-60 text-sm")
                        ui.label("configuration/configuration.json").classes(
                            "text-sm font-mono"
                        )
                with ui.stepper_navigation():
                    ui.button("Terug", on_click=self._stepper.previous).props("flat")
                    ui.button("Volgende", on_click=self._stepper.next)

            # --- Stap 3: aanmaken --------------------------------------------
            with ui.step("Aanmaken"):
                self._demo_checkbox = ui.checkbox(
                    "Demodata downloaden (≈4 MB, om het model direct te proberen)",
                    value=False,
                )
                self._demo_progress = ui.linear_progress(value=0.0, show_value=False)
                self._demo_progress.set_visibility(False)
                self._panel = ProcessPanel()
                with ui.stepper_navigation():
                    ui.button("Terug", on_click=self._stepper.previous).props("flat")
                    self._create_btn = ui.button(
                        "Project aanmaken",
                        icon="build",
                        on_click=self._create_project,
                    )
                self._next_slot = ui.row().classes("mt-2")

    # --- Stap-overgangen ------------------------------------------------------

    def _goto_confirm(self) -> None:
        self._project_dir = os.path.abspath(self._path_input.value.strip())
        self._confirm_path.set_text(self._project_dir)

        # Waarschuw als de map al een project bevat: init overschrijft niets,
        # maar de gebruiker moet weten dat de bestaande configuratie blijft staan.
        self._conflict_slot.clear()
        existing = os.path.join(
            self._project_dir, "configuration", "configuration.json"
        )
        if os.path.isfile(existing):
            with self._conflict_slot:
                info_banner(
                    "Deze map bevat al een project. De bestaande configuratie "
                    "blijft behouden — er wordt niets overschreven."
                )

        self._stepper.next()

    def _on_dir_selected(self, path: str) -> None:
        self._path_input.set_value(path)

    # --- Uitvoeren ------------------------------------------------------------

    async def _create_project(self) -> None:
        self._create_btn.props("loading")
        self._next_slot.clear()
        try:
            os.makedirs(self._project_dir, exist_ok=True)
            returncode = await self._panel.run(["init"], cwd=self._project_dir)

            if returncode == 0 and self._demo_checkbox.value:
                await self._download_demodata()

            if returncode == 0:
                STATE.project_dir = self._project_dir
                self._show_next()
        finally:
            self._create_btn.props(remove="loading")

    async def _download_demodata(self) -> None:
        dest = os.path.join(self._project_dir, "data", "input_raw")
        self._demo_progress.set_visibility(True)
        holder = {"value": 0.0}
        timer = ui.timer(
            0.1, lambda: self._demo_progress.set_value(max(holder["value"], 0.0))
        )

        def _work() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                zip_path = os.path.join(tmp, "demo-data.zip")
                demodata.download_file(
                    demodata.DEMO_URL,
                    zip_path,
                    progress_cb=lambda f: holder.__setitem__("value", f),
                )
                demodata.extract_zip(zip_path, dest)

        try:
            await asyncio.to_thread(_work)
            self._demo_progress.set_value(1.0)
            with self._next_slot:
                info_banner("Demodata gedownload naar data/input_raw/.")
        except Exception as exc:  # noqa: BLE001 — toon nette melding, geen crash
            from gui.components.states import error_banner

            with self._next_slot:
                error_banner(
                    "Demodata downloaden mislukt.",
                    f"Controleer je internetverbinding. Details: {exc}",
                )
        finally:
            timer.cancel()

    def _show_next(self) -> None:
        with self._next_slot:
            with ui.row().classes("items-center gap-2"):
                ui.icon("check_circle").props("color=positive")
                ui.label("Project klaar.").classes("text-positive font-medium")
                ui.button(
                    "Ga naar configuratie",
                    icon="arrow_forward",
                    on_click=lambda: ui.navigate.to("/config"),
                ).props("unelevated")
