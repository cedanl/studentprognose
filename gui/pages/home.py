"""Startpagina van de GUI.

Referentie-implementatie van het design system (#272): welkomst, projectstatus
en — bij een leeg project — een leegsituatie met call-to-action naar de eerste
wizard-stap. Biedt daarnaast een één-klik demo (#274) die de volledige flow
automatisch met demodata uitvoert in een tijdelijke, willekeurig genoemde map.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from nicegui import ui

from gui import demodata, filtering_io, nav
from gui.components.layout import page_shell
from gui.components.log_stream import ProcessPanel
from gui.components.states import empty_state, error_banner, section_title, status_badge
from gui.state import STATE

#: Jaar/week voor de demo-voorspelling (de demodata dekt 2020–2026).
_DEMO_YEAR = "2024"
_DEMO_WEEK = "6"

#: De demo scopt bewust op een subset (Master + Niet-EER) zodat de volledige
#: pipeline in seconden klaar is in plaats van minuten — snappy én het houdt de
#: verbinding levend (geen client-reconnect halverwege een lange run).
_DEMO_FILTERING = {
    "filtering": {"programme": [], "herkomst": ["Niet-EER"], "examentype": ["Master"]}
}


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
            _HomeView()


class _HomeView:
    """Beheert de startpagina: intro/CTA's en de één-klik demo-uitvoering."""

    def __init__(self) -> None:
        self._container = ui.column().classes("w-full gap-4")
        self._render_intro()

    def _render_intro(self) -> None:
        self._container.clear()
        with self._container:
            # Één-klik demo.
            with ui.card().classes("w-full bg-blue-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("bolt").classes("text-2xl").style("color: #3f51b5")
                    ui.label("Direct proberen").classes("text-lg font-medium")
                ui.label(
                    "Draai de pipeline met meegeleverde demodata — in een "
                    "tijdelijke map, zonder iets in te stellen. Voor de snelheid "
                    "beperkt de demo zich tot een subset (Master, Niet-EER)."
                ).classes("text-sm opacity-70")
                ui.button(
                    "Probeer direct met demodata",
                    icon="rocket_launch",
                    on_click=self._run_demo,
                ).props("unelevated")

            # Projectstatus / eigen project.
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
                        title="Eigen project opzetten",
                        message="Werk je met je eigen data? Zet een projectmap op "
                        "met configuratie en de juiste mappenstructuur.",
                        action_label="Project opzetten",
                        on_action=lambda: ui.navigate.to("/wizard"),
                    )

    async def _run_demo(self) -> None:
        """Voer init → demodata → pipeline automatisch uit in een tijdelijke map."""
        # Willekeurig genoemde tijdelijke map — bewust géén vaste 'studentprognose'-map,
        # zodat niets wordt overschreven en demo-runs elkaar niet bijten (#274).
        project_dir = tempfile.mkdtemp(prefix="sp-demo-")
        STATE.project_dir = project_dir

        self._container.clear()
        with self._container:
            with ui.card().classes("w-full"):
                ui.label("Demo wordt uitgevoerd").classes("text-lg font-medium")
                ui.label(project_dir).classes("text-xs font-mono opacity-60")
                self._steps = ui.column().classes("w-full gap-1")
                self._progress = ui.linear_progress(value=0.0, show_value=False)
                self._progress.set_visibility(False)
                panel = ProcessPanel()
                self._error = ui.column().classes("w-full")

        try:
            # 1. init
            self._step("Projectmap aanmaken…")
            if await panel.run(["init"], cwd=project_dir) != 0:
                return

            # Scope de filtering zodat de demo snel klaar is (subset).
            filtering_io.save_filtering(
                os.path.join(project_dir, "configuration", "filtering", "base.json"),
                _DEMO_FILTERING,
            )

            # 2. demodata
            self._step("Demodata downloaden…")
            await self._download_demodata(project_dir)

            # 3. pipeline
            self._step("Voorspelling draaien…")
            rc = await panel.run(
                ["-d", "c", "-w", _DEMO_WEEK, "-y", _DEMO_YEAR, "--yes"],
                cwd=project_dir,
            )
        except Exception as exc:  # noqa: BLE001 — nette melding i.p.v. crash
            with self._error:
                error_banner("De demo kon niet worden voltooid.", f"Details: {exc}")
            return

        if rc == 0:
            self._on_demo_done()

    def _on_demo_done(self) -> None:
        """Toon het succesresultaat met een resultatenknop + auto-navigatie.

        Auto-navigeren gebeurt via een client-gebonden one-shot timer (niet direct
        vanuit de coroutine): dat voorkomt de 'slot deleted'-race terwijl de
        browser de pagina afbreekt. De knop is de betrouwbare terugval.
        """
        self._step("Klaar!", done=True)
        with self._steps:
            ui.button(
                "Bekijk resultaten",
                icon="insights",
                on_click=lambda: ui.navigate.to("/output"),
            ).props("unelevated")
        ui.timer(0.8, lambda: ui.navigate.to("/output"), once=True)

    def _step(self, text: str, *, done: bool = False) -> None:
        with self._steps:
            with ui.row().classes("items-center gap-2"):
                ui.icon("check_circle" if done else "arrow_right").props(
                    "color=positive" if done else "color=primary"
                )
                ui.label(text).classes("text-sm")

    async def _download_demodata(self, project_dir: str) -> None:
        dest = os.path.join(project_dir, "data", "input_raw")
        # Onbepaalde voortgangsbalk: robuust binnen de async-keten (een op een
        # timer gepolde waarde verliest hier zijn slot-context).
        self._progress.props("indeterminate")
        self._progress.set_visibility(True)

        def _work() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                zip_path = os.path.join(tmp, "demo-data.zip")
                demodata.download_file(demodata.DEMO_URL, zip_path)
                demodata.extract_zip(zip_path, dest)

        try:
            await asyncio.to_thread(_work)
        finally:
            self._progress.set_visibility(False)
