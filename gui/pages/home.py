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

from gui import demodata, filtering_io, nav, theme, tracks
from gui.components import pipeline_rail
from gui.components.layout import page_shell
from gui.components.log_stream import ProcessPanel
from gui.components.progress_card import ProgressCard
from gui.components.states import error_banner, section_title, status_badge
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
            # Pipeline-hub — de primaire navigatie. Klik een stap om te starten;
            # de status per stap volgt de echte projectstatus.
            section_title(
                "De pipeline",
                "Klik een stap om te beginnen — of probeer eerst de demo hieronder.",
            )
            pipeline_rail.render()

            # Één-klik demo.
            with (
                ui.card()
                .classes("w-full")
                .style(
                    f"background: {theme.ACCENT}14; border: 1px solid {theme.ACCENT}55"
                )
            ):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("bolt").classes("text-2xl").style(f"color: {theme.ACCENT}")
                    ui.label("Direct proberen").classes("text-lg font-medium")
                ui.label(
                    "Draai de pipeline met meegeleverde demodata — in een "
                    "tijdelijke map, zonder iets in te stellen. De demo draait het "
                    "cumulatieve spoor (de demodata bevat geen individuele "
                    "aanmelddata) en beperkt zich voor de snelheid tot een subset "
                    "(Master, Niet-EER)."
                ).classes("text-sm opacity-70")
                ui.button(
                    "Probeer direct met demodata",
                    icon="rocket_launch",
                    on_click=self._run_demo,
                ).props("unelevated")

            # Actief project (alleen als er een project is; de Project-knoop in de
            # rail dekt het opzetten af voor nieuwe gebruikers).
            if STATE.is_initialised:
                with ui.card().classes("w-full"):
                    with ui.row().classes("items-center justify-between w-full"):
                        section_title("Actief project")
                        status_badge("ready")
                    ui.label(STATE.project_dir).classes("text-sm opacity-70")

            # Hoe werkt het — de drie voorspelsporen (diepere uitleg).
            self._render_tracks_explainer()

    def _render_tracks_explainer(self) -> None:
        """Leg de drie voorspelsporen uit; knop leidt naar het schema (methodologie)."""
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("alt_route").classes("text-xl").style(f"color: {theme.ACCENT}")
                ui.label("Hoe werkt het? — drie voorspelsporen").classes(
                    "text-lg font-medium"
                )
            ui.label(
                "De tool voorspelt studentinstroom via drie sporen. Kies er één "
                "bij Uitvoeren:"
            ).classes("text-sm opacity-70")
            for t in tracks.TRACKS:
                with ui.row().classes("items-start gap-3 no-wrap w-full"):
                    ui.icon(t.icon).classes("text-xl mt-1").style(
                        f"color: {theme.ACCENT}"
                    )
                    with ui.column().classes("gap-0"):
                        with ui.row().classes("items-center gap-2 no-wrap"):
                            ui.label(t.label).classes("font-medium")
                            if t.label == tracks.RECOMMENDED:
                                ui.badge("aanbevolen", color="accent").props("outline")
                        ui.label(t.short).classes("text-sm opacity-70")
            ui.button(
                "Methodologie",
                icon="schema",
                on_click=lambda: ui.navigate.to("/methodologie"),
            ).props("outline color=accent").classes("mt-1")

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
                self._pipeline_progress = ProgressCard()
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

            # 3. pipeline — start voortgangsbalk
            self._step("Voorspelling draaien (cumulatief spoor)…")
            self._pipeline_progress.start()
            rc = await panel.run(
                ["-d", "c", "-w", _DEMO_WEEK, "-y", _DEMO_YEAR, "--yes"],
                cwd=project_dir,
                on_line=self._pipeline_progress.on_line,
            )
            self._pipeline_progress.complete(success=rc == 0)
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
