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

_ACTION_CARDS = [
    ("tune", "Configuratie", "/config", "Model- en pipelineparameters"),
    ("play_circle", "Uitvoeren", "/run", "Voorspelling draaien"),
    ("insights", "Resultaten", "/output", "Bekijk de laatste uitvoer"),
]


def create() -> None:
    """Registreer de route ``/``."""
    nav.register_route("/")

    @ui.page("/")
    def home_page() -> None:
        ui.add_head_html("""<style>
@keyframes sp-shimmer {
    0%   { background-position: 150% 50%; }
    100% { background-position: -50% 50%; }
}
.sp-demo-btn {
    background: linear-gradient(100deg,
        #a84a15 0%, #dd784b 30%, #ffd166 52%, #dd784b 70%, #a84a15 100%) !important;
    background-size: 300% 100% !important;
    animation: sp-shimmer 1.6s linear infinite !important;
    color: #fff !important;
    font-weight: 700 !important;
    letter-spacing: 0.4px !important;
    box-shadow: 0 4px 18px rgba(221,120,75,0.55), inset 0 1px 0 rgba(255,255,255,0.18) !important;
}
.sp-action-card {
    transition: box-shadow 0.2s, transform 0.15s;
    cursor: pointer;
}
.sp-action-card:hover {
    box-shadow: 0 10px 28px rgba(0,0,0,0.14) !important;
    transform: translateY(-3px);
}
.sp-demo-step-dot {
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; font-size: 12px; font-weight: 700; color: #fff;
}
</style>""")
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
        self._step_n: int = 0
        self._render_intro()

    def _render_intro(self) -> None:
        self._container.clear()
        with self._container:
            if STATE.is_initialised:
                self._render_dashboard()
            else:
                with ui.row().classes("items-center gap-2"):
                    ui.icon("info").style(f"color: {theme.ACCENT}")
                    ui.label(
                        "Nieuw hier? Begin met 'Project opzetten' of probeer eerst de demo."
                    ).classes("text-sm opacity-70")
                self._render_tracks_explainer()
                self._render_cta()

    # ── Dashboard (terugkerende gebruiker) ────────────────────────────────────

    def _render_dashboard(self) -> None:
        """Projectdashboard: name, snelkoppelingen en CTA."""
        project_name = os.path.basename(STATE.project_dir or "")

        # Project-statuskaart
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-4 no-wrap w-full"):
                with ui.element("div").style(
                    f"width:52px;height:52px;border-radius:12px;"
                    f"background:{theme.ACCENT}1a;"
                    f"display:flex;align-items:center;justify-content:center;flex-shrink:0"
                ):
                    ui.icon("folder_open").classes("text-3xl").style(
                        f"color:{theme.ACCENT}"
                    )
                with ui.column().classes("gap-0 flex-1 overflow-hidden"):
                    ui.label(project_name).classes("text-xl font-bold")
                status_badge("ready")

        # Prominente CTA — direct naar de pipeline
        ui.button(
            "Ga naar Uitvoeren",
            icon="play_arrow",
            on_click=lambda: ui.navigate.to("/run"),
        ).props("unelevated")

        # Snelkoppelingen
        with ui.row().classes("w-full gap-3"):
            for icon, label, route, desc in _ACTION_CARDS:
                with ui.card().classes("flex-1 sp-action-card") as card:
                    card.on("click", lambda r=route: ui.navigate.to(r))
                    with ui.column().classes(
                        "items-center text-center gap-2 py-5 px-3"
                    ):
                        with ui.element("div").style(
                            f"width:48px;height:48px;border-radius:50%;"
                            f"background:{theme.ACCENT}18;"
                            f"display:flex;align-items:center;justify-content:center"
                        ):
                            ui.icon(icon).classes("text-2xl").style(
                                f"color:{theme.ACCENT}"
                            )
                        ui.label(label).classes("font-semibold text-sm")
                        ui.label(desc).classes("text-xs opacity-60")

    # ── Tracks-uitleg ─────────────────────────────────────────────────────────

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

    # ── CTA (nieuwe gebruiker) ────────────────────────────────────────────────

    def _render_cta(self) -> None:
        with ui.card().classes("w-full"):
            with ui.column().classes("w-full items-center text-center gap-3 py-12"):
                ui.icon("rocket_launch").classes("text-6xl opacity-40")
                ui.label("Eigen project opzetten").classes("text-xl font-medium")
                ui.label(
                    "Werk je met je eigen data? Zet een projectmap op "
                    "met configuratie en de juiste mappenstructuur."
                ).classes("text-sm opacity-70 max-w-md")
                ui.button(
                    "Project opzetten",
                    on_click=lambda: ui.navigate.to("/wizard"),
                ).props("unelevated")
                ui.button(
                    "Probeer direct met demodata",
                    icon="bolt",
                    on_click=self._run_demo,
                ).props("unelevated").classes("sp-demo-btn")

    # ── Demo-uitvoering ───────────────────────────────────────────────────────

    async def _run_demo(self) -> None:
        """Voer init → demodata → pipeline automatisch uit in een tijdelijke map."""
        project_dir = tempfile.mkdtemp(prefix="sp-demo-")
        STATE.project_dir = project_dir
        self._step_n = 0

        self._container.clear()
        with self._container:
            with ui.card().classes("w-full"):
                # Header
                with ui.row().classes("items-center gap-3 no-wrap mb-3"):
                    with ui.element("div").style(
                        f"width:44px;height:44px;border-radius:10px;"
                        f"background:{theme.ACCENT}1a;"
                        "display:flex;align-items:center;justify-content:center;flex-shrink:0"
                    ):
                        ui.icon("rocket_launch").classes("text-2xl").style(
                            f"color:{theme.ACCENT}"
                        )
                    with ui.column().classes("gap-0"):
                        ui.label("Demo wordt uitgevoerd").classes(
                            "text-lg font-semibold"
                        )
                        ui.label(project_dir).classes("text-xs font-mono opacity-50")

                ui.separator().classes("my-1")

                # Stappen
                self._steps = ui.column().classes("w-full gap-2 my-2")
                self._progress = ui.linear_progress(value=0.0, show_value=False)
                self._progress.set_visibility(False)
                self._pipeline_progress = ProgressCard()

                # Terminal output ingeklapt — analisten hoeven de rauwe log niet te zien
                with ui.expansion("Uitvoerlog", icon="terminal").props(
                    "dense"
                ).classes("w-full mt-2"):
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
        """Toon het succesresultaat met een resultatenknop + auto-navigatie."""
        self._step("Klaar!", done=True)
        with self._steps:
            ui.button(
                "Bekijk resultaten",
                icon="insights",
                on_click=lambda: ui.navigate.to("/output"),
            ).props("unelevated").classes("mt-1")
        ui.timer(0.8, lambda: ui.navigate.to("/output"), once=True)

    def _step(self, text: str, *, done: bool = False) -> None:
        self._step_n += 1
        n = self._step_n
        with self._steps:
            with ui.row().classes("items-center gap-3 py-1"):
                if done:
                    ui.icon("check_circle").props("color=positive").classes("text-2xl")
                else:
                    with ui.element("div").classes("sp-demo-step-dot").style(
                        f"background:{theme.ACCENT}"
                    ):
                        ui.label(str(n)).style(
                            "color:white;font-size:12px;font-weight:700"
                        )
                ui.label(text).classes("text-sm font-medium" if done else "text-sm")

    async def _download_demodata(self, project_dir: str) -> None:
        dest = os.path.join(project_dir, "data", "input_raw")
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
