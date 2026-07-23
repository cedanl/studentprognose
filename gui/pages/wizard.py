"""Init-wizard (#266, #data-upload): zet een nieuw project op en upload inputdata.

Vier sub-stappen:
  1. Projectmap kiezen
  2. Bevestigen wat aangemaakt wordt
  3. Aanmaken + optioneel demodata downloaden
  4. Modus kiezen + bijbehorende bestanden uploaden en valideren
"""

from __future__ import annotations

import asyncio
import datetime
import os
import tempfile
from collections.abc import Callable

from nicegui import ui

from gui import demodata, nav, theme
from gui.components.file_picker import DirectoryPicker
from gui.components.layout import page_shell
from gui.components.log_stream import ProcessPanel
from gui.components.states import error_banner, info_banner, section_title
from gui.data_upload import (
    FileCheckResult,
    FileStatus,
    save_and_validate_individueel,
    save_and_validate_oktober,
    save_and_validate_telbestand,
    scan_existing_files,
)
from gui.state import STATE

_CREATED_DIRS = [
    "configuration/filtering/",
    "data/input/",
    "data/input_raw/telbestanden/",
    "data/output/",
]

# Visuele configuratie per FileStatus.
_STATUS_VISUAL: dict[FileStatus, tuple[str, str, str]] = {
    FileStatus.CHECKING: ("hourglass_top", theme.INFO,     "Controleren…"),
    FileStatus.VALID:    ("check_circle",  theme.POSITIVE, "Geldig"),
    FileStatus.WARNINGS: ("warning",       theme.WARNING,  "Geldig (met opmerkingen)"),
    FileStatus.ERRORS:   ("error",         theme.NEGATIVE, "Fouten gevonden"),
}

# Beschikbare pipeline-modi met bijbehorende metadata.
_MODE_OPTS: list[tuple[str, str, str, str, str]] = [
    # (key, label, icon, cli_flag, beschrijving)
    ("cumulative", "Cumulatief",  "bar_chart", "-d cumulative", "Alleen telbestanden"),
    ("individual", "Individueel", "person",    "-d individual", "Alleen aanmelddata"),
    ("both",       "Beide",       "bolt",      "-d both",       "Telbestanden + aanmelddata"),
]


def create() -> None:
    """Registreer de route ``/wizard``."""
    nav.register_route("/wizard")

    @ui.page("/wizard")
    def wizard_page() -> None:
        with page_shell(active="/wizard", title="Project opzetten"):
            section_title(
                "Nieuw project",
                "Zet een projectmap op en upload je inputbestanden.",
            )
            _WizardView()


# ---------------------------------------------------------------------------
# Upload-zone component
# ---------------------------------------------------------------------------

class _UploadZone:
    """Upload-zone voor één bestandstype met directe validatiefeedback."""

    def __init__(
        self,
        *,
        title: str,
        description: str,
        hint: str,
        icon: str,
        required: bool,
        accept: str,
        multiple: bool,
        project_dir_getter: Callable[[], str],
        validate_fn: Callable[[str, str, bytes], FileCheckResult],
        on_change: Callable[[], None],
    ) -> None:
        self._project_dir_getter = project_dir_getter
        self._validate_fn = validate_fn
        self._on_change = on_change
        self._results: dict[str, FileCheckResult] = {}
        self._build(title, description, hint, icon, required, accept, multiple)

    # --- Public interface ---------------------------------------------------

    def load_existing(
        self,
        existing: dict[str, FileCheckResult] | FileCheckResult | None,
    ) -> None:
        if isinstance(existing, dict):
            self._results = dict(existing)
        elif existing is not None:
            self._results[existing.filename] = existing
        self._refresh_results()

    def set_required(self, required: bool) -> None:
        """Wissel het 'Vereist'/'Optioneel'-badge live."""
        text = "Vereist" if required else "Optioneel"
        color = "accent" if required else "grey-6"
        self._badge.set_text(text)
        self._badge.props(f"color={color}")

    @property
    def has_valid(self) -> bool:
        return any(
            r.status in (FileStatus.VALID, FileStatus.WARNINGS)
            for r in self._results.values()
        )

    @property
    def count(self) -> int:
        return len(self._results)

    @property
    def valid_count(self) -> int:
        return sum(
            1 for r in self._results.values()
            if r.status in (FileStatus.VALID, FileStatus.WARNINGS)
        )

    # --- Build -------------------------------------------------------------

    def _build(
        self,
        title: str,
        description: str,
        hint: str,
        icon: str,
        required: bool,
        accept: str,
        multiple: bool,
    ) -> None:
        with (
            ui.card()
            .classes("w-full")
            .style("border: 1px solid #e8e8e8; border-radius: 8px;")
        ):
            with ui.row().classes("w-full items-start justify-between gap-2 mb-1"):
                with ui.row().classes("items-center gap-2 no-wrap"):
                    ui.icon(icon).classes("text-2xl").style(f"color: {theme.ACCENT}")
                    with ui.column().classes("gap-0"):
                        ui.label(title).classes("font-medium")
                        ui.label(description).classes("text-xs opacity-60")
                badge_text = "Vereist" if required else "Optioneel"
                badge_color = "accent" if required else "grey-6"
                self._badge = ui.badge(badge_text).props(
                    f"color={badge_color}"
                ).classes("text-xs self-start mt-1 flex-none")

            if hint:
                ui.label(hint).classes("text-xs opacity-50 mb-2").style(
                    "font-family: monospace"
                )

            (
                ui.upload(
                    on_upload=self._handle_upload,
                    multiple=multiple,
                    auto_upload=True,
                )
                .props(f"flat color=grey-3 text-color=grey-9 accept='{accept}'")
                .classes("w-full")
                .style(
                    "border: 2px dashed #d0d0d0; border-radius: 6px; min-height: 72px;"
                )
            )

            self._results_slot = ui.column().classes("w-full gap-1 mt-2")

    # --- Upload-handler (async) -------------------------------------------

    async def _handle_upload(self, e) -> None:
        content = await e.file.read()
        filename = e.file.name

        self._results[filename] = FileCheckResult(
            filename=filename, status=FileStatus.CHECKING
        )
        self._refresh_results()

        result = await asyncio.to_thread(
            self._validate_fn,
            self._project_dir_getter(),
            filename,
            content,
        )

        self._results[filename] = result
        self._refresh_results()
        self._on_change()

    # --- UI-rendering -------------------------------------------------------

    def _refresh_results(self) -> None:
        self._results_slot.clear()
        with self._results_slot:
            for result in self._results.values():
                self._render_file_row(result)

    def _render_file_row(self, result: FileCheckResult) -> None:
        icon_name, color, status_text = _STATUS_VISUAL.get(
            result.status, ("radio_button_unchecked", theme.MUTED, ""),
        )
        is_checking = result.status == FileStatus.CHECKING

        with ui.column().classes("w-full gap-0"):
            with ui.row().classes("w-full items-center gap-2 no-wrap py-1"):
                if is_checking:
                    ui.spinner(size="xs").style(f"color: {color}")
                else:
                    ui.icon(icon_name).style(f"color: {color}").classes("text-base flex-none")
                ui.label(result.filename).classes("text-sm font-mono grow truncate")
                meta_parts = [status_text]
                if result.row_count is not None:
                    meta_parts.append(f"{result.row_count:,} rijen".replace(",", "."))
                ui.label(" · ".join(meta_parts)).classes("text-xs flex-none").style(
                    f"color: {color}"
                )

            if result.hard_errors or result.soft_errors:
                with ui.column().classes("ml-6 gap-0.5 mb-1"):
                    for msg in result.hard_errors + result.soft_errors:
                        with ui.row().classes("items-start gap-1 no-wrap"):
                            ui.icon("subdirectory_arrow_right").classes(
                                "text-xs flex-none mt-0.5 opacity-40"
                            )
                            ui.label(msg).classes("text-xs leading-snug").style(
                                f"color: {theme.NEGATIVE}"
                            )

            if result.warnings:
                with ui.column().classes("ml-6 gap-0.5 mb-1"):
                    for msg in result.warnings:
                        with ui.row().classes("items-start gap-1 no-wrap"):
                            ui.icon("subdirectory_arrow_right").classes(
                                "text-xs flex-none mt-0.5 opacity-40"
                            )
                            ui.label(msg).classes("text-xs leading-snug").style(
                                f"color: {theme.WARNING}"
                            )


# ---------------------------------------------------------------------------
# Wizard
# ---------------------------------------------------------------------------

class _WizardView:
    """Houdt de wizard-state en rendert de vier stappen."""

    def __init__(self) -> None:
        stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self._project_dir = os.path.join(os.getcwd(), "tmp", f"studentprognose{stamp}")
        self._mode: str = "both"
        self._picker = DirectoryPicker(on_select=self._on_dir_selected)
        self._build()

    def _build(self) -> None:
        with ui.stepper().props("vertical").classes("w-full") as self._stepper:
            self._build_step1()
            self._build_step2()
            self._build_step3()
            self._build_step4()

    # ── Stap 1: map kiezen ──────────────────────────────────────────────────

    def _build_step1(self) -> None:
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

    # ── Stap 2: bevestigen ──────────────────────────────────────────────────

    def _build_step2(self) -> None:
        with ui.step("Bevestigen"):
            self._conflict_slot = ui.column().classes("w-full")
            ui.label("De volgende structuur wordt aangemaakt:").classes("text-sm opacity-70")
            self._confirm_path = ui.label().classes("text-sm font-mono")
            with ui.column().classes("gap-0 ml-2"):
                for d in _CREATED_DIRS:
                    with ui.row().classes("items-center gap-1"):
                        ui.icon("folder").classes("text-amber-8 text-sm")
                        ui.label(d).classes("text-sm font-mono")
                with ui.row().classes("items-center gap-1"):
                    ui.icon("description").classes("opacity-60 text-sm")
                    ui.label("configuration/configuration.json").classes("text-sm font-mono")
            with ui.stepper_navigation():
                ui.button("Terug", on_click=self._stepper.previous).props("flat")
                ui.button("Volgende", on_click=self._stepper.next)

    # ── Stap 3: aanmaken ────────────────────────────────────────────────────

    def _build_step3(self) -> None:
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
            self._create_feedback = ui.column().classes("mt-2 w-full")

    # ── Stap 4: modus kiezen + data uploaden ────────────────────────────────

    def _build_step4(self) -> None:
        with ui.step("Data uploaden"):

            # ── Modus-selectie ─────────────────────────────────────────────
            ui.label("Welke data ga je gebruiken?").classes("font-medium")
            ui.label(
                "De gekozen modus bepaalt welke bestanden verplicht zijn."
            ).classes("text-sm opacity-60 mb-3")

            self._mode_cards: dict[str, ui.card] = {}
            with ui.row().classes("w-full gap-3 mb-5"):
                for key, label, icon, cli_flag, desc in _MODE_OPTS:
                    with (
                        ui.card()
                        .classes("flex-1 cursor-pointer")
                        .style("border-radius: 8px; border: 2px solid #e8e8e8;")
                    ) as card:
                        card.on("click", lambda k=key: self._select_mode(k))
                        with ui.column().classes("items-center text-center gap-1"):
                            ui.icon(icon).classes("text-3xl").style(
                                f"color: {theme.ACCENT}"
                            )
                            ui.label(label).classes("font-medium text-sm")
                            ui.label(cli_flag).classes("text-xs font-mono opacity-40")
                            ui.label(desc).classes("text-xs opacity-60")
                    self._mode_cards[key] = card

            # ── Upload-zones ───────────────────────────────────────────────
            self._tel_wrapper = ui.column().classes("w-full")
            with self._tel_wrapper:
                self._zone_tel = _UploadZone(
                    title="Telbestanden",
                    description="Weekelijkse Studielink-exports (één CSV per week).",
                    hint="bijv. telbestandY2024W10.csv of telbestand_sl_20241007_v01_2024.csv",
                    icon="bar_chart",
                    required=True,
                    accept=".csv",
                    multiple=True,
                    project_dir_getter=lambda: self._project_dir,
                    validate_fn=save_and_validate_telbestand,
                    on_change=self._refresh_summary,
                )

            ui.space().classes("h-3")

            self._ind_wrapper = ui.column().classes("w-full")
            with self._ind_wrapper:
                self._zone_ind = _UploadZone(
                    title="Individuele aanmelddata",
                    description="Eén CSV-bestand met aanmeldinformatie per student.",
                    hint="Wordt opgeslagen als: individuele_aanmelddata.csv",
                    icon="person",
                    required=True,
                    accept=".csv",
                    multiple=False,
                    project_dir_getter=lambda: self._project_dir,
                    validate_fn=save_and_validate_individueel,
                    on_change=self._refresh_summary,
                )

            ui.space().classes("h-3")

            self._zone_okt = _UploadZone(
                title="Oktober-bestand",
                description="Excel-bestand met studentaantallen — labels voor het model.",
                hint="Wordt opgeslagen als: oktober_bestand.xlsx",
                icon="calendar_month",
                required=True,
                accept=".xlsx",
                multiple=False,
                project_dir_getter=lambda: self._project_dir,
                validate_fn=save_and_validate_oktober,
                on_change=self._refresh_summary,
            )

            ui.space().classes("h-4")

            # ── Statuskaart ────────────────────────────────────────────────
            self._summary_card = (
                ui.card()
                .classes("w-full")
                .style("border: 1px solid #e8e8e8; border-radius: 8px;")
            )

            with ui.stepper_navigation():
                ui.button("Terug", on_click=self._stepper.previous).props("flat")
                ui.button(
                    "Overslaan",
                    icon="skip_next",
                    on_click=lambda: ui.navigate.to("/config"),
                ).props("flat color=grey")
                self._proceed_btn = ui.button(
                    "Naar configuratie",
                    icon="arrow_forward",
                    on_click=lambda: ui.navigate.to("/config"),
                ).props("unelevated color=accent")
                self._proceed_btn.set_visibility(False)

            # Initialiseer visuele staat nadat alle elementen bestaan.
            self._apply_mode()

    # ── Modus-logica ────────────────────────────────────────────────────────

    def _select_mode(self, mode: str) -> None:
        self._mode = mode
        self._apply_mode()

    def _apply_mode(self) -> None:
        """Pas kaartrand, zichtbaarheid van zones en badge-teksten aan."""
        for key, card in self._mode_cards.items():
            if key == self._mode:
                card.style(
                    f"border-radius: 8px; border: 2px solid {theme.ACCENT};"
                    f"background: {theme.ACCENT}0d;"
                )
            else:
                card.style("border-radius: 8px; border: 2px solid #e8e8e8;")

        needs_tel = self._mode in ("cumulative", "both")
        needs_ind = self._mode in ("individual", "both")

        self._tel_wrapper.set_visibility(needs_tel)
        self._ind_wrapper.set_visibility(needs_ind)

        self._refresh_summary()

    # ── Samenvattingskaart ───────────────────────────────────────────────────

    def _refresh_summary(self) -> None:
        tel_ok = self._zone_tel.has_valid
        ind_ok = self._zone_ind.has_valid
        okt_ok = self._zone_okt.has_valid

        needs_tel = self._mode in ("cumulative", "both")
        needs_ind = self._mode in ("individual", "both")

        # Oktober is altijd vereist (bevat de labels voor het model).
        ready = (
            (not needs_tel or tel_ok)
            and (not needs_ind or ind_ok)
            and okt_ok
        )

        self._summary_card.clear()
        with self._summary_card:
            rows: list[tuple[str, bool, str]] = []

            if needs_tel:
                if tel_ok and self._zone_tel.count > self._zone_tel.valid_count:
                    detail = (
                        f"{self._zone_tel.valid_count} van "
                        f"{self._zone_tel.count} bestanden geldig"
                    )
                elif tel_ok:
                    detail = f"{self._zone_tel.valid_count} bestand(en) geldig"
                else:
                    detail = "Nog uploaden"
                rows.append(("Telbestanden", tel_ok, detail))

            if needs_ind:
                rows.append((
                    "Individuele aanmelddata",
                    ind_ok,
                    "Aanwezig" if ind_ok else "Nog uploaden",
                ))

            rows.append((
                "Oktober-bestand",
                okt_ok,
                "Aanwezig" if okt_ok else "Nog uploaden",
            ))

            with ui.row().classes("items-center gap-2 mb-2"):
                ui.icon("checklist").classes("text-lg").style(f"color: {theme.ACCENT}")
                ui.label("Status geselecteerde modus").classes("font-medium")

            for label, ok, detail in rows:
                icon_name = "check_circle" if ok else "radio_button_unchecked"
                color = theme.POSITIVE if ok else theme.MUTED
                with ui.row().classes("items-center gap-2 py-0.5"):
                    ui.icon(icon_name).style(f"color: {color}").classes("text-base flex-none")
                    ui.label(label).classes("text-sm flex-none w-52")
                    ui.label(detail).classes("text-xs opacity-60")

            if not ready:
                ui.separator().classes("my-2")
                with ui.row().classes("items-center gap-2"):
                    ui.icon("info").style(f"color: {theme.INFO}").classes("text-base")
                    ui.label(
                        "Upload de vereiste bestanden of klik 'Overslaan' als je "
                        "ze al handmatig hebt geplaatst."
                    ).classes("text-xs opacity-70")

        self._proceed_btn.set_visibility(ready)

    # ── Stap-overgangen ─────────────────────────────────────────────────────

    def _goto_confirm(self) -> None:
        self._project_dir = os.path.abspath(self._path_input.value.strip())
        self._confirm_path.set_text(self._project_dir)

        self._conflict_slot.clear()
        if os.path.isfile(
            os.path.join(self._project_dir, "configuration", "configuration.json")
        ):
            with self._conflict_slot:
                info_banner(
                    "Deze map bevat al een project. "
                    "De bestaande configuratie blijft behouden — "
                    "er wordt niets overschreven."
                )

        self._stepper.next()

    def _on_dir_selected(self, path: str) -> None:
        self._path_input.set_value(path)

    # ── Project aanmaken ────────────────────────────────────────────────────

    async def _create_project(self) -> None:
        self._create_btn.props("loading")
        self._create_feedback.clear()
        try:
            os.makedirs(self._project_dir, exist_ok=True)
            returncode = await self._panel.run(["init"], cwd=self._project_dir)

            if returncode == 0 and self._demo_checkbox.value:
                await self._download_demodata()

            if returncode == 0:
                STATE.project_dir = self._project_dir
                await self._goto_upload_step()
        finally:
            self._create_btn.props(remove="loading")

    async def _goto_upload_step(self) -> None:
        self._stepper.next()
        existing = await asyncio.to_thread(scan_existing_files, self._project_dir)
        self._zone_tel.load_existing(existing["telbestanden"])
        self._zone_ind.load_existing(existing["individueel"])
        self._zone_okt.load_existing(existing["oktober"])
        self._refresh_summary()

    async def _download_demodata(self) -> None:
        dest = os.path.join(self._project_dir, "data", "input_raw")
        self._demo_progress.set_visibility(True)
        holder: dict = {"value": 0.0}
        timer = ui.timer(
            0.1,
            lambda: self._demo_progress.set_value(max(holder["value"], 0.0)),
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
            with self._create_feedback:
                info_banner("Demodata gedownload naar data/input_raw/.")
        except Exception as exc:  # noqa: BLE001
            with self._create_feedback:
                error_banner(
                    "Demodata downloaden mislukt.",
                    f"Controleer je internetverbinding. Details: {exc}",
                )
        finally:
            timer.cancel()
