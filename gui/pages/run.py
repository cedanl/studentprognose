"""Pipeline-runner (#268): stel parameters in en draai de voorspelling.

Bouwt een ``studentprognose``-commando, toont een live preview, vraagt
bevestiging en streamt de uitvoer live. Parameters worden per sessie bewaard.
"""

from __future__ import annotations

from nicegui import app, ui

from gui import nav, process, theme, tracks
from gui.components.layout import page_shell
from gui.components.log_stream import ProcessPanel
from gui.components.progress_card import ProgressCard
from gui.components.states import empty_state, section_title
from gui.state import STATE

#: Sleutel waaronder de laatst gebruikte parameters bewaard worden.
_STORAGE_KEY = "run_settings"

_DEFAULTS = {
    "dataset": "Beide",
    "cohort": "Eerstejaars",
    "years": "",
    "weeks": "",
    "institutions": [],
    "skip_years": 0,
    "noetl": False,
    "dashboard": False,
    "no_warnings": False,
    "yes": True,
}


def create() -> None:
    """Registreer de route ``/run``."""
    nav.register_route("/run")

    @ui.page("/run")
    def run_page() -> None:
        with page_shell(active="/run", title="Uitvoeren"):
            section_title(
                "Uitvoeren", "Stel de voorspelling in en volg de voortgang live."
            )
            if not STATE.is_initialised:
                empty_state(
                    icon="folder_off",
                    title="Nog geen project",
                    message="Kies eerst een project voordat je de pipeline draait.",
                    action_label="Project opzetten",
                    on_action=lambda: ui.navigate.to("/wizard"),
                )
                return
            _RunView()


class _RunView:
    """Rendert het parameterformulier, de preview en de runner."""

    def __init__(self) -> None:
        self._settings = {**_DEFAULTS, **app.storage.general.get(_STORAGE_KEY, {})}
        self._build()

    def _build(self) -> None:
        with ui.card().classes("w-full"):
            with ui.grid(columns=2).classes("w-full gap-4"):
                self._dataset = ui.select(
                    ["Individueel", "Cumulatief", "Beide"],
                    value=self._settings["dataset"],
                    label="Dataset (voorspelspoor)",
                ).classes("w-full")
                self._dataset.tooltip(tracks.dataset_tooltip())
                self._dataset.on_value_change(lambda _e: self._update_dataset_hint())
                self._cohort = ui.select(
                    ["Eerstejaars", "Hogerejaars", "Volume"],
                    value=self._settings["cohort"],
                    label="Cohort",
                ).classes("w-full")
                self._years = ui.input(
                    "Jaren",
                    value=self._settings["years"],
                    placeholder="bijv. 2024 of 2023 2024",
                ).classes("w-full")
                self._weeks = ui.input(
                    "Weken",
                    value=self._settings["weeks"],
                    placeholder="bijv. 6 of 1:38",
                ).classes("w-full")
                self._institutions = (
                    ui.select(
                        options=list(self._settings["institutions"]),
                        value=list(self._settings["institutions"]),
                        multiple=True,
                        with_input=True,
                        label="Instellingsfilter (leeg = alle)",
                    )
                    .props("use-chips new-value-mode=add-unique")
                    .classes("w-full")
                )
                self._skip_years = ui.number(
                    "Jaren overslaan (backtesting)",
                    value=self._settings["skip_years"],
                    min=0,
                ).classes("w-full")

            # Uitleg van het gekozen voorspelspoor (werkt zonder de docs te openen).
            with ui.row().classes("items-center gap-2 w-full"):
                self._dataset_hint_icon = (
                    ui.icon("info").classes("text-sm").style(f"color: {theme.ACCENT}")
                )
                self._dataset_hint = ui.label("").classes("text-sm opacity-80")
            self._update_dataset_hint()

            with ui.row().classes("w-full gap-6 mt-2"):
                self._noetl = ui.checkbox(
                    "ETL overslaan (--noetl)", value=self._settings["noetl"]
                )
                self._dashboard = ui.checkbox(
                    "Dashboards genereren (--dashboard)",
                    value=self._settings["dashboard"],
                )
                self._no_warnings = ui.checkbox(
                    "Waarschuwingen onderdrukken", value=self._settings["no_warnings"]
                )
                self._yes = ui.checkbox(
                    "Validatieprompt overslaan (--yes)", value=self._settings["yes"]
                )
                self._yes.tooltip(
                    "Aanbevolen aan: een GUI-run heeft geen interactieve invoer."
                )

        # Live command-preview.
        with ui.card().classes("w-full bg-grey-2"):
            ui.label("Commando").classes("text-xs uppercase opacity-60")
            self._preview = ui.label("").classes("font-mono text-sm break-all")

        with ui.row().classes("items-center gap-3"):
            self._start_btn = ui.button(
                "Start voorspelling", icon="play_arrow", on_click=self._confirm
            ).props("unelevated")
            self._result_slot = ui.row().classes("items-center gap-2")

        self._progress = ProgressCard()
        self._panel = ProcessPanel()

        # Reageer op wijzigingen: preview verversen.
        for widget in (
            self._dataset,
            self._cohort,
            self._years,
            self._weeks,
            self._institutions,
            self._skip_years,
            self._noetl,
            self._dashboard,
            self._no_warnings,
            self._yes,
        ):
            widget.on_value_change(lambda _e: self._update_preview())
        self._update_preview()

    def _current_args(self) -> list[str]:
        return process.build_run_args(
            dataset=self._dataset.value,
            cohort=self._cohort.value,
            years=self._years.value or "",
            weeks=self._weeks.value or "",
            institutions=list(self._institutions.value or []),
            skip_years=int(self._skip_years.value or 0),
            noetl=self._noetl.value,
            dashboard=self._dashboard.value,
            no_warnings=self._no_warnings.value,
            yes=self._yes.value,
        )

    def _update_dataset_hint(self) -> None:
        t = tracks.track(self._dataset.value)
        if t is None:
            self._dataset_hint.set_text("")
            return
        suffix = " (aanbevolen)" if t.label == tracks.RECOMMENDED else ""
        self._dataset_hint.set_text(f"{t.label} — {t.short}{suffix}")

    def _update_preview(self) -> None:
        self._preview.set_text(process.preview_command(self._current_args()))

    def _persist(self) -> None:
        app.storage.general[_STORAGE_KEY] = {
            "dataset": self._dataset.value,
            "cohort": self._cohort.value,
            "years": self._years.value or "",
            "weeks": self._weeks.value or "",
            "institutions": list(self._institutions.value or []),
            "skip_years": int(self._skip_years.value or 0),
            "noetl": self._noetl.value,
            "dashboard": self._dashboard.value,
            "no_warnings": self._no_warnings.value,
            "yes": self._yes.value,
        }

    def _confirm(self) -> None:
        command = process.preview_command(self._current_args())
        with ui.dialog() as dialog, ui.card():
            ui.label("Deze voorspelling uitvoeren?").classes("text-lg font-medium")
            ui.label(command).classes("font-mono text-sm break-all")

            async def _run_confirmed() -> None:
                # Sluit de dialoog en start de run; als async handler zodat de
                # coroutine daadwerkelijk geawait wordt (een lambda zou 'm droppen).
                dialog.close()
                await self._start()

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Annuleren", on_click=dialog.close).props("flat")
                ui.button("Uitvoeren", on_click=_run_confirmed).props("unelevated")
        dialog.open()

    async def _start(self) -> None:
        self._persist()
        self._result_slot.clear()
        self._start_btn.props("loading")
        self._progress.start()
        try:
            args = self._current_args()
            returncode = await self._panel.run(
                args,
                cwd=STATE.project_dir,
                on_line=self._progress.on_line,
            )
            self._progress.complete(success=returncode == 0)
            if returncode == 0:
                self._show_results_link()
        finally:
            self._start_btn.props(remove="loading")

    def _show_results_link(self) -> None:
        with self._result_slot:
            ui.icon("check_circle").props("color=positive")
            if nav.is_available("/output"):
                ui.button(
                    "Bekijk resultaten",
                    icon="insights",
                    on_click=lambda: ui.navigate.to("/output"),
                ).props("unelevated")
            else:
                ui.label("Voorspelling klaar.").classes("text-positive")
