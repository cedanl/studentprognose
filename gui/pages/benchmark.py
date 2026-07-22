"""Benchmark & tune (#270).

Twee secties: benchmark (vergelijk alternatieve modellen; gesorteerde
resultaattabel met de winnaar gemarkeerd) en tune (hyperparameter-zoektocht;
gevonden parameters kopieerbaar naar de configuratie).
"""

from __future__ import annotations

import json
import os

from nicegui import ui

from gui import benchmark_io, config_io, nav
from gui.components.layout import page_shell
from gui.components.log_stream import ProcessPanel
from gui.components.states import empty_state, section_title
from gui.state import STATE

_DATASET_CODE = {"Cumulatief": "c", "Individueel": "i"}
_TUNE_CODE = {"Regressor": "regressor", "SARIMA": "sarima", "Beide": "both"}


def create() -> None:
    """Registreer de route ``/benchmark``."""
    nav.register_route("/benchmark")

    @ui.page("/benchmark")
    def benchmark_page() -> None:
        with page_shell(
            active="/benchmark", title="Benchmark & tune", show_stepper=False
        ):
            section_title(
                "Benchmark & tune",
                "Vergelijk alternatieve modellen en stem hyperparameters af.",
            )
            if not STATE.is_initialised:
                empty_state(
                    icon="folder_off",
                    title="Nog geen project",
                    message="Kies eerst een project.",
                    action_label="Project opzetten",
                    on_action=lambda: ui.navigate.to("/wizard"),
                )
                return

            with ui.tabs().classes("w-full") as tabs:
                tab_bench = ui.tab("Benchmark", icon="leaderboard")
                tab_tune = ui.tab("Tune", icon="tune")
            with ui.tab_panels(tabs, value=tab_bench).classes("w-full"):
                with ui.tab_panel(tab_bench):
                    _BenchmarkSection()
                with ui.tab_panel(tab_tune):
                    _TuneSection()


class _BenchmarkSection:
    """Draai een benchmark en toon de gesorteerde resultaattabel(len)."""

    def __init__(self) -> None:
        with ui.row().classes("w-full items-end gap-4"):
            self._dataset = ui.select(
                ["Cumulatief", "Individueel"], value="Cumulatief", label="Dataset"
            ).classes("w-48")
            self._week = ui.number("Predict week", value=12, min=1, max=52).classes(
                "w-40"
            )
            self._start = ui.button(
                "Start benchmark", icon="play_arrow", on_click=self._run
            ).props("unelevated")
        self._panel = ProcessPanel()
        self._results = ui.column().classes("w-full")
        self._render_existing()  # toon eventuele eerdere resultaten

    def _render_existing(self) -> None:
        self._results.clear()
        with self._results:
            any_shown = False
            for fname, (
                metric,
                higher,
                label,
            ) in benchmark_io.BENCHMARK_METRICS.items():
                path = os.path.join(STATE.output_dir, fname)
                if os.path.isfile(path):
                    self._render_table(path, metric, higher, label, fname)
                    any_shown = True
            if not any_shown:
                ui.label("Nog geen benchmarkresultaten.").classes("text-sm opacity-60")

    def _render_table(self, path, metric, higher, label, fname) -> None:
        import pandas as pd

        try:
            df = pd.read_csv(path)
        except (OSError, ValueError):
            return
        records = benchmark_io.summarize(df, metric, higher_is_better=higher)
        if not records:
            return
        ui.label(f"{fname} — gemiddelde {label} per model").classes("font-medium mt-2")
        columns = [
            {"name": "model", "label": "Model", "field": "model", "align": "left"},
            {"name": "metric", "label": label, "field": "metric", "sortable": True},
        ]
        table = ui.table(columns=columns, rows=records, row_key="model").classes(
            "w-full"
        )
        table.add_slot(
            "body-cell-model",
            r"""
            <q-td :props="props">
                <q-badge v-if="props.row.winner" color="positive" label="beste" class="q-mr-sm" />
                {{ props.value }}
            </q-td>
            """,
        )

    async def _run(self) -> None:
        self._start.props("loading")
        try:
            code = _DATASET_CODE[self._dataset.value]
            week = int(self._week.value or 12)
            rc = await self._panel.run(
                ["benchmark", "-d", code, "-w", str(week)], cwd=STATE.project_dir
            )
            if rc == 0:
                self._render_existing()
        finally:
            self._start.props(remove="loading")


class _TuneSection:
    """Draai hyperparameter-tuning en maak de gevonden parameters kopieerbaar."""

    def __init__(self) -> None:
        with ui.row().classes("w-full items-end gap-4"):
            self._target = ui.select(
                ["Regressor", "SARIMA", "Beide"], value="Regressor", label="Tune-doel"
            ).classes("w-48")
            self._week = ui.number("Predict week", value=12, min=1, max=52).classes(
                "w-40"
            )
            self._start = ui.button(
                "Start tuning", icon="play_arrow", on_click=self._run
            ).props("unelevated")
        ui.label("Tuning betreft het cumulatieve spoor (-d c).").classes(
            "text-sm opacity-60"
        )
        self._panel = ProcessPanel()
        self._snippet_slot = ui.column().classes("w-full")

    async def _run(self) -> None:
        self._start.props("loading")
        self._snippet_slot.clear()
        try:
            target = _TUNE_CODE[self._target.value]
            week = int(self._week.value or 12)
            rc = await self._panel.run(
                ["tune", "-d", "c", "-w", str(week), "--tune-target", target],
                cwd=STATE.project_dir,
            )
            if rc == 0:
                self._show_snippets()
        finally:
            self._start.props(remove="loading")

    def _show_snippets(self) -> None:
        snippets = benchmark_io.extract_config_snippets(self._panel.output_text())
        with self._snippet_slot:
            if not snippets:
                ui.label(
                    "Geen parameters gevonden in de uitvoer (mogelijk te weinig "
                    "trainingsjaren voor een geldige zoektocht)."
                ).classes("text-sm opacity-70")
                return
            # Voeg alle gevonden snippets samen tot één config-fragment.
            merged: dict = {}
            for snip in snippets:
                merged = benchmark_io.deep_merge(merged, snip)
            ui.label("Aanbevolen parameters").classes("font-medium")
            ui.code(
                json.dumps(merged, ensure_ascii=False, indent=2), language="json"
            ).classes("w-full")
            ui.button(
                "Kopieer naar configuratie",
                icon="content_copy",
                on_click=lambda: self._apply(merged),
            ).props("unelevated")

    def _apply(self, merged: dict) -> None:
        try:
            config = config_io.load_config(STATE.config_path)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            ui.notify(f"Configuratie niet leesbaar: {exc}", type="negative")
            return
        updated = benchmark_io.deep_merge(config, merged)
        try:
            config_io.save_config(STATE.config_path, updated)
        except OSError as exc:
            ui.notify(f"Opslaan mislukt: {exc}", type="negative")
            return
        ui.notify("Parameters toegevoegd aan configuration.json.", type="positive")
