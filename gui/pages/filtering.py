"""Filtering-editor (#271).

Bewerkt ``configuration/filtering/base.json``: op welke opleidingen, herkomst en
examentypes de pipeline draait. Toont live hoeveel opleidingen na de filters
overblijven, berekend uit het student_count-bestand zonder de pipeline te draaien.
"""

from __future__ import annotations

import json
import os

import pandas as pd
from nicegui import ui

from gui import config_io, filtering_io, nav
from gui.components.dirty_guard import DirtyGuard
from gui.components.layout import page_shell
from gui.components.states import empty_state, error_banner, info_banner, section_title
from gui.state import STATE


def create() -> None:
    """Registreer de route ``/filtering``."""
    nav.register_route("/filtering")

    @ui.page("/filtering")
    def filtering_page() -> None:
        with page_shell(active="/filtering", title="Filteren"):
            section_title(
                "Filteren",
                "Bepaal op welke opleidingen, herkomst en examentypes je draait.",
            )
            if not STATE.is_initialised:
                empty_state(
                    icon="folder_off",
                    title="Nog geen project",
                    message="Kies eerst een project voordat je filters instelt.",
                    action_label="Project opzetten",
                    on_action=lambda: ui.navigate.to("/wizard"),
                )
                return
            _FilteringView()


class _FilteringView:
    """Houdt de filtering-state en rendert de editor met live preview."""

    def __init__(self) -> None:
        self._path = STATE.filtering_path
        self._data = filtering_io.load_filtering(self._path)
        self._filtering = self._data["filtering"]
        self._roles = self._load_roles()
        self._df = self._load_student_count()
        self._build()

    def _load_roles(self) -> dict:
        """Kolomrollen uit de config (met veilige defaults)."""
        defaults = {
            "programme": "Croho groepeernaam",
            "origin": "Herkomst",
            "exam_type": "Examentype",
        }
        try:
            config = config_io.load_config(STATE.config_path)
        except (FileNotFoundError, json.JSONDecodeError):
            return defaults
        roles = config.get("column_roles", {})
        return {k: roles.get(k, v) for k, v in defaults.items()}

    def _load_student_count(self) -> pd.DataFrame | None:
        """Laad student_count voor de opleidingenlijst + preview (of None)."""
        path = os.path.join(
            STATE.project_dir, "data", "input", "student_count_first-years.xlsx"
        )
        if not os.path.isfile(path):
            return None
        try:
            return pd.read_excel(path)
        except (OSError, ValueError):
            return None

    # --- Opbouw ---------------------------------------------------------------

    def _build(self) -> None:
        with ui.row().classes("w-full items-center justify-between"):
            ui.button("Opslaan", icon="save", on_click=self._save).props("unelevated")
            self._guard = DirtyGuard()

        if self._df is None:
            info_banner(
                "De opleidingenlijst is nog niet beschikbaar (student_count "
                "ontbreekt). Draai eerst de pipeline/ETL. Je kunt opleidingen "
                "hieronder wel handmatig invoeren."
            )

        programme_options = self._programme_options()

        with ui.card().classes("w-full"):
            ui.label("Opleidingen").classes("font-medium")
            ui.label(
                "Leeg = alle opleidingen. Typ om te zoeken of voer handmatig in."
            ).classes("text-sm opacity-60")
            self._programme_select = (
                ui.select(
                    options=programme_options,
                    value=list(self._filtering.get("programme", [])),
                    multiple=True,
                    with_input=True,
                    label="Opleidingen",
                )
                .props("use-chips new-value-mode=add-unique")
                .classes("w-full")
            )
            self._programme_select.on_value_change(self._on_programme_change)

        with ui.row().classes("w-full gap-4 no-wrap"):
            with ui.card().classes("grow"):
                ui.label("Herkomst").classes("font-medium")
                self._herkomst_boxes = self._checkbox_group(
                    filtering_io.HERKOMST_CHOICES,
                    self._filtering.get("herkomst", []),
                    self._on_herkomst_change,
                )
            with ui.card().classes("grow"):
                ui.label("Examentype").classes("font-medium")
                self._examentype_boxes = self._checkbox_group(
                    filtering_io.EXAMENTYPE_CHOICES,
                    self._filtering.get("examentype", []),
                    self._on_examentype_change,
                )

        with ui.card().classes("w-full bg-blue-1"):
            self._preview = ui.label().classes("text-base font-medium")
        self._update_preview()

    def _programme_options(self) -> list[str]:
        options = set(self._filtering.get("programme", []))
        if self._df is not None:
            col = self._roles["programme"]
            options |= set(self._df[col].dropna().astype(str).unique())
        return sorted(options)

    def _checkbox_group(self, choices, selected, on_change) -> dict:
        boxes = {}
        for choice in choices:
            cb = ui.checkbox(choice, value=choice in selected)
            cb.on_value_change(on_change)
            boxes[choice] = cb
        return boxes

    # --- Change-handlers ------------------------------------------------------

    def _on_programme_change(self, e) -> None:
        self._filtering["programme"] = list(e.value or [])
        self._guard.mark()
        self._update_preview()

    def _on_herkomst_change(self, _e) -> None:
        self._filtering["herkomst"] = [
            c for c, box in self._herkomst_boxes.items() if box.value
        ]
        self._guard.mark()
        self._update_preview()

    def _on_examentype_change(self, _e) -> None:
        self._filtering["examentype"] = [
            c for c, box in self._examentype_boxes.items() if box.value
        ]
        self._guard.mark()
        self._update_preview()

    # --- Preview & opslaan ----------------------------------------------------

    def _update_preview(self) -> None:
        if self._df is None:
            self._preview.set_text(
                "Live preview niet beschikbaar (geen student_count-bestand)."
            )
            return
        remaining, total = filtering_io.count_programmes(
            self._df,
            programme_col=self._roles["programme"],
            origin_col=self._roles["origin"],
            exam_col=self._roles["exam_type"],
            programme=self._filtering.get("programme", []),
            herkomst=self._filtering.get("herkomst", []),
            examentype=self._filtering.get("examentype", []),
        )
        self._preview.set_text(
            f"{remaining} van {total} opleidingen geselecteerd na filtering."
        )

    def _save(self) -> None:
        errors = filtering_io.validate_filtering(self._data)
        if errors:
            with ui.column():
                for err in errors:
                    error_banner(err)
            ui.notify("Kan niet opslaan: ongeldige filterwaarden.", type="negative")
            return
        try:
            filtering_io.save_filtering(self._path, self._data)
        except OSError as exc:
            ui.notify(f"Opslaan mislukt: {exc}", type="negative")
            return
        self._guard.clear()
        ui.notify("Filtering opgeslagen.", type="positive")
