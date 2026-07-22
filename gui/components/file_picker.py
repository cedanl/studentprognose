"""Lokale mappenkiezer als NiceGUI-dialoog.

De GUI draait lokaal, dus de "server" is de machine van de gebruiker: door de
mappen aan de serverkant te tonen krijg je effectief een lokale mappenkiezer.
Gebruikt door de init-wizard (#266) om een projectmap te kiezen.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from nicegui import ui


class DirectoryPicker:
    """Modale dialoog waarmee de gebruiker een map selecteert.

    Example:
        picker = DirectoryPicker(on_select=lambda path: ...)
        picker.open(start=os.getcwd())
    """

    def __init__(self, on_select: Callable[[str], None]) -> None:
        self._on_select = on_select
        self._current = os.getcwd()
        self._dialog = ui.dialog()
        with self._dialog, ui.card().classes("w-[36rem] max-w-full"):
            ui.label("Kies een map").classes("text-lg font-medium")
            self._path_label = ui.label().classes("text-xs opacity-60 break-all")
            self._list = ui.column().classes(
                "w-full h-72 overflow-auto border rounded p-1 gap-0"
            )
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Annuleren", on_click=self._dialog.close).props("flat")
                ui.button("Deze map kiezen", on_click=self._choose).props("unelevated")

    def open(self, start: str | None = None) -> None:
        """Open de dialoog, startend in ``start`` (default: huidige werkmap)."""
        if start and os.path.isdir(start):
            self._current = os.path.abspath(start)
        self._render()
        self._dialog.open()

    def _render(self) -> None:
        self._path_label.set_text(self._current)
        self._list.clear()
        with self._list:
            parent = os.path.dirname(self._current)
            if parent and parent != self._current:
                self._row("arrow_upward", ".. (bovenliggende map)", parent)
            try:
                entries = sorted(
                    e
                    for e in os.listdir(self._current)
                    if os.path.isdir(os.path.join(self._current, e))
                    and not e.startswith(".")
                )
            except OSError:
                entries = []
            for name in entries:
                full = os.path.join(self._current, name)
                self._row("folder", name, full)

    def _row(self, icon: str, label: str, target: str) -> None:
        with (
            ui.row()
            .classes(
                "w-full items-center gap-2 cursor-pointer hover:bg-blue-1 rounded px-2 py-1"
            )
            .on("click", lambda t=target: self._enter(t))
        ):
            ui.icon(icon).classes("text-amber-8")
            ui.label(label).classes("text-sm")

    def _enter(self, target: str) -> None:
        self._current = target
        self._render()

    def _choose(self) -> None:
        self._dialog.close()
        self._on_select(self._current)
