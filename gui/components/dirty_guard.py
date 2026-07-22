"""Herbruikbare 'niet-opgeslagen wijzigingen'-bewaker.

Zet een statuslabel neer en registreert een ``beforeunload``-listener die de
gebruiker waarschuwt bij het verlaten van de pagina met onopgeslagen
wijzigingen. Gebruikt ``addEventListener`` (niet ``window.onbeforeunload =``)
zodat NiceGUI's eigen unload-handler intact blijft.
"""

from __future__ import annotations

from nicegui import ui

_GUARD_JS = """
    if (!window.__spUnloadGuard) {
        window.__spDirty = false;
        window.__spUnloadGuard = (e) => {
            if (window.__spDirty) { e.preventDefault(); e.returnValue = ''; }
        };
        window.addEventListener('beforeunload', window.__spUnloadGuard);
    }
"""


class DirtyGuard:
    """Beheert een dirty-vlag met bijbehorend statuslabel en unload-waarschuwing.

    Rendert direct een statuslabel op de huidige plek in de layout.
    """

    def __init__(self) -> None:
        self._dirty = False
        self._status = ui.label("").classes("text-sm")

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark(self) -> None:
        """Markeer als gewijzigd en activeer de unload-waarschuwing."""
        if not self._dirty:
            self._dirty = True
            ui.run_javascript(_GUARD_JS + "window.__spDirty = true;")
        self._status.set_text("Niet-opgeslagen wijzigingen").style("color: #ed6c02")

    def clear(self) -> None:
        """Markeer als opgeslagen en schakel de unload-waarschuwing uit."""
        self._dirty = False
        ui.run_javascript("window.__spDirty = false;")
        self._status.set_text("Opgeslagen").style("color: #2e7d32")
