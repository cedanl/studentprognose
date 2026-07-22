"""Gedeelde paginaschil: header en zijbalknavigatie.

Basisversie voor de projectsetup (#265). De volledige navigatieflow met
stap-indicator en states wordt uitgewerkt in het UX-design (#272).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from nicegui import ui

from gui.theme import QUASAR_COLORS

#: Navigatie-items: (route, label, icoon). Pagina's uit latere issues voegen zich
#: hier toe zodra ze bestaan.
NAV_ITEMS: list[tuple[str, str, str]] = [
    ("/", "Start", "home"),
]


@contextmanager
def page_shell(active: str, title: str) -> Iterator[None]:
    """Render de header + zijbalk en yield de content-container.

    Args:
        active: Route van de actieve pagina (markeert het navigatie-item).
        title: Titel die in de header verschijnt.
    """
    ui.colors(**QUASAR_COLORS)

    with ui.header().classes("items-center justify-between"):
        ui.label("🎓 Studentprognose").classes("text-lg font-medium")
        ui.label(title).classes("text-sm opacity-80")

    with ui.left_drawer(fixed=False).classes("bg-grey-1"):
        for route, label, icon in NAV_ITEMS:
            classes = "w-full justify-start"
            if route == active:
                classes += " text-primary font-medium"
            ui.button(
                label, icon=icon, on_click=lambda r=route: ui.navigate.to(r)
            ).props("flat align=left").classes(classes)

    with ui.column().classes("w-full max-w-5xl mx-auto p-6 gap-4"):
        yield
