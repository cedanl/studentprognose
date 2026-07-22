"""Gedeelde paginaschil: header, zijbalknavigatie en wizard-stepper.

Implementeert het navigatiemodel uit :mod:`gui.nav`:

* **Zijbalk** — vrije navigatie voor terugkerende gebruikers. Nog-niet-gebouwde
  of (bij een ontbrekend project) nog-niet-toegankelijke bestemmingen staan
  uitgeschakeld, zodat de gebruiker de volledige flow ziet zonder in een 404 te
  lopen.
* **Stepper** — lineaire voortgang voor nieuwe gebruikers, met een vinkje op
  afgeronde stappen.

Zie ``gui/DESIGN.md`` voor de stijlgids.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from nicegui import ui

from gui import nav
from gui.state import STATE
from gui.theme import QUASAR_COLORS


def _navigate(route: str) -> None:
    ui.navigate.to(route)


def _requires_project(route: str) -> bool:
    """True als de route pas zin heeft nadat er een project is gekozen."""
    return route not in ("/", "/wizard")


def _drawer(active: str) -> None:
    """Render de zijbalk met alle navigatie-items."""
    with ui.left_drawer(fixed=False).classes("bg-grey-1 gap-1"):
        ui.label("Navigatie").classes("text-xs uppercase opacity-50 px-3 pt-2")
        for item in nav.all_items():
            built = nav.is_available(item.route)
            locked = _requires_project(item.route) and not STATE.is_initialised
            enabled = built and not locked

            # Kleur draagt de betekenis: primair = actief, donkergrijs = klikbaar,
            # lichtgrijs = uitgeschakeld. Op een uitgeschakelde q-btn (opacity 0.7)
            # leest lichtgrijs duidelijk als "nu niet beschikbaar".
            if item.route == active:
                color = "primary"
            elif enabled:
                color = "grey-9"
            else:
                color = "grey-5"

            classes = "w-full justify-start"
            if item.route == active:
                classes += " font-medium bg-blue-1 rounded"

            reason = (
                "Nog in ontwikkeling"
                if not built
                else "Kies eerst een project (stap 1)"
            )

            # Een uitgeschakelde q-btn vangt geen hover-events, dus de tooltip
            # hangt aan een wrapper zodat de uitleg tóch verschijnt.
            with ui.element("div").classes("w-full") as wrapper:
                btn = (
                    ui.button(
                        item.label,
                        icon=item.icon,
                        on_click=(lambda r=item.route: _navigate(r))
                        if enabled
                        else None,
                    )
                    .props(f"flat align=left color={color}")
                    .classes(classes)
                )
                if not enabled:
                    btn.props("disable")
                    wrapper.tooltip(reason)


def _stepper(active: str) -> None:
    """Render een horizontale stap-indicator voor de wizard-flow."""
    active_step = next((i.step for i in nav.WIZARD_FLOW if i.route == active), None)
    if active_step is None:
        return

    with ui.row().classes("w-full items-center gap-1 mb-2"):
        for item in nav.WIZARD_FLOW:
            done = item.step < active_step
            current = item.step == active_step
            if done:
                color, icon = "positive", "check_circle"
            elif current:
                color, icon = "primary", "radio_button_checked"
            else:
                color, icon = "grey-5", "radio_button_unchecked"
            with ui.row().classes("items-center gap-1 no-wrap"):
                ui.icon(icon).props(f"color={color}")
                ui.label(item.label).classes(
                    "text-sm " + ("font-medium" if current else "opacity-60")
                )
            if item.step < len(nav.WIZARD_FLOW):
                ui.separator().props("vertical").classes("mx-1")


@contextmanager
def page_shell(active: str, title: str, *, show_stepper: bool = True) -> Iterator[None]:
    """Render de header + zijbalk (+ optioneel stepper) en yield de content.

    Args:
        active: Route van de actieve pagina.
        title: Titel in de header.
        show_stepper: Toon de wizard-stepper (alleen zinvol op flow-pagina's).
    """
    ui.colors(**QUASAR_COLORS)

    with ui.header().classes("items-center justify-between q-px-md"):
        with ui.row().classes("items-center gap-2 no-wrap"):
            ui.icon("school").classes("text-2xl")
            ui.label("Studentprognose").classes("text-lg font-medium")
        ui.label(title).classes("text-sm opacity-80")

    _drawer(active)

    with ui.column().classes("w-full max-w-5xl mx-auto p-6 gap-4"):
        if show_stepper:
            _stepper(active)
        yield
