"""Gedeelde paginaschil: header en wizard-stepper.

De primaire navigatie is de pipeline-hub op de startpagina (``/``, zie
:mod:`gui.components.pipeline_rail`); er is geen zijbalk. Deze schil levert:

* **Header** — het Npuls-logo en de toolnaam zijn klikbaar en brengen je vanaf
  elke pagina terug naar de pipeline-hub.
* **Stepper** — lineaire voortgang voor nieuwe gebruikers, met een vinkje op
  afgeronde stappen.

Zie ``gui/DESIGN.md`` voor de stijlgids.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from nicegui import ui

from gui import nav, theme
from gui.theme import QUASAR_COLORS


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
                color, icon = "accent", "radio_button_checked"
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

    # Zwart app-bar (CEDA/Npuls) met een oranje accentlijn onderaan.
    with (
        ui.header()
        .classes("items-center justify-between q-px-md")
        .style(f"background: {theme.PRIMARY}; border-bottom: 3px solid {theme.ACCENT}")
    ):
        # Logo + toolnaam vormen samen de weg terug naar de pipeline-hub (``/``):
        # zonder zijbalk is dit de vaste navigatie-affordance op elke pagina.
        brand = (
            ui.row()
            .classes("items-center gap-3 no-wrap cursor-pointer")
            .on("click", lambda: ui.navigate.to("/"))
        )
        with brand:
            # Officieel Npuls-logo (wit) — co-branding met de toolnaam.
            ui.image("/gui-assets/npuls-logo-white.svg").classes("w-8 h-8")
            ui.label("Studentprognose").classes("text-lg font-medium text-white")
        brand.tooltip("Terug naar de pipeline-hub")
        ui.label(title).classes("text-sm text-white opacity-70")

    with ui.column().classes("w-full max-w-5xl mx-auto p-6 gap-4"):
        if show_stepper:
            _stepper(active)
        yield
