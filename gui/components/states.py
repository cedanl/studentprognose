"""Herbruikbare UX-state-componenten.

Elke pagina gebruikt deze primitieven zodat de vijf states (leeg, klaar, bezig,
geslaagd, gefaald) er overal hetzelfde uitzien. Zie ``gui/DESIGN.md`` voor de
achterliggende stijlgids.
"""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from gui import theme

#: Status → (label, Quasar-kleur, icoon). Eén definitie voor alle statusbadges.
_STATUS = {
    "idle": ("Nog niet gestart", "grey", "radio_button_unchecked"),
    "ready": ("Klaar om te starten", "positive", "check_circle"),
    "running": ("Bezig…", "info", "hourglass_top"),
    "success": ("Geslaagd", "positive", "task_alt"),
    "failed": ("Mislukt", "negative", "error"),
}


def status_badge(state: str) -> ui.badge:
    """Render een gekleurde statusbadge voor één van de vijf states.

    Args:
        state: ``idle`` | ``ready`` | ``running`` | ``success`` | ``failed``.
    """
    label, color, icon = _STATUS.get(state, _STATUS["idle"])
    with ui.badge(color=color).classes("px-2 py-1") as badge:
        with ui.row().classes("items-center gap-1 no-wrap"):
            ui.icon(icon).classes("text-xs")
            ui.label(label).classes("text-xs")
    return badge


def empty_state(
    *,
    icon: str,
    title: str,
    message: str,
    action_label: str | None = None,
    on_action: Callable[[], None] | None = None,
) -> None:
    """Render een leegsituatie met icoon, uitleg en optionele call-to-action.

    Gebruikt door pagina's die nog geen data/project hebben, zodat de gebruiker
    altijd een volgende stap ziet in plaats van een leeg scherm.
    """
    with ui.column().classes("w-full items-center text-center gap-3 py-12"):
        ui.icon(icon).classes("text-6xl opacity-40")
        ui.label(title).classes("text-xl font-medium")
        ui.label(message).classes("text-sm opacity-70 max-w-md")
        if action_label and on_action is not None:
            ui.button(action_label, on_click=on_action).props("unelevated")


def error_banner(message: str, hint: str | None = None) -> None:
    """Render een foutbanner met een actiegerichte hersteltekst.

    Args:
        message: Wat er misging, in gewone taal.
        hint: Concrete herstelstap (bijv. "Open Configuratie om het pad aan te
            passen"). Optioneel maar sterk aangeraden.
    """
    with (
        ui.card()
        .classes("w-full border-l-4")
        .style(f"border-color: {theme.NEGATIVE}; background: #fdecea")
    ):
        with ui.row().classes("items-start gap-2 no-wrap"):
            ui.icon("error").style(f"color: {theme.NEGATIVE}").classes("text-xl mt-1")
            with ui.column().classes("gap-1"):
                ui.label(message).classes("font-medium")
                if hint:
                    ui.label(hint).classes("text-sm opacity-80")


def info_banner(message: str) -> None:
    """Render een neutrale informatiebanner."""
    with (
        ui.card()
        .classes("w-full border-l-4")
        .style(f"border-color: {theme.INFO}; background: #e8f4fb")
    ):
        with ui.row().classes("items-center gap-2 no-wrap"):
            ui.icon("info").style(f"color: {theme.INFO}").classes("text-xl")
            ui.label(message).classes("text-sm")


def section_title(text: str, subtitle: str | None = None) -> None:
    """Render een consistente sectiekop."""
    ui.label(text).classes("text-lg font-medium mt-2")
    if subtitle:
        ui.label(subtitle).classes("text-sm opacity-60 -mt-1")
