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

from gui import nav, theme
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

            # Kleur draagt de betekenis: accent (oranje) = actief, donkergrijs =
            # klikbaar, lichtgrijs = uitgeschakeld. Op een uitgeschakelde q-btn
            # (opacity 0.7) leest lichtgrijs duidelijk als "nu niet beschikbaar".
            if item.route == active:
                color = "accent"
            elif enabled:
                color = "grey-9"
            else:
                color = "grey-5"

            classes = "w-full justify-start"
            if item.route == active:
                # Zachte oranje tint + linker accentrand voor de actieve pagina.
                classes += " font-medium rounded"

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
                if item.route == active:
                    btn.style(
                        f"background: {theme.ACCENT}1a; "
                        f"border-left: 3px solid {theme.ACCENT}"
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


def _feedback_fab() -> None:
    """Zwevende feedbackknop rechtsonder + dialoogvenster."""
    with ui.dialog() as dialog, ui.card().style(
        "width: 440px; max-width: 95vw; border-radius: 16px; overflow: hidden; padding: 0;"
    ):
        # ── Koptekst ────────────────────────────────────────────────────────
        with ui.row().classes("w-full items-center justify-between px-5 pt-4 pb-3").style(
            "border-bottom: 1px solid #f0f0f0;"
        ):
            with ui.row().classes("items-center gap-2 no-wrap"):
                ui.icon("chat_bubble_outline").classes("text-xl").style(
                    f"color: {theme.ACCENT}"
                )
                ui.label("Feedback geven").classes("font-semibold text-base")
            ui.button(icon="close", on_click=dialog.close).props(
                "flat round dense color=grey-6"
            )

        # ── Formulier ───────────────────────────────────────────────────────
        with ui.column().classes("w-full gap-4 px-5 pt-4 pb-5"):
            ui.label(
                "Deel jouw gedachten, ideeën of meldingen — wij lezen alles."
            ).classes("text-sm leading-relaxed").style("color: #666;")

            ui.textarea(placeholder="Typ hier jouw feedback…").props(
                "outlined autogrow"
            ).classes("w-full").style("font-size: 14px;")

            # ── Voettekst: e-mail links, knoppen rechts ──────────────────
            with ui.row().classes("w-full items-center justify-between no-wrap gap-2"):
                with ui.row().classes("items-center gap-1 no-wrap"):
                    ui.icon("mail_outline").classes("text-sm").style("color: #aaa;")
                    ui.label("ceda@surf.nl").classes("text-xs font-mono").style(
                        "color: #aaa;"
                    )
                with ui.row().classes("gap-2 no-wrap"):
                    ui.button("Annuleren", on_click=dialog.close).props(
                        "flat color=grey-7"
                    )
                    ui.button("Versturen", icon="send").props(
                        "unelevated color=accent"
                    )

    # ── FAB ─────────────────────────────────────────────────────────────────
    (
        ui.button("Feedback", icon="chat_bubble_outline", on_click=dialog.open)
        .props("unelevated")
        .style(
            f"position: fixed; bottom: 28px; right: 28px; z-index: 900;"
            f"background: {theme.PRIMARY}; color: #fff;"
            f"border-radius: 100px; padding: 0 20px; height: 44px;"
            f"font-size: 13px; letter-spacing: 0.02em;"
            f"box-shadow: 0 4px 16px rgba(0,0,0,0.22);"
        )
    )


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
        with ui.row().classes("items-center gap-3 no-wrap"):
            # Officieel Npuls-logo (wit) — co-branding met de toolnaam.
            ui.image("/gui-assets/npuls-logo-white.svg").classes("w-8 h-8")
            ui.label("Studentprognose").classes("text-lg font-medium text-white")
        with ui.row().classes("items-center gap-3 no-wrap"):
            ui.label(title).classes("text-sm text-white opacity-70")
            def _reset() -> None:
                STATE.project_dir = None
                ui.navigate.to("/")
            ui.button(icon="restart_alt", on_click=_reset).props(
                "flat round dense color=white"
            ).tooltip("Reset — terug naar start")

    _drawer(active)

    _feedback_fab()

    with ui.column().classes("w-full max-w-5xl mx-auto p-6 gap-4"):
        if show_stepper:
            _stepper(active)
        yield
