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

import os
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
    """Render de zijbalk met projectcontext, navigatie-items en feedbacklink."""
    feedback = _feedback_dialog()

    with ui.left_drawer(fixed=False).classes("bg-grey-1 gap-1").style(
        "display: flex; flex-direction: column;"
    ):
        # Project-contextblok — toont welk project actief is zodat de gebruiker
        # altijd weet in welke werkmap de pipeline draait.
        if STATE.is_initialised:
            project_name = os.path.basename(STATE.project_dir or "")
            path = STATE.project_dir or ""
            short_path = ("…" + path[-28:]) if len(path) > 30 else path
            with ui.element("div").classes("mx-3 mt-3 mb-1 px-3 py-2 rounded-lg").style(
                f"background:{theme.ACCENT}12; border-left:3px solid {theme.ACCENT}"
            ):
                with ui.row().classes("items-center gap-2 no-wrap"):
                    ui.icon("folder_open").classes("text-base").style(
                        f"color:{theme.ACCENT}"
                    )
                    with ui.column().classes("gap-0"):
                        ui.label(project_name).classes(
                            "text-sm font-semibold"
                        ).style("max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap")
                        ui.label(short_path).classes("text-xs font-mono opacity-50").style(
                            "max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                        )
            ui.separator().classes("mx-3 my-1")
        ui.label("Navigatie").classes("text-xs uppercase opacity-50 px-3 pt-2")
        prev_was_enabled = True
        for item in nav.all_items():
            built = nav.is_available(item.route)
            locked = _requires_project(item.route) and not STATE.is_initialised
            enabled = built and not locked

            # Visuele scheiding tussen beschikbare en uitgeschakelde items.
            if not enabled and prev_was_enabled:
                ui.separator().classes("mx-3 my-1 opacity-30")
            prev_was_enabled = enabled

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
                    btn.style("font-size: 12px; opacity: 0.55;")
                    wrapper.tooltip(reason)

        # ── Concept-features — verborgen op de startpagina ──────────────────
        if active != "/":
            ui.separator().classes("mx-3 mt-3 mb-1 opacity-40")
            is_concept_active = active == "/concept"
            with (
                ui.row()
                .classes("items-center gap-1 px-3 pt-1 pb-1 cursor-pointer rounded")
                .style("background:#FFEBEE;" if is_concept_active else "")
                .on("click", lambda: _navigate("/concept"))
            ):
                ui.label("Concept").classes("text-xs uppercase font-semibold").style(
                    "color: #E53935; letter-spacing: 0.06em;"
                )
                ui.badge("preview").props("color=red-2 text-color=red-8 dense outline").classes(
                    "text-xs"
                )
            for item in nav.CONCEPTS:
                is_active = item.route == active
                with ui.element("div").classes("w-full"):
                    (
                        ui.button(
                            item.label,
                            icon=item.icon,
                            on_click=lambda r=item.route: _navigate(r),
                        )
                        .props("flat align=left")
                        .classes("w-full justify-start" + (" font-medium rounded" if is_active else ""))
                        .style(
                            f"color: {'#B71C1C' if is_active else '#E53935'};"
                            + ("background: #FFEBEE; border-left: 3px solid #E53935;" if is_active else "")
                        )
                    )

        # ── Feedback — gepind onderaan de zijbalk ───────────────────────────
        ui.element("div").style("flex: 1;")  # duwt feedback naar beneden
        ui.separator().classes("mx-3 opacity-50")
        with ui.element("div").classes("w-full px-2 pt-1 pb-3"):
            (
                ui.button(
                    "Feedback geven",
                    icon="chat_bubble_outline",
                    on_click=feedback.open,
                )
                .props("flat align=left")
                .classes("w-full justify-start")
                .style("color: #9e9e9e; font-size: 13px;")
            )


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


def _feedback_dialog() -> ui.dialog:
    """Bouw het feedback-dialoogvenster en geef de referentie terug."""
    with ui.dialog() as dialog, ui.card().style(
        "width: 440px; max-width: 95vw; border-radius: 16px; overflow: hidden; padding: 0;"
    ):
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

        with ui.column().classes("w-full gap-4 px-5 pt-4 pb-5"):
            ui.label(
                "Deel jouw gedachten, ideeën of meldingen — wij lezen alles."
            ).classes("text-sm leading-relaxed").style("color: #666;")
            ui.textarea(placeholder="Typ hier jouw feedback…").props(
                "outlined autogrow"
            ).classes("w-full").style("font-size: 14px;")
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
                    ui.button("Versturen", icon="send").props("unelevated color=accent")
    return dialog


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
            ui.button("Reset", icon="restart_alt", on_click=_reset).props(
                "flat dense color=white"
            ).tooltip("Reset — terug naar start")

    _drawer(active)

    with ui.column().classes("w-full max-w-5xl mx-auto p-6 gap-4"):
        if show_stepper:
            _stepper(active)
        yield
