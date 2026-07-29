"""Methodologie-pagina: het interactieve drie-sporen-schema.

Bereikbaar via de "Methodologie"-knop op de startpagina. Toont de flow van de
drie voorspelsporen; hover geeft korte uitleg, klik op een spoor opent de
uitgebreide methodologie-documentatie.
"""

from __future__ import annotations

from nicegui import ui

from gui import nav, theme, viz
from gui.components.layout import page_shell
from gui.components.states import section_title
from gui.pages.home import _USE_CASES


def create() -> None:
    """Registreer de route ``/methodologie``."""
    nav.register_route("/methodologie")

    @ui.page("/methodologie")
    def methodology_page() -> None:
        A = theme.ACCENT
        with page_shell(active="/methodologie", title="Methodologie", show_stepper=False):

            # ── Waarvoor gebruik je instroom-prognoses? ───────────────────────
            section_title(
                "Waarvoor gebruik je instroom-prognoses?",
                "Acht concrete toepassingen voor data-analisten en beleidsmakers.",
            )
            with ui.card().classes("w-full"):
                with ui.column().classes("w-full gap-0"):
                    for i, (icon, title, desc) in enumerate(_USE_CASES):
                        sep = i < len(_USE_CASES) - 1
                        with ui.element("div").style(
                            f"padding: 12px 16px;"
                            + ("border-bottom: 1px solid #f3f3f3;" if sep else "")
                        ):
                            with ui.row().classes("items-start gap-3 no-wrap"):
                                with ui.element("div").style(
                                    f"width:34px;height:34px;border-radius:8px;"
                                    f"background:{A}12;display:flex;"
                                    "align-items:center;justify-content:center;flex-shrink:0;"
                                ):
                                    ui.icon(icon).classes("text-lg").style(
                                        f"color:{A}"
                                    )
                                with ui.column().classes("gap-0"):
                                    ui.label(title).classes("text-sm font-semibold")
                                    ui.label(desc).classes(
                                        "text-xs leading-relaxed"
                                    ).style("color:#666;")

            # ── Drie voorspelsporen ───────────────────────────────────────────
            section_title(
                "Methodologie — drie voorspelsporen",
                "Zo voorspelt de tool studentinstroom. Hover over een spoor voor "
                "uitleg, klik erop voor de uitgebreide documentatie.",
            )
            with ui.card().classes("w-full"):
                ui.html(viz.flow_svg()).classes("w-full")
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("Hover over een spoor voor uitleg.").classes(
                        "text-xs opacity-60"
                    )
                    ui.link("Meer over de methodologie →", viz.DOCS).props(
                        "target=_blank"
                    ).classes("text-sm").style(f"color: {theme.ACCENT}")

            with ui.row().classes("gap-2 flex-wrap"):
                ui.button(
                    "Ga naar documentatie",
                    icon="open_in_new",
                    on_click=lambda: ui.navigate.to(
                        "https://cedanl.github.io/studentprognose"
                    ),
                ).props("outline color=accent")
                ui.button(
                    "Naar Uitvoeren",
                    icon="play_arrow",
                    on_click=lambda: ui.navigate.to("/run"),
                ).props("unelevated")
