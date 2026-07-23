"""NiceGUI entry point + routing voor de studentprognose-GUI.

Start de webapp op ``localhost:8080``. De app is een schil rond de CLI: elke
pagina bouwt een ``studentprognose``-commando en voert dat via een subprocess uit
(zie :mod:`gui.components.cli_runner`). De GUI importeert geen pipeline-interne
modules.
"""

from __future__ import annotations

import os

from nicegui import app, ui

#: Poort waarop de GUI draait. Vast, zodat de gedocumenteerde URL klopt.
PORT = 8080

#: Map met huisstijl-assets (logo, hero, favicon), geserveerd op ``/gui-assets``.
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
ASSETS_URL = "/gui-assets"


def _register_pages() -> None:
    """Registreer alle paginaroutes.

    Elke pagina leeft in :mod:`gui.pages` en registreert zichzelf via een
    ``create()``-functie. Zo blijft dit bestand een dunne router en groeit het
    niet mee met elke feature-issue.
    """
    from gui.pages import (
        benchmark,
        config_page,
        filtering,
        home,
        methodology,
        nav_prototypes,
        output,
        run,
        wizard,
    )

    home.create()
    wizard.create()
    config_page.create()
    filtering.create()
    run.create()
    output.create()
    benchmark.create()
    methodology.create()
    nav_prototypes.create()


def main() -> None:
    """Registreer de pagina's en start de NiceGUI-server.

    Het thema (``ui.colors``) wordt per pagina toegepast in
    :func:`gui.components.layout.page_shell`. In de globale scope aanroepen zou
    NiceGUI's script-mode triggeren en botsen met ``@ui.page``.
    """
    app.add_static_files(ASSETS_URL, ASSETS_DIR)
    _register_pages()
    ui.run(
        title="Studentprognose",
        port=PORT,
        reload=False,
        show=False,
        favicon=os.path.join(ASSETS_DIR, "favicon.svg"),
    )


# NiceGUI draait onder `python -m gui` in het hoofdproces (__main__) en, met
# reload, in een subprocess (__mp_main__). Beide moeten de pagina's registreren.
if __name__ in {"__main__", "__mp_main__"}:
    main()
