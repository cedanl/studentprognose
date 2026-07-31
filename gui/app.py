"""NiceGUI entry point + routing voor de studentprognose-GUI.

Start de webapp op ``localhost:8080``. De app is een schil rond de CLI: elke
pagina bouwt een ``studentprognose``-commando en voert dat via een subprocess uit
(zie :mod:`gui.components.cli_runner`). De GUI importeert geen pipeline-interne
modules.
"""

from __future__ import annotations

import asyncio
import os

from fastapi import File, Query, UploadFile
from nicegui import app, ui

#: Poort waarop de GUI draait. Vast, zodat de gedocumenteerde URL klopt.
PORT = 8080

#: Map met huisstijl-assets (logo, hero, favicon), geserveerd op ``/gui-assets``.
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
ASSETS_URL = "/gui-assets"


@app.post("/api/upload-telbestand")
async def _api_upload_telbestand(
    file: UploadFile = File(...),
    project_dir: str = Query(...),
) -> dict:
    """Verwerk één telbestand-CSV (voor de map-uploadmodus in de wizard)."""
    from gui.data_upload import save_and_validate_telbestand

    content = await file.read()
    result = await asyncio.to_thread(
        save_and_validate_telbestand, project_dir, file.filename, content
    )
    return {
        "filename": result.filename,
        "status": result.status.value,
        "hard_errors": result.hard_errors,
        "soft_errors": result.soft_errors,
        "warnings": result.warnings,
        "row_count": result.row_count,
    }


def _register_pages() -> None:
    """Registreer alle paginaroutes.

    Elke pagina leeft in :mod:`gui.pages` en registreert zichzelf via een
    ``create()``-functie. Zo blijft dit bestand een dunne router en groeit het
    niet mee met elke feature-issue.
    """
    from gui.pages import (
        api_explorer,
        benchmark,
        concept_hub,
        config_page,
        explainability,
        filtering,
        home,
        methodology,
        output,
        peer_benchmark,
        run,
        scenarios,
        uxflow,
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
    # UX Flow
    uxflow.create()
    # Concept-features
    concept_hub.create()
    scenarios.create()
    peer_benchmark.create()
    api_explorer.create()
    explainability.create()


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
        host="127.0.0.1",
        port=PORT,
        reload=False,
        show=False,
        favicon=os.path.join(ASSETS_DIR, "favicon.svg"),
    )


# NiceGUI draait onder `python -m gui` in het hoofdproces (__main__) en, met
# reload, in een subprocess (__mp_main__). Beide moeten de pagina's registreren.
if __name__ in {"__main__", "__mp_main__"}:
    main()
