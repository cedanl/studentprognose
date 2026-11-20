"""NiceGUI entry point + routing voor de studentprognose-GUI.

Start de webapp op ``localhost:8080``. De app is een schil rond de CLI: elke
pagina bouwt een ``studentprognose``-commando en voert dat via een subprocess uit
(zie :mod:`gui.components.cli_runner`). De GUI importeert geen pipeline-interne
modules.
"""

from __future__ import annotations

import asyncio
import os

from fastapi import File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
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
    """Verwerk één telbestand-CSV (voor de map-uploadmodus in de wizard).

    Beveiliging: dit endpoint luistert op localhost en heeft geen auth, dus het
    is bereikbaar voor een cross-origin POST (CSRF) vanuit een kwaadaardige site
    in de browser van de gebruiker. Daarom schrijven we **alleen** binnen het
    server-side actieve project (``STATE.project_dir``) en negeren we de door de
    client opgegeven ``project_dir`` als die niet exact daarmee overeenkomt. De
    bestandsnaam wordt bovendien gesaneerd in :func:`save_and_validate_telbestand`
    (padtraversal), zodat een write nooit buiten ``telbestanden/`` kan landen.
    """
    from gui.data_upload import save_and_validate_telbestand
    from gui.state import STATE

    active = STATE.project_dir
    if active is None or os.path.realpath(project_dir) != os.path.realpath(active):
        raise HTTPException(status_code=403, detail="Onbekende of inactieve projectmap.")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Geen bestandsnaam ontvangen.")

    content = await file.read()
    result = await asyncio.to_thread(
        save_and_validate_telbestand, active, file.filename, content
    )
    return {
        "filename": result.filename,
        "status": result.status.value,
        "hard_errors": result.hard_errors,
        "soft_errors": result.soft_errors,
        "warnings": result.warnings,
        "row_count": result.row_count,
        "actual_columns": result.actual_columns,
        "missing_required": result.missing_required,
    }


@app.get("/project-dashboards/{rel_path:path}")
async def _serve_dashboard(rel_path: str) -> FileResponse:
    """Serveer een gegenereerd dashboard uit het actieve project.

    Alleen ``.html`` binnen ``<project>/data/output/visualisations`` (zie
    :func:`gui.results_io.resolve_dashboard`), zodat de route geen willekeurige
    bestanden van de schijf kan uitlezen.
    """
    from gui import results_io
    from gui.state import STATE

    if STATE.output_dir is None:
        raise HTTPException(status_code=404, detail="Geen actief project.")
    path = results_io.resolve_dashboard(STATE.output_dir, rel_path)
    if path is None:
        raise HTTPException(status_code=404, detail="Dashboard niet gevonden.")
    return FileResponse(path, media_type="text/html")


def _register_pages() -> None:
    """Registreer alle paginaroutes.

    Elke pagina leeft in :mod:`gui.pages` en registreert zichzelf via een
    ``create()``-functie. Zo blijft dit bestand een dunne router en groeit het
    niet mee met elke feature-issue.
    """
    from gui.pages import (
        benchmark,
        config_page,
        home,
        methodology,
        output,
        run,
        wizard,
    )

    home.create()
    wizard.create()
    config_page.create()
    run.create()
    output.create()
    benchmark.create()
    methodology.create()


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
        uvicorn_logging_level="debug",
    )


# NiceGUI draait onder `python -m gui` in het hoofdproces (__main__) en, met
# reload, in een subprocess (__mp_main__). Beide moeten de pagina's registreren.
if __name__ in {"__main__", "__mp_main__"}:
    main()
