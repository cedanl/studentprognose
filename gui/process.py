"""Lokaliseren en samenstellen van CLI-aanroepen.

De GUI voert de bestaande ``studentprognose``-console-script uit via een
subprocess. Deze module is pure logica (geen NiceGUI) zodat ze getest kan worden;
het live streamen naar de UI gebeurt in :mod:`gui.components.log_stream`.
"""

from __future__ import annotations

import os
import shutil
import sys


def locate_cli() -> str:
    """Vind het pad naar de ``studentprognose``-console-script.

    Zoekt eerst naast de draaiende Python-interpreter (de venv-``bin``/``Scripts``),
    zodat de GUI en de CLI gegarandeerd dezelfde omgeving delen; valt terug op de
    ``PATH``.

    Returns:
        Absoluut pad naar het uitvoerbare bestand.

    Raises:
        FileNotFoundError: Als de console-script nergens gevonden wordt.
    """
    bindir = os.path.dirname(sys.executable)
    for name in ("studentprognose", "studentprognose.exe"):
        candidate = os.path.join(bindir, name)
        if os.path.isfile(candidate):
            return candidate

    found = shutil.which("studentprognose")
    if found:
        return found

    raise FileNotFoundError(
        "De 'studentprognose'-CLI is niet gevonden. Installeer het pakket in "
        "dezelfde omgeving als de GUI (bijv. `uv sync --extra gui`)."
    )


def preview_command(args: list[str]) -> str:
    """Bouw een leesbare weergave van het commando voor de UI-preview.

    Args:
        args: Argumenten ná ``studentprognose`` (bijv. ``["-d", "c", "-w", "6"]``).

    Returns:
        Een string zoals ``studentprognose -d c -w 6``.
    """
    parts = ["studentprognose", *args]
    return " ".join(parts)
