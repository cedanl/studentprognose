"""Grafische gebruikersinterface (NiceGUI) voor studentprognose.

Deze package is een dunne schil rond de bestaande ``studentprognose``-CLI. Ze
importeert geen interne pipeline-modules maar bouwt CLI-commando's en stuurt die
via subprocessen aan — de CLI blijft het contract (zie issue #273).

Start met::

    uv run --extra gui python -m gui
"""

__all__ = ["main"]


def main() -> None:
    """Start de NiceGUI-webapp. Zie :mod:`gui.app`."""
    from gui.app import main as _main

    _main()
