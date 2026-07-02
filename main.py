"""Convenience shim om de pipeline vanuit een broncode-checkout te draaien.

Dit is NIET het echte entry point. De geïnstalleerde tool draait via de
console-script ``studentprognose`` (zie ``pyproject.toml`` →
``studentprognose.main:cli``). Dit bestand bestaat zodat ``uv run main.py``
of ``python main.py`` in een clone blijven werken; alle logica leeft in
``src/studentprognose/main.py``.
"""

import sys

from studentprognose.main import main

if __name__ == "__main__":
    main(sys.argv)
