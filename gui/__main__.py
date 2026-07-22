"""Maakt `python -m gui` mogelijk."""

from gui.app import main

if __name__ in {"__main__", "__mp_main__"}:
    main()
