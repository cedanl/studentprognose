"""Smoke test: de GUI-modules importeren en pagina's registreren zonder fouten.

Slaat over als NiceGUI niet geïnstalleerd is (de `gui`-extra is optioneel en zit
niet in de standaard test-dependencies).
"""

import pytest

pytest.importorskip("nicegui", reason="gui-extra niet geïnstalleerd")


def test_pages_register_without_error():
    """`_register_pages` moet de routes registreren zonder te crashen."""
    from gui.app import _register_pages

    _register_pages()


def test_theme_tokens_are_hex_colors():
    from gui.theme import QUASAR_COLORS

    for name, value in QUASAR_COLORS.items():
        assert value.startswith("#"), f"{name} is geen hex-kleur: {value}"


def test_demo_filtering_is_valid():
    """De demo-scope moet geldige filterwaarden bevatten."""
    from gui import filtering_io
    from gui.pages.home import _DEMO_FILTERING

    assert filtering_io.validate_filtering(_DEMO_FILTERING) == []
