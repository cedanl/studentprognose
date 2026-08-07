"""Tests voor gui.errors.humanize_error — pure logica, geen NiceGUI nodig."""

from gui.errors import _GENERIC, humanize_error


def test_file_not_found_is_actionable():
    result = humanize_error("FileNotFoundError: no such file 'x.xlsx'")
    assert "ontbreekt" in result.message.lower()
    assert result.hint  # er is altijd een herstelstap


def test_permission_denied():
    result = humanize_error("PermissionError: permission denied")
    assert "schrijfrechten" in result.message.lower()


def test_final_week():
    result = humanize_error("week 36 is de laatste week van het academisch jaar")
    assert "laatste week" in result.message.lower()


def test_range_mismatch():
    result = humanize_error("year/week valt buiten de range van de data")
    assert result.hint


def test_unknown_falls_back_to_generic():
    assert humanize_error("iets volstrekt onbekends") is _GENERIC


def test_empty_input_is_safe():
    assert humanize_error("") is _GENERIC
    assert humanize_error(None) is _GENERIC
