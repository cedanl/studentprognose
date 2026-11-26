"""Tests voor de feedbacklink en de instellingskiezer (#322)."""

import urllib.parse

from gui.components.layout import FEEDBACK_ADDRESS, feedback_mailto
from gui.pages.config_page import _institution_choice


def test_feedback_mailto_contains_address_and_text():
    url = feedback_mailto("  Grafiek is onduidelijk & te klein  ")
    assert url.startswith(f"mailto:{FEEDBACK_ADDRESS}?")
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert query["body"] == ["Grafiek is onduidelijk & te klein"]
    assert query["subject"] == ["Feedback Studentprognose-GUI"]


def test_institution_choice_single_code():
    assert _institution_choice("21PK") == ["21PK"]
    assert _institution_choice(" 28DN ") == ["28DN"]


def test_institution_choice_empty():
    assert _institution_choice(None) == []
    assert _institution_choice("") == []


def test_separator_name_is_readable():
    from gui.pages.config_page import _separator_name

    assert _separator_name(",") == "komma"
    assert _separator_name(";") == "puntkomma"
    assert _separator_name("#") == "'#'"
