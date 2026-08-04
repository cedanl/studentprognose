"""Tests voor de rungeschiedenis-parsing en het menselijk leesbare label."""

from datetime import datetime

from gui.run_history import RunEntry, _parse_stamp


# ── _parse_stamp: mapnaam → timestamp ──────────────────────────────────────


def test_parse_stamp_valid():
    ts = _parse_stamp("studentprognose20260804153012")
    assert ts == datetime(2026, 8, 4, 15, 30, 12)


def test_parse_stamp_wrong_prefix():
    assert _parse_stamp("prognose20260804153012") is None


def test_parse_stamp_too_few_digits():
    assert _parse_stamp("studentprognose202608") is None


def test_parse_stamp_trailing_chars_rejected():
    # De regex is verankerd op het einde ($): extra tekens tellen niet.
    assert _parse_stamp("studentprognose20260804153012_kopie") is None


def test_parse_stamp_impossible_date():
    # 13e maand bestaat niet → strptime faalt → None.
    assert _parse_stamp("studentprognose20261304153012") is None


# ── RunEntry.label: Nederlandse datum + correct meervoud ────────────────────


def test_label_plural_files():
    entry = RunEntry(output_dir="/x", timestamp=datetime(2026, 8, 4, 15, 30), n_files=3)
    assert entry.label == "4 aug 2026, 15:30 — 3 bestanden"


def test_label_singular_file():
    entry = RunEntry(output_dir="/x", timestamp=datetime(2026, 1, 9, 9, 5), n_files=1)
    assert entry.label == "9 jan 2026, 09:05 — 1 bestand"


def test_label_zero_files_uses_plural():
    entry = RunEntry(output_dir="/x", timestamp=datetime(2026, 12, 31, 0, 0), n_files=0)
    assert entry.label == "31 dec 2026, 00:00 — 0 bestanden"
