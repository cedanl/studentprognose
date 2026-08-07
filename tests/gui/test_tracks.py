"""Tests voor de sporen-metadata (één bron van waarheid over de drie sporen)."""

from gui import tracks


def test_track_lookup_by_label():
    t = tracks.track("Cumulatief")
    assert t is not None
    assert t.label == "Cumulatief"
    assert t.icon  # niet leeg
    assert t.short


def test_track_unknown_label_returns_none():
    assert tracks.track("Bestaat niet") is None


def test_all_three_tracks_present():
    labels = {t.label for t in tracks.TRACKS}
    assert labels == {"Cumulatief", "Individueel", "Beide"}
    assert tracks.RECOMMENDED in labels


def test_dataset_tooltip_default_lists_all_and_hint():
    tip = tracks.dataset_tooltip()
    lines = tip.splitlines()
    # Drie sporen + één hint-regel (Beide zit erin).
    assert len(lines) == 4
    for t in tracks.TRACKS:
        assert any(line.startswith(f"{t.label}: ") for line in lines)
    assert lines[-1] == f"Niet zeker? Kies '{tracks.RECOMMENDED}'."


def test_dataset_tooltip_subset_without_recommended_has_no_hint():
    tip = tracks.dataset_tooltip(["Cumulatief", "Individueel"])
    lines = tip.splitlines()
    assert len(lines) == 2
    assert not any("Niet zeker" in line for line in lines)


def test_dataset_tooltip_subset_with_recommended_appends_hint():
    tip = tracks.dataset_tooltip(["Beide"])
    lines = tip.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("Beide: ")
    assert lines[-1] == f"Niet zeker? Kies '{tracks.RECOMMENDED}'."
