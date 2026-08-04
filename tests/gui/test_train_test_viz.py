"""Tests voor de dataverdeling-visualisatie en het afgeleide traindata-bereik.

Bewaakt dat het traindata-bereik uit de werkelijke data-overlap komt (niet een
hardgecodeerd startjaar) en dat de training nooit voorbij het laatste jaar met
realisatiedata reikt.
"""

import os

import pandas as pd

from gui.components import train_test_viz as tvz
from gui.data_upload import scan_data_year_bounds


# ── _compute: traindata binnen de overlap ──────────────────────────────────


def test_compute_uses_data_start_not_hardcoded():
    """Traindata begint bij het meegegeven startjaar, niet bij DATA_START."""
    d = tvz._compute("2026", 0, data_start=2020, data_end=2025)
    assert d is not None
    assert d["train"] == (2020, 2025)
    assert d["test"] is None
    assert d["pred"] == (2026, 2026)


def test_compute_training_capped_at_data_end():
    """Bij een gat tussen data en prognose stopt de training op het overlap-eind."""
    d = tvz._compute("2028", 0, data_start=2020, data_end=2025)
    assert d is not None
    assert d["train"] == (2020, 2025)  # niet t/m 2027
    assert d["pred"] == (2028, 2028)


def test_compute_backtest_from_tail_of_overlap():
    """Backtest-jaren komen uit de staart van de overlap; training ervoor."""
    d = tvz._compute("2026", 2, data_start=2020, data_end=2025)
    assert d is not None
    assert d["train"] == (2020, 2023)
    assert d["test"] == (2024, 2025)
    assert d["test_n"] == 2


def test_compute_backtest_beyond_overlap_is_dropped():
    """Backtest voorbij het laatste realisatiejaar heeft geen truth → geen test."""
    d = tvz._compute("2028", 2, data_start=2020, data_end=2025)
    assert d is not None
    assert d["test"] is None
    assert d["train"] == (2020, 2025)


def test_compute_returns_none_without_training_room():
    """Prognose direct op het startjaar laat geen trainingsjaar over."""
    assert tvz._compute("2020", 0, data_start=2020, data_end=2025) is None


def test_render_v1_reflects_overlap_not_2016():
    """De gerenderde HTML toont het echte startjaar, niet het oude 2016."""
    html = tvz.render_v1("2026", 0, "26", data_start=2020, data_end=2025)
    assert "2020" in html
    assert "2016" not in html


# ── scan_data_year_bounds: overlap uit projectbestanden ─────────────────────


def _make_project(tmp_path, tel_years, okt_years, min_training_year=None):
    """Bouw een minimaal project met telbestand-namen en een oktober-bestand."""
    tel_dir = tmp_path / "data" / "input_raw" / "telbestanden"
    tel_dir.mkdir(parents=True)
    for year in tel_years:
        # Studielink-formaat 'telbestandY{year}W{week}'.
        (tel_dir / f"telbestandY{year}W10.csv").write_text("x", encoding="utf-8")

    okt_path = tmp_path / "data" / "input_raw" / "oktober_bestand.xlsx"
    pd.DataFrame({"Collegejaar": list(okt_years)}).to_excel(okt_path, index=False)

    if min_training_year is not None:
        cfg_dir = tmp_path / "configuration"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "configuration.json").write_text(
            f'{{"model_config": {{"min_training_year": {min_training_year}}}}}',
            encoding="utf-8",
        )
    return str(tmp_path)


def test_scan_bounds_intersection(tmp_path):
    project = _make_project(tmp_path, tel_years=[2020, 2021, 2022, 2023], okt_years=[2020, 2021, 2022])
    bounds = scan_data_year_bounds(project)
    assert bounds is not None
    assert bounds.overlap == [2020, 2021, 2022]
    assert bounds.train_start == 2020
    assert bounds.train_end == 2022


def test_scan_bounds_respects_config_floor(tmp_path):
    project = _make_project(
        tmp_path, tel_years=[2020, 2021, 2022, 2023], okt_years=[2020, 2021, 2022, 2023],
        min_training_year=2022,
    )
    bounds = scan_data_year_bounds(project)
    assert bounds is not None
    assert bounds.train_start == 2022  # vloer uit config
    assert bounds.train_end == 2023


def test_scan_bounds_none_without_overlap(tmp_path):
    project = _make_project(tmp_path, tel_years=[2024, 2025], okt_years=[2020, 2021])
    assert scan_data_year_bounds(project) is None


def test_scan_bounds_none_without_telbestanden(tmp_path):
    (tmp_path / "data" / "input_raw").mkdir(parents=True)
    okt = tmp_path / "data" / "input_raw" / "oktober_bestand.xlsx"
    pd.DataFrame({"Collegejaar": [2020, 2021]}).to_excel(okt, index=False)
    assert scan_data_year_bounds(str(tmp_path)) is None
