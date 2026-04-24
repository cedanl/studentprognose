"""Tests for the Baseline column in PostProcessor.postprocess()."""

import pandas as pd
import pytest

from studentprognose.output.postprocessor import PostProcessor
from studentprognose.utils.weeks import DataOption


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_postprocessor():
    cfg = {
        "numerus_fixus": {},
        "ensemble_override_cumulative": [],
        "ensemble_weights": {
            "master_week_17_23": {"individual": 0.5, "cumulative": 0.5},
            "week_30_34": {"individual": 0.5, "cumulative": 0.5},
            "week_35_37": {"individual": 0.5, "cumulative": 0.5},
            "default": {"individual": 0.5, "cumulative": 0.5},
        },
    }
    return PostProcessor(
        configuration=cfg,
        data_latest=None,
        ensemble_weights=None,
        data_studentcount=None,
        cwd="/tmp",
        data_option=DataOption.CUMULATIVE,
        ci_test_n=None,
    )


def _make_data(prognose_ratio=100.0):
    return pd.DataFrame({
        "Collegejaar": [2024],
        "Weeknummer": [10],
        "Croho groepeernaam": ["B Opleiding"],
        "Herkomst": ["NL"],
        "Examentype": ["Bachelor"],
        "Prognose_ratio": [prognose_ratio],
        "SARIMA_cumulative": [95.0],
        "Aantal_studenten": [98.0],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaseline:
    def test_baseline_equals_prognose_ratio_after_postprocess(self):
        """Baseline must be written as a copy of Prognose_ratio by postprocess()."""
        pp = _make_postprocessor()
        pp.data = _make_data(prognose_ratio=120.0)

        pp.postprocess(predict_year=2024, predict_week=10)

        assert "Baseline" in pp.data.columns
        assert pp.data["Baseline"].iloc[0] == pytest.approx(120.0)

    def test_baseline_tracks_prognose_ratio_per_row(self):
        """Each row's Baseline must equal that row's Prognose_ratio."""
        pp = _make_postprocessor()
        pp.data = pd.DataFrame({
            "Collegejaar": [2024, 2024],
            "Weeknummer": [10, 10],
            "Croho groepeernaam": ["B Opleiding A", "B Opleiding B"],
            "Herkomst": ["NL", "NL"],
            "Examentype": ["Bachelor", "Bachelor"],
            "Prognose_ratio": [80.0, 200.0],
            "SARIMA_cumulative": [75.0, 190.0],
            "Aantal_studenten": [78.0, 195.0],
        })

        pp.postprocess(predict_year=2024, predict_week=10)

        result = pp.data.set_index("Croho groepeernaam")
        assert result.loc["B Opleiding A", "Baseline"] == pytest.approx(80.0)
        assert result.loc["B Opleiding B", "Baseline"] == pytest.approx(200.0)
