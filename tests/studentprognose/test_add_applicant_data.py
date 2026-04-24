"""Tests for PostProcessor.add_applicant_data."""

import pandas as pd

from studentprognose.output.postprocessor import PostProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KEY_COLS = ["Weeknummer", "Examentype", "Croho groepeernaam", "Herkomst", "Collegejaar"]
UPDATE_COLS = [
    "Gewogen vooraanmelders",
    "Ongewogen vooraanmelders",
    "Aantal aanmelders met 1 aanmelding",
    "Inschrijvingen",
]


def _make_postprocessor():
    cfg = {"numerus_fixus": {}, "ensemble_override_cumulative": [], "ensemble_weights": {
        "master_week_17_23": {"individual": 0.5, "cumulative": 0.5},
        "week_30_34": {"individual": 0.5, "cumulative": 0.5},
        "week_35_37": {"individual": 0.5, "cumulative": 0.5},
        "default": {"individual": 0.5, "cumulative": 0.5},
    }}
    from studentprognose.utils.weeks import DataOption
    pp = PostProcessor(
        configuration=cfg,
        data_latest=None,
        ensemble_weights=None,
        data_studentcount=None,
        cwd="/tmp",
        data_option=DataOption.CUMULATIVE,
        ci_test_n=None,
    )
    return pp


def _make_data_row(year=2024, week=10, programme="B Opleiding", herkomst="NL",
                   examentype="Bachelor", gewogen=0.0):
    row = {c: [0.0] for c in UPDATE_COLS}
    row.update({
        "Collegejaar": [year],
        "Weeknummer": [week],
        "Croho groepeernaam": [programme],
        "Herkomst": [herkomst],
        "Examentype": [examentype],
        "Gewogen vooraanmelders": [gewogen],
        "Ongewogen vooraanmelders": [gewogen],
        "Aantal aanmelders met 1 aanmelding": [gewogen],
        "Inschrijvingen": [gewogen],
    })
    return pd.DataFrame(row)


def _make_snapshot(year=2024, week=10, programme="B Opleiding", herkomst="NL",
                   examentype="Bachelor", gewogen=99.0):
    return _make_data_row(year, week, programme, herkomst, examentype, gewogen)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAddApplicantData:
    def test_updates_matching_row(self):
        pp = _make_postprocessor()
        pp.data = _make_data_row(gewogen=0.0)
        snapshot = _make_snapshot(gewogen=99.0)

        pp.add_applicant_data(snapshot, predict_year=2024, predict_week=10)

        assert pp.data.loc[0, "Gewogen vooraanmelders"] == 99.0

    def test_does_not_update_other_years(self):
        pp = _make_postprocessor()
        pp.data = pd.concat([
            _make_data_row(year=2023, week=10, gewogen=0.0),
            _make_data_row(year=2024, week=10, gewogen=0.0),
        ]).reset_index(drop=True)
        snapshot = _make_snapshot(year=2024, week=10, gewogen=99.0)

        pp.add_applicant_data(snapshot, predict_year=2024, predict_week=10)

        assert pp.data.loc[pp.data["Collegejaar"] == 2023, "Gewogen vooraanmelders"].iloc[0] == 0.0
        assert pp.data.loc[pp.data["Collegejaar"] == 2024, "Gewogen vooraanmelders"].iloc[0] == 99.0

    def test_does_not_update_other_weeks(self):
        pp = _make_postprocessor()
        pp.data = pd.concat([
            _make_data_row(year=2024, week=9, gewogen=0.0),
            _make_data_row(year=2024, week=10, gewogen=0.0),
        ]).reset_index(drop=True)
        snapshot = _make_snapshot(year=2024, week=10, gewogen=99.0)

        pp.add_applicant_data(snapshot, predict_year=2024, predict_week=10)

        assert pp.data.loc[pp.data["Weeknummer"] == 9, "Gewogen vooraanmelders"].iloc[0] == 0.0
        assert pp.data.loc[pp.data["Weeknummer"] == 10, "Gewogen vooraanmelders"].iloc[0] == 99.0

    def test_no_match_in_snapshot_leaves_data_unchanged(self):
        pp = _make_postprocessor()
        pp.data = _make_data_row(programme="B Opleiding", gewogen=7.0)
        snapshot = _make_snapshot(programme="B Andere Opleiding", gewogen=99.0)

        pp.add_applicant_data(snapshot, predict_year=2024, predict_week=10)

        assert pp.data.loc[0, "Gewogen vooraanmelders"] == 7.0

    def test_empty_snapshot_leaves_data_unchanged(self):
        pp = _make_postprocessor()
        pp.data = _make_data_row(gewogen=7.0)
        empty_snapshot = pd.DataFrame(columns=pp.data.columns)

        pp.add_applicant_data(empty_snapshot, predict_year=2024, predict_week=10)

        assert pp.data.loc[0, "Gewogen vooraanmelders"] == 7.0

    def test_none_data_returns_early(self):
        pp = _make_postprocessor()
        pp.data = None
        pp.add_applicant_data(_make_snapshot(), predict_year=2024, predict_week=10)  # no exception

    def test_none_cumulative_returns_early(self):
        pp = _make_postprocessor()
        pp.data = _make_data_row(gewogen=7.0)
        pp.add_applicant_data(None, predict_year=2024, predict_week=10)
        assert pp.data.loc[0, "Gewogen vooraanmelders"] == 7.0

    def test_duplicate_snapshot_rows_keep_last(self):
        pp = _make_postprocessor()
        pp.data = _make_data_row(gewogen=0.0)
        duplicate = pd.concat([
            _make_snapshot(gewogen=10.0),
            _make_snapshot(gewogen=99.0),
        ]).reset_index(drop=True)

        pp.add_applicant_data(duplicate, predict_year=2024, predict_week=10)

        assert pp.data.loc[0, "Gewogen vooraanmelders"] == 99.0

    def test_updates_all_update_cols(self):
        pp = _make_postprocessor()
        pp.data = _make_data_row(gewogen=0.0)
        snapshot = _make_snapshot(gewogen=42.0)

        pp.add_applicant_data(snapshot, predict_year=2024, predict_week=10)

        for col in UPDATE_COLS:
            assert pp.data.loc[0, col] == 42.0, f"{col} not updated"

    def test_non_contiguous_index_updates_correct_row(self):
        """reset_index(drop=True) in add_applicant_data must realign idx with merge position."""
        pp = _make_postprocessor()
        # Simulate index [5, 6] — as if rows 0-4 were dropped earlier
        data = pd.concat([
            _make_data_row(year=2024, week=10, programme="B Opleiding", gewogen=0.0),
            _make_data_row(year=2024, week=10, programme="B Andere", gewogen=0.0),
        ])
        data.index = [5, 6]
        pp.data = data
        snapshot = pd.concat([
            _make_snapshot(year=2024, week=10, programme="B Opleiding", gewogen=11.0),
            _make_snapshot(year=2024, week=10, programme="B Andere", gewogen=22.0),
        ])

        pp.add_applicant_data(snapshot, predict_year=2024, predict_week=10)

        assert pp.data.loc[5, "Gewogen vooraanmelders"] == 11.0
        assert pp.data.loc[6, "Gewogen vooraanmelders"] == 22.0
