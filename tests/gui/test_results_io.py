"""Tests voor gui.results_io — samenvatten van de voorspellingsoutput."""

import pandas as pd

from gui import results_io


def _sample_output():
    # Twee opleidingen × NL, voorspelling verankerd op week 10; week 11 = NaN.
    return pd.DataFrame(
        {
            "Croho groepeernaam": ["B A", "B A", "B B", "B B"],
            "Herkomst": ["NL", "NL", "NL", "NL"],
            "Examentype": ["Bachelor", "Bachelor", "Master", "Master"],
            "Collegejaar": [2024, 2024, 2024, 2024],
            "Weeknummer": [10, 11, 10, 11],
            "SARIMA_cumulative": [100.0, None, 50.0, None],
            "Aantal_studenten": [80.0, 80.0, 60.0, 60.0],
            "Prognose_ratio": [90.0, 88.0, 55.0, 54.0],
            "MAPE_SARIMA_cumulative": [0.25, None, 0.17, None],
            "MAPE_Prognose_ratio": [0.12, None, 0.08, None],
            "MAE_SARIMA_cumulative": [20.0, None, 10.0, None],
            "MAE_Prognose_ratio": [10.0, None, 5.0, None],
        }
    )


def test_prediction_rows_anchors_on_prediction_week():
    rows = results_io.prediction_rows(_sample_output())
    assert len(rows) == 2  # alleen de week-10 verankerrijen
    assert set(rows["Weeknummer"]) == {10}


def test_compute_kpis():
    kpis = results_io.compute_kpis(_sample_output())
    assert kpis["n_programmes"] == 2
    assert kpis["year"] == 2024
    assert kpis["week"] == 10
    # Prognose_ratio heeft de laagste gemiddelde MAPE ((0.12+0.08)/2=0.10).
    assert kpis["best_model"] == "Prognose_ratio"
    assert abs(kpis["best_mape"] - 0.10) < 1e-9


def test_model_comparison():
    rows = results_io.prediction_rows(_sample_output())
    comp = results_io.model_comparison(rows)
    assert comp["SARIMA_cumulative"] == 15.0  # (20+10)/2
    assert comp["Prognose_ratio"] == 7.5  # (10+5)/2


def test_mape_bucket_thresholds():
    assert results_io.mape_bucket(0.05) == "positive"
    assert results_io.mape_bucket(0.20) == "warning"
    assert results_io.mape_bucket(0.40) == "negative"
    assert results_io.mape_bucket(None) == "grey"


def test_programme_rows_sorted_by_deviation():
    records = results_io.programme_rows(_sample_output())
    assert len(records) == 2
    # B A: |100-80|=20 > B B: |50-60|=10 → B A eerst.
    assert records[0]["opleiding"] == "B A"
    assert records[0]["afwijking"] == 20.0
    assert records[0]["kleur"] == "negative"  # MAPE 0.25


def test_audit_diff_needs_two_runs():
    single = _sample_output().assign(Run_date="2024-01-01")
    assert results_io.audit_diff(single)["available"] is False


def test_audit_diff_detects_new_and_shifts():
    run1 = pd.DataFrame(
        {
            "Croho groepeernaam": ["B A"],
            "Herkomst": ["NL"],
            "Examentype": ["Bachelor"],
            "Weeknummer": [10],
            "SARIMA_cumulative": [100.0],
            "Run_date": ["2024-01-01"],
        }
    )
    run2 = pd.DataFrame(
        {
            "Croho groepeernaam": ["B A", "B NEW"],
            "Herkomst": ["NL", "NL"],
            "Examentype": ["Bachelor", "Master"],
            "Weeknummer": [10, 10],
            "SARIMA_cumulative": [130.0, 40.0],  # B A +30%
            "Run_date": ["2024-01-08", "2024-01-08"],
        }
    )
    diff = results_io.audit_diff(pd.concat([run1, run2], ignore_index=True))
    assert diff["available"] is True
    assert any("B NEW" in p for p in diff["new_programmes"])
    assert diff["shifts"] and abs(diff["shifts"][0]["verschil"] - 0.30) < 1e-9
