"""Tests voor studentprognose.output.evaluation (aggregatie van output naar metrieken)."""

import math

import numpy as np
import pandas as pd
import pytest

from studentprognose.output.evaluation import (
    evaluate_predictions,
    pivot_metrics,
    to_mlflow_metrics,
)


def _sample_output():
    """Mini-output die de structuur van een cumulatieve run nabootst.

    - 'A' en 'B' zijn reguliere opleidingen; 'NF' is numerus fixus.
    - SARIMA_cumulative staat (net als in productie) alleen op de peilweek (week 12);
      latere weken (week 20) zijn NaN.
    - Prognose_ratio is op meerdere weken gevuld.
    - De MAE_*-kolommen zijn al door de pipeline berekend en zetten de NF-rij op NaN.
    """
    return pd.DataFrame(
        {
            "Croho groepeernaam": ["A", "A", "B", "B", "NF"],
            "Herkomst": ["NL", "NL", "NL", "NL", "NL"],
            "Examentype": ["Bachelor", "Bachelor", "Master", "Master", "Bachelor"],
            "Collegejaar": [2023, 2023, 2023, 2023, 2023],
            "Weeknummer": [12, 20, 12, 20, 12],
            "SARIMA_cumulative": [100.0, np.nan, 200.0, np.nan, 50.0],
            "Prognose_ratio": [110.0, 95.0, 190.0, 205.0, 55.0],
            "Aantal_studenten": [120.0, 120.0, 180.0, 180.0, 60.0],
            "MAE_SARIMA_cumulative": [20.0, np.nan, 20.0, np.nan, np.nan],
            "MAE_Prognose_ratio": [10.0, 25.0, 10.0, 25.0, np.nan],
            "MAPE_SARIMA_cumulative": [20 / 120, np.nan, 20 / 180, np.nan, np.nan],
            "MAPE_Prognose_ratio": [10 / 120, 25 / 120, 10 / 180, 25 / 180, np.nan],
        }
    )


def test_returns_row_per_prediction_column():
    metrics = evaluate_predictions(_sample_output(), week=12)
    assert set(metrics["prediction"]) == {"SARIMA_cumulative", "Prognose_ratio"}
    assert list(metrics.columns) == [
        "prediction",
        "n",
        "mae",
        "mape",
        "wape",
        "rmse",
        "bias",
        "r2",
    ]


def test_week_filter_selects_peilweek_only():
    metrics = evaluate_predictions(_sample_output(), week=12)
    sarima = metrics[metrics["prediction"] == "SARIMA_cumulative"].iloc[0]
    # Alleen A en B op week 12 tellen mee (NF uitgesloten via MAE-kolom).
    assert sarima["n"] == 2
    assert sarima["mae"] == pytest.approx(20.0)
    # WAPE = (20 + 20) / (120 + 180)
    assert sarima["wape"] == pytest.approx(40 / 300)
    # bias = mean(100-120, 200-180) = 0
    assert sarima["bias"] == pytest.approx(0.0)


def test_numerus_fixus_excluded_via_existing_mae_column():
    metrics = evaluate_predictions(_sample_output(), week=12)
    # NF-rij heeft Aantal_studenten 60 en pred 50; als die meetelde zou n=3 zijn.
    sarima = metrics[metrics["prediction"] == "SARIMA_cumulative"].iloc[0]
    assert sarima["n"] == 2


def test_nan_predictions_do_not_crash_and_are_skipped():
    # Zonder week-filter: SARIMA_cumulative is NaN op week 20 → alleen week-12-rijen tellen.
    metrics = evaluate_predictions(_sample_output())
    sarima = metrics[metrics["prediction"] == "SARIMA_cumulative"].iloc[0]
    assert sarima["n"] == 2
    # Prognose_ratio is op week 12 én 20 gevuld → 4 rijen (NF uitgesloten).
    ratio = metrics[metrics["prediction"] == "Prognose_ratio"].iloc[0]
    assert ratio["n"] == 4
    assert ratio["mae"] == pytest.approx((10 + 25 + 10 + 25) / 4)


def test_group_by_segments():
    metrics = evaluate_predictions(_sample_output(), week=12, group_by="Examentype")
    assert "Examentype" in metrics.columns
    sarima = metrics[metrics["prediction"] == "SARIMA_cumulative"]
    assert set(sarima["Examentype"]) == {"Bachelor", "Master"}
    bachelor = sarima[sarima["Examentype"] == "Bachelor"].iloc[0]
    assert bachelor["n"] == 1
    assert bachelor["mae"] == pytest.approx(20.0)


def test_custom_prediction_column_with_numerus_fixus_list():
    df = _sample_output().drop(columns=["MAE_SARIMA_cumulative", "MAE_Prognose_ratio"])
    df["MijnVoorspelling"] = [100.0, np.nan, 200.0, np.nan, 50.0]
    metrics = evaluate_predictions(
        df,
        week=12,
        prediction_columns=["MijnVoorspelling"],
        numerus_fixus=["NF"],
    )
    row = metrics.iloc[0]
    assert row["prediction"] == "MijnVoorspelling"
    assert row["n"] == 2  # NF uitgesloten via meegegeven lijst


def test_missing_actual_column_raises():
    df = _sample_output().drop(columns=["Aantal_studenten"])
    with pytest.raises(ValueError, match="Aantal_studenten"):
        evaluate_predictions(df, week=12)


def test_none_data_raises():
    with pytest.raises(ValueError, match="None"):
        evaluate_predictions(None)


def test_unknown_prediction_column_raises():
    with pytest.raises(ValueError, match="ontbreken"):
        evaluate_predictions(_sample_output(), prediction_columns=["Bestaat_niet"])


def test_to_mlflow_metrics_flat_keys_and_no_nan():
    metrics = evaluate_predictions(_sample_output(), week=12)
    flat = to_mlflow_metrics(metrics)
    assert flat["MAE_SARIMA_cumulative"] == pytest.approx(20.0)
    assert "WAPE_SARIMA_cumulative" in flat
    assert all(isinstance(v, float) for v in flat.values())
    assert all(not math.isnan(v) for v in flat.values())


def test_to_mlflow_metrics_grouped_keys_sanitized():
    metrics = evaluate_predictions(_sample_output(), week=12, group_by="Examentype")
    flat = to_mlflow_metrics(metrics, prefix="cumulatief/")
    # Sleutel bevat het prefix, de metriek, het model en de groepswaarde.
    assert "cumulatief/MAE_SARIMA_cumulative_Bachelor" in flat
    # Geen ongeldige tekens in de sleutels.
    import re

    assert all(re.fullmatch(r"[A-Za-z0-9_\-./ ]+", k) for k in flat)


def test_empty_after_week_filter_returns_empty_with_columns():
    with pytest.warns(UserWarning):
        metrics = evaluate_predictions(_sample_output(), week=99)
    assert metrics.empty
    assert "prediction" in metrics.columns


# --- pivot_metrics (View 3: model × jaar matrix + spreiding) ---------------------


def _sample_metrics():
    """Long-format metrieken zoals evaluate_predictions(..., group_by='Collegejaar').

    Twee modellen over drie backtest-jaren; 'Prognose_ratio' mist 2023 om de
    gaten-afhandeling (n_groups / min / max) te testen.
    """
    return pd.DataFrame(
        {
            "prediction": [
                "SARIMA_cumulative",
                "SARIMA_cumulative",
                "SARIMA_cumulative",
                "Prognose_ratio",
                "Prognose_ratio",
            ],
            "Collegejaar": [2021, 2022, 2023, 2021, 2022],
            "n": [10, 11, 12, 10, 11],
            "mae": [8.0, 7.0, 6.0, 9.0, 8.0],
            "mape": [0.10, 0.09, 0.08, 0.12, 0.11],
            "wape": [0.06, 0.05, 0.04, 0.08, 0.07],
            "rmse": [12.0, 11.0, 10.0, 14.0, 13.0],
            "bias": [1.0, -1.0, 0.5, -2.0, 1.0],
            "r2": [0.97, 0.98, 0.99, 0.95, 0.96],
        }
    )


def test_pivot_shape_index_and_columns():
    table = pivot_metrics(_sample_metrics(), value="wape", over="Collegejaar")
    assert table.index.name == "prediction"
    assert set(table.index) == {"SARIMA_cumulative", "Prognose_ratio"}
    assert [c for c in (2021, 2022, 2023)] == [c for c in table.columns if isinstance(c, int)]
    assert ["mean", "min", "max", "n_groups"] == list(table.columns)[-4:]


def test_pivot_cell_values_are_the_chosen_metric():
    table = pivot_metrics(_sample_metrics(), value="wape", over="Collegejaar")
    assert table.loc["SARIMA_cumulative", 2022] == pytest.approx(0.05)
    assert table.loc["Prognose_ratio", 2021] == pytest.approx(0.08)


def test_pivot_summary_mean_min_max_and_n_groups():
    table = pivot_metrics(_sample_metrics(), value="wape", over="Collegejaar")
    sarima = table.loc["SARIMA_cumulative"]
    assert sarima["mean"] == pytest.approx((0.06 + 0.05 + 0.04) / 3)
    assert sarima["min"] == pytest.approx(0.04)
    assert sarima["max"] == pytest.approx(0.06)
    assert sarima["n_groups"] == 3
    # Prognose_ratio mist 2023 → die cel is NaN en telt niet mee in summary.
    ratio = table.loc["Prognose_ratio"]
    assert math.isnan(ratio[2023])
    assert ratio["n_groups"] == 2
    assert ratio["mean"] == pytest.approx((0.08 + 0.07) / 2)


def test_pivot_value_selects_other_metric():
    table = pivot_metrics(_sample_metrics(), value="mae", over="Collegejaar")
    assert table.loc["SARIMA_cumulative", 2021] == pytest.approx(8.0)


def test_pivot_summary_false_only_group_columns():
    table = pivot_metrics(_sample_metrics(), value="wape", over="Collegejaar", summary=False)
    assert list(table.columns) == [2021, 2022, 2023]


def test_pivot_unknown_metric_raises():
    with pytest.raises(ValueError, match="Onbekende metriek"):
        pivot_metrics(_sample_metrics(), value="bestaat_niet", over="Collegejaar")


def test_pivot_missing_over_column_hints_group_by():
    with pytest.raises(ValueError, match="group_by"):
        pivot_metrics(_sample_metrics(), value="wape", over="Weeknummer")


def test_pivot_missing_prediction_column_raises():
    df = _sample_metrics().drop(columns=["prediction"])
    with pytest.raises(ValueError, match="prediction"):
        pivot_metrics(df, value="wape", over="Collegejaar")


def test_pivot_empty_metrics_returns_empty():
    empty = _sample_metrics().iloc[0:0]
    table = pivot_metrics(empty, value="wape", over="Collegejaar")
    assert table.empty


def test_pivot_end_to_end_from_pipeline_output():
    # Twee jaren output samenvoegen, per jaar evalueren, dan pivoteren.
    out_2022 = _sample_output().assign(Collegejaar=2022)
    out_2023 = _sample_output()  # heeft al Collegejaar 2023
    combined = pd.concat([out_2022, out_2023], ignore_index=True)
    metrics = evaluate_predictions(combined, week=12, group_by="Collegejaar")
    table = pivot_metrics(metrics, value="mae", over="Collegejaar")
    assert set(table.index) == {"SARIMA_cumulative", "Prognose_ratio"}
    assert {2022, 2023}.issubset(set(table.columns))
    # Identieke input per jaar → identieke MAE en dus nul spreiding.
    assert table.loc["SARIMA_cumulative", "min"] == pytest.approx(
        table.loc["SARIMA_cumulative", "max"]
    )


def test_pivot_casts_float_period_labels_to_int():
    # Collegejaar kan als float binnenkomen → kolomlabels moeten 2021 zijn, niet 2021.0.
    metrics = _sample_metrics().astype({"Collegejaar": "float64"})
    table = pivot_metrics(metrics, value="wape", over="Collegejaar", summary=False)
    assert list(table.columns) == [2021, 2022, 2023]
    assert table.columns.dtype == np.dtype("int64")


def test_pivot_single_group_has_zero_spread():
    metrics = _sample_metrics()
    one_year = metrics[metrics["Collegejaar"] == 2021]
    table = pivot_metrics(one_year, value="wape", over="Collegejaar")
    sarima = table.loc["SARIMA_cumulative"]
    assert sarima["n_groups"] == 1
    assert sarima["mean"] == sarima["min"] == sarima["max"]


def test_pivot_value_n_returns_counts_not_averaged():
    # Eén rij per (prediction, jaar) → aggfunc='mean' geeft de count ongewijzigd terug.
    table = pivot_metrics(_sample_metrics(), value="n", over="Collegejaar", summary=False)
    assert table.loc["SARIMA_cumulative", 2022] == 11
