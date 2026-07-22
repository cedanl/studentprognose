"""Tests voor gui.benchmark_io — samenvatten en tune-output parsen."""

import pandas as pd

from gui import benchmark_io


def test_summarize_lower_is_better():
    df = pd.DataFrame(
        {
            "model": ["sarima", "sarima", "ets", "ets"],
            "mape": [0.20, 0.30, 0.10, 0.14],
        }
    )
    records = benchmark_io.summarize(df, "mape", higher_is_better=False)
    assert records[0]["model"] == "ets"  # laagste gemiddelde MAPE (0.12)
    assert records[0]["winner"] is True
    assert records[1]["winner"] is False


def test_summarize_higher_is_better():
    df = pd.DataFrame({"model": ["a", "b"], "auc_roc": [0.7, 0.9]})
    records = benchmark_io.summarize(df, "auc_roc", higher_is_better=True)
    assert records[0]["model"] == "b"
    assert records[0]["winner"] is True


def test_summarize_ignores_inf():
    df = pd.DataFrame({"model": ["a", "a", "b"], "mape": [float("inf"), 0.2, 0.5]})
    records = benchmark_io.summarize(df, "mape", higher_is_better=False)
    # 'a' gemiddelde = 0.2 (inf genegeerd) < 'b' = 0.5
    assert records[0]["model"] == "a"


def test_extract_config_snippets():
    text = """
    ── Regressor (stap 2: xgboost) ──
    Beste parameters voor 'xgboost' (MAPE=0.1234, 12 kandidaten):
    {
        "model_config": {
            "regressor_params": {
                "xgboost": {"learning_rate": 0.1, "n_estimators": 200}
            }
        }
    }
    """
    snippets = benchmark_io.extract_config_snippets(text)
    assert len(snippets) == 1
    assert (
        snippets[0]["model_config"]["regressor_params"]["xgboost"]["n_estimators"]
        == 200
    )


def test_extract_config_snippets_ignores_non_config_json():
    text = 'wat tekst {"iets": 1} en meer {"model_config": {"x": 1}}'
    snippets = benchmark_io.extract_config_snippets(text)
    assert len(snippets) == 1
    assert snippets[0] == {"model_config": {"x": 1}}


def test_extract_config_snippets_none():
    assert benchmark_io.extract_config_snippets("geen json hier") == []


def test_deep_merge_nested():
    base = {"model_config": {"a": 1, "regressor_params": {"x": {"lr": 0.1}}}}
    overlay = {"model_config": {"regressor_params": {"y": {"lr": 0.2}}}}
    merged = benchmark_io.deep_merge(base, overlay)
    assert merged["model_config"]["a"] == 1
    assert merged["model_config"]["regressor_params"]["x"] == {"lr": 0.1}
    assert merged["model_config"]["regressor_params"]["y"] == {"lr": 0.2}


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1}}
    overlay = {"a": {"c": 2}}
    benchmark_io.deep_merge(base, overlay)
    assert base == {"a": {"b": 1}}  # ongewijzigd
