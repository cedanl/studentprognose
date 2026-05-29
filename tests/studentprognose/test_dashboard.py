"""Resilience tests voor dashboard-generatie.

Een fout in de dashboard-stap mag de rest van de pipeline niet laten falen — de
voorspelling zelf moet alsnog beschikbaar zijn. Deze tests pinnen dat gedrag vast.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

import studentprognose.main as main_module
from studentprognose.cli import PipelineConfig


def _make_strategy_stub():
    postprocessor = SimpleNamespace(
        data=pd.DataFrame({"Croho groepeernaam": ["X"], "Collegejaar": [2024], "Weeknummer": [10]}),
        numerus_fixus_list={},
        data_studentcount=None,
    )
    return SimpleNamespace(
        postprocessor=postprocessor,
        get_dashboard_data=lambda: {
            "data_cumulative": None,
            "xgboost_curve": None,
            "xgb_classifier_importance": None,
            "xgb_regressor_importance": None,
        },
    )


def test_dashboard_failure_is_logged_to_file(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    class _BoomDashboard:
        def __init__(self, **_kwargs):
            pass

        def build_and_save(self):
            raise RuntimeError("synthetic dashboard failure")

    monkeypatch.setattr(main_module, "DashboardBuilder", _BoomDashboard)

    cfg = PipelineConfig(weeks=[10], years=[2024], dashboard=True)

    main_module._try_build_dashboard(_make_strategy_stub(), cfg)

    log_path = tmp_path / "data" / "output" / "dashboard_error.log"
    assert log_path.exists(), "dashboard_error.log moet zijn aangemaakt bij een crash"
    log_contents = log_path.read_text(encoding="utf-8")
    assert "synthetic dashboard failure" in log_contents
    assert "Traceback" in log_contents

    captured = capsys.readouterr()
    assert "Waarschuwing: dashboard niet gegenereerd" in captured.out
    assert str(log_path) in captured.out


def test_dashboard_failure_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class _BoomDashboard:
        def __init__(self, **_kwargs):
            raise RuntimeError("crash in __init__")

    monkeypatch.setattr(main_module, "DashboardBuilder", _BoomDashboard)

    cfg = PipelineConfig(weeks=[10], years=[2024], dashboard=True)

    try:
        main_module._try_build_dashboard(_make_strategy_stub(), cfg)
    except Exception as exc:  # pragma: no cover - vangnet test
        pytest.fail(f"_try_build_dashboard mag niet propageren: {exc!r}")


def test_save_results_skips_dashboard_by_default(monkeypatch):
    """Zonder --dashboard mag _save_results de dashboard-builder niet aanroepen."""
    called = {"n": 0}

    def _spy(strategy, cfg):
        called["n"] += 1

    monkeypatch.setattr(main_module, "_try_build_dashboard", _spy)

    postprocessor = SimpleNamespace(
        data=pd.DataFrame({"x": [1]}),
        save_output=lambda _pred: None,
        save_totaal_audit_trail=lambda _pred: None,
    )
    strategy = SimpleNamespace(postprocessor=postprocessor)
    cfg = PipelineConfig(weeks=[10], years=[2024])  # dashboard=False (default)

    main_module._save_results(strategy, cfg)

    assert called["n"] == 0, "dashboard mag niet draaien zonder --dashboard"


def test_save_results_runs_dashboard_when_enabled(monkeypatch):
    """Met --dashboard moet _save_results de dashboard-builder triggeren."""
    called = {"n": 0}

    def _spy(strategy, cfg):
        called["n"] += 1

    monkeypatch.setattr(main_module, "_try_build_dashboard", _spy)

    postprocessor = SimpleNamespace(
        data=pd.DataFrame({"x": [1]}),
        save_output=lambda _pred: None,
        save_totaal_audit_trail=lambda _pred: None,
    )
    strategy = SimpleNamespace(postprocessor=postprocessor)
    cfg = PipelineConfig(weeks=[10], years=[2024], dashboard=True)

    main_module._save_results(strategy, cfg)

    assert called["n"] == 1, "dashboard moet draaien met --dashboard"
