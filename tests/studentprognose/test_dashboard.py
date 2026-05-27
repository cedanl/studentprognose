"""Resilience tests voor dashboard-generatie.

Een fout in de dashboard-stap mag de rest van de pipeline niet laten falen — de
voorspelling zelf moet alsnog beschikbaar zijn. Deze tests pinnen dat gedrag vast.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import studentprognose.main as main_module
from studentprognose.cli import PipelineConfig
from studentprognose.output.dashboard import DashboardBuilder
from studentprognose.utils.weeks import DataOption


def _make_builder_stub(tmp_path, data: pd.DataFrame, data_option: DataOption) -> DashboardBuilder:
    """Construct a DashboardBuilder without running __init__ — alleen de
    attributen die ``build_and_save`` raadpleegt worden geprikt."""
    builder = DashboardBuilder.__new__(DashboardBuilder)
    builder.data = data
    builder.data_option = data_option
    builder.cwd = str(tmp_path)
    builder._save_individual = MagicMock()
    builder._save_cumulative = MagicMock()
    builder._save_final = MagicMock()
    return builder


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
    )
    strategy = SimpleNamespace(postprocessor=postprocessor)
    cfg = PipelineConfig(weeks=[10], years=[2024], dashboard=True)

    main_module._save_results(strategy, cfg)

    assert called["n"] == 1, "dashboard moet draaien met --dashboard"


# ──────────────────────────────────────────────────────────────────────
# Per-pagina fault-isolation in DashboardBuilder.build_and_save
# ──────────────────────────────────────────────────────────────────────


def test_dashboard_partial_failure_does_not_skip_other_pages(tmp_path, capsys):
    """Als _save_cumulative crasht, moeten _save_individual en _save_final
    nog steeds aangeroepen worden, met een duidelijke stdout-waarschuwing
    en een per-pagina sectie in dashboard_error.log."""
    data = pd.DataFrame({"SARIMA_individual": [1], "SARIMA_cumulative": [1]})
    builder = _make_builder_stub(tmp_path, data, DataOption.BOTH_DATASETS)
    builder._save_cumulative.side_effect = RuntimeError("synthetic cumulative failure")

    builder.build_and_save()

    builder._save_individual.assert_called_once()
    builder._save_cumulative.assert_called_once()
    builder._save_final.assert_called_once(), "final-pagina mag niet skippen omdat cumulative faalde"

    captured = capsys.readouterr()
    assert "Waarschuwing: dashboard 'cumulative' niet gegenereerd" in captured.out
    assert "fouten in: cumulative" in captured.out

    log_path = tmp_path / "data" / "output" / "dashboard_error.log"
    assert log_path.exists()
    contents = log_path.read_text(encoding="utf-8")
    assert "=== cumulative ===" in contents
    assert "synthetic cumulative failure" in contents


def test_dashboard_gate_skip_prints_explicit_reason(tmp_path, capsys):
    """Als SARIMA_cumulative en Prognose_ratio beide ontbreken, moet de
    cumulative-pagina worden overgeslagen met expliciete reden in stdout —
    geen stille skip."""
    data = pd.DataFrame({"SARIMA_individual": [1]})  # geen cumulative-kolommen
    builder = _make_builder_stub(tmp_path, data, DataOption.BOTH_DATASETS)

    builder.build_and_save()

    builder._save_individual.assert_called_once()
    builder._save_cumulative.assert_not_called()
    builder._save_final.assert_called_once()

    captured = capsys.readouterr()
    assert "Dashboard 'cumulative' overgeslagen" in captured.out
    assert "SARIMA_cumulative" in captured.out
    assert "Prognose_ratio" in captured.out


def test_dashboard_clears_stale_log_on_success(tmp_path, capsys):
    """Een eerdere fout-log mag niet blijven staan na een succesvolle run."""
    log_path = tmp_path / "data" / "output" / "dashboard_error.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("=== oude fout ===\nstale traceback\n", encoding="utf-8")

    data = pd.DataFrame({"SARIMA_individual": [1], "SARIMA_cumulative": [1]})
    builder = _make_builder_stub(tmp_path, data, DataOption.BOTH_DATASETS)

    builder.build_and_save()

    assert not log_path.exists(), "log met verouderde fout moet bij een succesvolle run worden opgeruimd"
