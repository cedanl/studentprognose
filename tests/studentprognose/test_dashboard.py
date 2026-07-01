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
        final_academic_week=38,
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


# ── build_dashboard_from_dataframes (issue #253) ─────────────────────


def _install_dashboard_fake_strategy(monkeypatch, *, result="frame"):
    """Vervang create_strategy door een lichtgewicht fake met echte PostProcessor.

    Zo testen we de wiring van ``build_dashboard_from_dataframes`` (setup -> strategy
    -> DashboardBuilder -> HTML) end-to-end, zónder de zware SARIMA/XGBoost-compute
    (conform de projectconventie dat unit tests de volledige pipeline niet draaien).
    """
    import studentprognose.main as main_mod
    from studentprognose.config import load_defaults
    from studentprognose.output.postprocessor import PostProcessor
    from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK

    def _frame():
        return pd.DataFrame({
            "Croho groepeernaam": ["B Foo"],
            "Faculteit": ["FdM"],
            "Examentype": ["Bachelor"],
            "Collegejaar": [2025],
            "Herkomst": ["NL"],
            "Weeknummer": [10],
            "SARIMA_cumulative": [100.0],
            "SARIMA_individual": [110.0],
            "Voorspelde vooraanmelders": [120.0],
        })

    class _FakeStrategy:
        def __init__(self, cwd, opt):
            configuration = load_defaults()
            self.postprocessor = PostProcessor(
                configuration=configuration, data_latest=None, ensemble_weights=None,
                data_studentcount=None, cwd=cwd, data_option=opt, ci_test_n=None,
            )
            self.numerus_fixus_list = configuration["numerus_fixus"]
            self.final_academic_week = configuration.get("model_config", {}).get(
                "final_academic_week", FINAL_ACADEMIC_WEEK
            )
            self.programme_filtering = []

        def preprocess(self):
            return None

        def set_filtering(self, *args, **kwargs):
            pass

        def predict_nr_of_students(self, year, week, skip_years=0):
            if result == "none":
                return None
            if result == "empty":
                return _frame().iloc[0:0]  # zelfde kolommen, geen rijen
            return _frame()

        def get_dashboard_data(self):
            return {
                "data_cumulative": None, "xgboost_curve": None,
                "xgb_classifier_importance": None, "xgb_regressor_importance": None,
            }

    monkeypatch.setattr(
        main_mod, "create_strategy",
        lambda cfg, datasets, configuration, cwd: _FakeStrategy(cwd, cfg.data_option),
    )


def test_build_dashboard_from_dataframes_importable():
    from studentprognose import build_dashboard_from_dataframes

    assert callable(build_dashboard_from_dataframes)


def test_build_dashboard_from_dataframes_rejects_final_week():
    """De gedeelde year/week-validatie geldt ook voor de dashboard-route."""
    from studentprognose import build_dashboard_from_dataframes
    from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK

    with pytest.raises(ValueError, match=str(FINAL_ACADEMIC_WEEK)):
        build_dashboard_from_dataframes(year=2025, week=FINAL_ACADEMIC_WEEK)


def test_build_dashboard_from_dataframes_writes_html(tmp_path, monkeypatch):
    from studentprognose import DataOption, build_dashboard_from_dataframes

    _install_dashboard_fake_strategy(monkeypatch)
    monkeypatch.chdir(tmp_path)

    data_individual = pd.DataFrame({"Collegejaar": [2025], "Croho groepeernaam": ["B Foo"]})

    out = build_dashboard_from_dataframes(
        year=2025, week=10, data_individual=data_individual, dataset=DataOption.INDIVIDUAL,
    )

    vis = tmp_path / "data" / "output" / "visualisations"
    assert out == str(vis), "moet de visualisatiemap teruggeven"
    assert (vis / "final" / "dashboard.html").exists()
    # save_output wordt intern op False gezet: geen excel-/csv-uitvoer.
    assert not list((tmp_path / "data" / "output").glob("*.xlsx"))


@pytest.mark.parametrize("result", ["none", "empty"])
def test_build_dashboard_from_dataframes_raises_without_data(tmp_path, monkeypatch, result):
    """Geen voorspelrijen (None óf lege DataFrame) -> heldere ValueError i.p.v. een
    leeg/stil dashboard of een cryptische int(NaN)-crash in DashboardBuilder."""
    from studentprognose import DataOption, build_dashboard_from_dataframes

    _install_dashboard_fake_strategy(monkeypatch, result=result)
    monkeypatch.chdir(tmp_path)

    data_individual = pd.DataFrame({"Collegejaar": [2025], "Croho groepeernaam": ["B Foo"]})

    with pytest.raises(ValueError, match="Geen data om een dashboard"):
        build_dashboard_from_dataframes(
            year=2025, week=10, data_individual=data_individual, dataset=DataOption.INDIVIDUAL,
        )
