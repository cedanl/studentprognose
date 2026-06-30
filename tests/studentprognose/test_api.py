"""Tests for the public Python API exported from studentprognose.__init__."""

import importlib

import pandas as pd
import pytest


def test_load_configuration_importable():
    from studentprognose import load_configuration

    config = load_configuration("__nonexistent_path__.json")
    assert isinstance(config, dict)
    assert "paths" in config


def test_load_filtering_importable():
    from studentprognose import load_filtering

    filtering = load_filtering("__nonexistent_path__.json")
    assert isinstance(filtering, dict)
    assert "filtering" in filtering


def test_load_defaults_filtering_importable():
    from studentprognose import load_defaults_filtering

    filtering = load_defaults_filtering()
    assert isinstance(filtering, dict)
    assert "filtering" in filtering


def test_pipeline_config_importable():
    from studentprognose import PipelineConfig

    cfg = PipelineConfig()
    assert cfg.noetl is False


def test_data_option_importable():
    from studentprognose import DataOption

    assert DataOption.CUMULATIVE is not None
    assert DataOption.INDIVIDUAL is not None
    assert DataOption.BOTH_DATASETS is not None


def test_student_year_prediction_importable():
    from studentprognose import StudentYearPrediction

    assert StudentYearPrediction.FIRST_YEARS is not None
    assert StudentYearPrediction.HIGHER_YEARS is not None
    assert StudentYearPrediction.VOLUME is not None


def test_run_pipeline_cli_importable():
    from studentprognose import run_pipeline_cli

    assert callable(run_pipeline_cli)


def test_run_pipeline_from_dataframes_importable():
    from studentprognose import run_pipeline_from_dataframes

    assert callable(run_pipeline_from_dataframes)


def test_run_pipeline_from_dataframes_rejects_final_week():
    from studentprognose import run_pipeline_from_dataframes
    from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK

    with pytest.raises(ValueError, match=str(FINAL_ACADEMIC_WEEK)):
        run_pipeline_from_dataframes(year=2025, week=FINAL_ACADEMIC_WEEK)


def _make_prediction_frame():
    """Minimale voorspelling met exact de kolommen die de postprocessor selecteert."""
    return pd.DataFrame({
        "Croho groepeernaam": ["B Foo"],
        "Faculteit":          ["FdM"],
        "Examentype":         ["Bachelor"],
        "Collegejaar":        [2025],
        "Herkomst":           ["NL"],
        "Weeknummer":         [10],
        "SARIMA_cumulative":  [100.0],
        "SARIMA_individual":  [110.0],
        "Voorspelde vooraanmelders": [120.0],
    })


def _install_fake_strategy(monkeypatch, data_option):
    """Vervang create_strategy door een lichtgewicht fake.

    Doel: de save_output-gating in de orchestratielaag end-to-end via de publieke
    ``run_pipeline_from_dataframes`` testen, zónder de zware SARIMA/XGBoost-compute
    (conform de projectconventie dat unit tests de volledige pipeline niet draaien).
    De fake gebruikt een ECHTE PostProcessor, zodat de echte ``save_output_prelim``-
    schrijfpoging daadwerkelijk wordt gegate.
    """
    import studentprognose.main as main_mod
    from studentprognose.config import load_defaults
    from studentprognose.output.postprocessor import PostProcessor
    from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK

    class _FakeStrategy:
        def __init__(self, cwd, opt):
            configuration = load_defaults()
            self.postprocessor = PostProcessor(
                configuration=configuration,
                data_latest=None,
                ensemble_weights=None,
                data_studentcount=None,
                cwd=cwd,
                data_option=opt,
                ci_test_n=None,
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
            return _make_prediction_frame()

    def _fake_create_strategy(cfg, datasets, configuration, cwd):
        return _FakeStrategy(cwd, cfg.data_option)

    monkeypatch.setattr(main_mod, "create_strategy", _fake_create_strategy)


@pytest.mark.parametrize("save_output, expect_files", [(False, False), (True, True)])
def test_run_pipeline_respects_save_output(tmp_path, monkeypatch, save_output, expect_files):
    """save_output=False schrijft niets naar data/output/; save_output=True wel (issue #219)."""
    from studentprognose import run_pipeline_from_dataframes, DataOption

    _install_fake_strategy(monkeypatch, DataOption.INDIVIDUAL)
    # tmp_path als cwd: zowel check_output_writable (os.getcwd()) als de
    # postprocessor (self.CWD) schrijven dan binnen tmp_path.
    monkeypatch.chdir(tmp_path)

    data_individual = pd.DataFrame({
        "Collegejaar": [2025],
        "Croho groepeernaam": ["B Foo"],
    })

    run_pipeline_from_dataframes(
        year=2025,
        week=10,
        data_individual=data_individual,
        dataset=DataOption.INDIVIDUAL,
        save_output=save_output,
    )

    output_dir = tmp_path / "data" / "output"
    written = {p.name for p in output_dir.glob("*")} if output_dir.exists() else set()

    if expect_files:
        # Criterium: bij save_output=True blijven álle outputtypen behouden —
        # het tussenresultaat, het eindresultaat én de audittrail.
        expected = {
            "output_prelim_individueel.xlsx",
            "output_first-years_individueel.xlsx",
            "_totaal_first-years_individueel.xlsx",
        }
        assert expected <= written, f"ontbrekende outputs: {expected - written}"
    else:
        assert written == set(), (
            f"save_output=False mag niets schrijven, maar vond {sorted(written)}"
        )


def test_run_pipeline_from_dataframes_normalizes_programme_key(monkeypatch):
    """Regressie: de in-memory route normaliseert de programmesleutel (object/str
    -> Int64), net als load_data. Zonder dit lekt een object-sleutel uit de
    meegegeven DataFrames in self.data en crasht een latere merge op str-vs-Int64
    (de fout die cloud-/Fabric-gebruikers zagen)."""
    import studentprognose.main as main_mod
    from studentprognose import run_pipeline_from_dataframes, DataOption
    from studentprognose.config import load_defaults

    captured = {}

    def _fake_core(cfg, datasets, configuration, filtering, cwd, save_output=True):
        captured["datasets"] = datasets
        return None

    monkeypatch.setattr(main_mod, "_run_pipeline_core", _fake_core)

    # Object/string-sleutel zoals een Fabric-bron levert.
    df_cum = pd.DataFrame({
        "Collegejaar": [2024], "Weeknummer": [10],
        "Groepeernaam Croho": ["56604"],
    })
    label = pd.DataFrame({
        "Collegejaar": [2024], "Herkomst": ["NL"], "Examentype": ["Bachelor"],
        "Croho groepeernaam": ["56604"], "Aantal_studenten": [120],
    })

    run_pipeline_from_dataframes(
        year=2024, week=10,
        data_cumulative=df_cum, data_student_numbers=label,
        dataset=DataOption.CUMULATIVE, configuration=load_defaults(),
        save_output=False,
    )

    _, cum, lbl, _, _ = captured["datasets"]
    assert str(cum["Groepeernaam Croho"].dtype) == "Int64"
    assert str(lbl["Croho groepeernaam"].dtype) == "Int64"
    # De DataFrames van de aanroeper mogen niet in place gemuteerd zijn.
    assert str(df_cum["Groepeernaam Croho"].dtype) != "Int64"
    assert str(label["Croho groepeernaam"].dtype) != "Int64"


def test_all_exports_present():
    import studentprognose

    for name in studentprognose.__all__:
        obj = getattr(studentprognose, name, None)
        assert obj is not None, (
            f"'{name}' is listed in __all__ but cannot be imported from studentprognose"
        )
        assert hasattr(importlib.import_module("studentprognose"), name)


# --- Range-validatie van year/week in run_pipeline_from_dataframes (issue #218) ---


def _cumulative_frame(years=(2024, 2025), weeks=range(1, 53)):
    """Minimale cumulatieve trainingsdata: alleen de kolommen die de range-check leest.

    De range-check draait vóór ``_run_pipeline_core``, dus meer kolommen zijn niet nodig.
    """
    rows = [(year, week) for year in years for week in weeks]
    return pd.DataFrame(rows, columns=["Collegejaar", "Weeknummer"])


def test_rejects_year_before_training_data():
    from studentprognose import DataOption, run_pipeline_from_dataframes

    with pytest.raises(ValueError) as exc:
        run_pipeline_from_dataframes(
            year=2000,
            week=10,
            data_cumulative=_cumulative_frame(),
            dataset=DataOption.CUMULATIVE,
            save_output=False,
        )
    message = str(exc.value)
    assert "year=2000" in message
    assert "2024-2025" in message


def test_rejects_year_after_training_data():
    from studentprognose import DataOption, run_pipeline_from_dataframes

    with pytest.raises(ValueError) as exc:
        run_pipeline_from_dataframes(
            year=2030,
            week=10,
            data_cumulative=_cumulative_frame(),
            dataset=DataOption.CUMULATIVE,
            save_output=False,
        )
    message = str(exc.value)
    assert "year=2030" in message
    assert "2024-2025" in message


def test_rejects_week_below_range():
    from studentprognose import DataOption, run_pipeline_from_dataframes

    # week=0 passeert de FINAL_ACADEMIC_WEEK (38) guard en moet door de range-check vallen.
    with pytest.raises(ValueError) as exc:
        run_pipeline_from_dataframes(
            year=2025,
            week=0,
            data_cumulative=_cumulative_frame(),
            dataset=DataOption.CUMULATIVE,
            save_output=False,
        )
    message = str(exc.value)
    assert "week=0" in message
    assert "1-52" in message


def test_rejects_week_above_range():
    from studentprognose import DataOption, run_pipeline_from_dataframes

    with pytest.raises(ValueError) as exc:
        run_pipeline_from_dataframes(
            year=2025,
            week=53,
            data_cumulative=_cumulative_frame(),
            dataset=DataOption.CUMULATIVE,
            save_output=False,
        )
    message = str(exc.value)
    assert "week=53" in message
    assert "1-52" in message


def test_detector_individual_only_missing_year_no_indexerror():
    from studentprognose.data.range_check import (
        detect_data_range_mismatch,
        format_api_range_error,
    )

    # Individuele dataset heeft geen Weeknummer-kolom; een ontbrekend jaar mag geen
    # IndexError geven (regressie op de oude available_weeks[0]).
    data_individual = pd.DataFrame(
        {"Collegejaar": [2024, 2025], "Croho groepeernaam": ["B Foo", "B Foo"]}
    )
    mismatch = detect_data_range_mismatch((data_individual, None), [2030], [10])

    assert mismatch is not None
    assert mismatch.week_range == "n.v.t."
    assert mismatch.missing_weeks == []
    assert mismatch.missing_years == [2030]
    # De API-melding noemt alleen het jaar — geen verwarrend "n.v.t." voor weken.
    assert "n.v.t." not in format_api_range_error(2030, 10, mismatch)


def test_detector_in_range_returns_none():
    from studentprognose.data.range_check import detect_data_range_mismatch

    data_cumulative = pd.DataFrame({"Collegejaar": [2025], "Weeknummer": [10]})
    assert detect_data_range_mismatch((None, data_cumulative), [2025], [10]) is None


def test_format_cli_range_warning_with_weeks_full_string():
    from studentprognose.data.range_check import DataRangeMismatch, format_cli_range_warning

    mismatch = DataRangeMismatch(
        year_range="2024-2025", week_range="1-52", missing_years=[2030], missing_weeks=[]
    )
    # Volledige string-assert: bewaakt zowel de regelopbouw als de byte-identieke output
    # die de CLI verwacht (zie ook test_cli_check_data_range_exits).
    assert format_cli_range_warning(mismatch) == (
        "\nWaarschuwing: de gevraagde combinatie is niet (volledig) beschikbaar in de data."
        "\n  Beschikbare data: jaren 2024-2025, weken 1-52."
        "\n  Pas je flags aan tussen -y 2024-2025 en -w 1-52,"
        "\n  of voeg nieuwe trainingsdata toe in data/input_raw/ om je gewenste tijdstip te voorspellen."
    )


def test_format_cli_range_warning_nvt_branch_omits_week_flag():
    from studentprognose.data.range_check import DataRangeMismatch, format_cli_range_warning

    # Individuele dataset zonder Weeknummer-kolom -> week_range "n.v.t.".
    # Deze tak werd voorheen door geen enkele test geraakt.
    mismatch = DataRangeMismatch(
        year_range="2024-2025", week_range="n.v.t.", missing_years=[2030], missing_weeks=[]
    )
    warning = format_cli_range_warning(mismatch)
    assert "  Pas je flags aan tussen -y 2024-2025," in warning
    assert "-w" not in warning  # geen weekvlag wanneer weken onbekend zijn
    assert "n.v.t." not in warning  # geen verwarrend "n.v.t." in de gebruikerstekst
    assert "  Beschikbare data: jaren 2024-2025." in warning


def test_cli_check_data_range_exits(capsys):
    from studentprognose import PipelineConfig
    from studentprognose.main import _check_data_range

    data_cumulative = pd.DataFrame({"Collegejaar": [2024, 2025], "Weeknummer": [10, 11]})
    cfg = PipelineConfig(
        years=[2030], weeks=[10], years_specified=True, weeks_specified=True
    )

    with pytest.raises(SystemExit) as exc:
        _check_data_range((None, data_cumulative), cfg)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert (
        "Waarschuwing: de gevraagde combinatie is niet (volledig) beschikbaar in de data."
        in out
    )
    assert "jaren 2024-2025" in out


class TestResolveTuneTargets:
    """Bewaakt de vertaling van de publieke ``tune``-parameter naar targets."""

    def _resolve(self):
        from studentprognose.main import _resolve_tune_targets
        return _resolve_tune_targets

    def test_falsy_means_no_tuning(self):
        r = self._resolve()
        assert r(False) == {} and r(None) == {} and r({}) == {}

    def test_true_and_regressor_string(self):
        r = self._resolve()
        assert r(True) == {"regressor": None}
        assert r("regressor") == {"regressor": None}

    def test_sarima_and_both(self):
        r = self._resolve()
        assert r("sarima") == {"sarima": None}
        assert r("both") == {"regressor": None, "sarima": None}

    def test_dict_custom_grids(self):
        r = self._resolve()
        out = r({"regressor": {"max_depth": [3]}, "sarima": {"order": [(1, 0, 1)]}})
        assert out == {"regressor": {"max_depth": [3]}, "sarima": {"order": [(1, 0, 1)]}}

    def test_dict_non_dict_value_means_default_grid(self):
        r = self._resolve()
        assert r({"regressor": True, "sarima": None}) == {"regressor": None, "sarima": None}

    def test_unknown_string_raises(self):
        r = self._resolve()
        with pytest.raises(ValueError, match="Onbekende tune-waarde"):
            r("xgboost")

    def test_unknown_dict_key_raises(self):
        r = self._resolve()
        with pytest.raises(ValueError, match="onbekende sleutels"):
            r({"ensemble": {}})

    def test_wrong_type_raises(self):
        r = self._resolve()
        with pytest.raises(ValueError, match="bool, str of dict"):
            r(5)
