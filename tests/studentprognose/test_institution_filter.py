"""Tests voor het instellingsfilter (issue #200).

Dekt zowel de pure filterfunctie ``apply_institution_filter`` als de
configuratievalidatie ``_validate_institution_filter`` en de integratie via
``load_configuration``.
"""

import json
import warnings

import pandas as pd
import pytest

from studentprognose.config import load_configuration, _validate_institution_filter
from studentprognose.data.preprocessing.institution_filter import apply_institution_filter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _multi_institution_df():
    """Teldata met dezelfde opleiding bij twee instellingen (Radboud + UvA)."""
    return pd.DataFrame({
        "Korte naam instelling": ["21PB", "21PB", "21AB", "21AB"],
        "Collegejaar": [2024, 2024, 2024, 2024],
        "Croho groepeernaam": ["B Psychologie", "B Biologie", "B Psychologie", "B Biologie"],
        "Herkomst": ["NL", "NL", "NL", "NL"],
        "Ongewogen vooraanmelders": [30, 10, 50, 20],
    })


# ---------------------------------------------------------------------------
# apply_institution_filter
# ---------------------------------------------------------------------------

class TestApplyInstitutionFilter:
    def test_empty_filter_single_institution_returns_all_no_warning(self):
        df = _multi_institution_df()
        df = df[df["Korte naam instelling"] == "21PB"]  # één instelling
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_institution_filter(df, [])
        assert len(result) == len(df)
        assert len(w) == 0  # geen waarschuwing bij één instelling

    def test_empty_filter_multi_institution_warns(self):
        df = _multi_institution_df()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_institution_filter(df, [])
        assert len(result) == len(df)  # niets verwijderd
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "2 instellingen" in str(w[0].message)

    def test_filter_to_one_institution_keeps_only_that_institution(self):
        df = _multi_institution_df()
        result = apply_institution_filter(df, ["21PB"])
        assert set(result["Korte naam instelling"]) == {"21PB"}
        assert len(result) == 2

    def test_filter_prevents_cross_institution_aggregation(self):
        # Kern van issue #200: zonder filter zou een latere groupby per opleiding
        # de twee instellingen optellen (30 + 50 = 80 voor B Psychologie). Na
        # filteren op 21PB blijft alleen Radboud's 30 over.
        df = _multi_institution_df()
        result = apply_institution_filter(df, ["21PB"])
        psy = result[result["Croho groepeernaam"] == "B Psychologie"]
        assert psy["Ongewogen vooraanmelders"].sum() == 30

    def test_filter_multiple_institutions(self):
        df = _multi_institution_df()
        result = apply_institution_filter(df, ["21PB", "21AB"])
        assert set(result["Korte naam instelling"]) == {"21PB", "21AB"}
        assert len(result) == 4

    def test_filter_no_match_raises_with_available(self):
        df = _multi_institution_df()
        with pytest.raises(ValueError) as exc:
            apply_institution_filter(df, ["99XX"])
        # De foutmelding noemt de wél beschikbare instellingen.
        assert "21PB" in str(exc.value)
        assert "21AB" in str(exc.value)

    def test_missing_column_with_filter_raises(self):
        df = _multi_institution_df().drop(columns=["Korte naam instelling"])
        with pytest.raises(ValueError) as exc:
            apply_institution_filter(df, ["21PB"])
        assert "Korte naam instelling" in str(exc.value)

    def test_missing_column_without_filter_is_noop(self):
        # Geen instellingskolom én geen filter (bv. individuele data): geen fout.
        df = _multi_institution_df().drop(columns=["Korte naam instelling"])
        result = apply_institution_filter(df, [])
        assert len(result) == len(df)

    def test_custom_column_name(self):
        # Studielink gebruikt "Brincode" i.p.v. "Korte naam instelling".
        df = _multi_institution_df().rename(columns={"Korte naam instelling": "Brincode"})
        result = apply_institution_filter(df, ["21PB"], institution_column="Brincode")
        assert set(result["Brincode"]) == {"21PB"}

    def test_does_not_mutate_input(self):
        df = _multi_institution_df()
        original_len = len(df)
        apply_institution_filter(df, ["21PB"])
        assert len(df) == original_len


# ---------------------------------------------------------------------------
# _validate_institution_filter
# ---------------------------------------------------------------------------

class TestValidateInstitutionFilter:
    def test_empty_list_is_valid(self):
        _validate_institution_filter([], "cfg.json")  # geen exception

    def test_valid_list_passes(self):
        _validate_institution_filter(["21PB", "21AB"], "cfg.json")

    def test_non_list_exits(self):
        with pytest.raises(SystemExit) as exc:
            _validate_institution_filter("21PB", "cfg.json")
        assert exc.value.code == 1

    def test_non_string_element_exits(self):
        with pytest.raises(SystemExit) as exc:
            _validate_institution_filter([21], "cfg.json")
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# load_configuration — institution_filter integratie
# ---------------------------------------------------------------------------

class TestLoadConfigurationInstitutionFilter:
    def test_default_is_empty_list(self, tmp_path):
        # Geen institution_filter in het gebruikersbestand: defaults leveren [].
        cfg = {"numerus_fixus": {}}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        result = load_configuration(str(f))
        assert result.get("institution_filter") == []

    def test_valid_institution_filter_loads(self, tmp_path):
        cfg = {"numerus_fixus": {}, "institution_filter": ["21PB"]}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        result = load_configuration(str(f))
        assert result["institution_filter"] == ["21PB"]

    def test_invalid_institution_filter_exits(self, tmp_path):
        cfg = {"numerus_fixus": {}, "institution_filter": "21PB"}
        f = tmp_path / "configuration.json"
        f.write_text(json.dumps(cfg))
        with pytest.raises(SystemExit):
            load_configuration(str(f))

    def test_default_config_exposes_institution_role(self):
        from studentprognose.config import load_defaults

        defaults = load_defaults()
        assert defaults["column_roles"]["institution"] == "Korte naam instelling"
        assert defaults["institution_filter"] == []


# ---------------------------------------------------------------------------
# End-to-end: CumulativeStrategy.preprocess() met multi-instelling teldata
# ---------------------------------------------------------------------------

def _multi_institution_cumulative():
    """Volledige cumulatieve teldata voor twee instellingen, één opleiding."""
    from studentprognose.config import load_defaults

    def rows(inst, base):
        return [
            {
                "Korte naam instelling": inst, "Collegejaar": 2024,
                "Weeknummer rapportage": wk, "Weeknummer": wk, "Faculteit": "SOW",
                "Type hoger onderwijs": "Bachelor", "Groepeernaam Croho": "B Psychologie",
                "Naam Croho opleiding Nederlands": "B Psychologie", "Croho": "56604",
                "Herinschrijving": "Nee", "Hogerejaars": "Nee", "Herkomst": "NL",
                "Gewogen vooraanmelders": base + wk, "Ongewogen vooraanmelders": base + wk,
                "Aantal aanmelders met 1 aanmelding": 1, "Inschrijvingen": 0,
            }
            for wk in range(1, 39)
        ]

    data = pd.DataFrame(rows("21PB", 100) + rows("21AB", 500))
    return data, load_defaults()


def _make_cumulative_strategy(data, cfg):
    from studentprognose.strategies.cumulative import CumulativeStrategy
    from studentprognose.utils.weeks import DataOption

    return CumulativeStrategy(
        data, None, cfg, None, cfg["ensemble_weights"], "/tmp",
        DataOption.CUMULATIVE, None,
    )


class TestCumulativeStrategyInstitutionFilter:
    def test_filter_prevents_cross_institution_summing_in_preprocess(self):
        # Issue #200 acceptatiecriterium: getest met data van meerdere instellingen.
        # Met filter op 21PB blijft alleen Radboud's reeks over (week 38 = 138),
        # niet de som over beide instellingen (138 + 538 = 676).
        data, cfg = _multi_institution_cumulative()
        cfg["institution_filter"] = ["21PB"]
        strategy = _make_cumulative_strategy(data, cfg)

        assert strategy.institution_filter == ["21PB"]
        out = strategy.preprocess()
        wk38 = out[(out["Weeknummer"] == 38) & (out["Croho groepeernaam"] == "B Psychologie")]
        assert wk38["Gewogen vooraanmelders"].sum() == 138

    def test_no_filter_sums_across_institutions_and_warns(self):
        data, cfg = _multi_institution_cumulative()  # geen institution_filter
        strategy = _make_cumulative_strategy(data, cfg)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = strategy.preprocess()

        assert any(
            issubclass(x.category, UserWarning) and "instellingen" in str(x.message)
            for x in w
        )
        wk38 = out[(out["Weeknummer"] == 38) & (out["Croho groepeernaam"] == "B Psychologie")]
        assert wk38["Gewogen vooraanmelders"].sum() == 676
