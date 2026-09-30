"""Interactieve dashboards voor de studentprognose-pipeline.

Schrijft zelfstandige HTML-pagina's (geen server nodig) naar
``data/output/visualisations[_ci_test_N{n}]/{individual,cumulative,final}/dashboard.html``.

Alle pagina's delen één opzet rond modelperformance en betrouwbaarheid; zie
:mod:`studentprognose.output.performance_page` voor de inhoud en
:mod:`studentprognose.output.model_performance` voor de berekeningen.
"""

import os

import pandas as pd

from studentprognose.output import performance_page
from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK
from studentprognose.utils.weeks import (
    DataOption,
    StudentYearPrediction,
    get_all_weeks_ordered,
)


def _academic_weeks(final_week: int = FINAL_ACADEMIC_WEEK) -> list[str]:
    """Weekvolgorde voor de week-x-as, afgeleid van de academische jaargrens.

    ``get_all_weeks_ordered`` geeft de seizoensvolgorde (UvA 37→36, legacy 39→38);
    ISO-week 53 (lange ISO-jaren) wordt tussen 52 en 1 ingevoegd — consistent met
    ``week_sort_key`` — zodat hij op de juiste plek op de x-as staat i.p.v. als
    losse categorie achteraan.
    """
    weeks = get_all_weeks_ordered(final_week)
    i = weeks.index("52")
    return weeks[: i + 1] + ["53"] + weeks[i + 1 :]


class DashboardBuilder:
    """Bouwt de dashboardpagina's uit de output van een uitgevoerde strategy."""

    def __init__(
        self,
        data: pd.DataFrame,
        data_option: DataOption,
        numerus_fixus_list: dict,
        student_year_prediction: StudentYearPrediction,
        ci_test_n: int | None,
        cwd: str,
        predict_week: int | None,
        data_cumulative: pd.DataFrame | None,
        data_studentcount: pd.DataFrame | None,
        data_xgboost_curve: pd.DataFrame | None = None,
        final_academic_week: int = FINAL_ACADEMIC_WEEK,
    ):
        self.data = data.copy()
        self.data["Weeknummer"] = self.data["Weeknummer"].astype(int)
        # De programmesleutel is in het dashboard zowel label als filtersleutel. We
        # brengen hem overal naar string, zodat numerieke isatcodes (Int64) niet
        # botsen met stringvergelijkingen en de numerus-fixuslijst.
        self.data["Croho groepeernaam"] = self.data["Croho groepeernaam"].astype(str)
        self.data_option = data_option
        self.numerus_fixus_list = {
            str(k): v for k, v in numerus_fixus_list.items() if v > 0
        }
        self.student_year_prediction = student_year_prediction
        self.ci_test_n = ci_test_n
        self.cwd = cwd
        self.predict_week = predict_week
        # Academische jaargrens (UvA 36, legacy 38) bepaalt de week-x-as-volgorde.
        self.final_academic_week = final_academic_week
        self.academic_weeks = _academic_weeks(final_academic_week)
        if data_cumulative is not None:
            data_cumulative = data_cumulative.copy()
            data_cumulative["Weeknummer"] = data_cumulative["Weeknummer"].astype(int)
            data_cumulative["Croho groepeernaam"] = data_cumulative[
                "Croho groepeernaam"
            ].astype(str)
        self.data_cumulative = data_cumulative
        if (
            data_studentcount is not None
            and "Croho groepeernaam" in data_studentcount.columns
        ):
            data_studentcount = data_studentcount.copy()
            data_studentcount["Croho groepeernaam"] = data_studentcount[
                "Croho groepeernaam"
            ].astype(str)
        self.data_studentcount = data_studentcount
        if data_xgboost_curve is not None:
            data_xgboost_curve = data_xgboost_curve.copy()
            data_xgboost_curve["Croho groepeernaam"] = data_xgboost_curve[
                "Croho groepeernaam"
            ].astype(str)
        self.data_xgboost_curve = data_xgboost_curve

        self.prediction_year = int(self.data["Collegejaar"].max())

        # replace_latest_data() zet voorspellingen op individuele studentrijen; het
        # dashboard heeft één rij per groep nodig.
        group_cols = [
            "Croho groepeernaam",
            "Collegejaar",
            "Herkomst",
            "Examentype",
            "Weeknummer",
        ]
        self.data = self.data.drop_duplicates(subset=group_cols)

        ci_suffix = f"_ci_test_N{ci_test_n}" if ci_test_n is not None else ""
        self.base_dir = os.path.join(
            cwd, "data", "output", f"visualisations{ci_suffix}"
        )

    # ── Welke pagina's ────────────────────────────────────────────────

    def _builds_individual(self) -> bool:
        return (
            self.data_option in (DataOption.INDIVIDUAL, DataOption.BOTH_DATASETS)
            and "SARIMA_individual" in self.data.columns
            and self.data["SARIMA_individual"].notna().any()
        )

    def _builds_cumulative(self) -> bool:
        return self.data_option in (
            DataOption.CUMULATIVE,
            DataOption.BOTH_DATASETS,
        ) and any(
            c in self.data.columns for c in ("SARIMA_cumulative", "Prognose_ratio")
        )

    def _pages(self) -> list[tuple[str, str, str]]:
        """``(track, label, href)`` van de pagina's die :meth:`build_and_save` schrijft.

        De navigatie linkt alleen hiernaar, zodat bijvoorbeeld een ``-d c``-run geen
        dode link naar de individuele pagina toont.
        """
        pages = []
        if self._builds_individual():
            pages.append(("individual", "Individueel", "../individual/dashboard.html"))
        if self._builds_cumulative():
            pages.append(("cumulative", "Cumulatief", "../cumulative/dashboard.html"))
        pages.append(("final", "Eindoverzicht", "../final/dashboard.html"))
        return pages

    # ── Schrijven ─────────────────────────────────────────────────────

    def _save(self, track: str) -> None:
        payload = performance_page.build_payload(
            data=self.data,
            data_cumulative=self.data_cumulative,
            data_studentcount=self.data_studentcount,
            prediction_year=self.prediction_year,
            predict_week=self.predict_week,
            weeks=self.academic_weeks,
            final_week=self.final_academic_week,
            numerus_fixus=self.numerus_fixus_list,
            track=track,
            xgboost_curve=self.data_xgboost_curve,
        )
        nav = [(label, href, t == track) for t, label, href in self._pages()]
        out_dir = os.path.join(self.base_dir, track)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "dashboard.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(performance_page.render_html(payload, nav))
        print(f"  Dashboard saved: {path}")

    def _save_individual(self) -> None:
        self._save("individual")

    def _save_cumulative(self) -> None:
        self._save("cumulative")

    def _save_final(self) -> None:
        self._save("final")

    def build_and_save(self) -> None:
        """Schrijf alle van toepassing zijnde dashboardpagina's."""
        print("Generating dashboards...")
        for track, _, _ in self._pages():
            self._save(track)
        print("Dashboards done.")
