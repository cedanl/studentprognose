import datetime
import numpy as np
import pandas as pd
import os
import sys
from statistics import mean

from studentprognose.utils.weeks import (
    DataOption, StudentYearPrediction, increment_week,
)
from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK, LOOKBACK_YEARS
from studentprognose.models.ratio import predict_with_ratio as _predict_with_ratio
from studentprognose.data.transforms import replace_latest_data


class PostProcessor:
    """Handles output preparation, ensemble calculation, error metrics, and file saving.

    Replaces the old HelperMethods god-object with the same interface but importing
    from the new src/ modules.
    """

    _SY_LABELS = {
        StudentYearPrediction.FIRST_YEARS: "first-years",
        StudentYearPrediction.HIGHER_YEARS: "higher-years",
        StudentYearPrediction.VOLUME: "volume",
    }

    @staticmethod
    def check_output_writable(data_option, student_year_prediction, ci_test_n):
        """Verify output files can be written (not locked by Excel)."""
        mode_suffix = data_option.filename_suffix
        ci_suffix = f"_ci_test_N{ci_test_n}" if ci_test_n is not None else ""
        output_dir = os.path.join(os.getcwd(), "data", "output")
        os.makedirs(output_dir, exist_ok=True)
        try:
            open(f"data/output/output_prelim_{mode_suffix}{ci_suffix}.xlsx", "w").close()
            if ci_test_n is None:
                sy_label = PostProcessor._SY_LABELS.get(student_year_prediction)
                if sy_label is not None:
                    open(f"data/output/output_{sy_label}_{mode_suffix}{ci_suffix}.xlsx", "w").close()
                    # _totaal mag NIET worden getrunceerd: het bevat de
                    # historie van eerdere runs. "a" opent zonder leeg te
                    # maken en faalt alsnog als Excel het bestand
                    # vergrendelt — exact wat we hier willen detecteren.
                    totaal_path = os.path.join(
                        output_dir, f"_totaal_{sy_label}_{mode_suffix}.xlsx"
                    )
                    if os.path.exists(totaal_path):
                        open(totaal_path, "a").close()
        except IOError:
            print(
                "Fout: outputbestand kan niet geopend worden, waarschijnlijk staat het nog open in Excel. "
                "Sluit het bestand en probeer opnieuw."
            )
            sys.exit(1)

    def __init__(self, configuration, data_latest, ensemble_weights,
                 data_studentcount, cwd, data_option, ci_test_n):
        self.data_latest = data_latest
        self.ensemble_weights = ensemble_weights
        self.data_studentcount = data_studentcount
        self.numerus_fixus_list = configuration["numerus_fixus"]
        self.final_academic_week = configuration.get("model_config", {}).get(
            "final_academic_week", FINAL_ACADEMIC_WEEK
        )
        self.ensemble_override_cumulative = configuration.get("ensemble_override_cumulative", [])
        self.ensemble_weights_config = configuration.get("ensemble_weights", {
            "master_week_17_23": {"individual": 0.5, "cumulative": 0.5},
            "week_30_34":        {"individual": 0.5, "cumulative": 0.5},
            "week_35_37":        {"individual": 0.5, "cumulative": 0.5},
            "default":           {"individual": 0.5, "cumulative": 0.5},
        })
        self._column_roles = configuration.get("column_roles", {})
        self.CWD = cwd
        self.data_option = data_option
        self.ci_test_n = ci_test_n
        self.data = None

    def add_predicted_preregistrations(self, data, predicted_preregistrations):
        dict = {
            "Collegejaar": [],
            "Faculteit": [],
            "Examentype": [],
            "Herkomst": [],
            "Croho groepeernaam": [],
            "Weeknummer": [],
            "SARIMA_cumulative": [],
            "SARIMA_individual": [],
            "Voorspelde vooraanmelders": [],
        }

        index = 0
        for _, row in data.iterrows():
            if index >= len(predicted_preregistrations):
                print(f"Index {index} out of range: {len(predicted_preregistrations)}")
                continue

            current_predicted_preregistrations = predicted_preregistrations[index]

            current_week = increment_week(row["Weeknummer"])
            for current_prediction in current_predicted_preregistrations:
                dict["Collegejaar"].append(row["Collegejaar"])
                dict["Faculteit"].append(row["Faculteit"])
                dict["Examentype"].append(row["Examentype"])
                dict["Herkomst"].append(row["Herkomst"])
                dict["Croho groepeernaam"].append(row["Croho groepeernaam"])
                dict["Weeknummer"].append(current_week)
                dict["SARIMA_cumulative"].append(np.nan)
                dict["SARIMA_individual"].append(np.nan)
                dict["Voorspelde vooraanmelders"].append(current_prediction)

                current_week = increment_week(current_week)

            index += 1

        return pd.concat([data, pd.DataFrame(dict)], ignore_index=True)

    def _numerus_fixus_cap(self, data, year, week):
        for nf in self.numerus_fixus_list:
            nf_data = data[
                (data["Collegejaar"] == year)
                & (data["Weeknummer"] == week)
                & (data["Croho groepeernaam"] == nf)
            ]
            if "SARIMA_individual" in data.columns and np.sum(nf_data["SARIMA_individual"]) > self.numerus_fixus_list[nf]:
                data = self._nf_students_based_on_distribution_of_last_years(
                    data, self.data_latest, nf, year, week, "SARIMA_individual"
                )

            if "SARIMA_cumulative" in data.columns and np.sum(nf_data["SARIMA_cumulative"]) > self.numerus_fixus_list[nf]:
                data = self._nf_students_based_on_distribution_of_last_years(
                    data, self.data_latest, nf, year, week, "SARIMA_cumulative"
                )

        return data

    def _nf_students_based_on_distribution_of_last_years(
        self, data, data_latest, nf, year, week, method
    ):
        last_years_data = data_latest[
            (data_latest["Collegejaar"] < year)
            & (data_latest["Collegejaar"] >= year - LOOKBACK_YEARS)
            & (data_latest["Weeknummer"] == week)
            & (data_latest["Croho groepeernaam"] == nf)
        ].fillna(0)
        distribution_per_herkomst = {"EER": [], "NL": [], "Niet-EER": []}
        for last_year in range(year - LOOKBACK_YEARS, year):
            total_students = last_years_data[last_years_data["Collegejaar"] == last_year][
                "Aantal_studenten"
            ].sum()
            for herkomst in distribution_per_herkomst:
                distribution_per_herkomst[herkomst].append(
                    last_years_data[
                        (last_years_data["Collegejaar"] == last_year)
                        & (last_years_data["Herkomst"] == herkomst)
                    ]["Aantal_studenten"].values[0]
                    / total_students
                )
        for herkomst in distribution_per_herkomst:
            data.loc[
                (data["Collegejaar"] == year)
                & (data["Weeknummer"] == week)
                & (data["Croho groepeernaam"] == nf)
                & (data["Herkomst"] == herkomst),
                method,
            ] = self.numerus_fixus_list[nf] * mean(distribution_per_herkomst[herkomst])
        return data

    def prepare_data_for_output_prelim(self, data, year, week, data_cumulative=None, skip_years=0):
        """Bereken het voorlopige tussenresultaat in ``self.data`` (numerus-fixus-cap,
        kolomselectie en merges met studentaantallen/cumulatieve aanmelddata).

        Deze methode is puur compute: ze schrijft niets naar schijf. Het wegschrijven
        gebeurt in :meth:`save_output_prelim`, die de pipeline alleen aanroept wanneer
        ``save_output=True``. Zo blijft een puur in-memory run (bijv. cloud-pipelines
        met een read-only bestandssysteem) vrij van disk-writes.
        """
        self.data = data
        self.data = self._numerus_fixus_cap(self.data, year, week)

        columns_to_select = [
            "Croho groepeernaam",
            "Faculteit",
            "Examentype",
            "Collegejaar",
            "Herkomst",
            "Weeknummer",
            "SARIMA_cumulative",
            "SARIMA_individual",
            "Voorspelde vooraanmelders",
        ]
        if skip_years > 0:
            columns_to_select = columns_to_select + ["Skip_prediction"]
        self.data = self.data[columns_to_select]

        if self.data_studentcount is not None:
            # Faculteit kan in studentcount voorkomen (bijv. bij Radboud). We
            # droppen 'm en aggregeren naar opleidings-niveau zodat:
            #   1) de merge geen Faculteit_x/_y suffixes produceert
            #      (self.data behoudt de eigen Faculteit uit de aanmelddata),
            #   2) meerdere Faculteit-rijen per opleiding niet leiden tot
            #      rij-multiplicatie maar tot een correcte som.
            studentcount = self.data_studentcount.drop(columns="Faculteit", errors="ignore")
            join_cols = ["Croho groepeernaam", "Collegejaar", "Herkomst", "Examentype"]
            studentcount = studentcount.groupby(join_cols, as_index=False)["Aantal_studenten"].sum()
            self.data = self.data.merge(studentcount, on=join_cols, how="left")

        if data_cumulative is not None:
            data_cumulative = data_cumulative.drop(columns="Faculteit", errors="ignore")

            self.data = self.data.merge(
                data_cumulative,
                on=[
                    "Croho groepeernaam",
                    "Collegejaar",
                    "Herkomst",
                    "Weeknummer",
                    "Examentype",
                ],
                how="left",
            )

    def save_output_prelim(self):
        """Schrijf het voorlopige tussenresultaat naar
        ``data/output/output_prelim_<modus>.xlsx``.

        Gescheiden van :meth:`prepare_data_for_output_prelim` zodat de berekening
        (numerus-fixus-cap + merges) los staat van de I/O. De pipeline roept deze
        methode alleen aan wanneer ``save_output=True``; bij een puur in-memory run
        (``save_output=False``) wordt er niets geschreven — net zoals bij de overige
        outputs (:meth:`save_output`, :meth:`save_totaal_audit_trail`, dashboard).
        """
        if self.data is None:
            return
        ci_suffix = f"_ci_test_N{self.ci_test_n}" if self.ci_test_n is not None else ""
        output_path = os.path.join(
            self.CWD,
            "data",
            "output",
            f"output_prelim_{self.data_option.filename_suffix}{ci_suffix}.xlsx",
        )
        self.data.to_excel(output_path, index=False)

    def predict_with_ratio(self, data_cumulative, predict_year):
        self.data = _predict_with_ratio(
            self.data, data_cumulative, self.data_studentcount,
            self.numerus_fixus_list, predict_year
        )

    def add_applicant_data(self, data_cumulative: "pd.DataFrame", predict_year: int, predict_week: int) -> None:
        """Overwrite applicant columns in self.data with actuals from the cumulative snapshot.

        prepare_data_for_output_prelim merges the full cumulative dataset (alle jaren
        en weken) al in self.data, waardoor de aanmeldkolommen voor predict_year/week
        in de meeste gevallen al correct zijn. Deze methode is een expliciete hersynced
        na predict_with_ratio: die stap kan in afgeleide strategieën self.data muteren
        op manieren die de merge-waarden overschrijven. De aanroep hier garandeert dat
        de aanmeldcijfers altijd de actuele snapshot-waarden zijn, ongeacht wat er
        tussen prepare_data_for_output_prelim en postprocess() is aangepast.

        Only rows matching predict_year/predict_week are updated. Rows without a
        match in data_cumulative are left unchanged.
        """
        if self.data is None or data_cumulative is None:
            return

        key_cols = ["Weeknummer", "Examentype", "Croho groepeernaam", "Herkomst", "Collegejaar"]
        update_cols = [
            "Gewogen vooraanmelders",
            "Ongewogen vooraanmelders",
            "Aantal aanmelders met 1 aanmelding",
            "Inschrijvingen",
        ]

        snapshot = data_cumulative[
            (data_cumulative["Collegejaar"] == predict_year)
            & (data_cumulative["Weeknummer"] == predict_week)
        ]
        if snapshot.empty:
            return

        # keep="last": telbestanden zijn cumulatief en kunnen bij herverwerking
        # meerdere keren dezelfde week bevatten. De laatste verschijning is
        # altijd de meest recente en wint.
        snapshot = (
            snapshot
            .sort_values(key_cols)
            .drop_duplicates(subset=key_cols, keep="last")
            [key_cols + update_cols]
        )

        mask = (
            (self.data["Collegejaar"] == predict_year)
            & (self.data["Weeknummer"] == predict_week)
        )
        if not mask.any():
            return

        idx = self.data.index[mask]
        # reset_index zodat merged positie 0,1,2... exact overeenkomt met
        # idx[0],idx[1],idx[2]... — self.data kan na eerder filteren een
        # niet-aaneengesloten index hebben waardoor idx[has_value] anders
        # naar de verkeerde rijen zou wijzen.
        merged = (
            self.data.loc[idx, key_cols]
            .reset_index(drop=True)
            .merge(snapshot, on=key_cols, how="left")
        )

        for col in update_cols:
            if col in merged.columns:
                has_value = merged[col].notna().values
                self.data.loc[idx[has_value], col] = merged.loc[has_value, col].values

    def postprocess(self, predict_year, predict_week):
        if self.data_latest is not None:
            self.data = replace_latest_data(
                self.data_latest, self.data, predict_year, predict_week
            )

        if self.data_option == DataOption.BOTH_DATASETS:
            self._create_ensemble_columns(predict_year, predict_week)

        if "Prognose_ratio" in self.data.columns:
            self.data["Baseline"] = self.data["Prognose_ratio"]

        self._create_error_columns()

        self.data = self.data.drop_duplicates()

        self.data_latest = self.data

    def _create_ensemble_columns(self, predict_year, predict_week):
        self.data = self.data.sort_values(
            by=["Croho groepeernaam", "Herkomst", "Collegejaar", "Weeknummer"]
        ).reset_index(drop=True)

        base_mask = (
            (self.data["Collegejaar"] == predict_year)
            & (self.data["Weeknummer"] == predict_week)
        )

        self.data.loc[base_mask, "Ensemble_prediction"] = np.nan
        self.data.loc[base_mask, "Weighted_ensemble_prediction"] = -1.0

        self._compute_normal_ensemble(base_mask)

        if self.ensemble_weights is not None:
            self._compute_weighted_ensemble(base_mask)

        self.data["Average_ensemble_prediction"] = self.data.groupby(
            ["Croho groepeernaam", "Examentype", "Herkomst", "Collegejaar"]
        )["Ensemble_prediction"].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean().shift().bfill()
        )

        self.data["Weighted_ensemble_prediction"] = np.where(
            self.data["Weighted_ensemble_prediction"] == -1.0,
            self.data["Average_ensemble_prediction"],
            self.data["Weighted_ensemble_prediction"],
        )

    def _compute_normal_ensemble(self, mask):
        """Vectorized ensemble-berekening (vervangt row-by-row .apply)."""
        d = self.data
        w = self.ensemble_weights_config

        cum = d.loc[mask, "SARIMA_cumulative"].fillna(0)
        ind = d.loc[mask, "SARIMA_individual"].fillna(0)
        wk = d.loc[mask, "Weeknummer"]
        ex = d.loc[mask, "Examentype"]
        prog = d.loc[mask, "Croho groepeernaam"]

        override = prog.isin(self.ensemble_override_cumulative)
        master_17_23 = ~override & wk.between(17, 23) & (ex == "Master")
        week_30_34 = ~override & ~master_17_23 & wk.between(30, 34)
        week_35_37 = ~override & ~master_17_23 & ~week_30_34 & wk.between(35, self.final_academic_week - 1)
        week_final = ~override & ~master_17_23 & ~week_30_34 & ~week_35_37 & (wk == self.final_academic_week)

        conditions = [override, master_17_23, week_30_34, week_35_37, week_final]
        choices = [
            cum,
            ind * w["master_week_17_23"]["individual"] + cum * w["master_week_17_23"]["cumulative"],
            ind * w["week_30_34"]["individual"] + cum * w["week_30_34"]["cumulative"],
            ind * w["week_35_37"]["individual"] + cum * w["week_35_37"]["cumulative"],
            ind,
        ]
        default = ind * w["default"]["individual"] + cum * w["default"]["cumulative"]

        d.loc[mask, "Ensemble_prediction"] = np.select(conditions, choices, default=default)

    def _compute_weighted_ensemble(self, base_mask):
        """Vectorized weighted ensemble berekening."""
        weights = self.ensemble_weights.rename(columns={"Programme": "Croho groepeernaam"})
        if "Average_ensemble_prediction" not in self.data.columns:
            weights = weights.rename(
                columns={"Average_ensemble_prediction": "Average_ensemble_prediction_weight"}
            )

        self.data = self.data.merge(
            weights,
            on=["Collegejaar", "Croho groepeernaam", "Examentype", "Herkomst"],
            how="left",
            suffixes=("", "_weight"),
        )

        d = self.data.loc[base_mask]
        weighted = (
            d["SARIMA_cumulative"].fillna(0) * d["SARIMA_cumulative_weight"].fillna(0)
            + d["SARIMA_individual"].fillna(0) * d["SARIMA_individual_weight"].fillna(0)
            + d["Prognose_ratio"].fillna(0) * d["Prognose_ratio_weight"].fillna(0)
        )

        self.data.loc[base_mask, "Weighted_ensemble_prediction"] = np.where(
            d["Average_ensemble_prediction_weight"] != 1,
            weighted,
            self.data.loc[base_mask, "Weighted_ensemble_prediction"],
        )

        self.data = self.data.drop(
            columns=[
                "SARIMA_cumulative_weight",
                "SARIMA_individual_weight",
                "Prognose_ratio_weight",
                "Average_ensemble_prediction_weight",
            ],
        )

    def _create_error_columns(self):
        if "Aantal_studenten" not in self.data.columns:
            return

        if self.data_option == DataOption.BOTH_DATASETS:
            predictions = [
                "Weighted_ensemble_prediction",
                "Average_ensemble_prediction",
                "Ensemble_prediction",
                "Prognose_ratio",
                "SARIMA_cumulative",
                "SARIMA_individual",
            ]
        elif self.data_option == DataOption.INDIVIDUAL:
            predictions = [
                "SARIMA_individual",
            ]
        elif self.data_option == DataOption.CUMULATIVE:
            predictions = [
                "Prognose_ratio",
                "SARIMA_cumulative",
            ]

        predictions = [p for p in predictions if p in self.data.columns]

        mae_columns = [f"MAE_{pred}" for pred in predictions]
        mape_columns = [f"MAPE_{pred}" for pred in predictions]

        for col in mae_columns + mape_columns:
            self.data[col] = np.nan

        valid_rows = ~self.data["Croho groepeernaam"].isin(self.numerus_fixus_list)

        for pred in predictions:
            predicted = self.data[pred]
            self.data[f"MAE_{pred}"] = abs(self.data["Aantal_studenten"] - predicted).where(
                valid_rows
            )
            self.data[f"MAPE_{pred}"] = (
                abs(self.data["Aantal_studenten"] - predicted) / self.data["Aantal_studenten"]
            ).where(valid_rows, np.nan)

    def ready_new_data(self):
        self.data_latest = self.data

    def save_output(self, student_year_prediction):
        sy_label = self._SY_LABELS.get(student_year_prediction, "")
        ci_suffix = f"_ci_test_N{self.ci_test_n}" if self.ci_test_n is not None else ""
        output_filename = f"output_{sy_label}_{self.data_option.filename_suffix}{ci_suffix}.xlsx"

        output_path = os.path.join(self.CWD, "data", "output", output_filename)

        self.data.sort_values(
            by=["Croho groepeernaam", "Examentype", "Collegejaar", "Weeknummer", "Herkomst"],
            inplace=True,
            ignore_index=True,
        )

        self.data.to_excel(output_path, index=False)

    # Sleutel- en sorteer-volgorde van de audittrail, uitgedrukt in
    # column_roles. De concrete kolomnamen verschillen per instelling
    # (Radboud vs. CEDA-default) maar de rolinvulling is universeel:
    # één rij = uniek (jaar, week, opleiding, herkomst, examentype).
    _TOTAAL_KEY_ROLES = ("academic_year", "week", "programme", "origin", "exam_type")
    _TOTAAL_SORT_ROLES = ("programme", "exam_type", "academic_year", "week", "origin")

    def save_totaal_audit_trail(self, student_year_prediction):
        """Append de huidige run aan een doorlopend `_totaal_<sy>_<do>.xlsx`.

        Idempotent per sleutelcombo uit ``column_roles`` (jaar, week,
        opleiding, herkomst, examentype): rijen voor dezelfde combo worden
        overschreven, niet gedupliceerd. Voegt een ``Run_date``-kolom toe
        zodat traceerbaar blijft wanneer een rij is gegenereerd. Het
        bestand wordt nooit als input door de pipeline gelezen (de loader
        leest enkel paden uit ``configuration.json``), dus het is
        structureel veilig om naast de reguliere outputs te bestaan.
        """
        if self.data is None:
            return

        sy_label = self._SY_LABELS.get(student_year_prediction)
        if sy_label is None:
            return

        try:
            key_cols = [self._column_roles[r] for r in self._TOTAAL_KEY_ROLES]
            sort_cols = [self._column_roles[r] for r in self._TOTAAL_SORT_ROLES]
        except KeyError as missing:
            raise RuntimeError(
                f"configuratie mist column_roles[{missing.args[0]!r}] — "
                "vereist voor de _totaal-audittrail."
            ) from missing

        filename = f"_totaal_{sy_label}_{self.data_option.filename_suffix}.xlsx"
        path = os.path.join(self.CWD, "data", "output", filename)

        new_rows = self.data.copy()
        new_rows["Run_date"] = datetime.date.today().isoformat()

        if os.path.exists(path):
            try:
                existing = pd.read_excel(path)
            except Exception as e:
                # Corrupte of door Excel vergrendelde file: vroege check
                # ving dat normaal al af. Hier alsnog defensief: log en sla
                # over zodat de hoofdpipeline-output niet verloren gaat.
                print(
                    f"Waarschuwing: kon bestaande audittrail niet lezen ({path}): {e}. "
                    "Audittrail wordt deze run niet bijgewerkt."
                )
                return
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows

        # keep="last": de nieuwe run staat altijd achteraan in de concat,
        # dus zijn waarden winnen bij gelijke sleutel. Dit realiseert het
        # "overschrijven per week"-gedrag zonder rij-duplicatie.
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
        combined = combined.sort_values(by=sort_cols, ignore_index=True)
        combined.to_excel(path, index=False)
