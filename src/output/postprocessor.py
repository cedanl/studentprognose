import numpy as np
import pandas as pd
import os
from statistics import mean

from src.utils.weeks import (
    DataOption, StudentYearPrediction, increment_week, convert_nan_to_zero,
)
from src.models.ratio import predict_with_ratio as _predict_with_ratio
from src.data.transforms import replace_latest_data


class PostProcessor:
    """Handles output preparation, ensemble calculation, error metrics, and file saving.

    Replaces the old HelperMethods god-object with the same interface but importing
    from the new src/ modules.
    """

    @staticmethod
    def check_output_writable(data_option, student_year_prediction, ci_test_n, filtering_path):
        """Verify output files can be written (not locked by Excel)."""
        mode_suffix = data_option.filename_suffix
        ci_suffix = f"_ci_test_N{ci_test_n}" if ci_test_n is not None else ""
        try:
            open(f"data/output/output_prelim_{mode_suffix}{ci_suffix}.xlsx", "w").close()
            if "test" not in filtering_path:
                match student_year_prediction:
                    case StudentYearPrediction.FIRST_YEARS:
                        open(f"data/output/output_first-years_{mode_suffix}{ci_suffix}.xlsx", "w").close()
                    case StudentYearPrediction.HIGHER_YEARS:
                        open(f"data/output/output_higher-years_{mode_suffix}{ci_suffix}.xlsx", "w").close()
                    case StudentYearPrediction.VOLUME:
                        open(f"data/output/output_volume_{mode_suffix}{ci_suffix}.xlsx", "w").close()
        except IOError:
            input(
                "Could not open output files because they are (probably) opened by another process. "
                "Please close Excel. Press Enter to continue."
            )

    def __init__(self, configuration, data_latest, ensemble_weights,
                 data_studentcount, cwd, data_option, ci_test_n):
        self.data_latest = data_latest
        self.ensemble_weights = ensemble_weights
        self.data_studentcount = data_studentcount
        self.numerus_fixus_list = configuration["numerus_fixus"]
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
            & (data_latest["Collegejaar"] >= year - 3)
            & (data_latest["Weeknummer"] == week)
            & (data_latest["Croho groepeernaam"] == nf)
        ].fillna(0)
        distribution_per_herkomst = {"EER": [], "NL": [], "Niet-EER": []}
        for last_year in range(year - 3, year):
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
            self.data = self.data.merge(
                self.data_studentcount,
                on=["Croho groepeernaam", "Collegejaar", "Herkomst", "Examentype"],
                how="left",
            )

        if data_cumulative is not None:
            data_cumulative = data_cumulative.drop(columns="Faculteit")

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

        ci_suffix = f"_ci_test_N{self.ci_test_n}" if self.ci_test_n is not None else ""
        output_path = os.path.join(self.CWD, "data", "output", f"output_prelim_{self.data_option.filename_suffix}{ci_suffix}.xlsx")
        self.data.to_excel(output_path, index=False)

    def predict_with_ratio(self, data_cumulative, predict_year):
        self.data = _predict_with_ratio(
            self.data, data_cumulative, self.data_studentcount,
            self.numerus_fixus_list, predict_year
        )

    def postprocess(self, predict_year, predict_week):
        if self.data_latest is not None:
            self.data = replace_latest_data(
                self.data_latest, self.data, predict_year, predict_week
            )

        if self.data_option == DataOption.BOTH_DATASETS:
            self._create_ensemble_columns(predict_year, predict_week)

        self._create_error_columns()

        self.data = self.data.drop_duplicates()

        self.data_latest = self.data

    def _create_ensemble_columns(self, predict_year, predict_week):
        self.data = self.data.sort_values(
            by=["Croho groepeernaam", "Herkomst", "Collegejaar", "Weeknummer"]
        )
        self.data = self.data.reset_index(drop=True)

        self.data.loc[
            (self.data["Collegejaar"] == predict_year) & (self.data["Weeknummer"] == predict_week),
            "Ensemble_prediction",
        ] = np.nan
        self.data.loc[
            (self.data["Collegejaar"] == predict_year) & (self.data["Weeknummer"] == predict_week),
            "Weighted_ensemble_prediction",
        ] = -1.0

        for examentype in self.data[
            (self.data["Collegejaar"] == predict_year) & (self.data["Weeknummer"] == predict_week)
        ]["Examentype"].unique():
            self.data.loc[
                (self.data["Collegejaar"] == predict_year)
                & (self.data["Weeknummer"] == predict_week)
                & (self.data["Examentype"] == examentype),
                "Ensemble_prediction",
            ] = self.data[
                (self.data["Collegejaar"] == predict_year)
                & (self.data["Weeknummer"] == predict_week)
                & (self.data["Examentype"] == examentype)
            ].apply(
                self._get_normal_ensemble, axis=1
            )

            if self.ensemble_weights is not None:
                weights = self.ensemble_weights.rename(columns={"Programme": "Croho groepeernaam"})
                if "Average_ensemble_prediction" not in self.data.columns:
                    weights = weights.rename(
                        columns={
                            "Average_ensemble_prediction": "Average_ensemble_prediction_weight"
                        }
                    )
                self.data = self.data.merge(
                    weights,
                    on=["Collegejaar", "Croho groepeernaam", "Examentype", "Herkomst"],
                    how="left",
                    suffixes=("", "_weight"),
                )

                weighted_ensemble = (
                    self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Weeknummer"] == predict_week)
                        & (self.data["Examentype"] == examentype)
                    ]["SARIMA_cumulative"].fillna(0)
                    * self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Weeknummer"] == predict_week)
                        & (self.data["Examentype"] == examentype)
                    ]["SARIMA_cumulative_weight"].fillna(0)
                    + self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Weeknummer"] == predict_week)
                        & (self.data["Examentype"] == examentype)
                    ]["SARIMA_individual"].fillna(0)
                    * self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Weeknummer"] == predict_week)
                        & (self.data["Examentype"] == examentype)
                    ]["SARIMA_individual_weight"].fillna(0)
                    + self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Weeknummer"] == predict_week)
                        & (self.data["Examentype"] == examentype)
                    ]["Prognose_ratio"].fillna(0)
                    * self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Weeknummer"] == predict_week)
                        & (self.data["Examentype"] == examentype)
                    ]["Prognose_ratio_weight"].fillna(0)
                )

                self.data.loc[
                    (self.data["Collegejaar"] == predict_year)
                    & (self.data["Weeknummer"] == predict_week)
                    & (self.data["Examentype"] == examentype),
                    "Weighted_ensemble_prediction",
                ] = np.where(
                    self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Weeknummer"] == predict_week)
                        & (self.data["Examentype"] == examentype)
                    ]["Average_ensemble_prediction_weight"]
                    != 1,
                    weighted_ensemble,
                    self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Weeknummer"] == predict_week)
                        & (self.data["Examentype"] == examentype)
                    ]["Weighted_ensemble_prediction"],
                )

                self.data = self.data.drop(
                    [
                        "SARIMA_cumulative_weight",
                        "SARIMA_individual_weight",
                        "Prognose_ratio_weight",
                        "Average_ensemble_prediction_weight",
                    ],
                    axis=1,
                )

        self.data["Average_ensemble_prediction"] = np.nan

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

    def _get_normal_ensemble(self, row):
        sarima_cumulative = convert_nan_to_zero(row["SARIMA_cumulative"])
        sarima_individual = convert_nan_to_zero(row["SARIMA_individual"])

        if row["Croho groepeernaam"] in [
            "B Geneeskunde",
            "B Biomedische Wetenschappen",
            "B Tandheelkunde",
        ]:
            return sarima_cumulative
        elif row["Weeknummer"] in range(17, 24) and row["Examentype"] == "Master":
            return sarima_individual * 0.2 + sarima_cumulative * 0.8
        elif row["Weeknummer"] in range(30, 35):
            return sarima_individual * 0.6 + sarima_cumulative * 0.4
        elif row["Weeknummer"] in range(35, 38):
            return sarima_individual * 0.7 + sarima_cumulative * 0.3
        elif row["Weeknummer"] == 38:
            return sarima_individual
        else:
            return sarima_individual * 0.5 + sarima_cumulative * 0.5

    def _create_error_columns(self):
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

        mae_columns = [f"MAE_{pred}" for pred in predictions]
        mape_columns = [f"MAPE_{pred}" for pred in predictions]

        for col in mae_columns + mape_columns:
            self.data[col] = np.nan

        valid_rows = ~self.data["Croho groepeernaam"].isin(self.numerus_fixus_list)

        for pred in predictions:
            predicted = self.data[pred].apply(convert_nan_to_zero)
            self.data[f"MAE_{pred}"] = abs(self.data["Aantal_studenten"] - predicted).where(
                valid_rows
            )
            self.data[f"MAPE_{pred}"] = (
                abs(self.data["Aantal_studenten"] - predicted) / self.data["Aantal_studenten"]
            ).where(valid_rows, np.nan)

    def ready_new_data(self):
        self.data_latest = self.data

    def save_output(self, student_year_prediction):
        output_filename = "output_"

        if student_year_prediction == StudentYearPrediction.FIRST_YEARS:
            output_filename += "first-years"
        elif student_year_prediction == StudentYearPrediction.HIGHER_YEARS:
            output_filename += "higher-years"
        elif student_year_prediction == StudentYearPrediction.VOLUME:
            output_filename += "volume"

        ci_suffix = f"_ci_test_N{self.ci_test_n}" if self.ci_test_n is not None else ""
        output_filename += f"_{self.data_option.filename_suffix}{ci_suffix}.xlsx"

        output_path = os.path.join(self.CWD, "data", "output", output_filename)

        self.data.sort_values(
            by=["Croho groepeernaam", "Examentype", "Collegejaar", "Weeknummer", "Herkomst"],
            inplace=True,
            ignore_index=True,
        )

        self.data.to_excel(output_path, index=False)
