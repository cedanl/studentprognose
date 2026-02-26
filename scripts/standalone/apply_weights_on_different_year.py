import os as os
import pandas as pd
import numpy as np
import sys

import numpy as np
import pandas as pd
import os
from statistics import mean

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.load_data import load_configuration


class ApplyWeightsOnDifferentYear:
    """This class consists of the methods used for operations on dataholders and is loaded in
    the initialisation of DataHolderSuperclass (dataholder_superclass.py) as a variable. The
    methods in this class are used for several varying tasks in main.py and for adding the
    predicted preregistrations when calculating the cumulative values.
    """

    def __init__(
        self,
        data_latest,
        ensemble_weights,
        data_studentcount,
        year_to_calculate_ensemble_values,
        numerus_fixus_list,
    ):
        self.data_latest = data_latest
        self.ensemble_weights = ensemble_weights
        self.data_studentcount = data_studentcount
        self.year_to_calculate_ensemble_values = year_to_calculate_ensemble_values
        self.data = self.data_latest.copy()
        self.numerus_fixus_list = numerus_fixus_list

    def postprocess(self):
        self._create_ensemble_columns()
        self._create_error_columns()

        self.data = self.data.drop_duplicates()

        self.data_latest = self.data

    def _create_ensemble_columns(self):
        predict_year = self.year_to_calculate_ensemble_values

        self.data = self.data.sort_values(
            by=["Croho groepeernaam", "Herkomst", "Collegejaar", "Weeknummer"]
        )
        self.data = self.data.reset_index(drop=True)

        # Initialize columns
        self.data.loc[
            (self.data["Collegejaar"] == predict_year),
            "Ensemble_prediction",
        ] = np.nan
        self.data.loc[
            (self.data["Collegejaar"] == predict_year),
            "Weighted_ensemble_prediction",
        ] = -1.0

        # Compute ensemble predictions using vectorized operations

        for examentype in self.data[(self.data["Collegejaar"] == predict_year)][
            "Examentype"
        ].unique():
            self.data.loc[
                (self.data["Collegejaar"] == predict_year)
                & (self.data["Examentype"] == examentype),
                "Ensemble_prediction",
            ] = self.data[
                (self.data["Collegejaar"] == predict_year)
                & (self.data["Examentype"] == examentype)
            ].apply(
                self._get_normal_ensemble, axis=1
            )

            if self.ensemble_weights is not None:
                # Merge weights into data for vectorized calculations
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
                        & (self.data["Examentype"] == examentype)
                    ]["SARIMA_cumulative"].fillna(0)
                    * self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Examentype"] == examentype)
                    ]["SARIMA_cumulative_weight"].fillna(0)
                    + self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Examentype"] == examentype)
                    ]["SARIMA_individual"].fillna(0)
                    * self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Examentype"] == examentype)
                    ]["SARIMA_individual_weight"].fillna(0)
                    + self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Examentype"] == examentype)
                    ]["Prognose_ratio"].fillna(0)
                    * self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Examentype"] == examentype)
                    ]["Prognose_ratio_weight"].fillna(0)
                )

                self.data.loc[
                    (self.data["Collegejaar"] == predict_year)
                    & (self.data["Examentype"] == examentype),
                    "Weighted_ensemble_prediction",
                ] = np.where(
                    self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Examentype"] == examentype)
                    ]["Average_ensemble_prediction_weight"]
                    != 1,
                    weighted_ensemble,
                    self.data[
                        (self.data["Collegejaar"] == predict_year)
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

        # Compute average ensemble predictions
        self.data["Average_ensemble_prediction"] = np.nan

        # Use groupby and rolling to calculate averages
        self.data["Average_ensemble_prediction"] = self.data.groupby(
            ["Croho groepeernaam", "Examentype", "Herkomst", "Collegejaar"]
        )["Ensemble_prediction"].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean().shift().bfill()
        )

        # Update Weighted_ensemble_prediction where needed
        self.data["Weighted_ensemble_prediction"] = np.where(
            self.data["Weighted_ensemble_prediction"] == -1.0,
            self.data["Average_ensemble_prediction"],
            self.data["Weighted_ensemble_prediction"],
        )

    def _get_normal_ensemble(self, row):
        sarima_cumulative = self.convert_nan_to_zero(row["SARIMA_cumulative"])
        sarima_individual = self.convert_nan_to_zero(row["SARIMA_individual"])

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
        predictions = [
            "Weighted_ensemble_prediction",
            "Average_ensemble_prediction",
            "Ensemble_prediction",
            "Prognose_ratio",
            "SARIMA_cumulative",
            "SARIMA_individual",
        ]

        mae_columns = [f"MAE_{pred}" for pred in predictions]
        mape_columns = [f"MAPE_{pred}" for pred in predictions]

        for col in mae_columns + mape_columns:
            self.data[col] = np.nan

        valid_rows = ~self.data["Croho groepeernaam"].isin(self.numerus_fixus_list)

        for pred in predictions:
            predicted = self.data[pred].apply(self.convert_nan_to_zero)
            self.data[f"MAE_{pred}"] = abs(self.data["Aantal_studenten"] - predicted).where(
                valid_rows
            )
            self.data[f"MAPE_{pred}"] = (
                abs(self.data["Aantal_studenten"] - predicted) / self.data["Aantal_studenten"]
            ).where(valid_rows, np.nan)

    def _calculate_errors(self, row):
        actual = row["Aantal_studenten"]
        errors = {}
        for key in [
            "Weighted_ensemble_prediction",
            "Average_ensemble_prediction",
            "Ensemble_prediction",
            "Prognose_ratio",
            "SARIMA_cumulative",
            "SARIMA_individual",
        ]:
            predicted = self.convert_nan_to_zero(row[key])
            errors[f"MAE_{key}"] = self._mean_absolute_error(actual, predicted)
            errors[f"MAPE_{key}"] = self._mean_absolute_percentage_error(actual, predicted)
        return errors

    def _mean_absolute_error(self, actual, predicted):
        return abs(actual - predicted)

    def _mean_absolute_percentage_error(self, actual, predicted):
        return abs((actual - predicted) / actual) if actual != 0 else np.nan

    def convert_nan_to_zero(self, number):
        if pd.isnull(number):
            return 0
        else:
            return number


if __name__ == "__main__":
    year_to_calculate_ensemble_values = 2021
    year_to_use_ensemble_weights_from = 2022

    configuration = load_configuration("configuration/configuration.json")
    # NOTE: This script uses SARIMA_cumulative and Prognose_ratio (from cumulative variant)
    # and SARIMA_individual (from individual variant). Currently only loads the cumulative file.
    data_latest = pd.read_excel(configuration["paths"]["path_latest_cumulative"])
    data_studentcount = pd.read_excel(configuration["paths"]["path_student_count_first-years"])

    ensemble_weights = pd.read_excel(configuration["paths"]["path_ensemble_weights"])
    new_ensemble_weights = ensemble_weights[
        ensemble_weights["Collegejaar"] == year_to_use_ensemble_weights_from
    ].copy()
    new_ensemble_weights["Collegejaar"] = year_to_calculate_ensemble_values
    ensemble_weights = ensemble_weights[
        ensemble_weights["Collegejaar"] != year_to_calculate_ensemble_values
    ].copy()
    ensemble_weights = pd.concat([ensemble_weights, new_ensemble_weights], ignore_index=True)
    ensemble_weights = ensemble_weights.sort_values(
        by=["Collegejaar", "Programme", "Examentype", "Herkomst"],
        ignore_index=True,
    )

    apply_weights_on_different_year = ApplyWeightsOnDifferentYear(
        data_latest,
        ensemble_weights,
        data_studentcount,
        year_to_calculate_ensemble_values,
        configuration["numerus_fixus"],
    )
    apply_weights_on_different_year.postprocess()
    data_latest = apply_weights_on_different_year.data_latest
    data_latest.to_excel(configuration["paths"]["path_latest_cumulative"], index=False)
