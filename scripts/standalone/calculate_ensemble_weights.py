import pandas as pd
import numpy as np
import os
import sys
import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.load_data import load_configuration

CONFIGURATION = load_configuration("configuration/configuration.json")


class CalculateEnsembleWeights:
    def __init__(
        self,
        data_latest,
        years,
        percentage_range,
        methods,
        weight_data,
        error_rates,
        ensemble_weights,
    ):
        self.data_latest = data_latest
        self.years = years
        self.percentage_range = percentage_range
        self.methods = methods
        self.weight_data = weight_data
        self.error_rates = error_rates
        self.ensemble_weights = ensemble_weights

    def calculate_weight_distribution(self):
        data = self.data_latest.copy()

        weight_distribution = {
            "Collegejaar": [],
            "Programme": [],
            "Examentype": [],
            "Herkomst": [],
            "Percentage": [],
            "Second percentage": [],
        }

        MAEs = []
        MAPEs = []

        print("Calculating weight distribution...")
        for year in self.years:
            print(f"Year: {year}")
            data_temp = data[data["Collegejaar"] < year]
            for percentage in self.percentage_range:
                print(f"Percentage: {percentage}")
                for second_percentage in self.percentage_range:
                    for programme in data_temp["Croho groepeernaam"].unique():
                        for examentype in data_temp[data_temp["Croho groepeernaam"] == programme][
                            "Examentype"
                        ].unique():
                            for herkomst in data_temp["Herkomst"].unique():
                                weight_distribution["Collegejaar"].append(year)
                                weight_distribution["Programme"].append(programme)
                                weight_distribution["Examentype"].append(examentype)
                                weight_distribution["Herkomst"].append(herkomst)
                                weight_distribution["Percentage"].append(percentage)
                                weight_distribution["Second percentage"].append(second_percentage)

                                MAE, MAPE = self._get_metric_weight_distribution(
                                    data_temp,
                                    programme,
                                    examentype,
                                    herkomst,
                                    percentage,
                                    second_percentage,
                                    self.methods,
                                )

                                MAEs.append(MAE)
                                MAPEs.append(MAPE)

        for m in self.methods:
            weight_distribution[f"MAE_{m}"] = [dict(MAE)[m] for MAE in MAEs]
            weight_distribution[f"MAPE_{m}"] = [dict(MAPE)[m] for MAPE in MAPEs]

        new_weight_data = pd.DataFrame(weight_distribution)

        existing_weight_data = self.weight_data.copy()
        existing_weight_data = existing_weight_data[
            ~existing_weight_data["Collegejaar"].isin(self.years)
        ]
        weight_data = pd.concat([existing_weight_data, new_weight_data], ignore_index=True)
        weight_data = weight_data.sort_values(
            by=[
                "Collegejaar",
                "Programme",
                "Examentype",
                "Herkomst",
                "Percentage",
                "Second percentage",
            ],
            ignore_index=True,
        )

        self.weight_data = weight_data
        weight_data.to_excel(CONFIGURATION["paths"]["path_weight_distribution"], index=False)

    def calculate_error_rates(self):
        data = self.data_latest.copy()
        weight_data = self.weight_data.copy()

        error_rates = {
            "Collegejaar": [],
            "Percentage": [],
            "Second percentage": [],
            "MAE": [],
            "MAPE": [],
            "Herkomst": [],
            "MAE of AE": [],
            "MAPE of AE": [],
        }

        print("Calculating error rates...")
        for year in self.years:
            print(f"Year: {year}")
            temp_filtered_weight_data_year = weight_data[weight_data["Collegejaar"] == year]
            for herkomst in ["EER", "NL", "Niet-EER"]:
                temp_filtered_weight_data_herkomst = temp_filtered_weight_data_year[
                    temp_filtered_weight_data_year["Herkomst"] == herkomst
                ]
                print(f"Herkomst: {herkomst}")
                for percentage in self.percentage_range:
                    temp_filtered_weight_data_percentage = temp_filtered_weight_data_herkomst[
                        temp_filtered_weight_data_herkomst["Percentage"] == percentage
                    ]
                    for second_percentage in self.percentage_range:
                        filtered_weight_data = temp_filtered_weight_data_percentage[
                            temp_filtered_weight_data_percentage["Second percentage"]
                            == second_percentage
                        ]

                        error_rates["Collegejaar"].append(year)
                        error_rates["Percentage"].append(percentage)
                        error_rates["Second percentage"].append(second_percentage)

                        MAE_total = 0.0
                        MAPE_total = 0.0

                        MAE_total_AE = 0.0
                        MAPE_total_AE = 0.0

                        for i, row_weight in filtered_weight_data.iterrows():
                            filtered_data = data[
                                (data["Croho groepeernaam"] == row_weight["Programme"])
                                & (data["Examentype"] == row_weight["Examentype"])
                                & (data["Collegejaar"] < row_weight["Collegejaar"])
                                & (data["Collegejaar"] >= 2021)
                                & (data["Herkomst"] == row_weight["Herkomst"])
                            ]

                            use_average_ensemble = False
                            if sum([row_weight[f"MAE_{method}"] for method in self.methods]) != 1:
                                use_average_ensemble = True

                            MAE_subtotal = 0.0
                            MAPE_subtotal = 0.0

                            MAE_subtotal_AE = 0.0
                            MAPE_subtotal_AE = 0.0
                            for j, row in filtered_data.iterrows():
                                if use_average_ensemble:
                                    if not np.isnan(row["MAE_Average_ensemble_prediction"]):
                                        MAE_subtotal += row["MAE_Average_ensemble_prediction"]
                                    if not np.isnan(row["MAPE_Average_ensemble_prediction"]):
                                        MAPE_total += row["MAPE_Average_ensemble_prediction"]
                                else:
                                    if all(
                                        [(not np.isnan(row[method])) for method in self.methods]
                                    ) and not np.isnan(row["Aantal_studenten"]):
                                        new_ensemble_prediction = sum(
                                            [
                                                (row[method] * row_weight[f"MAE_{method}"])
                                                for method in self.methods
                                            ]
                                        )
                                        MAE_subtotal += self._mean_absolute_error(
                                            row["Aantal_studenten"], new_ensemble_prediction
                                        )
                                        MAPE_subtotal += self._mean_absolute_percentage_error(
                                            row["Aantal_studenten"], new_ensemble_prediction
                                        )

                                if not np.isnan(row["MAE_Average_ensemble_prediction"]):
                                    MAE_subtotal_AE += row["MAE_Average_ensemble_prediction"]
                                if not np.isnan(row["MAPE_Average_ensemble_prediction"]):
                                    MAPE_subtotal_AE += row["MAPE_Average_ensemble_prediction"]

                            count = len(filtered_data)
                            if count == 0:
                                count = 1

                            MAE_total += MAE_subtotal / count
                            MAPE_total += MAPE_subtotal / count * 100

                            # print(MAE_total)

                            MAE_total_AE += MAE_subtotal_AE / count
                            MAPE_total_AE += MAPE_subtotal_AE / count * 100

                        error_rates["MAE"].append(MAE_total / len(filtered_weight_data))
                        error_rates["MAPE"].append(MAPE_total / len(filtered_weight_data))
                        error_rates["Herkomst"].append(herkomst)
                        error_rates["MAE of AE"].append(MAE_total_AE / len(filtered_weight_data))
                        error_rates["MAPE of AE"].append(MAPE_total_AE / len(filtered_weight_data))

        new_error_rates = pd.DataFrame(error_rates)

        existing_error_rates = self.error_rates.copy()
        existing_error_rates = existing_error_rates[
            ~existing_error_rates["Collegejaar"].isin(self.years)
        ]
        error_rates = pd.concat([existing_error_rates, new_error_rates], ignore_index=True)
        error_rates = error_rates.sort_values(
            by=["Collegejaar", "Percentage", "Second percentage"], ignore_index=True
        )

        self.error_rates = error_rates
        error_rates.to_excel(CONFIGURATION["paths"]["path_error_rates"], index=False)

    def calculate_ensemble_weights(self):
        error_rates = self.error_rates.copy()
        weight_data = self.weight_data.copy()

        print("Calculating ensemble weights...")
        ensemble_weights = {
            "Collegejaar": [],
            "Programme": [],
            "Examentype": [],
            "Herkomst": [],
            "SARIMA_cumulative": [],
            "SARIMA_individual": [],
            "Prognose_ratio": [],
            "Average_ensemble_prediction": [],
        }
        for year in years:
            year_temp_data = error_rates[error_rates["Collegejaar"] == year]
            percentage_per_herkomst = {"EER": (), "NL": (), "Niet-EER": ()}
            for herkomst in percentage_per_herkomst.keys():
                temp_data = year_temp_data[year_temp_data["Herkomst"] == herkomst]
                temp_data = temp_data[temp_data["MAE"] == temp_data["MAE"].min()]
                percentage_per_herkomst[herkomst] = (
                    float(temp_data["Percentage"].iloc[0]),
                    float(temp_data["Second percentage"].iloc[0]),
                )

                temp_data = weight_data[
                    (weight_data["Herkomst"] == herkomst) & (weight_data["Collegejaar"] == year)
                ]
                temp_data = temp_data[
                    temp_data["Percentage"] == percentage_per_herkomst[herkomst][0]
                ]
                temp_data = temp_data[
                    temp_data["Second percentage"] == percentage_per_herkomst[herkomst][1]
                ]

                for i, row in temp_data.iterrows():
                    ensemble_weights["Collegejaar"].append(year)
                    ensemble_weights["Programme"].append(row["Programme"])
                    ensemble_weights["Examentype"].append(row["Examentype"])
                    ensemble_weights["Herkomst"].append(row["Herkomst"])

                    average_ensemble_weight = 0.0

                    MAEs = [row[f"MAE_{method}"] for method in methods]
                    sum_MAE = sum(MAEs)

                    if sum_MAE != 1 or sum_MAE == 0:
                        average_ensemble_weight = 1.0
                        MAEs = [0.0] * len(methods)

                    for m in range(len(methods)):
                        ensemble_weights[methods[m]].append(MAEs[m])

                    ensemble_weights["Average_ensemble_prediction"].append(average_ensemble_weight)

        new_ensemble_weights = pd.DataFrame(ensemble_weights)

        existing_ensemble_weights = self.ensemble_weights.copy()
        existing_ensemble_weights = existing_ensemble_weights[
            ~existing_ensemble_weights["Collegejaar"].isin(years)
        ]
        ensemble_weights = pd.concat(
            [existing_ensemble_weights, new_ensemble_weights], ignore_index=True
        )
        ensemble_weights = ensemble_weights.sort_values(
            by=["Collegejaar", "Programme", "Examentype", "Herkomst"],
            ignore_index=True,
        )

        self.ensemble_weights = ensemble_weights
        ensemble_weights.to_excel(CONFIGURATION["paths"]["path_ensemble_weights"], index=False)

    def _get_metric_weight_distribution(
        self, data, programme, examentype, herkomst, percentage, second_percentage, methods
    ):
        # print(programme, herkomst)
        data = data[
            (data["Croho groepeernaam"] == programme)
            & (data["Examentype"] == examentype)
            & (data["Herkomst"] == herkomst)
        ]

        MAE = [(method, np.nanmean(data[f"MAE_{method}"])) for method in methods]
        MAE = self._compute_metric_weights(MAE, percentage, second_percentage)
        # print(f"MAE: {MAE}")

        MAPE = [
            (method, np.nanmean(data[f"MAPE_{method}"]) / len(data) * 100) for method in methods
        ]
        MAPE = self._compute_metric_weights(MAPE, percentage, second_percentage)
        # print(f"MAPE: {MAPE}")

        return MAE, MAPE

    def _compute_metric_weights(self, metrics, percentage, second_percentage):
        # print(f"Metrics {metrics}")
        sorted(metrics, key=lambda x: x[1])
        # print(f"Metrics {metrics}")
        result = [(m[0], 0.0) for m in metrics]

        first_threshold = 1.0 + (float(percentage) / 100)
        second_threshold = 1.0 + (float(second_percentage) / 100)

        # Check if first method is best
        if metrics[1][1] > metrics[0][1] * first_threshold:
            result[0] = (result[0][0], 1.0)
            return result

        # Check if second method is still better than third
        elif metrics[2][1] > metrics[1][1] * second_threshold:
            m1 = metrics[0][1]
            m2 = metrics[1][1]
            total = m1 + m2
            new_total = total / m1 + total / m2

            result[0] = (result[0][0], (total / m1) / new_total)
            result[1] = (result[1][0], (total / m2) / new_total)
            return result

        # Combine all
        else:
            total = sum([m[1] for m in metrics])
            new_total = sum([total / m[1] for m in metrics])
            result = [(m[0], (total / m[1]) / new_total) for m in metrics]
            return result

    def _mean_absolute_error(self, true, prediction):
        return abs(true - prediction)

    def _mean_absolute_percentage_error(self, true, prediction):
        if true == 0:
            return np.nan  

        return abs((true - prediction) / true)

 


def get_year_to_calculate():
    arguments = sys.argv
    years = []
    year_slice = False

    for i in range(1, len(arguments)):
        arg = arguments[i]
        if arg == "-y" or arg == "-year":
            continue
        elif arg == ":":
            year_slice = True
        elif arg.isnumeric():
            if year_slice:
                last_year = years.pop(-1)
                years = years + list(range(last_year, int(arg) + 1))
                year_slice = False
            else:
                years.append(int(arg))

    if years == []:
        years.append(datetime.date.today().year)

    return years


def load_current_ensemble_data():
    weight_data = pd.DataFrame()
    error_rates = pd.DataFrame()
    ensemble_weights = pd.DataFrame()

    if os.path.isfile(CONFIGURATION["paths"]["path_weight_distribution"]):
        weight_data = pd.read_excel(CONFIGURATION["paths"]["path_weight_distribution"])
    if os.path.isfile(CONFIGURATION["paths"]["path_error_rates"]):
        error_rates = pd.read_excel(CONFIGURATION["paths"]["path_error_rates"])
    if os.path.isfile(CONFIGURATION["paths"]["path_ensemble_weights"]):
        ensemble_weights = pd.read_excel(CONFIGURATION["paths"]["path_ensemble_weights"])

    return weight_data, error_rates, ensemble_weights


if __name__ == "__main__":
    # NOTE: This script uses SARIMA_cumulative and Prognose_ratio (from cumulative variant)
    # and SARIMA_individual (from individual variant). Currently only loads the cumulative file.
    data_latest = pd.read_excel(CONFIGURATION["paths"]["path_latest_cumulative"])
    data_latest = data_latest.replace(np.inf, np.nan)

    years = get_year_to_calculate()
    percentage_range = range(10, 150, 10)
    methods = ["SARIMA_cumulative", "SARIMA_individual", "Prognose_ratio"]
    weight_data, error_rates, ensemble_weights = load_current_ensemble_data()

    calcule_ensemble_weights = CalculateEnsembleWeights(
        data_latest, years, percentage_range, methods, weight_data, error_rates, ensemble_weights
    )

    calcule_ensemble_weights.calculate_weight_distribution()
    calcule_ensemble_weights.calculate_error_rates()
    calcule_ensemble_weights.calculate_ensemble_weights()
