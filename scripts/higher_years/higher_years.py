import os
import pandas as pd
import numpy as np
import sys
from fill_in_ratiofile import calculate_ratios
import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.load_data import load_configuration


class PredictHigherYearsBasedOnLastYearNumbers:
    def __init__(
        self,
        data_student_numbers_first_years,
        data_student_numbers_higher_years,
        data_october,
        data_ratios,
        data_latest,
    ):
        self.data_student_numbers_first_years = data_student_numbers_first_years
        self.data_student_numbers_higher_years = data_student_numbers_higher_years
        self.data_october = data_october
        self.data_ratios = data_ratios
        self.data_latest = data_latest

    def run_predict_with_last_year_numbers(self, predict_year, all_data, skip_years=False):
        for programme in self.data_october["Groepeernaam Croho"].unique():
            for examtype in self.data_october[
                self.data_october["Groepeernaam Croho"] == programme
            ]["Examentype code"].unique():
                for herkomst in self.data_october[
                    (self.data_october["Groepeernaam Croho"] == programme)
                    & (self.data_october["Examentype code"] == examtype)
                ]["EER-NL-nietEER"].unique():
                    if examtype in ["Bachelor eerstejaars", "Bachelor hogerejaars"]:
                        examtype = "Bachelor"

                    data_filtered = self.data_latest[
                        (self.data_latest["Croho groepeernaam"] == programme)
                        & (self.data_latest["Examentype"] == examtype)
                        & (self.data_latest["Herkomst"] == herkomst)
                        & (self.data_latest["Collegejaar"] == predict_year - 1)
                    ]

                    try:
                        faculteit = self.data_latest[
                            self.data_latest["Croho groepeernaam"] == programme
                        ]["Faculteit"].values[0]
                    except:
                        faculteit = None  # Van toepassing bij nieuwe opleidingen waarvan faculteit niet bekend is in totaalbestand.

                    for week in range(1, 53):
                        if data_filtered[data_filtered["Weeknummer"] == week].empty:
                            all_data = all_data._append(
                                {
                                    "Examentype": examtype,
                                    "Croho groepeernaam": programme,
                                    "Herkomst": herkomst,
                                    "Faculteit": faculteit,
                                    "Collegejaar": predict_year,
                                    "Weeknummer": week,
                                },
                                ignore_index=True,
                            )

                    if not skip_years:
                        nextyear_higher_years = self.predict_with_last_year_numbers(
                            predict_year, programme, examtype, herkomst
                        )

                        all_data.loc[
                            (all_data["Collegejaar"] == predict_year)
                            & (all_data["Croho groepeernaam"] == programme)
                            & (all_data["Herkomst"] == herkomst)
                            & (all_data["Examentype"] == examtype),
                            "Higher_years_prediction",
                        ] = nextyear_higher_years

                    elif skip_years:
                        weeks_with_prediction = all_data[
                            all_data["Average_ensemble_prediction"] > 0
                        ]["Weeknummer"].unique()

                        for week in weeks_with_prediction:
                            nextyear_higher_years = self.predict_with_last_year_numbers(
                                predict_year,
                                programme,
                                examtype,
                                herkomst,
                                skip_years=True,
                                week=week,
                            )

                            all_data.loc[
                                (all_data["Collegejaar"] == predict_year)
                                & (all_data["Croho groepeernaam"] == programme)
                                & (all_data["Herkomst"] == herkomst)
                                & (all_data["Examentype"] == examtype)
                                & (all_data["Weeknummer"] == week),
                                "Higher_years_prediction",
                            ] = nextyear_higher_years

        return all_data

    def predict_with_last_year_numbers(
        self, predict_year, programme, examtype, herkomst, skip_years=False, week=0
    ):
        row = self.data_ratios[
            (self.data_ratios["Collegejaar"] == predict_year)
            & (self.data_ratios["Croho groepeernaam"] == programme)
            & (self.data_ratios["Examentype"] == examtype)
            & (self.data_ratios["Herkomst"] == herkomst)
        ]
        if not row.empty:
            avg_ratio_advancing_to_higher_year = row["Ratio dat doorstroomt"].values[0]
            avg_ratio_dropping_out_in_higher_year = row["Ratio dat uitvalt"].values[0]
        else:
            avg_ratio_advancing_to_higher_year = 0
            avg_ratio_dropping_out_in_higher_year = 0

        if not skip_years:
            currentyear_higher_years = self.data_student_numbers_higher_years[
                (self.data_student_numbers_higher_years["Collegejaar"] == predict_year - 1)
                & (self.data_student_numbers_higher_years["Croho groepeernaam"] == programme)
                & (self.data_student_numbers_higher_years["Examentype"] == examtype)
                & (self.data_student_numbers_higher_years["Herkomst"] == herkomst)
            ]
            if not currentyear_higher_years.empty:
                currentyear_higher_years = currentyear_higher_years["Aantal_studenten"].values[0]
            else:
                currentyear_higher_years = 0

            currentyear_first_years = self.data_student_numbers_first_years[
                (self.data_student_numbers_first_years["Collegejaar"] == predict_year - 1)
                & (self.data_student_numbers_first_years["Croho groepeernaam"] == programme)
                & (self.data_student_numbers_first_years["Examentype"] == examtype)
                & (self.data_student_numbers_first_years["Herkomst"] == herkomst)
            ]
            if not currentyear_first_years.empty:
                currentyear_first_years = currentyear_first_years["Aantal_studenten"].values[0]
            else:
                currentyear_first_years = 0

        if skip_years:
            predicted_row = self.data_latest[
                (self.data_latest["Collegejaar"] == predict_year - skip_years - 1)
                & (self.data_latest["Croho groepeernaam"] == programme)
                & (self.data_latest["Examentype"] == examtype)
                & (self.data_latest["Herkomst"] == herkomst)
                & (self.data_latest["Weeknummer"] == week)
            ]

            currentyear_higher_years = predicted_row["Higher_years_prediction"]
            if not currentyear_higher_years.empty:
                currentyear_higher_years = currentyear_higher_years.values[0]
            else:
                currentyear_higher_years = 0

            currentyear_first_years = predicted_row["Average_ensemble_prediction"]
            if not currentyear_first_years.empty:
                currentyear_first_years = currentyear_first_years.values[0]
            else:
                currentyear_first_years = 0

        nextyear_higher_years = (
            currentyear_higher_years
            + (currentyear_first_years * avg_ratio_advancing_to_higher_year)
            - (currentyear_higher_years * avg_ratio_dropping_out_in_higher_year)
        )

        return nextyear_higher_years


if __name__ == "__main__":
    arguments = sys.argv
    predict_years = []
    year_slice = False

    for i in range(1, len(arguments)):
        arg = arguments[i]
        if arg == "-y" or arg == "-year":
            continue
        elif arg == ":":
            year_slice = True
        elif arg.isnumeric():
            if year_slice:
                last_year = predict_years.pop(-1)
                predict_years = predict_years + list(range(last_year, int(arg) + 1))
                year_slice = False
            else:
                predict_years.append(int(arg))

    if predict_years == []:
        predict_years = datetime.date.today().year

    configuration = load_configuration("configuration/configuration.json")

    data_student_numbers_first_years = pd.read_excel(
        configuration["paths"]["path_student_count_first-years"]
    )
    data_student_numbers_higher_years = pd.read_excel(
        configuration["paths"]["path_student_count_higher-years"]
    )

    data_october = pd.read_excel(configuration["paths"]["path_october"])

    data_ratios = pd.read_excel(configuration["paths"]["path_ratios"])
    data_latest = pd.read_excel(configuration["paths"]["path_latest_cumulative"])

    data_october = data_october[data_october["Examentype code"] != "Master post-initieel"]
    data_october[
        data_october["Groepeernaam Croho"] == "M LVHO in de Taal- en Cultuurwetenschappen"
    ]["Groepeernaam Croho"] = "M LVHO in de Taal en Cultuurwetenschappen"

    print("Predicting higher years based on last year numbers...")
    predict_higher_years_based_on_last_year_numbers = PredictHigherYearsBasedOnLastYearNumbers(
        data_student_numbers_first_years,
        data_student_numbers_higher_years,
        data_october,
        data_ratios,
        data_latest,
    )

    for predict_year in predict_years:
        skip_year = (predict_year - max(data_october["Collegejaar"]) - 2) == 0
        if (predict_year - max(data_october["Collegejaar"]) - 2) > 0:
            raise ValueError(
                f"Predict year {predict_year} is too far in the future. The maximum year in the data is {max(data_october['Collegejaar'])+2}."
            )
        print(f"Skip year: {skip_year}")

        if not skip_year:
            calculate_ratios([predict_year])
        elif skip_year:
            calculate_ratios([predict_year - 1])

        print(f"Predicting year {predict_year}")
        data_latest = (
            predict_higher_years_based_on_last_year_numbers.run_predict_with_last_year_numbers(
                predict_year, data_latest, skip_year  # , week
            )
        )

    print("Adding MAE and MAPE errors (if applicable)")
    data_latest["MAE_higher_years"] = data_latest.apply(
        lambda row: (
            abs(row["Aantal_studenten_higher_years"] - row["Higher_years_prediction"])
            if pd.notna(row["Higher_years_prediction"])
            and pd.notna(row["Aantal_studenten_higher_years"])
            else np.nan
        ),
        axis=1,
    )
    data_latest["MAPE_higher_years"] = data_latest.apply(
        lambda row: (
            abs(
                (row["Aantal_studenten_higher_years"] - row["Higher_years_prediction"])
                / row["Aantal_studenten_higher_years"]
            )
            if pd.notna(row["Higher_years_prediction"])
            and pd.notna(row["Aantal_studenten_higher_years"])
            and row["Aantal_studenten_higher_years"] != 0
            else np.nan
        ),
        axis=1,
    )

    data_latest.sort_values(
        by=["Croho groepeernaam", "Examentype", "Collegejaar", "Weeknummer", "Herkomst"],
        inplace=True,
        ignore_index=True,
    )

    print("Saving output...")
    CWD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    outfile = os.path.join(CWD, "data/output/output_higher-years.xlsx")

    data_latest.to_excel(outfile, index=False)
