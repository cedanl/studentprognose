import os
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.load_data import load_configuration


class FillInRatioFile:
    def __init__(
        self,
        data_student_numbers_first_years,
        data_student_numbers_higher_years,
        data_october,
        data_ratios,
    ):
        self.data_student_numbers_first_years = data_student_numbers_first_years
        self.data_student_numbers_higher_years = data_student_numbers_higher_years
        self.data_october = data_october
        self.data_ratios = data_ratios

    def calculate_ratios_and_fill_in_dataframe(self, predict_year):
        data_october = self.data_october.copy(deep=True)
        data_ratios = self.data_ratios.copy(deep=True)

        if data_ratios.empty:
            data_ratios = pd.DataFrame(
                columns=[
                    "Collegejaar",
                    "Croho groepeernaam",
                    "Herkomst",
                    "Examentype",
                    "Ratio dat doorstroomt",
                    "Ratio dat uitvalt",
                ]
            )

        for programme in data_october["Groepeernaam Croho"].unique():
            for examtype in data_october[data_october["Groepeernaam Croho"] == programme][
                "Examentype code"
            ].unique():
                for herkomst in data_october[
                    (data_october["Groepeernaam Croho"] == programme)
                    & (data_october["Examentype code"] == examtype)
                ]["EER-NL-nietEER"].unique():
                    if examtype in ["Bachelor eerstejaars", "Bachelor hogerejaars"]:
                        examtype = "Bachelor"

                    avg_ratio_advancing_to_higher_year, avg_ratio_dropping_out_in_higher_year = (
                        self.ratio_last_3_years(predict_year, programme, examtype, herkomst)
                    )

                    if data_ratios[
                        (data_ratios["Collegejaar"] == predict_year)
                        & (data_ratios["Croho groepeernaam"] == programme)
                        & (data_ratios["Herkomst"] == herkomst)
                        & (data_ratios["Examentype"] == examtype)
                    ].empty:
                        data_ratios = data_ratios._append(
                            {
                                "Collegejaar": predict_year,
                                "Croho groepeernaam": programme,
                                "Herkomst": herkomst,
                                "Examentype": examtype,
                                "Ratio dat doorstroomt": avg_ratio_advancing_to_higher_year,
                                "Ratio dat uitvalt": avg_ratio_dropping_out_in_higher_year,
                            },
                            ignore_index=True,
                        )
                    else:
                        data_ratios.loc[
                            (data_ratios["Collegejaar"] == predict_year)
                            & (data_ratios["Croho groepeernaam"] == programme)
                            & (data_ratios["Herkomst"] == herkomst)
                            & (data_ratios["Examentype"] == examtype),
                            "Ratio dat doorstroomt",
                        ] = avg_ratio_advancing_to_higher_year
                        data_ratios.loc[
                            (data_ratios["Collegejaar"] == predict_year)
                            & (data_ratios["Croho groepeernaam"] == programme)
                            & (data_ratios["Herkomst"] == herkomst)
                            & (data_ratios["Examentype"] == examtype),
                            "Ratio dat uitvalt",
                        ] = avg_ratio_dropping_out_in_higher_year

        self.data_ratios = data_ratios.copy()

    def ratio_last_3_years(self, predict_year, programme, examtype, herkomst):
        avg_ratio_advancing_to_higher_year = 0
        avg_ratio_dropping_out_in_higher_year = 0
        for year in range(predict_year - 4, predict_year - 1):
            avg_ratio_advancing_to_higher_year += self.ratio_advancing_to_higher_year(
                year, programme, examtype, herkomst
            )
            avg_ratio_dropping_out_in_higher_year += self.ratio_dropping_out_in_higher_year(
                year, programme, examtype, herkomst
            )
        avg_ratio_advancing_to_higher_year /= 3
        avg_ratio_dropping_out_in_higher_year /= 3

        return avg_ratio_advancing_to_higher_year, avg_ratio_dropping_out_in_higher_year

    def ratio_advancing_to_higher_year(self, year_firstyear, programme, examtype, herkomst):
        data_october = self.data_october.copy(deep=True)

        if examtype == "Bachelor":
            examtypes = ["Bachelor eerstejaars", "Bachelor hogerejaars"]
            eerstejaars_examtype = "Bachelor eerstejaars"
        else:
            examtypes = [examtype]
            eerstejaars_examtype = examtype

        next_year = data_october[
            (data_october["Collegejaar"] == year_firstyear + 1)
            & (data_october["Groepeernaam Croho"] == programme)
            & (data_october["Examentype code"].isin(examtypes))
            & (data_october["EER-NL-nietEER"] == herkomst)
            & (data_october["Aantal Hoofdinschrijvingen"] == 1)
        ]

        this_year = data_october[
            (data_october["Collegejaar"] == year_firstyear)
            & (data_october["Groepeernaam Croho"] == programme)
            & (data_october["Examentype code"] == eerstejaars_examtype)
            & (data_october["EER-NL-nietEER"] == herkomst)
            & (data_october["Aantal Hoofdinschrijvingen"] == 1)
            & (data_october["Aantal eerstejaars croho"] == 1)
        ]

        students_advanced = next_year.merge(this_year, on="ID", how="inner")

        if len(this_year) > 0:
            ratio_advancing = len(students_advanced) / len(this_year)
        else:
            ratio_advancing = 1

        return ratio_advancing

    def ratio_dropping_out_in_higher_year(self, year, programme, examtype, herkomst):
        data_october = self.data_october.copy(deep=True)

        if examtype == "Bachelor":
            hogerejaars_examtype = "Bachelor hogerejaars"
        else:
            hogerejaars_examtype = examtype

        this_year = data_october[
            (data_october["Collegejaar"] == year)
            & (data_october["Groepeernaam Croho"] == programme)
            & (data_october["Examentype code"] == hogerejaars_examtype)
            & (data_october["EER-NL-nietEER"] == herkomst)
            & (data_october["Aantal Hoofdinschrijvingen"] == 1)
            & (data_october["Aantal eerstejaars croho"] == 0)
        ]

        next_year = data_october[
            (data_october["Collegejaar"] == year + 1)
            & (data_october["Groepeernaam Croho"] == programme)
            & (data_october["Examentype code"] == hogerejaars_examtype)
            & (data_october["EER-NL-nietEER"] == herkomst)
            & (data_october["Aantal Hoofdinschrijvingen"] == 1)
            & (data_october["Aantal eerstejaars croho"] == 0)
        ]

        students_dropped_out = this_year[~this_year["ID"].isin(next_year["ID"])]

        if len(this_year) > 0:
            ratio_dropping_out = len(students_dropped_out) / len(this_year)
        else:
            ratio_dropping_out = 0

        return ratio_dropping_out


def calculate_ratios(years_to_predict):
    print("Calculating ratios for the years: ", years_to_predict)
    configuration = load_configuration("configuration/configuration.json")

    data_student_numbers_first_years = pd.read_excel(
        configuration["paths"]["path_student_count_first-years"]
    )
    data_student_numbers_higher_years = pd.read_excel(
        configuration["paths"]["path_student_count_higher-years"]
    )

    data_october = pd.read_excel(configuration["paths"]["path_october"])
    try:
        data_ratios = pd.read_excel(configuration["paths"]["path_ratios"])
    except:
        data_ratios = pd.DataFrame()

    fill_in_ratio_file = FillInRatioFile(
        data_student_numbers_first_years,
        data_student_numbers_higher_years,
        data_october,
        data_ratios,
    )

    for year in years_to_predict:
        fill_in_ratio_file.calculate_ratios_and_fill_in_dataframe(year)

    new_data_ratios = fill_in_ratio_file.data_ratios

    new_data_ratios.sort_values(
        by=["Croho groepeernaam", "Examentype", "Collegejaar", "Herkomst"],
        inplace=True,
        ignore_index=True,
    )
    new_data_ratios = new_data_ratios[
        [
            "Croho groepeernaam",
            "Examentype",
            "Collegejaar",
            "Herkomst",
            "Ratio dat doorstroomt",
            "Ratio dat uitvalt",
        ]
    ]

    new_data_ratios.to_excel(configuration["paths"]["path_ratios"], index=False)
