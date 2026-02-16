import os as os
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# from scripts.load_data import load_configuration


class AddWeeksWherePreapplicantsAreZero:
    def __init__(self, data_cumulative, years, weeks):
        self.data_cumulative = data_cumulative
        self.years = years
        self.weeks = weeks

    def add_weeks(self):
        data_cumulative = self.data_cumulative

        # Convert numeric columns to appropriate types before using them
        numeric_columns = [
            "Gewogen vooraanmelders",
            "Ongewogen vooraanmelders",
            "Aantal aanmelders met 1 aanmelding",
            "Inschrijvingen",
            "Collegejaar",
            "Weeknummer",
            "Weeknummer rapportage"
        ]
        for col in numeric_columns:
            if col in data_cumulative.columns:
                data_cumulative[col] = pd.to_numeric(data_cumulative[col], errors='coerce').fillna(0).astype(int)

        data_cumulative.loc[
            (data_cumulative["Type hoger onderwijs"] == "Pre-master"),
            "Hogerejaars",
        ] = "Nee"

        data_cumulative = data_cumulative[data_cumulative["Hogerejaars"] == "Nee"]

        herkomsten = ["NL", "EER", "Niet-EER"]
        zero_cols = [
            "Gewogen vooraanmelders",
            "Ongewogen vooraanmelders",
            "Aantal aanmelders met 1 aanmelding",
            "Inschrijvingen",
        ]

        new_rows = []

        for year in self.years:
            data_previous_year = data_cumulative[
                (data_cumulative["Collegejaar"] == year - 1)
                & (data_cumulative["Inschrijvingen"] > 0)
            ]

            for programme in data_previous_year["Groepeernaam Croho"].unique():
                for examtype in data_previous_year[
                    data_previous_year["Groepeernaam Croho"] == programme
                ]["Type hoger onderwijs"].unique():
                    data_this_year = data_cumulative[
                        (data_cumulative["Collegejaar"] == year)
                        & (data_cumulative["Type hoger onderwijs"] == examtype)
                        & (data_cumulative["Groepeernaam Croho"] == programme)
                    ]

                    programme_row = (
                        data_previous_year[
                            (data_previous_year["Groepeernaam Croho"] == programme)
                            & (data_previous_year["Type hoger onderwijs"] == examtype)
                        ]
                        .iloc[0]
                        .copy()
                    )
                    programme_row["Collegejaar"] = year
                    for col in zero_cols:
                        programme_row[col] = 0

                    for herkomst in herkomsten:
                        programme_row["Herkomst"] = herkomst

                        data_herkomst = data_this_year[data_this_year["Herkomst"] == herkomst]
                        missing_weeks = list(
                            set(self.weeks) - set(data_herkomst["Weeknummer"].unique())
                        )
                        for week in missing_weeks:
                            new_row = programme_row.copy()
                            new_row["Weeknummer"] = week
                            new_row["Weeknummer rapportage"] = week
                            new_rows.append(new_row)

        data_cumulative = pd.concat([data_cumulative, pd.DataFrame(new_rows)], ignore_index=True)

        data_cumulative = data_cumulative.sort_values(
            by=[
                "Groepeernaam Croho",
                "Type hoger onderwijs",
                "Collegejaar",
                "Weeknummer",
                "Herkomst",
            ],
            ignore_index=True,
        )

        data_cumulative["Weeknummer"] = data_cumulative["Weeknummer"].astype(int)
        data_cumulative["Weeknummer rapportage"] = data_cumulative["Weeknummer rapportage"].astype(
            int
        )
        data_cumulative["Collegejaar"] = data_cumulative["Collegejaar"].astype(int)

        self.data_cumulative = data_cumulative


if __name__ == "__main__":
    # configuration = load_configuration("configuration/configuration.json")
    path_cumulative = "\\\\ru.nl\\wrkgrp\\TeamIR\\Man_info\\Student Analytics\\Prognosemodel RU\\Syntax\\Python\\studentprognose\\data\\input\\vooraanmeldingen_cumulatief.csv"
    data_cumulative = pd.read_csv(path_cumulative, sep=";", skiprows=[1], low_memory=True)

    years = [2024]
    weeks = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
    ]

    add_weeks_where_preapplicants_are_zero = AddWeeksWherePreapplicantsAreZero(
        data_cumulative, years, weeks
    )

    add_weeks_where_preapplicants_are_zero.add_weeks()

    new_data_cumulative = add_weeks_where_preapplicants_are_zero.data_cumulative

    CWD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    outfile = os.path.join(CWD, "data/output/vooraanmeldingen_cumulatief_zerorows.csv")
    new_data_cumulative.to_csv(outfile, sep=";", index=False)
