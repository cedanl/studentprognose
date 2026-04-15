import numpy as np
import pandas as pd


def predict_with_ratio(data: pd.DataFrame, data_cumulative: pd.DataFrame, data_studentcount: pd.DataFrame | None, numerus_fixus_list: dict, predict_year: int) -> pd.DataFrame:
    """
    Predict student influx using the ratio between pre-registrants and actual enrollments.

    Uses a 3-year historical average ratio.

    Args:
        data: The main output data (will be modified in place and returned).
        data_cumulative: Cumulative pre-application data.
        data_studentcount: Actual student count data.
        numerus_fixus_list: Dict of numerus fixus programmes and their caps.
        predict_year: The year being predicted.

    Returns:
        pd.DataFrame: The data with Prognose_ratio column added.
    """
    average_ratio_between = (predict_year - 3, predict_year - 1)
    if data_studentcount is not None:
        data_vooraanmeldingen = data_cumulative[
            [
                "Collegejaar",
                "Weeknummer",
                "Examentype",
                "Croho groepeernaam",
                "Herkomst",
                "Ongewogen vooraanmelders",
                "Inschrijvingen",
            ]
        ]

        data_vooraanmeldingen["Ongewogen vooraanmelders"] = data_vooraanmeldingen[
            "Ongewogen vooraanmelders"
        ].astype("float")

        data_vooraanmeldingen["Aanmeldingen"] = (
            data_vooraanmeldingen["Ongewogen vooraanmelders"]
            + data_vooraanmeldingen["Inschrijvingen"]
        )

        data_vooraanmeldingen = data_vooraanmeldingen[
            (data_vooraanmeldingen["Collegejaar"] >= average_ratio_between[0])
            & (data_vooraanmeldingen["Collegejaar"] <= average_ratio_between[1])
        ]

        data_merged = pd.merge(
            left=data_vooraanmeldingen,
            right=data_studentcount,
            on=["Collegejaar", "Examentype", "Croho groepeernaam", "Herkomst"],
            how="left",
        )

        data_merged["Average_Ratio"] = data_merged["Aanmeldingen"].divide(
            data_merged["Aantal_studenten"]
        )

        data_merged["Average_Ratio"] = data_merged["Average_Ratio"].replace(np.inf, np.nan)

        average_ratios = (
            data_merged[
                [
                    "Croho groepeernaam",
                    "Examentype",
                    "Herkomst",
                    "Weeknummer",
                    "Average_Ratio",
                ]
            ]
            .groupby(
                ["Croho groepeernaam", "Examentype", "Herkomst", "Weeknummer"],
                as_index=False,
            )
            .mean()
        )

        data = data.rename(columns={"Aantal eerstejaarsopleiding": "EOI_vorigjaar"})

        data["Ongewogen vooraanmelders"] = data["Ongewogen vooraanmelders"].astype(
            "float"
        )
        data["Aanmelding"] = (
            data["Ongewogen vooraanmelders"] + data["Inschrijvingen"]
        )

        data["Ratio"] = (data["Aanmelding"]).divide(data["Aantal_studenten"])
        data["Ratio"] = data["Ratio"].replace(np.inf, np.nan)

        data = pd.merge(
            left=data,
            right=average_ratios,
            how="left",
            on=["Croho groepeernaam", "Examentype", "Herkomst", "Weeknummer"],
        )

        for i, row in data.iterrows():
            if row["Average_Ratio"] != 0:
                data.at[i, "Prognose_ratio"] = row["Aanmelding"] / row["Average_Ratio"]

        NFs = numerus_fixus_list
        for year in data["Collegejaar"].unique():
            for week in data["Weeknummer"].unique():
                for nf in NFs:
                    nf_data = data[
                        (data["Collegejaar"] == year)
                        & (data["Weeknummer"] == week)
                        & (data["Croho groepeernaam"] == nf)
                    ]

                    if np.sum(nf_data["Prognose_ratio"]) > NFs[nf]:
                        data.loc[
                            (data["Collegejaar"] == year)
                            & (data["Weeknummer"] == week)
                            & (data["Croho groepeernaam"] == nf)
                            & (data["Herkomst"] == "NL"),
                            "Prognose_ratio",
                        ] = nf_data[nf_data["Herkomst"] == "NL"]["Prognose_ratio"] - (
                            np.sum(nf_data["Prognose_ratio"]) - NFs[nf]
                        )

    return data
