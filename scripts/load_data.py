import pandas as pd
import json
import os
from scripts.helper import DataOption
from scripts.standalone.add_weeks_where_preapplicants_are_zero import (
    AddWeeksWherePreapplicantsAreZero,
)


def get_path_latest(paths, data_option):
    if data_option == DataOption.INDIVIDUAL:
        return paths.get("path_latest_individual", "")
    elif data_option == DataOption.CUMULATIVE:
        return paths.get("path_latest_cumulative", "")
    return ""


def load_data(configuration, data_option):
    paths = configuration["paths"]

    data_individual = None
    data_distances = None
    if data_option == DataOption.INDIVIDUAL or data_option == DataOption.BOTH_DATASETS:
        data_individual = (
            pd.read_csv(paths["path_individual"], sep=";", skiprows=[1])
            if (paths["path_individual"] != "" and os.path.exists(paths["path_individual"]))
            else None
        )

        data_distances = (
            pd.read_excel(paths["path_distances"])
            if (paths["path_distances"] != "" and os.path.exists(paths["path_distances"]))
            else None
        )

    data_cumulative = None
    if data_option == DataOption.CUMULATIVE or data_option == DataOption.BOTH_DATASETS:
        data_cumulative = (
            pd.read_csv(paths["path_cumulative"], sep=";", skiprows=[1], low_memory=True)
            if (paths["path_cumulative"] != "" and os.path.exists(paths["path_cumulative"]))
            else None
        )
        if os.path.exists(paths["path_cumulative_new"]):
            data_cumulative_new = pd.read_csv(paths["path_cumulative_new"], sep=";", skiprows=[1])

            years = data_cumulative_new["Collegejaar"].unique()
            weeks = data_cumulative_new["Weeknummer"].unique()

            data_cumulative = pd.concat([data_cumulative, data_cumulative_new], ignore_index=True)
            data_cumulative = data_cumulative.drop_duplicates(
                subset=[
                    "Collegejaar",
                    "Weeknummer",
                    "Groepeernaam Croho",
                    "Type hoger onderwijs",
                    "Herinschrijving",
                    "Hogerejaars",
                    "Herkomst",
                ],
                keep="last",
            )

            add_weeks_where_preapplicants_are_zero = AddWeeksWherePreapplicantsAreZero(
                data_cumulative, years, weeks
            )

            add_weeks_where_preapplicants_are_zero.add_weeks()

            data_cumulative = add_weeks_where_preapplicants_are_zero.data_cumulative

            data_cumulative.to_csv(paths["path_cumulative"], sep=";", index=False)
            os.remove(paths["path_cumulative_new"])

    path_latest = get_path_latest(paths, data_option)
    data_latest = (
        pd.read_excel(path_latest)
        if (path_latest != "" and os.path.exists(path_latest))
        else None
    )

    data_weighted_ensemble = (
        pd.read_excel(paths["path_ensemble_weights"])
        if (
            paths["path_ensemble_weights"] != "" and os.path.exists(paths["path_ensemble_weights"])
        )
        else None
    )

    data_student_numbers_first_years = (
        pd.read_excel(paths["path_student_count_first-years"])
        if paths["path_student_count_first-years"] != ""
        else None
    )

    if data_individual is not None:
        columns_i = configuration["columns"]["individual"]
        data_individual = data_individual.rename(
            columns={
                columns_i["Sleutel"]: "Sleutel",
                columns_i["Datum Verzoek Inschr"]: "Datum Verzoek Inschr",
                columns_i["Ingangsdatum"]: "Ingangsdatum",
                columns_i["Collegejaar"]: "Collegejaar",
                columns_i["Datum intrekking vooraanmelding"]: "Datum intrekking vooraanmelding",
                columns_i["Inschrijfstatus"]: "Inschrijfstatus",
                columns_i["Faculteit"]: "Faculteit",
                columns_i["Examentype"]: "Examentype",
                columns_i["Croho"]: "Croho",
                columns_i["Croho groepeernaam"]: "Croho groepeernaam",
                columns_i["Opleiding"]: "Opleiding",
                columns_i["Hoofdopleiding"]: "Hoofdopleiding",
                columns_i["Eerstejaars croho jaar"]: "Eerstejaars croho jaar",
                columns_i["Is eerstejaars croho opleiding"]: "Is eerstejaars croho opleiding",
                columns_i["Is hogerejaars"]: "Is hogerejaars",
                columns_i["BBC ontvangen"]: "BBC ontvangen",
                columns_i["Type vooropleiding"]: "Type vooropleiding",
                columns_i["Nationaliteit"]: "Nationaliteit",
                columns_i["EER"]: "EER",
                columns_i["Geslacht"]: "Geslacht",
                columns_i["Geverifieerd adres postcode"]: "Geverifieerd adres postcode",
                columns_i["Geverifieerd adres plaats"]: "Geverifieerd adres plaats",
                columns_i["Geverifieerd adres land"]: "Geverifieerd adres land",
                columns_i["Studieadres postcode"]: "Studieadres postcode",
                columns_i["Studieadres land"]: "Studieadres land",
                columns_i["School code eerste vooropleiding"]: "School code eerste vooropleiding",
                columns_i["School eerste vooropleiding"]: "School eerste vooropleiding",
                columns_i["Plaats code eerste vooropleiding"]: "Plaats code eerste vooropleiding",
                columns_i["Land code eerste vooropleiding"]: "Land code eerste vooropleiding",
                columns_i["Aantal studenten"]: "Aantal studenten",
            }
        )

    if data_cumulative is not None:
        columns_c = configuration["columns"]["cumulative"]
        data_cumulative = data_cumulative.rename(
            columns={
                columns_c["Korte naam instelling"]: "Korte naam instelling",
                columns_c["Collegejaar"]: "Collegejaar",
                columns_c["Weeknummer rapportage"]: "Weeknummer rapportage",
                columns_c["Weeknummer"]: "Weeknummer",
                columns_c["Faculteit"]: "Faculteit",
                columns_c["Type hoger onderwijs"]: "Type hoger onderwijs",
                columns_c["Groepeernaam Croho"]: "Groepeernaam Croho",
                columns_c["Naam Croho opleiding Nederlands"]: "Naam Croho opleiding Nederlands",
                columns_c["Croho"]: "Croho",
                columns_c["Herinschrijving"]: "Herinschrijving",
                columns_c["Hogerejaars"]: "Hogerejaars",
                columns_c["Herkomst"]: "Herkomst",
                columns_c["Gewogen vooraanmelders"]: "Gewogen vooraanmelders",
                columns_c["Ongewogen vooraanmelders"]: "Ongewogen vooraanmelders",
                columns_c[
                    "Aantal aanmelders met 1 aanmelding"
                ]: "Aantal aanmelders met 1 aanmelding",
                columns_c["Inschrijvingen"]: "Inschrijvingen",
            }
        )

    return (
        data_individual,
        data_cumulative,
        data_student_numbers_first_years,
        data_latest,
        data_distances,
        data_weighted_ensemble,
    )


def load_configuration(file_path):
    f = open(file_path)

    data = json.load(f)

    f.close()

    return data
