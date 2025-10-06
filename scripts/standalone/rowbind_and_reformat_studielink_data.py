import os
import pandas as pd

directory = "PATH_TO_FOLDER_WITH_STUDIELINK_DATA"

if __name__ == "__main__":
    dataframes = []

    # Loop over alle bestanden in de directory en voeg ze samen in een dataframe
    for file in os.listdir(directory):
        if file != "rowbinded.csv":
            """
            Het weeknummer van de data staat enkel in de bestandsnaam en niet in de data zelf. Dit 
            is een workaround om het weeknummer aan de data toe te voegen. Dit
            gedeelte zal aangepast moeten worden indien het het format van de bestandsnaam anders
            is dan "telbestandY2024WXX.csv".
            """
            weeknummer = file.replace("telbestandY2024W", "").replace(".csv", "")

            data = pd.read_csv(directory + "/" + file, sep=";", low_memory=False)
            data["Weeknummer"] = weeknummer

            dataframes.append(data)

    data = pd.concat(dataframes)

    # Gewogen vooraanmelders berekenen
    data["Gewogen vooraanmelders"] = data["Aantal"] / data["meercode_V"]

    # Kolommen hernoemen
    data.rename(
        columns={
            "Brincode": "Korte naam instelling",
            "Studiejaar": "Collegejaar",
            "Type_HO": "Type hoger onderwijs",
            "Isatcode": "Croho",
            "Aantal": "Ongewogen vooraanmelders",
        },
        inplace=True,
    )

    # Kolommen verwijderen
    data = data[
        [
            "Korte naam instelling",
            "Croho",
            "Type hoger onderwijs",
            "Collegejaar",
            "Herkomst",
            "Hogerejaars",
            "Herinschrijving",
            "Ongewogen vooraanmelders",
            "Weeknummer",
            "Gewogen vooraanmelders",
        ]
    ]

    # Kolommen toevoegen en vullen met None
    data[
        [
            "Weeknummer rapportage",
            "Faculteit",
            "Groepeernaam Croho",
            "Naam Croho opleiding Nederlands",
            "Aantal aanmelders met 1 aanmelding",
            "Inschrijvingen",
        ]
    ] = None

    # Volgorde van kolommen aanpassen
    data = data[
        [
            "Korte naam instelling",
            "Collegejaar",
            "Weeknummer rapportage",
            "Weeknummer",
            "Faculteit",
            "Type hoger onderwijs",
            "Groepeernaam Croho",
            "Naam Croho opleiding Nederlands",
            "Croho",
            "Herinschrijving",
            "Hogerejaars",
            "Herkomst",
            "Gewogen vooraanmelders",
            "Ongewogen vooraanmelders",
            "Aantal aanmelders met 1 aanmelding",
            "Inschrijvingen",
        ]
    ]

    # Kolommen vullen met waarden van andere kolommen
    data["Weeknummer rapportage"] = data["Weeknummer"]
    data["Faculteit"] = "FACULTEIT"
    data["Groepeernaam Croho"] = data["Croho"]
    data["Naam Croho opleiding Nederlands"] = data["Croho"]

    # Examentype, herinschrijving, hogerejaars en herkomst omzetten naar waarden die we in het
    # script hanteren
    data["Type hoger onderwijs"] = data["Type hoger onderwijs"].replace(
        {"P": "Bachelor", "B": "Bachelor", "M": "Master"}
    )
    data["Herinschrijving"] = data["Herinschrijving"].replace({"J": "Ja", "N": "Nee"})
    data["Hogerejaars"] = data["Hogerejaars"].replace({"J": "Ja", "N": "Nee"})
    data["Herkomst"] = data["Herkomst"].replace({"N": "NL", "E": "EER", "R": "Niet-EER"})

    data.to_csv(
        "OUTPUT_PATH/rowbinded.csv",
        sep=";",
        index=False,
    )
