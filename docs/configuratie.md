# Configuratie

De pipeline wordt geconfigureerd via `configuration/configuration.json`. Het bestand heeft vier secties: `paths`, `model_config`, `numerus_fixus` en `columns`.

## `paths` — bestandspaden

Alle paden zijn relatief aan de werkmap waar je de pipeline uitvoert.

| Sleutel | Standaard | Omschrijving |
|---------|-----------|-------------|
| `path_cumulative` | `data/input/vooraanmeldingen_cumulatief.csv` | Cumulatieve vooraanmelddata (ETL-output) |
| `path_individual` | `data/input/vooraanmeldingen_individueel.csv` | Individuele aanmelddata (ETL-output) |
| `path_cumulative_new` | `""` | Optioneel nieuw cumulatief bestand; wordt ingemergt en daarna verwijderd — zie [waarschuwing](je-data-voorbereiden.md#bekende-valkuil-path_cumulative_new) |
| `path_latest_cumulative` | `data/input/totaal_cumulatief.xlsx` | Historische cumulatieve voorspellingen (post-processing output) |
| `path_latest_individual` | `data/input/totaal_individueel.xlsx` | Historische individuele voorspellingen (post-processing output) |
| `path_ensemble_weights` | `data/input/ensemble_weights.xlsx` | Ensemble-gewichten per opleiding/herkomst/examentype |
| `path_raw_october` | `data/input_raw/oktober_bestand.xlsx` | DUO oktober-bestand (ruwe bron) |
| `path_raw_telbestanden` | `data/input_raw/telbestanden` | Map met Studielink telbestanden |
| `path_raw_individueel` | `data/input_raw/individuele_aanmelddata.csv` | Ruwe individuele aanmelddata |
| `path_student_count_first-years` | `data/input/student_count_first-years.xlsx` | Werkelijk aantal eerstejaars (DUO) |
| `path_student_count_higher-years` | `data/input/student_count_higher-years.xlsx` | Werkelijk aantal hogerjaars (DUO) |
| `path_student_volume` | `data/input/student_volume.xlsx` | Totaal studentvolume (DUO) |
| `path_ratios` | `data/input/ratiobestand.xlsx` | Ratiobestand (optioneel) |

## `model_config` — modelparameters

### `status_mapping`

Bepaalt welke inschrijfstatussen als "ingeschreven" (1) of "niet-ingeschreven" (0) worden gelabeld voor het XGBoost-classificatiemodel.

Standaardwaarden:

| Status | Label |
|--------|-------|
| `Ingeschreven` | `1` |
| `Uitgeschreven` | `1` |
| `Geannuleerd` | `0` |
| `Verzoek tot inschrijving` | `0` |
| `Studie gestaakt` | `0` |
| `Aanmelding vervolgen` | `0` |

Pas dit aan als jouw instelling andere statuswaarden gebruikt. Onbekende statussen worden niet gemapt en veroorzaken een fout.

### `min_training_year`

Standaard: `2016`

Het vroegste collegejaar dat als trainingsdata wordt meegenomen. Data van vóór dit jaar wordt genegeerd. Verlaag dit alleen als je betrouwbare historische data hebt die verder teruggaat.

## `numerus_fixus`

Een object met opleidingsnamen als sleutels en het maximale inschrijvingsaantal als waarde:

```json
{
    "numerus_fixus": {
        "B Geneeskunde": 340,
        "B Tandheelkunde": 50
    }
}
```

Als de gesommeerde voorspelling over herkomstgroepen het maximum overschrijdt, wordt het overschot afgetrokken van de NL-herkomstgroep. Opleidingen die hier niet staan worden niet gecapped.

## `excluded_data_points` — anomaliejaren uitsluiten van trainingsdata

Optionele lijst van filterregels voor het verwijderen van bekende problematische datapunten uit de **trainingsdata**. De tool past de regels toe vóór elk model wordt getraind. Het voorspeljaar (`predict_year`) wordt **altijd** beschermd en nooit uitgesloten, ongeacht de regels.

Gebruik dit alleen voor aantoonbare datakwaliteitsproblemen: foutieve Studielink-snapshots, deadlineverschuivingen die een historische reeks onvergelijkbaar maken, of structureel afwijkende populaties. Uitsluiten om een betere modelfit te forceren is misbruik — documenteer altijd de reden.

```json
{
    "excluded_data_points": [
        {
            "year": 2024,
            "herkomst": "Niet-EER",
            "examentype": ["Bachelor", "Pre-master"]
        },
        {
            "year_before": 2024,
            "examentype": "Pre-master",
            "opleiding": "B Notarieel Recht"
        }
    ]
}
```

### Filtersleutels per regel

Alle sleutels binnen één regel worden gecombineerd met **AND**. Meerdere regels worden gecombineerd met **OR**.

| Sleutel | Type | Omschrijving |
|---------|------|-------------|
| `year` | `int` | Sluit exact dit collegejaar uit |
| `year_before` | `int` | Sluit alle jaren vóór (exclusief) deze waarde uit |
| `year_after` | `int` | Sluit alle jaren na (exclusief) deze waarde uit |
| `herkomst` | `str` of `list[str]` | Filter op herkomst (`"NL"`, `"EER"`, `"Niet-EER"`) |
| `examentype` | `str` of `list[str]` | Filter op examentype (`"Bachelor"`, `"Master"`, `"Pre-master"`) |
| `opleiding` | `str` of `list[str]` | Filter op `Croho groepeernaam` |

Een lege lijst (`[]`) schakelt uitsluiting volledig uit — dit is de standaard.

!!! warning "Gebruik dit bewust"
    Elk uitgesloten datapunt verkleint de trainingsset. Bij kleine opleidingen kan dat de modelkwaliteit ernstig verslechteren. Beperk uitsluiting tot jaren waarvan je kunt aantonen dat de data structureel fout of onvergelijkbaar is.

## `ensemble_override_cumulative` — ensemble-uitzondering per opleiding

Een lijst van opleidingsnamen (op `Croho groepeernaam`) waarvoor de ensemble-logica altijd het SARIMA-cumulatief model gebruikt, ongeacht weeknummer of examentype.

```json
{
    "ensemble_override_cumulative": [
        "B Geneeskunde",
        "B Biomedische Wetenschappen",
        "B Tandheelkunde"
    ]
}
```

Gebruik dit voor opleidingen met een numerus fixus of een sterk afwijkend aanmeldpatroon waarbij het cumulatieve SARIMA-model aantoonbaar beter presteert. Lege lijst (`[]`) schakelt de uitzondering uit voor alle opleidingen.

De waarden in de demo-configuratie zijn Radboud-specifiek. **Vervang of maak deze lijst leeg voor je eigen instelling.**

## `exclude_from_combined` — uitsluiting van combined-modus

Een lijst van opleidingsnamen (op `Croho groepeernaam`) die worden overgeslagen in de combined-modus (`-d both`). Opleidingen op deze lijst worden niet meegenomen in de combined-voorspelling.

```json
{
    "exclude_from_combined": [
        "M Educatie in de Mens- en Maatschappijwetenschappen"
    ]
}
```

Gebruik dit voor opleidingen waarvoor de combined-modus aantoonbaar slechter werkt dan het cumulatieve spoor alleen. Lege lijst schakelt de uitsluiting uit.

De waarde in de demo-configuratie is Radboud-specifiek. **Vervang of maak deze lijst leeg voor je eigen instelling.**

## `ensemble_weights` — weging SARIMA-individueel vs. SARIMA-cumulatief

Bepaalt per weekperiode hoe zwaar het individuele en het cumulatieve SARIMA-model meewegen in de ensemble-voorspelling. Elke sleutel is een situatie; de waarde is een object met `individual` en `cumulative` (moeten optellen tot 1.0).

```json
{
    "ensemble_weights": {
        "master_week_17_23": {"individual": 0.2, "cumulative": 0.8},
        "week_30_34":        {"individual": 0.6, "cumulative": 0.4},
        "week_35_37":        {"individual": 0.7, "cumulative": 0.3},
        "default":           {"individual": 0.5, "cumulative": 0.5}
    }
}
```

| Sleutel | Van toepassing wanneer |
|---------|----------------------|
| `master_week_17_23` | Examentype = Master én weeknummer 17–23 |
| `week_30_34` | Weeknummer 30–34 |
| `week_35_37` | Weeknummer 35–37 |
| `default` | Alle overige gevallen |

Week 38 is altijd 100% individueel en wordt niet door deze instelling beïnvloed. Zie [Ensemble](methodologie/ensemble.md) voor achtergrond bij de keuze van gewichten.

!!! note "Weekgrenzen zijn niet configureerbaar"
    De sleutelnamen (zoals `week_30_34`) beschrijven de weekperiode waarop een gewicht van toepassing is, maar de weekgrenzen zelf zijn in de code vastgelegd. Alleen de **gewichten** zijn instelbaar via dit blok. Wil je de weekgrenzen aanpassen, dan vereist dat een codewijziging in `output/postprocessor.py`.

## `columns` — kolomnamen mapping

Maakt het mogelijk dat instellingen andere kolomnamen gebruiken dan de kanonieke namen die de pipeline intern hanteert. Specificeer alleen de namen die afwijken — ontbrekende sleutels vallen terug op de kanonieke naam.

### `individual` — individuele aanmelddata

Volledige lijst van kanonieke kolomnamen die gemapped kunnen worden:

`Sleutel`, `Datum Verzoek Inschr`, `Ingangsdatum`, `Collegejaar`, `Datum intrekking vooraanmelding`, `Inschrijfstatus`, `Faculteit`, `Examentype`, `Croho`, `Croho groepeernaam`, `Opleiding`, `Hoofdopleiding`, `Eerstejaars croho jaar`, `Is eerstejaars croho opleiding`, `Is hogerejaars`, `BBC ontvangen`, `Type vooropleiding`, `Nationaliteit`, `EER`, `Geslacht`, `Geverifieerd adres postcode`, `Geverifieerd adres plaats`, `Geverifieerd adres land`, `Studieadres postcode`, `Studieadres land`, `School code eerste vooropleiding`, `School eerste vooropleiding`, `Plaats code eerste vooropleiding`, `Land code eerste vooropleiding`, `Aantal studenten`

### `oktober` — DUO oktober-bestand

`Collegejaar`, `Groepeernaam Croho`, `Aantal eerstejaars croho`, `EER-NL-nietEER`, `Examentype code`, `Aantal Hoofdinschrijvingen`

### `cumulative` — Studielink cumulatieve data

`Korte naam instelling`, `Collegejaar`, `Weeknummer rapportage`, `Weeknummer`, `Faculteit`, `Type hoger onderwijs`, `Groepeernaam Croho`, `Naam Croho opleiding Nederlands`, `Croho`, `Herinschrijving`, `Hogerejaars`, `Herkomst`, `Gewogen vooraanmelders`, `Ongewogen vooraanmelders`, `Aantal aanmelders met 1 aanmelding`, `Inschrijvingen`

## `validation` — validatiedrempels overschrijven

Optionele sectie. Hoeft niet in je configuratiebestand te staan. Voeg alleen toe wat afwijkt van de standaard:

```json
{
    "validation": {
        "nan_error_threshold": 0.20,
        "telbestand": {
            "herkomst_allowed": ["N", "E", "R", "ONBEKEND"]
        }
    }
}
```

| Sleutel | Standaard | Omschrijving |
|---------|-----------|-------------|
| `nan_warning_threshold` | `0.05` | Fractie ontbrekende waarden die een waarschuwing geeft |
| `nan_error_threshold` | `0.30` | Fractie ontbrekende waarden die een fout geeft |
| `collegejaar_min_offset` | `15` | Collegejaren ouder dan `huidig jaar - 15` geven een fout |
| `collegejaar_max_offset` | `2` | Collegejaren na `huidig jaar + 2` geven een fout |
| `telbestand.herkomst_allowed` | `["N","E","R"]` | Toegestane herkomstwaarden in telbestanden |

Zie [Validatie](validatie.md) voor een volledig overzicht van alle controles.

## Voorbeeld minimale configuratie

```json
{
    "paths": {
        "path_cumulative": "data/input/vooraanmeldingen_cumulatief.csv",
        "path_individual": "data/input/vooraanmeldingen_individueel.csv",
        "path_cumulative_new": "",
        "path_latest_individual": "data/input/totaal_individueel.xlsx",
        "path_latest_cumulative": "data/input/totaal_cumulatief.xlsx",
        "path_ensemble_weights": "data/input/ensemble_weights.xlsx",
        "path_raw_october": "data/input_raw/oktober_bestand.xlsx",
        "path_student_count_first-years": "data/input/student_count_first-years.xlsx",
        "path_student_count_higher-years": "data/input/student_count_higher-years.xlsx",
        "path_student_volume": "data/input/student_volume.xlsx",
        "path_ratios": "data/input/ratiobestand.xlsx",
        "path_raw_telbestanden": "data/input_raw/telbestanden",
        "path_raw_individueel": "data/input_raw/individuele_aanmelddata.csv"
    },
    "model_config": {
        "status_mapping": {
            "Ingeschreven": 1,
            "Geannuleerd": 0,
            "Uitgeschreven": 1,
            "Verzoek tot inschrijving": 0,
            "Studie gestaakt": 0,
            "Aanmelding vervolgen": 0
        },
        "min_training_year": 2016
    },
    "numerus_fixus": {},
    "columns": {
        "individual": {},
        "oktober": {},
        "cumulative": {}
    }
}
```
