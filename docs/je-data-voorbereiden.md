# Je data voorbereiden

De pipeline verwacht twee typen inputmappen: `data/input_raw/` voor de ruwe bronbestanden die de ETL verwerkt, en `data/input/` voor de verwerkte bestanden die het model direct inleest.

Zie de [volledige dataflow](https://github.com/cedanl/studentprognose/blob/main/doc/PIPELINE.md) voor een visueel Mermaid-diagram.

## Ruwe bronbestanden (`data/input_raw/`)

### Studielink telbestanden

Map: `data/input_raw/telbestanden/`
Bestandsnaampatroon: `telbestandY{jaar}W{week}.csv` (bijv. `telbestandY2024W10.csv`)
Scheidingsteken: `;`

| Kolom | Type | Omschrijving |
|-------|------|-------------|
| `Studiejaar` | int | Collegejaar (bijv. `2024`) |
| `Isatcode` | str | CROHO-code |
| `Groepeernaam` | str | Naam van de opleiding |
| `Aantal` | int | Aantal vooraanmelders |
| `meercode_V` | int | Weegfactor; mag niet 0 zijn (leidt tot deling door nul in ETL) |
| `Herinschrijving` | str | `J` of `N` |
| `Hogerejaars` | str | `J` of `N` |
| `Herkomst` | str | `N` (Nederland), `E` (EER), `R` (rest) |

### Individuele aanmelddata

Pad: `data/input_raw/individuele_aanmelddata.csv`
Bron: Osiris / Usis
Scheidingsteken: `;`

Verplichte kolommen (kanonieke namen — pas aan via `configuration.json` als jouw instelling andere namen gebruikt):

| Canonieke naam | Omschrijving |
|----------------|-------------|
| `Sleutel` | Unieke studentidentifier |
| `Datum Verzoek Inschr` | Datum van vooraanmelding |
| `Ingangsdatum` | Ingangsdatum inschrijving |
| `Collegejaar` | Collegejaar |
| `Datum intrekking vooraanmelding` | Weeknummer van intrekking (leeg als niet ingetrokken) |
| `Inschrijfstatus` | Zie `status_mapping` in configuratie |
| `Faculteit` | Faculteit |
| `Examentype` | `Bachelor` of `Master` |
| `Croho` | CROHO-code |
| `Croho groepeernaam` | Naam van de opleiding |
| `Nationaliteit` | Nationaliteit student |
| `EER` | EER-indicator |
| `Geslacht` | Geslacht |
| `Type vooropleiding` | Type vooropleiding |

### Oktober-bestand (DUO 1-cijfer HO)

Pad: `data/input_raw/oktober_bestand.xlsx`
Bron: DUO

| Canonieke naam | Omschrijving |
|----------------|-------------|
| `Collegejaar` | Collegejaar |
| `Groepeernaam Croho` | Naam van de opleiding |
| `Aantal eerstejaars croho` | Aantal eerstejaars |
| `EER-NL-nietEER` | Herkomstgroep |
| `Examentype code` | `B` (Bachelor) of `M` (Master) |
| `Aantal Hoofdinschrijvingen` | Aantal hoofdinschrijvingen |

## ETL-stappen

De ETL draait automatisch bij elke run (tenzij `--noetl` is opgegeven). De stappen:

| Stap | Actie | Input | Output |
|------|-------|-------|--------|
| 1 | Rowbind + reformat | `telbestanden/*.csv` | Samengevoegd cumulatief bestand |
| 2 | Interpolatie ontbrekende weken | Samengevoegd bestand | `vooraanmeldingen_cumulatief.csv` |
| 3 | Studentaantallen berekenen | `oktober_bestand.xlsx` | `student_count_*.xlsx`, `student_volume.xlsx` |
| 4 | Kopiëren individuele data | `individuele_aanmelddata.csv` | `vooraanmeldingen_individueel.csv` |

## Verwerkte inputbestanden (`data/input/`)

Dit zijn de bestanden die het model direct inleest. Ze worden aangemaakt door de ETL of handmatig aangeleverd als je `--noetl` gebruikt.

| Bestand | Wanneer nodig | Bron |
|---------|---------------|------|
| `vooraanmeldingen_cumulatief.csv` | `-d c` of `-d b` | ETL stap 1+2 |
| `vooraanmeldingen_individueel.csv` | `-d i` of `-d b` | ETL stap 4 |
| `student_count_first-years.xlsx` | Altijd | ETL stap 3 |
| `student_count_higher-years.xlsx` | Altijd | ETL stap 3 |
| `student_volume.xlsx` | Altijd | ETL stap 3 |
| `ensemble_weights.xlsx` | Optioneel | `archive/calculate_ensemble_weights.py` |
| `totaal_cumulatief.xlsx` | Optioneel | `archive/append_studentcount_and_compute_errors.py` |
| `totaal_individueel.xlsx` | Optioneel | `archive/append_studentcount_and_compute_errors.py` |

## Institutiespecifieke kolomnamen

Jouw instelling gebruikt mogelijk andere kolomnamen dan de kanonieke namen hierboven. Pas de `columns`-sectie in `configuration.json` aan:

```json
{
    "columns": {
        "individual": {
            "Croho groepeernaam": "CROHO_NAAM",
            "Inschrijfstatus": "STATUS"
        }
    }
}
```

Alleen de afwijkende namen hoeven opgegeven te worden. Zie [Configuratie](configuratie.md) voor de volledige kolommenlijst.

## Bekende valkuil: `path_cumulative_new`

!!! warning "path_cumulative_new overschrijft en verwijdert bestanden"
    Als `path_cumulative_new` in je configuratie een bestaand bestand aanwijst, **mergt de
    pipeline dit bestand in `vooraanmeldingen_cumulatief.csv` en verwijdert daarna het bronbestand**.
    De pipeline toont daarvoor een melding:

    ```
    Merging data/input/nieuw_bestand.csv into data/input/vooraanmeldingen_cumulatief.csv and removing source file...
    ```

    Bij een crash ná het schrijven maar vóór het verwijderen is de originele data onherstelbaar kwijt.
    Maak vooraf een backup als je dit mechanisme gebruikt.

    Gebruik dit mechanisme alleen als je bewust een nieuw cumulatief bestand wilt inladen.
    Zet `path_cumulative_new` op `""` als je dit niet nodig hebt.
