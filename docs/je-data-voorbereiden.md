# Je data voorbereiden

De pipeline verwacht twee typen inputmappen: `data/input_raw/` voor de ruwe bronbestanden die de ETL verwerkt, en `data/input/` voor de verwerkte bestanden die het model direct inleest.

!!! tip "Uitgebreide versie — Jupyter notebook"
    Wil je de schema's, pivot-transformatie en validatiechecks uitvoerbaar op demodata zien?
    Zie [`notebooks/01_data_voorbereiden.ipynb`](https://github.com/cedanl/studentprognose/blob/main/notebooks/01_data_voorbereiden.ipynb).

`data/input/` wordt automatisch aangemaakt door de ETL als de map nog niet bestaat. `data/input_raw/telbestanden/` wordt aangemaakt door `studentprognose init`. Je hoeft deze mappen niet zelf te maken.

Zie de [volledige dataflow](https://github.com/cedanl/studentprognose/blob/main/doc/PIPELINE.md) voor een visueel Mermaid-diagram.

!!! tip "Heb je al verwerkte data?"
    Heb je verwerkte bestanden uit een eigen ETL of van een collega, dan kun je de ingebouwde ETL overslaan met `--noetl`. **Let op:** de bestanden moeten voldoen aan het schema in [Verwerkte inputbestanden](#verwerkte-inputbestanden-data-input) — dezelfde kolomnamen, types en betekenis. Wijkende kolomnamen kun je remappen via `configuration.json` (zie [Institutiespecifieke kolomnamen](#institutiespecifieke-kolomnamen)).

## Ruwe bronbestanden (`data/input_raw/`)

### Studielink telbestanden

Map: `data/input_raw/telbestanden/`
Bestandsnaampatroon: `telbestandY{jaar}W{week}.csv` (bijv. `telbestandY2024W10.csv`) — configureerbaar, zie [Afwijkende bestandsnamen](#afwijkende-bestandsnamen) hieronder.
Scheidingsteken: `;`

| Kolom | Type | Omschrijving |
|-------|------|-------------|
| `Studiejaar` | int | Collegejaar (bijv. `2024`) |
| `Isatcode` | str | CROHO-code |
| `Groepeernaam` | str | Naam van de opleiding |
| `Aantal` | int | Aantal vooraanmelders |
| `meercode_V` | int | Aantal inschrijfverzoeken/inschrijvingen per student (status V/U/I). Wordt gebruikt als weegfactor in `Gewogen vooraanmelders = Aantal / meercode_V`. Voor status A-rijen is deze waarde 0 — die rijen worden door de ETL uitgefilterd. |
| `Status` | str | `V` (verzoek), `I` (inschrijving), `U` (uitgeschreven/gestaakt) of `A` (annulering). Rijen met `A` worden uit de vooraanmelderaggregatie gehouden. |
| `Herinschrijving` | str | `J` of `N` |
| `Herkomst` | str | `N` (Nederland), `E` (EER), `R` (rest) |

#### Afwijkende bestandsnamen

Gebruikt jouw instelling een andere conventie dan `telbestandY{jaar}W{week}.csv`? Geef in `configuration.json` één of meer eigen patronen op:

```json
{
    "telbestand_filename_patterns": [
        "telbestandY{year}W{week}",
        "VU_telbestand_{year}_W{week}"
    ]
}
```

Placeholders zijn `{year}` (vier cijfers) en `{week}` (één of twee cijfers). Andere karakters worden letterlijk gematcht — punten en streepjes zijn veilig. Meerdere patronen tegelijk zijn handig tijdens een migratie van oude naar nieuwe naamgeving. Zie [`telbestand_filename_patterns`](configuratie.md#telbestand_filename_patterns-bestandsnaampatronen-voor-telbestanden) voor de volledige uitleg.

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

### Telbestand studenten

Pad: `data/input_raw/oktober_bestand.xlsx`
Bron: door je instelling zelf aangeleverd

Dit bestand bevat de werkelijke inschrijvingen per opleiding, herkomst en collegejaar — de ground truth waarop het model wordt geëvalueerd en die het ratio-model voedt. De meeste instellingen genereren dit bestand uit hun SIS/datawarehouse (Osiris, Usis, of vergelijkbaar).

!!! note "Waarom heet het bestand `oktober_bestand.xlsx`?"
    Historische naam. De bestandsnaam en de configuratiesleutels (`path_raw_october`, `columns.oktober`) zijn ongewijzigd gelaten om bestaande installaties niet te breken. Inhoudelijk is het een **telbestand met studentaantallen**.

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
| 3 | Studentaantallen berekenen | `oktober_bestand.xlsx` (telbestand studenten) | `student_count_*.xlsx`, `student_volume.xlsx` |
| 4 | Kopiëren individuele data | `individuele_aanmelddata.csv` | `vooraanmeldingen_individueel.csv` |

## ETL overslaan

Als je al beschikt over verwerkte data — bijvoorbeeld uit een eigen ETL-pipeline, van een collega, of uit een eerdere run — kun je de ingebouwde ETL-stappen overslaan met de `--noetl`-vlag:

```bash
studentprognose --noetl -d cumulative -w 12 -y 2025
```

Zorg ervoor dat de vereiste bestanden in `data/input/` staan voordat je de tool draait. Welke bestanden je nodig hebt hangt af van de gekozen datamodus (`-d`). Zie de tabel hieronder.

## Verwerkte inputbestanden (`data/input/`)

Dit zijn de bestanden die het model direct inleest. Ze worden aangemaakt door de ETL, of je plaatst ze zelf in `data/input/` als je `--noetl` gebruikt.

| Bestand | Wanneer nodig | Bron |
|---------|---------------|------|
| `vooraanmeldingen_cumulatief.csv` | `-d c` of `-d b` | ETL stap 1+2 |
| `vooraanmeldingen_individueel.csv` | `-d i` of `-d b` | ETL stap 4 |
| `student_count_first-years.xlsx` | Altijd | ETL stap 3 |
| `student_volume.xlsx` | Alleen bij `-sy v` | ETL stap 3 |
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
