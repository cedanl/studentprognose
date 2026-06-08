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

!!! info "Officiële Studielink-specificatie"
    Studielink levert de telbestanden volgens het *Programma van Levering Telbestand Studielink* (Stichting Studielink, versie Definitief 2.8, 3 november 2022). Dat document beschrijft alle velden, codetabellen en leveringskalender. PDF: [pvl telbestand studielink.pdf](https://www.tignl.eu/downloads/studielink/pvl%20telbestand%20studielink.pdf).

    De §-verwijzingen in de tabel hieronder verwijzen naar paragrafen in dit document. De pipeline gebruikt een **subset** van de PvL-velden (zie [#206](https://github.com/cedanl/studentprognose/issues/206) voor de status van demo-data t.o.v. de volledige PvL).

De velden hieronder zijn de kolommen die de demo-telbestanden bevatten en die de ETL daadwerkelijk inleest:

| Kolom | Type | PvL § | Omschrijving |
|-------|------|-------|-------------|
| `Brincode` | str | §5.2 | BRIN-code van de instelling. ETL hernoemt naar `Korte naam instelling`. |
| `Studiejaar` | int | §5.7 | Collegejaar (bijv. `2024`). ETL hernoemt naar `Collegejaar`. |
| `Type_HO` | str | §5.5 | Type hoger onderwijs (`B` = Bachelor, `M` = Master, etc.). ETL hernoemt naar `Type hoger onderwijs`. |
| `Isatcode` | str | §5.4 | CROHO-code van de opleiding. ETL hernoemt naar `Croho`. |
| `Groepeernaam` | str | — | Naam van de opleiding (afgeleid/verrijkt veld, niet in PvL). ETL kopieert naar `Groepeernaam Croho` én `Naam Croho opleiding Nederlands`. |
| `Faculteit` | str | — | Faculteit van de opleiding (afgeleid/verrijkt veld, niet in PvL). Wordt door de ETL leeg gezet als de kolom ontbreekt. |
| `Herkomst` | str | §5.10 | `N` (Nederland), `E` (EER), `R` (rest). ETL hertaalt naar `NL` / `EER` / `Niet-EER`. |
| `Hogerejaars` | str | §5.14 | `J` of `N`. ETL hertaalt naar `Ja` / `Nee`. |
| `Herinschrijving` | str | §5.15 | `J` of `N`. ETL hertaalt naar `Ja` / `Nee`. |
| `Aantal` | int | §5.17 | Aantal (voor)aanmeldingen. ETL hernoemt naar `Ongewogen vooraanmelders`. |
| `meercode_V` | int | §5.12 | Gemiddeld aantal aanmeldingen per student met status V/U/I. Wordt door de ETL als deler gebruikt: `Gewogen vooraanmelders = Aantal / meercode_V`. Voor status A-rijen is deze waarde 0 — die rijen worden door de ETL uitgefilterd. |
| `Status` | str | §5.13 | `V` (verzoek), `I` (inschrijving), `U` (uitgeschreven/gestaakt) of `A` (annulering). Rijen met `A` worden uit de vooraanmelderaggregatie gehouden. |

#### Meerdere instellingen in de teldata

De volledige Studielink-teldata bevat rijen van **alle instellingen**, elk herkenbaar aan de `Brincode` (ná de ETL: `Korte naam instelling`). Wil je maar één instelling — meestal je eigen — modelleren, zet dan `institution_filter` in je configuratie. Laat je dat leeg terwijl er meerdere instellingen in de data zitten, dan worden hun cijfers per opleiding bij elkaar opgeteld; de pipeline waarschuwt hiervoor. Zie [`institution_filter`](configuratie.md#institution_filter-filteren-op-instelling).

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
Bron: Osiris / Usis (Studenten Informatie Systeem of datawarehouse van jouw instelling)
Scheidingsteken: `;`

Volledige kolommenlijst (kanonieke namen — pas aan via `configuration.json` als jouw instelling andere namen gebruikt). De *Rol* geeft aan hoe de pipeline het veld gebruikt: **sleutel** = unieke identifier per aanmelding; **basis** = gebruikt door alle individuele stappen (filtering, deadlinebepaling, output-aggregatie); **classifier** = feature in het XGBoost classifier-model voor inschrijfkans; **info** = wel ingelezen, niet door modellen gebruikt.

| Canonieke naam | Rol | Omschrijving |
|----------------|------|-------------|
| `Sleutel` | sleutel | Unieke studentidentifier |
| `Datum Verzoek Inschr` | basis | Datum van vooraanmelding (bepaalt aanmeldweek) |
| `Ingangsdatum` | basis | Ingangsdatum inschrijving |
| `Collegejaar` | basis, classifier | Collegejaar |
| `Datum intrekking vooraanmelding` | basis | Weeknummer van intrekking (leeg als niet ingetrokken) |
| `Inschrijfstatus` | basis | Zie `status_mapping` in configuratie |
| `Faculteit` | classifier | Faculteit |
| `Examentype` | basis, classifier | `Bachelor` of `Master` |
| `Croho` | basis | CROHO-code |
| `Croho groepeernaam` | basis, classifier | Naam van de opleiding |
| `Opleiding` | classifier | Specifieke opleidingsnaam (variant binnen CROHO-groep) |
| `Hoofdopleiding` | info | Markering hoofdopleiding bij duale studies |
| `Eerstejaars croho jaar` | basis | Jaar waarin student eerstejaars is in deze CROHO |
| `Is eerstejaars croho opleiding` | basis | Indicator eerstejaars binnen deze CROHO-opleiding |
| `Is hogerejaars` | basis | Indicator of student hogerejaars is (`Ja` / `Nee`) |
| `BBC ontvangen` | info | Indicator BBC (Bewijs van Betaald Collegegeld) ontvangen |
| `Type vooropleiding` | classifier | Type vooropleiding |
| `Nationaliteit` | classifier | Nationaliteit student |
| `EER` | classifier | EER-indicator |
| `Geslacht` | classifier | Geslacht |
| `Geverifieerd adres postcode` | classifier | Geverifieerde postcode woonadres |
| `Geverifieerd adres plaats` | classifier | Geverifieerde plaats woonadres |
| `Geverifieerd adres land` | classifier | Geverifieerd land woonadres |
| `Studieadres postcode` | classifier | Postcode studieadres |
| `Studieadres land` | classifier | Land studieadres |
| `School code eerste vooropleiding` | classifier | School-code eerste vooropleiding |
| `School eerste vooropleiding` | classifier | Naam school eerste vooropleiding |
| `Plaats code eerste vooropleiding` | classifier | Plaats-code eerste vooropleiding |
| `Land code eerste vooropleiding` | classifier | Land-code eerste vooropleiding |
| `Aantal studenten` | basis | Tellerveld (meestal `1` per rij) |

De classifier-features (`numeric`/`categorical`) zijn instelbaar in `configuration.json` onder `model_features.classifier` — zie [Configuratie](configuratie.md#columns-kolomnamen-mapping). Velden met rol *info* mogen leeg of ontbrekend zijn zonder dat het model breekt.

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
