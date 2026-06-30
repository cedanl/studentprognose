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

### UvA SQL-telbestand (fijnmazig Studielink-formaat)

Map: `data/input_raw/telbestanden/`
Bestandsnaampatroon: `telbestand_sl_{leverdatum}_v{volgnummer}_{studiejaar}.csv` (bijv. `telbestand_sl_20260525_v34_2026.csv`)
Scheidingsteken: `,` (komma — let op: het legacy-instellingsformaat hierboven gebruikt `;`)

De UvA levert de wekelijkse Studielink-telbestanden aan als **SQL-export met 22 kolommen**. Dit is een fijnmaziger, minder voorbewerkte variant van hetzelfde bronmateriaal: het bevat extra dimensies (`Geslacht`, `Opl_vorm`, `Voertaal`, `Fixus`, `1cHO_L`/`1cHO_K`) en mist de verrijkte velden `Groepeernaam` en `Faculteit`. De ETL zet dit formaat om naar **exact dezelfde 16 kolommen** als het legacy-formaat, zodat de modellen ongewijzigd blijven.

De omzetting is volledig **config-gedreven** via het `cumulative_input`-blok in `configuration.json` (separator, waardevertalingen, aggregatie); zie [`cumulative_input`](configuratie.md#cumulative_input-uva-sql-telbestand-omzetten).

| Kolom | Type | Omschrijving |
|-------|------|-------------|
| `Brincode` | str | BRIN-code instelling → `Korte naam instelling`. Het bestand is **multi-instelling**: één levering bevat alle instellingen waar een student zich aanmeldde. |
| `Isatcode` | str | CROHO-code opleiding → `Croho`. Wordt óók als placeholder voor `Groepeernaam Croho`/`Naam Croho opleiding Nederlands` gebruikt (zie hieronder). |
| `Type_HO` | str | `B`/`P` → Bachelor, `M` → Master, **`A` → Associate degree** (nieuw t.o.v. de 2016–2024 historie), `O` → Onbekend. |
| `Studiejaar` | int | Collegejaar → `Collegejaar`. |
| `Herkomst` | str | `N`/`E`/`R`(/`O`) → `NL`/`EER`/`Niet-EER`/`Onbekend`. |
| `Hogerejaars`, `Herinschrijving` | str | `J`/`N` → `Ja`/`Nee`. |
| `Status` | str | `V`/`I`/`U`/`A`. Rijen met `A` (annulering) worden uitgefilterd; voor die rijen is `meercode_V == 0`. |
| `meercode_V` | int | Deler voor `Gewogen vooraanmelders = Aantal / meercode_V`. Na het `Status != "A"`-filter komt geen `meercode_V == 0` meer voor (deel-door-nul kan niet optreden). |
| `Aantal` | int | (Voor)aanmeldingen → `Ongewogen vooraanmelders`. |
| `etl_is_deleted` | int | Soft-delete vlag; rijen met `1` worden uitgefilterd. |
| `Voertaal`, `Geslacht`, `Opl_vorm`, `Fixus`, `1cHO_L`, `1cHO_K` | — | Extra split-dimensies. Staan niet in de 16-koloms output; de ETL **aggregeert** ze weg (zie ETL-stappen). |

!!! info "Snapshot-semantiek (cumulatief, geen delta)"
    Elke wekelijkse levering is een **volledige cumulatieve momentopname**: `Aantal` is een lopend totaal vanaf de openstelling, geen wekelijkse aanwas. Per seizoen stijgt het monotoon en het reset rond de jaarwisseling van de Studielink-cyclus. **Tel daarom nooit over meerdere weekleveringen heen op** — dat zou overtellen.

!!! warning "Weeknummer uit de leverdatum, niet uit de data"
    De week zit niet in de SQL-kolommen. De ETL leidt het weeknummer af uit de **leverdatum** in de bestandsnaam (`telbestand_sl_20260525_...` → ISO-kalenderweek **22**), niet uit het Studielink-volgnummer. Zo sluit het aan op de ISO-week-indexering van het bestaande `vooraanmeldingen_cumulatief.csv`.

!!! note "Ontbrekende `Groepeernaam` en `Faculteit` (placeholder)"
    De SQL-levering bevat geen leesbare opleidingsnaam of faculteit en die zijn er niet uit af te leiden. De ETL vult `Groepeernaam Croho`/`Naam Croho opleiding Nederlands` met de **`Isatcode`** en `Faculteit` met een vaste placeholder (`"Onbekend"`). Een niet-lege placeholder is noodzakelijk: een lege (NaN) `Faculteit` zou in de cumulatieve `groupby`/pivot alle UvA-rijen laten vallen.

    **De join met het label is opgelost door op de `Isatcode` te koppelen i.p.v. op de leesbare naam.** Omdat de `Isatcode` een landelijke, stabiele CROHO-code is (en niet instellingsspecifiek zoals een opleidingsnaam), keyt zowel de feature-kant (`vooraanmeldingen_cumulatief.csv`) als het label (`student_count_first-years.xlsx`, afgeleid uit `oktober_bestand.xlsx`) op deze code. Daarvoor moet `oktober_bestand.xlsx` een `Isatcode`-kolom bevatten (zie [Telbestand studenten](#telbestand-studenten)). Een leesbare opleidingsnaam blijft wenselijk voor de dashboard-weergave; die mapping is belegd in [#232](https://github.com/cedanl/studentprognose/issues/232).

!!! note "Andere academische-jaargrens (week 36)"
    De UvA-aanmeldfase eindigt structureel rond **week 36**; er zijn geen leveringen in de weken 37–39. Het legacy-formaat loopt tot week 38. Daarom staat voor het UvA-formaat `model_config.final_academic_week` op `36` (zie [configuratie](configuratie.md#final_academic_week)). Dit raakt het cumulatieve spoor; het individuele spoor is hier niet op van toepassing (UvA is cumulatief-only).

    **Bekende beperking:** het Plotly-dashboard ordent de week-as nog volgens de vaste week 39→38-kalender. De voorspellingen en outputbestanden kloppen, maar bij een UvA-configuratie kan de week-as van het dashboard cosmetisch afwijken (weken 37/38 leeg aan het eind). Het meeschalen van het dashboard met `final_academic_week` is een aparte follow-up.

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
| `Isatcode` | **CROHO-code van de opleiding — de joinsleutel met de features.** Moet dezelfde code zijn als de `Isatcode` in de telbestanden. |
| `Groepeernaam Croho` | Naam van de opleiding (optioneel, voor leesbaarheid; niet langer de joinsleutel) |
| `Aantal eerstejaars croho` | Aantal eerstejaars |
| `EER-NL-nietEER` | Herkomstgroep |
| `Examentype code` | `B` (Bachelor) of `M` (Master) |
| `Aantal Hoofdinschrijvingen` | Aantal hoofdinschrijvingen |

!!! warning "`Isatcode` is verplicht en moet matchen met de telbestanden"
    De studentaantallen worden op de **`Isatcode`** gekoppeld aan de vooraanmeldingen, niet op de opleidingsnaam (die per instelling verschilt). Zorg dat de codes in `oktober_bestand.xlsx` exact overeenkomen met die in je telbestanden, anders blijft de koppeling leeg en traint het cumulatieve model zonder labels.

## ETL-stappen

De ETL draait automatisch bij elke run (tenzij `--noetl` is opgegeven). De stappen:

| Stap | Actie | Input | Output |
|------|-------|-------|--------|
| 1 | Rowbind + reformat (config-gedreven via [`cumulative_input`](configuratie.md#cumulative_input-uva-sql-telbestand-omzetten): separator, waardevertalingen, aggregatie naar de grain) | `path_raw_telbestanden/*.csv` (`telbestanden/`) | Samengevoegd cumulatief bestand |
| 2 | Interpolatie ontbrekende weken | Samengevoegd bestand | `vooraanmeldingen_cumulatief.csv` |
| 3 | Studentaantallen berekenen | `oktober_bestand.xlsx` (telbestand studenten) | `student_count_*.xlsx`, `student_volume.xlsx` |
| 4 | Kopiëren individuele data | `individuele_aanmelddata.csv` | `vooraanmeldingen_individueel.csv` |

## ETL overslaan

Als je al beschikt over verwerkte data — bijvoorbeeld uit een eigen ETL-pipeline, van een collega, of uit een eerdere run — kun je de ingebouwde ETL-stappen overslaan met de `--noetl`-vlag:

```bash
studentprognose --noetl -d cumulative -w 12 -y 2025
```

Zorg ervoor dat de vereiste bestanden in `data/input/` staan voordat je de tool draait. Welke bestanden je nodig hebt hangt af van de gekozen datamodus (`-d`). Zie de tabel hieronder.

!!! warning "De cumulatieve data heeft historische collegejaren nodig"
    Het cumulatieve spoor traint op data van vóór het voorspeljaar: het XGBoost-instroommodel (`SARIMA_cumulative`) en het ratio-model (`Prognose_ratio`) leiden beide hun voorspelling af uit eerdere collegejaren. Bevat je cumulatieve data (`vooraanmeldingen_cumulatief.csv`, of `df_cum` bij in-memory gebruik via [`run_pipeline_from_dataframes`](api/index.md)) **alleen het voorspeljaar**, dan blijven beide kolommen voor elke opleiding `NaN` — terwijl `Voorspelde vooraanmelders` zich wél vult en de output dus compleet *oogt*.

    Lever daarom altijd minstens één, idealiter de drie, collegejaren vóór het voorspeljaar mee. De [trainingshistorie-check](validatie.md#trainingshistorie-waarom-deze-check-bestaat) stopt de pipeline (of waarschuwt, op het in-memory pad) wanneer die historie ontbreekt. Snelle controle: `sorted(df_cum["Collegejaar"].unique())`.

## Verwerkte inputbestanden (`data/input/`)

Dit zijn de bestanden die het model direct inleest. Ze worden aangemaakt door de ETL, of je plaatst ze zelf in `data/input/` als je `--noetl` gebruikt.

| Bestand | Wanneer nodig | Bron |
|---------|---------------|------|
| `vooraanmeldingen_cumulatief.csv` | `-d c` of `-d b` | ETL stap 1+2 |
| `vooraanmeldingen_individueel.csv` | `-d i` of `-d b` | ETL stap 4 |
| `student_count_first-years.xlsx` | Altijd | ETL stap 3 |
| `student_volume.xlsx` | Alleen bij `-sy v` | ETL stap 3 |
| `ensemble_weights.xlsx` | Optioneel | `scripts/calculate_ensemble_weights.py` |
| `totaal_cumulatief.xlsx` | Optioneel | `scripts/append_studentcount_and_compute_errors.py` |
| `totaal_individueel.xlsx` | Optioneel | `scripts/append_studentcount_and_compute_errors.py` |

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
