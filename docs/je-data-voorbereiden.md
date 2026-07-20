# Je data klaarzetten

Je zet je **ruwe** bestanden in `data/input_raw/`. De tool verwerkt die automatisch (ETL) naar de modelklare bestanden in `data/input/` — die tweede map raak je zelf niet aan. Beide mappen worden voor je aangemaakt door `studentprognose init` en de ETL.

!!! tip "Minimaal nodig voor je eerste eigen run"
    - [ ] **Studielink-telbestanden** in `data/input_raw/telbestanden/` — voor het cumulatieve spoor (`-d c`/`-d b`)
    - [ ] **`individuele_aanmelddata.csv`** in `data/input_raw/` — voor het individuele spoor (`-d i`/`-d b`)
    - [ ] **`oktober_bestand.xlsx`** in `data/input_raw/` — altijd nodig (studentaantallen / ground truth)

    Alleen even uitproberen? Gebruik de [demodata via de Snelstart](snelstart.md) — dan heb je hier nog niks voor nodig.

!!! note "Twee begrippen vooraf"
    - **ETL** — de automatische stap die je ruwe bestanden omzet naar het modelklare formaat. Sla hem over met `--noetl` als je data al verwerkt is.
    - **Gewogen vs. ongewogen vooraanmelders** — *ongewogen* is het ruwe `Aantal`; *gewogen* corrigeert voor studenten die zich bij meerdere opleidingen aanmelden (`Gewogen = Aantal / meercode_V`).

    De volledige woordenlijst staat op [Begrippen](begrippen.md). Voor de uitvoerbare versie met schema's en validatiechecks: [`notebooks/01_data_voorbereiden.ipynb`](https://github.com/cedanl/studentprognose/blob/main/notebooks/01_data_voorbereiden.ipynb).

## Ruwe bronbestanden (`data/input_raw/`)

### Studielink telbestanden

Map: `data/input_raw/telbestanden/` · Bestandsnaam: `telbestandY{jaar}W{week}.csv` (configureerbaar, zie [Afwijkende bestandsnamen](#afwijkende-bestandsnamen)) · Scheidingsteken: `;`

De ETL leest een **subset** van de officiële Studielink-velden. De belangrijkste zijn `Brincode` (instelling), `Isatcode` (CROHO-code, de joinsleutel), `Studiejaar` (collegejaar), `Aantal` (ongewogen vooraanmelders) en `meercode_V` (deler voor de gewogen vooraanmelders).

!!! info "Officiële Studielink-specificatie"
    Studielink levert de telbestanden volgens het *Programma van Levering Telbestand Studielink* (versie 2.8, 3 nov. 2022): [pvl telbestand studielink.pdf](https://www.tignl.eu/downloads/studielink/pvl%20telbestand%20studielink.pdf). De §-verwijzingen in de tabel hieronder verwijzen daarnaar. De pipeline gebruikt een subset van de PvL-velden (status: [#206](https://github.com/cedanl/studentprognose/issues/206)).

??? note "Volledige veldbeschrijving (referentie)"
    | Kolom | Type | PvL § | Omschrijving |
    |-------|------|-------|-------------|
    | `Brincode` | str | §5.2 | BRIN-code van de instelling. ETL hernoemt naar `Korte naam instelling`. |
    | `Studiejaar` | int | §5.7 | Collegejaar (bijv. `2024`). ETL hernoemt naar `Collegejaar`. |
    | `Type_HO` | str | §5.5 | Type hoger onderwijs (`B` = Bachelor, `M` = Master, etc.). ETL hernoemt naar `Type hoger onderwijs`. |
    | `Isatcode` | str | §5.4 | CROHO-code van de opleiding. ETL hernoemt naar `Croho`. |
    | `Groepeernaam` | str | — | Naam van de opleiding (afgeleid/verrijkt veld, niet in PvL). ETL kopieert naar `Groepeernaam Croho` én `Naam Croho opleiding Nederlands`. |
    | `Faculteit` | str | — | Faculteit (afgeleid/verrijkt veld, niet in PvL). Wordt leeg gezet als de kolom ontbreekt. |
    | `Herkomst` | str | §5.10 | `N` (Nederland), `E` (EER), `R` (rest). ETL hertaalt naar `NL` / `EER` / `Niet-EER`. |
    | `Hogerejaars` | str | §5.14 | `J` of `N`. ETL hertaalt naar `Ja` / `Nee`. |
    | `Herinschrijving` | str | §5.15 | `J` of `N`. ETL hertaalt naar `Ja` / `Nee`. |
    | `Aantal` | int | §5.17 | Aantal (voor)aanmeldingen. ETL hernoemt naar `Ongewogen vooraanmelders`. |
    | `meercode_V` | int | §5.12 | Deler: `Gewogen vooraanmelders = Aantal / meercode_V`. Status A-rijen hebben waarde 0 en worden uitgefilterd. |
    | `Status` | str | §5.13 | `V` (verzoek), `I` (inschrijving), `U` (uitgeschreven) of `A` (annulering). `A`-rijen worden uit de aggregatie gehouden. |

!!! tip "Teldata bevat meerdere instellingen — scoop op je eigen Brincode"
    Een telbestand is landelijk: de kolom `Brincode` bevat álle instellingen. Wil je alleen je eigen instelling, gebruik dan [`institution_filter`](configuratie.md#institution_filter-beperk-de-teldata-tot-een-of-meer-instellingen) of de CLI-vlag [`--institution`](aan-de-slag.md#-institution). Zonder filter traint en voorspelt de tool over alle instellingen tegelijk.

#### Afwijkende bestandsnamen

Gebruikt jouw instelling een andere conventie? Geef in `configuration.json` eigen patronen op met placeholders `{year}` en `{week}`:

```json
{ "telbestand_filename_patterns": ["telbestandY{year}W{week}", "VU_telbestand_{year}_W{week}"] }
```

Andere karakters worden letterlijk gematcht. Zie [`telbestand_filename_patterns`](configuratie-referentie.md#telbestand_filename_patterns-bestandsnaampatronen-voor-telbestanden) voor de volledige uitleg.

### Individuele aanmelddata

Pad: `data/input_raw/individuele_aanmelddata.csv` · Bron: Osiris / Usis · Scheidingsteken: `;`

Eén rij per aanmelding. De *Rol* geeft aan hoe de pipeline een veld gebruikt: **sleutel** = unieke identifier; **basis** = filtering/deadline/aggregatie; **classifier** = feature in het XGBoost-model; **info** = ingelezen maar niet door modellen gebruikt (mag leeg zijn).

??? note "Volledige kolommenlijst (kanonieke namen — herbenoembaar via `configuration.json`)"
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
    | `Is hogerejaars` | basis | Indicator hogerejaars (`Ja` / `Nee`) |
    | `BBC ontvangen` | info | Indicator BBC (Bewijs van Betaald Collegegeld) ontvangen |
    | `Type vooropleiding` | classifier | Type vooropleiding |
    | `Nationaliteit` | classifier | Nationaliteit student |
    | `EER` | classifier | EER-indicator |
    | `Geslacht` | classifier | Geslacht |
    | `Geverifieerd adres postcode` / `plaats` / `land` | classifier | Geverifieerd woonadres |
    | `Studieadres postcode` / `land` | classifier | Studieadres |
    | `School code eerste vooropleiding` / `School eerste vooropleiding` | classifier | Eerste vooropleiding |
    | `Plaats code eerste vooropleiding` / `Land code eerste vooropleiding` | classifier | Locatie eerste vooropleiding |
    | `Aantal studenten` | basis | Tellerveld (meestal `1` per rij) |

    De classifier-features zijn instelbaar onder `model_features.classifier` — zie [Configuratie](configuratie-referentie.md#columns-kolomnamen-mapping).

### Telbestand studenten

Pad: `data/input_raw/oktober_bestand.xlsx` · Bron: door je instelling aangeleverd

Bevat de werkelijke inschrijvingen per collegejaar, opleiding (`Isatcode`), herkomst en examentype — de ground truth waarop het model wordt geëvalueerd en die het ratio-model voedt. Elke rij is een telling geaggregeerd op die combinatie van dimensies; de meeste instellingen genereren dit bestand uit hun SIS/datawarehouse (Osiris, Usis, of vergelijkbaar).

| Canonieke naam | Omschrijving |
|----------------|-------------|
| `Collegejaar` | Collegejaar |
| `Isatcode` | **CROHO-code — de joinsleutel met de features.** Moet gelijk zijn aan de `Isatcode` in de telbestanden. |
| `Groepeernaam Croho` | Naam van de opleiding (optioneel, voor leesbaarheid; niet de joinsleutel) |
| `Aantal eerstejaars croho` | Aantal eerstejaars |
| `EER-NL-nietEER` | Herkomstgroep |
| `Examentype code` | Examentype, bv. `Bachelor eerstejaars`, `Bachelor hogerejaars`, `Master` |
| `Aantal Hoofdinschrijvingen` | Aantal hoofdinschrijvingen |

!!! warning "`Isatcode` is verplicht en moet matchen met de telbestanden"
    De studentaantallen worden op de **`Isatcode`** gekoppeld aan de vooraanmeldingen, niet op de opleidingsnaam (die per instelling verschilt). Komen de codes niet overeen, dan blijft de koppeling leeg en traint het cumulatieve model zonder labels.

??? note "Waarom heet het bestand `oktober_bestand.xlsx`?"
    Historische naam. De bestandsnaam en configuratiesleutels (`path_raw_october`, `columns.oktober`) zijn ongewijzigd gelaten om bestaande installaties niet te breken. Inhoudelijk is het een telbestand met studentaantallen.

## ETL-stappen

De ETL draait automatisch bij elke run (tenzij `--noetl`):

| Stap | Actie | Input | Output |
|------|-------|-------|--------|
| 1 | Rowbind + reformat (config-gedreven via [`cumulative_input`](configuratie-referentie.md#cumulative_input-uva-sql-telbestand-omzetten)) | `telbestanden/*.csv` | Samengevoegd cumulatief bestand |
| 2 | Interpolatie ontbrekende weken | Samengevoegd bestand | `vooraanmeldingen_cumulatief.csv` |
| 3 | Studentaantallen berekenen | `oktober_bestand.xlsx` | `student_count_*.xlsx`, `student_volume.xlsx` |
| 4 | Kopiëren individuele data | `individuele_aanmelddata.csv` | `vooraanmeldingen_individueel.csv` |

Zie de [volledige dataflow](https://github.com/cedanl/studentprognose/blob/main/doc/PIPELINE.md) voor een Mermaid-diagram.

## ETL overslaan

Heb je al verwerkte data (eigen ETL, collega, eerdere run)? Sla ETL én validatie over met `--noetl`. De vereiste bestanden moeten dan in `data/input/` staan.

```bash
studentprognose --noetl -d cumulative -w 12 -y 2024
```

!!! warning "Heb je alleen ruwe bronbestanden? Gebruik `--noetl` dan níét"
    `--noetl` slaat de ETL over en verwacht kant-en-klare bestanden in `data/input/`. Heb je alleen de ruwe bronbestanden (Studielink-telbestanden, Osiris/Usis-export, telbestand studenten)? Laat `--noetl` dan weg, zodat de ingebouwde ETL de verwerkte bestanden voor je aanmaakt.

!!! warning "De cumulatieve data heeft historische collegejaren nodig"
    Het cumulatieve spoor traint op data van vóór het voorspeljaar: het XGBoost-instroommodel (`SARIMA_cumulative`) en het ratio-model (`Prognose_ratio`) leiden hun voorspelling af uit eerdere jaren. Bevat je data **alleen het voorspeljaar**, dan blijven beide kolommen `NaN` — terwijl `Voorspelde vooraanmelders` zich wél vult en de output dus compleet *oogt*. Lever altijd minstens één, idealiter drie collegejaren vóór het voorspeljaar mee. De [trainingshistorie-check](validatie.md#trainingshistorie-waarom-deze-check-bestaat) bewaakt dit. Snelle controle: `sorted(df_cum["Collegejaar"].unique())`.

## Verwerkte inputbestanden (`data/input/`)

Dit zijn de bestanden die het model direct inleest. De ETL maakt ze aan; bij `--noetl` plaats je ze zelf.

| Bestand | Wanneer nodig | Bron |
|---------|---------------|------|
| `vooraanmeldingen_cumulatief.csv` | `-d c` of `-d b` | ETL stap 1+2 |
| `vooraanmeldingen_individueel.csv` | `-d i` of `-d b` | ETL stap 4 |
| `student_count_first-years.xlsx` | Altijd | ETL stap 3 |
| `student_volume.xlsx` | Alleen bij `-sy v` | ETL stap 3 |
| `ensemble_weights.xlsx` | Optioneel | `scripts/calculate_ensemble_weights.py` |
| `totaal_cumulatief.xlsx` / `totaal_individueel.xlsx` | Optioneel | `scripts/append_studentcount_and_compute_errors.py` |

## Institutiespecifieke kolomnamen

Gebruikt jouw instelling andere kolomnamen dan de kanonieke? Map ze in de `columns`-sectie van `configuration.json` — alleen de afwijkende namen:

```json
{ "columns": { "individual": { "Croho groepeernaam": "CROHO_NAAM", "Inschrijfstatus": "STATUS" } } }
```

Zie [Configuratie — referentie](configuratie-referentie.md#columns-kolomnamen-mapping) voor de volledige kolommenlijst.

## Voor gevorderden

!!! info "Sla dit over tenzij het op jou van toepassing is"
    De onderstaande onderwerpen gelden alleen bij een afwijkend inputformaat (UvA SQL) of bij bewust gebruik van `path_cumulative_new`. Gebruik je de standaard Studielink-telbestanden, dan hoef je hier niets mee.

### Bekende valkuil: `path_cumulative_new`

!!! warning "`path_cumulative_new` overschrijft en verwijdert bestanden"
    Als `path_cumulative_new` in je configuratie een bestaand bestand aanwijst, **mergt de pipeline dit in `vooraanmeldingen_cumulatief.csv` en verwijdert daarna het bronbestand** (met een melding). Bij een crash ná het schrijven maar vóór het verwijderen is de originele data onherstelbaar kwijt. Maak vooraf een backup, en zet `path_cumulative_new` op `""` als je dit niet nodig hebt.

### UvA SQL-telbestand (fijnmazig Studielink-formaat)

Map: `data/input_raw/telbestanden/` · Bestandsnaam: `telbestand_sl_{leverdatum}_v{volgnummer}_{studiejaar}.csv` · Scheidingsteken: `,` (komma)

De UvA levert de wekelijkse telbestanden als **SQL-export met 22 kolommen**: een fijnmaziger, minder voorbewerkte variant met extra dimensies (`Geslacht`, `Opl_vorm`, `Voertaal`, `Fixus`, …) en zonder de verrijkte velden `Groepeernaam`/`Faculteit`. De ETL zet dit **config-gedreven** om naar exact dezelfde 16 kolommen als het legacy-formaat, zodat de modellen ongewijzigd blijven — zie [`cumulative_input`](configuratie-referentie.md#cumulative_input-uva-sql-telbestand-omzetten).

??? note "Kolommen, snapshot-semantiek en bekende beperkingen"
    | Kolom | Omschrijving |
    |-------|-------------|
    | `Brincode` | → `Korte naam instelling`. **Multi-instelling**: één levering bevat alle instellingen waar een student zich aanmeldde. |
    | `Isatcode` | → `Croho`. Ook placeholder voor `Groepeernaam Croho`/`Naam Croho opleiding Nederlands`. |
    | `Type_HO` | `B`/`P` → Bachelor, `M` → Master, **`A` → Associate degree**, `O` → Onbekend. |
    | `Studiejaar` | → `Collegejaar`. |
    | `Herkomst` | `N`/`E`/`R`/`O` → `NL`/`EER`/`Niet-EER`/`Onbekend`. |
    | `Status` | `A` (annulering) wordt uitgefilterd; voor die rijen is `meercode_V == 0`. |
    | `meercode_V` | Deler voor `Gewogen vooraanmelders`. Na het `Status != "A"`-filter komt geen deel-door-nul voor. |
    | `Aantal` | → `Ongewogen vooraanmelders`. |
    | `etl_is_deleted` | Soft-delete vlag; rijen met `1` worden uitgefilterd. |
    | `Voertaal`, `Geslacht`, `Opl_vorm`, `Fixus`, `1cHO_L`, `1cHO_K` | Extra split-dimensies; de ETL aggregeert ze weg. |

    - **Snapshot-semantiek** — elke levering is een volledige cumulatieve momentopname (`Aantal` is een lopend totaal, geen wekelijkse aanwas). **Tel nooit over meerdere weekleveringen op.**
    - **Weeknummer uit de leverdatum** — de week zit niet in de kolommen; de ETL leidt de **ISO-kalenderweek** af uit de leverdatum in de bestandsnaam (`telbestand_sl_20260525_...` → week 22).
    - **Ontbrekende `Groepeernaam`/`Faculteit`** — de ETL vult `Groepeernaam Croho` met de `Isatcode` en `Faculteit` met een placeholder (`"Onbekend"`); een lege Faculteit zou anders alle UvA-rijen uit de pivot laten vallen. De join loopt daarom op de landelijk-stabiele `Isatcode` (zie [#232](https://github.com/cedanl/studentprognose/issues/232)).
    - **Andere jaargrens (week 36)** — de UvA-aanmeldfase eindigt rond week 36; zet `model_config.final_academic_week` op `36` (zie [configuratie](configuratie-referentie.md#final_academic_week)). Het dashboard ordent de week-as nog volgens de vaste 39→38-kalender; de voorspellingen kloppen, maar de week-as kan cosmetisch afwijken.
