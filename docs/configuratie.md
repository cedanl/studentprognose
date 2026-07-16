# Configuratie

De pipeline wordt geconfigureerd via `configuration/configuration.json`.

## Beginnen

Voer eenmalig `studentprognose init` uit in je werkmap. Dit schrijft een startconfiguratie met alle standaardwaarden. Je hoeft daarna alleen de velden aan te passen die bij jouw instelling afwijken.

## Hoe configuratie laden werkt

De pipeline laadt altijd eerst de **ingebakken standaardwaarden** uit het package, en past jouw `configuration.json` daar bovenop als een *overschrijving*. Dat betekent:

- Je hoeft alleen te specificeren wat afwijkt van de standaard.
- Een ontbrekend veld in jouw bestand valt automatisch terug op de standaardwaarde.
- Als het configuratiebestand helemaal niet bestaat, worden de standaardwaarden gebruikt en verschijnt er een waarschuwing.

Het bestand heeft de volgende secties: `paths`, `telbestand_filename_patterns`, `cumulative_input`, `validation`, `runtime`, `model_config`, `numerus_fixus`, `excluded_data_points`, `ensemble_override_cumulative`, `exclude_from_combined`, `ensemble_weights`, `column_roles`, `model_features` en `columns`.

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
| `path_raw_october` | `data/input_raw/oktober_bestand.xlsx` | Telbestand studenten (ruwe bron, eigen levering instelling). De sleutel heet `_october` om historische redenen. |
| `path_raw_telbestanden` | `data/input_raw/telbestanden` | Map met Studielink telbestanden |
| `path_raw_individueel` | `data/input_raw/individuele_aanmelddata.csv` | Ruwe individuele aanmelddata |
| `path_student_count_first-years` | `data/input/student_count_first-years.xlsx` | Werkelijk aantal eerstejaars (afgeleid uit telbestand studenten) |
| `path_student_count_higher-years` | `data/input/student_count_higher-years.xlsx` | Werkelijk aantal hogerejaars (alleen nodig bij `-sy h`) |
| `path_student_volume` | `data/input/student_volume.xlsx` | Totaal studentvolume (afgeleid uit telbestand studenten) |
| `path_ratios` | `data/input/ratiobestand.xlsx` | Ratiobestand (optioneel) |

## `telbestand_filename_patterns` — bestandsnaampatronen voor telbestanden

De meegeleverde (`init`-)config herkent standaard zowel het legacy- als het UvA Studielink SQL-formaat: `["telbestandY{year}W{week}", "telbestand_sl_{date}_v{volgnummer}_{year}"]`. Laat je de sleutel volledig weg, dan valt de code terug op alleen `telbestandY{year}W{week}`.

De pipeline herkent telbestanden in `path_raw_telbestanden` aan hun bestandsnaam. Instellingen met afwijkende naamconventies kunnen hier hun eigen patroon opgeven. De ETL en de validatie gebruiken hetzelfde patroon om jaar en weeknummer uit de bestandsnaam te halen.

**Placeholder-syntax:**

- `{year}` — vierdigit collegejaar (bijv. `2024`); **verplicht**
- `{week}` — weeknummer (1–2 cijfers, gevalideerd op bereik 1–53)
- `{date}` — leverdatum `YYYYMMDD` (8 cijfers); de week wordt afgeleid als **ISO-kalenderweek** van die datum (gebruikt door het UvA SQL-telbestand)
- `{volgnummer}` — Studielink-volgnummer (genegeerd voor de week-afleiding, maar mag in het patroon staan)
- Alle andere karakters worden letterlijk gematcht. Punten, streepjes en underscores zijn veilig (`re.escape` op de achtergrond).

Een patroon moet `{year}` bevatten én een week kunnen opleveren: dus `{week}` **of** `{date}`.

**Voorbeelden:**

```json
{
    "telbestand_filename_patterns": [
        "telbestandY{year}W{week}",
        "VU_telbestand_{year}_W{week}",
        "telbestand_sl_{date}_v{volgnummer}_{year}"
    ]
}
```

Met meerdere patronen kan de pipeline bestanden van verschillende leveranciers naast elkaar verwerken — handig tijdens een migratie van oude naar nieuwe naamgeving. Het laatste voorbeeld is het UvA SQL-formaat: `telbestand_sl_20260525_v34_2026.csv` levert ISO-week 22 (uit leverdatum `20260525`).

**Foutgedrag:** Een patroon zonder `{year}`, of zonder zowel `{week}` als `{date}`, leidt tot een configuratiefout en stopt de pipeline direct, met een melding die het probleempatroon noemt.

## `runtime` — uitvoerparameters

Instellingen die bepalen hoe de pipeline zich gedraagt tijdens uitvoer (los van de modelkeuze).

### `cpu_count`

Standaard: `null`

Het aantal CPU-cores dat de pipeline gebruikt voor parallelle voorspellingen. De voorspelling van individuele SARIMA-modellen wordt verdeeld over deze cores via `joblib.Parallel`.

| Waarde | Gedrag |
|--------|--------|
| `null` | Automatisch — `os.cpu_count()` wordt gebruikt, of `1` als het besturingssysteem geen waarde teruggeeft (kan voorkomen in containers met cgroup-limieten). |
| Geheel getal `>= 1` | Gebruikt deze waarde. Als de waarde hoger is dan het aantal beschikbare cores, wordt deze automatisch verlaagd naar dat aantal en verschijnt er een waarschuwing. |

Pas dit aan als:

- De pipeline crasht omdat `os.cpu_count()` geen betrouwbare waarde teruggeeft op jouw hardware of in jouw container (achtergrond van [issue #162](https://github.com/cedanl/studentprognose/issues/162)).
- Je het CPU-gebruik wilt beperken op een gedeelde machine.

Ongeldige waarden (`0`, negatieve getallen, niet-gehele getallen) leiden tot een configuratiefout en stoppen de pipeline direct.

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

### `cumulative_timeseries`

Standaard: `"sarima"`

Het tijdreeksmodel voor de cumulatieve curve-extrapolatie (stap 1 van het cumulatieve spoor). De keuze bepaalt hoe de vooraanmelderscurve tot week 38 wordt geëxtrapoleerd.

| Waarde | Model | Omschrijving |
|--------|-------|-------------|
| `sarima` | SARIMA(1,0,1)×(1,1,1,52) | Standaard — vaste ordes, bewezen in productie |
| `ets` | AutoETS | Automatische componentkeuze (error/trend/seizoen), stabieler bij korte reeksen |
| `theta` | AutoTheta | Zeer simpel, competitief bij korte tijdreeksen |
| `auto_arima` | AutoARIMA | Automatische orde-selectie via AICc, flexibeler dan vaste SARIMA |

Gebruik `studentprognose benchmark` om te vergelijken welk model het best presteert op jouw data.

### `cumulative_regressor`

Standaard: `"xgboost"`

Het regressiemodel dat vooraanmelderscijfers vertaalt naar verwachte inschrijvingen (stap 2 van het cumulatieve spoor).

| Waarde | Model | Omschrijving |
|--------|-------|-------------|
| `xgboost` | XGBoost Regressor | Standaard — gradient boosting, krachtig bij voldoende data |
| `ridge` | Ridge Regression | L2-regularisatie, stabiel bij weinig trainingsdata en multicollineariteit |
| `random_forest` | Random Forest | Robuust bij kleine datasets, ingebouwde feature importance |

### `regressor_params` — hyperparameters vastleggen

Optioneel. Een object dat per regressor de hyperparameters vastlegt waarmee het model wordt geïnstantieerd. Niet-opgegeven parameters vallen terug op de modeldefaults. Dit is het **reproduceerbare** pad: de waarden worden bij elke run gebruikt, zonder opnieuw te tunen.

```json
"model_config": {
    "cumulative_regressor": "xgboost",
    "regressor_params": {
        "xgboost": { "learning_rate": 0.1, "n_estimators": 200, "max_depth": 5 }
    }
}
```

De parameters per regressor worden direct doorgegeven aan het onderliggende model (bijv. `XGBRegressor`, `Ridge`, `RandomForestRegressor`). Alleen het model dat in `cumulative_regressor` actief is, gebruikt zijn parameters; vermeldingen voor andere regressors worden genegeerd.

Vul je dit liever niet handmatig in? Het commando `studentprognose tune -d c` zoekt de beste waarden en print een kant-en-klaar snippet om hier te plakken (zie [Waarden vastleggen](#waarden-vastleggen) hieronder en [XGBoost → Hyperparameter tuning](methodologie/xgboost.md#hyperparameter-tuning)).

### `tuning_grid` — eigen zoekruimte voor tuning

Optioneel. Een object dat de zoekruimte voor `studentprognose tune` (en de API-parameter `tune`) overschrijft: per hyperparameter een lijst van te proberen waarden. Bij afwezigheid wordt de ingebouwde, regularisatie-gerichte standaardgrid gebruikt.

```json
"model_config": {
    "tuning_grid": {
        "max_depth": [3, 5, 7],
        "n_estimators": [100, 200, 400]
    }
}
```

### `forecaster_params` — SARIMA-ordes vastleggen

Optioneel. Het tijdreeks-equivalent van `regressor_params`: een object dat per forecaster de instantiatie-parameters vastlegt. Voor SARIMA zijn dat de ARIMA-ordes `order` en `seasonal_order`; voor de overige forecasters bijv. `season_length`. Niet-opgegeven waarden vallen terug op de modeldefaults. Ook dit is het **reproduceerbare** pad.

```json
"model_config": {
    "cumulative_timeseries": "sarima",
    "forecaster_params": {
        "sarima": { "order": [1, 1, 1], "seasonal_order": [1, 1, 0, 52] }
    }
}
```

De vierde waarde van `seasonal_order` is de seizoenslengte; die wordt bij fit nog afgestemd op de werkelijke jaar-periode van je data (net als in productie). Alleen de forecaster die in `cumulative_timeseries` actief is, gebruikt zijn parameters.

### `sarima_tuning_grid` — eigen zoekruimte voor SARIMA-tuning

Optioneel. Net als `tuning_grid`, maar voor `studentprognose tune --tune-target sarima` (en de API `tune="sarima"`/`"both"`): per orde een lijst van te proberen waarden. Bij afwezigheid wordt de ingebouwde SARIMA-grid gebruikt.

```json
"model_config": {
    "sarima_tuning_grid": {
        "order": [[1, 0, 1], [1, 1, 1], [2, 1, 1]],
        "seasonal_order": [[1, 1, 0, 52], [1, 1, 1, 52]]
    }
}
```

#### Waarden vastleggen

De aanbevolen werkwijze: draai `studentprognose tune -d c -w <week>` (eventueel met `--tune-target sarima` of `both`), kopieer de getoonde snippets naar je `configuration.json`, en draai daarna normaal. Zo zijn de getunede waarden expliciet, reproduceerbaar en deelbaar — in plaats van impliciet bij elke run opnieuw gezocht.

### `final_academic_week`

Standaard: `38`

De **laatste week van het academisch jaar** in de Studielink-cyclus. Bepaalt de seizoensvolgorde van de weken (het jaar loopt van week `final_academic_week + 1` via week 52 door naar `final_academic_week`), de voorspelhorizon (`pred_len`) en de reset-week-injectie in het cumulatieve spoor.

- Het legacy instellingsformaat loopt tot **week 38** (eind september) — dit is de standaard.
- Het UvA SQL-telbestand levert de aanmeldfase tot **week 36**; er zijn geen leveringen in de weken 37–39. Zet daarvoor `final_academic_week` op `36`.

Een verkeerde waarde laat de cumulatieve voorspelling crashen (de pipeline slicet kolommen tot deze week) of geeft een onjuiste voorspelhorizon. De waarde geldt voor het **cumulatieve spoor**; het individuele spoor gebruikt de vaste Studielink-kalender.

## `cumulative_input` — UvA SQL-telbestand omzetten

Stuurt hoe de ETL-rowbind (`_rowbind_and_reformat`) de ruwe telbestanden uit `path_raw_telbestanden` omzet naar de 16-koloms `vooraanmeldingen_cumulatief.csv`. **De meegeleverde (`init`-)config bevat dit blok al, ingesteld op het raw Studielink SQL-formaat** — dat is de standaard-doelgroep. De `Default`-kolom hieronder beschrijft de code-terugval wanneer je het blok (of een sleutel) volledig weglaat; dan vervalt de ETL naar de legacy-defaults (`;`-gescheiden). Voor het UvA SQL-formaat (zie [Je data voorbereiden](je-data-voorbereiden.md#uva-sql-telbestand-fijnmazig-studielink-formaat)):

```json
{
    "cumulative_input": {
        "separator": ",",
        "value_maps": {
            "Type hoger onderwijs": {"P": "Bachelor", "B": "Bachelor", "M": "Master", "A": "Associate degree", "O": "Onbekend"},
            "Herkomst": {"N": "NL", "E": "EER", "R": "Niet-EER", "O": "Onbekend"}
        },
        "faculteit_sentinel": "Onbekend",
        "aggregate": true,
        "drop_deleted": true
    }
}
```

| Sleutel | Default | Betekenis |
|---------|---------|-----------|
| `separator` | `";"` | Scheidingsteken van de ruwe telbestanden (UvA SQL: `","`). |
| `rename` | legacy-map | Bronkolom → canonieke kolomnaam. De UvA-bronkolommen (`Brincode`, `Studiejaar`, `Type_HO`, `Isatcode`, `Aantal`) zijn gelijk aan legacy, dus meestal niet nodig. |
| `value_maps` | legacy-vertalingen | Waardevertalingen per canonieke kolom. Per kolom volledig overschrijfbaar; niet-genoemde kolommen (bijv. `Herinschrijving`/`Hogerejaars`) houden hun default. Niet-gemapte waarden gaan onveranderd door (met een waarschuwing). |
| `faculteit_sentinel` | `null` | Vulwaarde voor een ontbrekende `Faculteit`. **Moet niet-leeg zijn** als de kolom in de bron ontbreekt: een lege (NaN) Faculteit laat de cumulatieve `groupby`/pivot de rijen vallen. |
| `programme_name_source` | `"Groepeernaam"` | Bronkolom voor de leesbare opleidingsnaam; valt terug op `Croho` (Isatcode) als de kolom ontbreekt. |
| `aggregate` | `false` | Tel de fijnmazige rijen op naar de canonieke grain (verplicht voor UvA: bereken `Gewogen` per rij, sommeer daarna; anders crasht de pivot op dubbele indexrijen). |
| `drop_deleted` | `false` | Filter rijen met `etl_is_deleted != 0` (UvA SQL soft-delete vlag). |

De canonieke output is altijd `;`-gescheiden, ongeacht de input-separator.

## `numerus_fixus`

Een object met **programmasleutels** als keys en het maximale inschrijvingsaantal (de NF-capaciteit) als waarde. Numerus-fixus-opleidingen krijgen een aparte behandeling: een eigen regressor (los van de gedeelde pool), een capaciteitsplafond op de voorspelling, en aparte foutrapportage.

```json
{
    "numerus_fixus": {
        "56604": 350,
        "50033": 260
    }
}
```

Als de gesommeerde voorspelling over herkomstgroepen het maximum overschrijdt, wordt het overschot afgetrokken van de NL-herkomstgroep. Opleidingen die hier niet staan worden niet gecapped.

!!! danger "De sleutel moet exact matchen met de programmakolom (`Croho groepeernaam`)"
    De NF-behandeling grijpt **alleen** aan als de sleutel exact overeenkomt met een waarde in de programmakolom van je data. Welk formaat dat is, hangt af van je instelling:

    - **Cumulatief spoor** — de programmakolom bevat sinds de Isatcode-migratie de numerieke **Isatcode** (CROHO-code, bijv. `56604`). Gebruik dan de Isatcode als sleutel.
    - **Individueel spoor / legacy** — de programmakolom bevat de **leesbare opleidingsnaam** (bijv. `"B Geneeskunde"`). Gebruik dan de naam als sleutel.

    Het dtype maakt niet uit: een numerieke sleutel wordt automatisch als getal geïnterpreteerd (`"56604"` en `56604` zijn equivalent). JSON kent geen inline-commentaar, dus noteer de leesbare naam bij voorkeur in je eigen documentatie of changelog naast de Isatcode.

!!! warning "Sleutelruimtes van de twee sporen overlappen niet (#238)"
    Het cumulatieve spoor keyt op Isatcodes, het individuele spoor op namen. Eén sleutel kan daardoor maar één van beide sporen matchen. Draai je `-d both`, dan grijpt een Isatcode-sleutel wél aan in het cumulatieve spoor maar niet in het individuele — je krijgt hierover een **waarschuwing**. Tot #238 is opgelost is er geen sleutel die beide sporen tegelijk bedient.

!!! info "Guard tegen stille misconfiguratie (#258)"
    Vóór de voorspelling controleert de pipeline of elke `numerus_fixus`-sleutel voorkomt in de programmakolom. Matcht een sleutel **geen enkel** geladen spoor (typefout of verkeerd formaat), dan **stopt de pipeline met een harde fout** — een niet-matchende sleutel wordt nooit meer stil genegeerd. Zie [validatie](validatie.md#numerus-fixus-sleutels).

## `institution_filter` — beperk de teldata tot één of meer instellingen

De landelijke Studielink-teldata bevat rijen van **álle** instellingen (elke Brincode staat in de kolom `Korte naam instelling`). Met `institution_filter` beperk je de pipeline tot je eigen instelling(en), zodat de modellen niet op andermans instroom trainen.

```json
{
    "institution_filter": ["21PC"]
}
```

- **Type:** lijst van instellingscodes (Brincodes, bijv. `"21PC"`) of leesbare instellingsnamen — precies zoals ze in de kolom `Korte naam instelling` staan.
- **Standaard (`[]`):** geen filter — **alle instellingen** in de data worden meegenomen. Dit is het backwards-compatibele default-gedrag.
- Meerdere instellingen: `["21PC", "00IC"]`.

Het filter grijpt aan op **load-tijd**, vóór preprocessing, zodat zowel de training als de voorspelling op de gekozen instelling(en) draaien.

!!! info "Welke sporen worden gefilterd?"
    Alleen sporen die een instellingskolom dragen — dat is het **cumulatieve spoor** (de landelijke teldata). Het **individuele spoor** is doorgaans de eigen aanmeldexport van één instelling en heeft geen instellingskolom; daar is het filter een no-op. Het studentaantallenbestand (`student_count`) wordt eveneens verondersteld al instelling-specifiek te zijn.

!!! warning "Onbekende instelling faalt hard"
    Komt een opgegeven code in het geheel niet voor in de data, dan stopt de pipeline met een duidelijke foutmelding (en de lijst van beschikbare instellingen) in plaats van stil een lege dataset te verwerken. Komt een code wél gedeeltelijk voor, dan zie je een waarschuwing voor de ontbrekende codes.

Je kunt deze config-waarde op de commandline overschrijven met de vlag [`--institution`](aan-de-slag.md#-institution) — handig om snel voor één instelling te draaien zonder de config aan te passen.

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
            "opleiding": "B Voorbeeldopleiding"
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
| `week_35_37` | Weeknummer 35 t/m `final_academic_week - 1` |
| `default` | Alle overige gevallen |

De einddeadline (`final_academic_week`, standaard week 38; UvA week 36) is altijd 100% individueel en wordt niet door deze instelling beïnvloed. Zie [Ensemble](methodologie/ensemble.md) voor achtergrond bij de keuze van gewichten.

!!! note "Weekgrenzen volgen `final_academic_week`"
    De sleutelnamen (zoals `week_30_34`) beschrijven de weekperiode waarop een gewicht van toepassing is; de **gewichten** zijn instelbaar via dit blok. De bovengrens van `week_35_37` en de 100%-individueel-eindweek schalen mee met [`model_config.final_academic_week`](#final_academic_week) (bij week 36 loopt `week_35_37` dus alleen over week 35). De overige weekgrenzen (17–23, 30–34) liggen vast in `output/postprocessor.py` en vereisen een codewijziging om aan te passen.

## `columns` — kolomnamen mapping

Maakt het mogelijk dat instellingen andere kolomnamen gebruiken dan de kanonieke namen die de pipeline intern hanteert. Specificeer alleen de namen die afwijken — ontbrekende sleutels vallen terug op de kanonieke naam.

### `individual` — individuele aanmelddata

Volledige lijst van kanonieke kolomnamen die gemapped kunnen worden:

`Sleutel`, `Datum Verzoek Inschr`, `Ingangsdatum`, `Collegejaar`, `Datum intrekking vooraanmelding`, `Inschrijfstatus`, `Faculteit`, `Examentype`, `Croho`, `Croho groepeernaam`, `Opleiding`, `Hoofdopleiding`, `Eerstejaars croho jaar`, `Is eerstejaars croho opleiding`, `Is hogerejaars`, `BBC ontvangen`, `Type vooropleiding`, `Nationaliteit`, `EER`, `Geslacht`, `Geverifieerd adres postcode`, `Geverifieerd adres plaats`, `Geverifieerd adres land`, `Studieadres postcode`, `Studieadres land`, `School code eerste vooropleiding`, `School eerste vooropleiding`, `Plaats code eerste vooropleiding`, `Land code eerste vooropleiding`, `Aantal studenten`

### `oktober` — telbestand studenten

`Collegejaar`, `Isatcode`, `Groepeernaam Croho`, `Aantal eerstejaars croho`, `EER-NL-nietEER`, `Examentype code`, `Aantal Hoofdinschrijvingen`

De sleutel heet `oktober` om historische redenen — zie [Je data voorbereiden](je-data-voorbereiden.md#telbestand-studenten). `Isatcode` is de **joinsleutel** met de vooraanmeldingen (de studentaantallen worden hierop gekoppeld, niet op `Groepeernaam Croho`); zorg dat de codes overeenkomen met die in je telbestanden.

### `cumulative` — Studielink cumulatieve data

`Korte naam instelling`, `Collegejaar`, `Weeknummer rapportage`, `Weeknummer`, `Faculteit`, `Type hoger onderwijs`, `Groepeernaam Croho`, `Naam Croho opleiding Nederlands`, `Croho`, `Herinschrijving`, `Hogerejaars`, `Herkomst`, `Gewogen vooraanmelders`, `Ongewogen vooraanmelders`, `Aantal aanmelders met 1 aanmelding`, `Inschrijvingen`

Dit zijn de kolommen ná de ETL-transformatie (na het inlezen en hernoemen van de ruwe Studielink-velden). Zie [Studielink telbestanden](je-data-voorbereiden.md#studielink-telbestanden) voor de mapping van ruwe PvL-velden naar deze kanonieke namen.

## `column_roles` — semantische rollen → kanonieke kolomnaam

Waar `columns` instellings-specifieke namen vertaalt naar kanonieke namen, koppelt `column_roles` een **semantische rol** (die de code intern gebruikt) aan de kanonieke kolomnaam. De pipeline benadert kolommen via deze rollen (bijv. `programme`, `academic_year`, `origin`), zodat een hernoeming op één plek volstaat.

| Rol | Standaard kolom | Rol | Standaard kolom |
|-----|-----------------|-----|-----------------|
| `academic_year` | `Collegejaar` | `week` | `Weeknummer` |
| `programme` | `Croho groepeernaam` | `exam_type` | `Examentype` |
| `origin` | `Herkomst` | `faculty` | `Faculteit` |
| `enrollment_status` | `Inschrijfstatus` | `cancellation_date` | `Datum intrekking vooraanmelding` |
| `student_count` | `Aantal_studenten` | `enrollments` | `Inschrijvingen` |
| `weighted_applicants` | `Gewogen vooraanmelders` | `unweighted_applicants` | `Ongewogen vooraanmelders` |
| `single_applicants` | `Aantal aanmelders met 1 aanmelding` | `higher_years` | `Hogerejaars` |
| `croho_source` | `Groepeernaam Croho` | `higher_education_type` | `Type hoger onderwijs` |
| `institution` | `Korte naam instelling` | | |

De rol `institution` bepaalt op welke kolom [`institution_filter`](#institution_filter-beperk-de-teldata-tot-een-of-meer-instellingen) filtert.

Pas dit alleen aan als je de **interne** kanonieke namen wijzigt — in de meeste gevallen gebruik je `columns` (voor je ruwe inputkolommen) en laat je `column_roles` ongemoeid.

## `model_features` — featurelijsten per model

Bepaalt welke kolommen als features in de modellen gaan. Twee subsleutels:

- `classifier` — de individuele XGBoost-classifier, met `numeric` en `categorical` featurelijsten (zie [XGBoost](methodologie/xgboost.md#features)).
- `regressor` — de cumulatieve XGBoost-regressor, met een `categorical` featurelijst; numerieke features (vooraanmelders, lags, acceleratie) worden afgeleid tijdens feature-engineering.

Voeg of verwijder features hier als je inputdata extra (of minder) kolommen bevat. Verwijderde features mogen niet meer in `columns`/`column_roles` als verplicht gemarkeerd staan.

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
| `telbestand.required_columns` | legacy-lijst | Vereiste ruwe kolommen. Het UvA SQL-formaat mist `Groepeernaam`, dus zet hier een lijst zonder die kolom. |
| `telbestand.separator` | `";"` | Scheidingsteken waarmee de validatie de telbestanden inleest. Zet op `","` voor het UvA SQL-formaat (gelijk aan `cumulative_input.separator`). |
| `telbestand.programme_column` | `"Groepeernaam"` | Kolom waarop categorale fouten worden gegroepeerd. Zet op `"Isatcode"` als `Groepeernaam` ontbreekt (UvA). |

Net als bij `cumulative_input` levert de meegeleverde (`init`-)config dit `validation.telbestand`-blok al mee, ingesteld op het raw Studielink SQL-formaat; de `Standaard`-kolom beschrijft de code-terugval als je een sleutel weglaat. De `validation`-overrides en het [`cumulative_input`](#cumulative_input-uva-sql-telbestand-omzetten)-blok horen samen; ze beschrijven hetzelfde ruwe formaat voor respectievelijk de validatie en de ETL:

```json
{
    "validation": {
        "telbestand": {
            "separator": ",",
            "programme_column": "Isatcode",
            "required_columns": ["Studiejaar", "Isatcode", "Aantal", "meercode_V", "Status", "Herinschrijving", "Hogerejaars", "Herkomst"],
            "herkomst_allowed": ["N", "E", "R", "O"]
        }
    }
}
```

Zie [Validatie](validatie.md) voor een volledig overzicht van alle controles.

## Voorbeeld minimale configuratie

Omdat de pipeline standaardwaarden inbouwt, hoef je alleen te specificeren wat afwijkt. Een typische startconfiguratie bevat alleen de instelling-specifieke velden:

```json
{
    "numerus_fixus": {
        "B Geneeskunde": 340
    },
    "ensemble_override_cumulative": [
        "B Geneeskunde"
    ]
}
```

Alle overige velden — paden, kolomnamen, modelparameters, ensemble-gewichten — vallen terug op de ingebakken standaardwaarden. Voer `studentprognose init` uit om een volledig ingevuld startbestand te krijgen dat je als referentie kunt gebruiken.
