# XGBoost

!!! abstract "In het kort"
    - **Wat het doet:** XGBoost voorspelt op twee plekken — als classifier de inschrijfkans per individuele student, en als regressor het totale cohortaantal uit het wekelijkse vooraanmeldpatroon.
    - **Vertrouw het als:** er genoeg historische inschrijfpatronen zijn die representatief zijn voor het huidige cohort.
    - **Wees voorzichtig als:** de samenstelling van vooraanmelders sterk afwijkt van de historie (nieuwe campagne, gewijzigd beleid), of bij opleidingen met weinig historische data.
    - **In het ensemble:** levert de classifier de curve van het individuele spoor en de regressor de inschrijvingsschatting van het cumulatieve spoor; beide gaan mee in de gewogen eindvoorspelling.

XGBoost wordt op twee plekken ingezet in de pipeline, met fundamenteel verschillende rollen: als **classifier** (individueel spoor) en als **regressor** (cumulatief spoor).

!!! tip "Uitgebreide versie — Jupyter notebook"
    Een uitvoerbare versie met feature-engineering stap voor stap (lagged features, acceleration,
    exclusivity ratio) en feature importance op de demodata staat in
    [`notebooks/03_xgboost.ipynb`](https://github.com/cedanl/studentprognose/blob/main/notebooks/03_xgboost.ipynb).

## XGBoost Classifier (individueel spoor)

!!! tip "End-to-end uitleg van het individueel spoor"
    Deze sectie beschrijft de classifier zelf. Voor de volledige pipeline (preprocessing → classifier → aggregatie → SARIMA-extrapolatie) en de filterregels die bepalen welke aanmeldingen meedoen, zie [Individueel model](individueel.md).

### Wat doet het?

De classifier voorspelt per individuele student de **kans dat deze zich uiteindelijk inschrijft**, gegeven persoonskenmerken en het moment van vooraanmelding. De uitkomsten per student worden gesommeerd tot een verwacht cohortaantal.

### Waarom XGBoost hier?

De inschrijfkans hangt af van meerdere factoren die op elkaar inwerken. XGBoost vangt zulke niet-lineaire samenhangen op zonder dat je die vooraf handmatig hoeft te specificeren (geen expliciete feature engineering). Bovendien gaat het goed om met het feit dat er veel meer 'wel'- dan 'niet'-inschrijvers zijn: de meeste vooraanmelders schrijven zich uiteindelijk in, dus de twee klassen zijn ongelijk verdeeld.

### Features

| Type | Features |
|------|----------|
| Numeriek | Collegejaar, Sleutel_count, is_numerus_fixus |
| Categorisch | Examentype, Faculteit, Croho groepeernaam, Deadlineweek, Herkomst, Weeknummer, Opleiding, Type vooropleiding, Nationaliteit, EER, Geslacht, Plaats code eerste vooropleiding, School code/naam eerste vooropleiding, Land code eerste vooropleiding, Studieadres postcode/land, Geverifieerd adres postcode/plaats/land |
| Cumulatief (optioneel) | Gewogen vooraanmelders (ratio), Ongewogen vooraanmelders (ratio), Aanmelders met 1 aanmelding (ratio), Inschrijvingen (ratio) |

Categorische features worden one-hot geëncodeerd. Onbekende categorieën in de testset worden genegeerd (`handle_unknown="ignore"`).

### Aannames

- Historische inschrijfpatronen zijn representatief voor het huidige cohort.
- Studenten die hun aanmelding hebben ingetrokken vóór de voorspelweek tellen als niet-ingeschreven (label 0).

---

## XGBoost Regressor (cumulatief spoor)

### Wat doet het?

De regressor voorspelt het **totaal aantal inschrijvingen** per opleiding/herkomst/examentype, op basis van het geaggregeerde wekelijkse vooraanmeldpatroon (tot en met week 38) en historische studentaantallen als target.

### Features

| Type | Features |
|------|----------|
| Numeriek | Collegejaar, weekkolommen (week 1 t/m 38 als afzonderlijke kolommen) |
| Afgeleid — lagged | `Gewogen_t-2`, `Gewogen_t-5`: gewogen vooraanmelders 2 en 5 weken vóór de voorspelweek |
| Afgeleid — dynamiek | `Gewogen_acceleration`: geeft aan of de aanmeldstroom versnelt of juist afvlakt — positief = instroom versnelt, negatief = instroom vlakt af. Berekend als de tweede afgeleide van de aanmeldstroom: `(huidig − t-2) − (t-2 − t-5)`. |
| Afgeleid — commitment | `exclusivity_ratio`: aandeel aanmelders dat exclusief voor deze opleiding kiest — `Aantal aanmelders met 1 aanmelding / (Ongewogen vooraanmelders + ε)`. Hogere waarde = grotere commitment. |
| Categorisch | Examentype, Faculteit, Croho groepeernaam, Herkomst |

De lagged en afgeleide features worden per (opleiding × herkomst × examentype × jaar) berekend ten opzichte van de voorspelweek. "2 weken eerder" telt in de volgorde van het academisch jaar: in week 1 is dat week 51 van het kalenderjaar ervoor, en aan het begin van het seizoen stopt de telling bij de startweek. Ontbreekt een referentieweek, dan wordt als fallback week 1 gebruikt, maar alleen als week 1 op dat moment al voorbij is. Anders, of als ook week 1 ontbreekt, wordt 0 ingevuld.

**Waarom in academische volgorde?** Tot [#318](https://github.com/cedanl/studentprognose/issues/318) werd gewoon `voorspelweek − 2` gerekend. Aan het begin van het seizoen (bijv. week 37 bij `final_academic_week = 36`) wees dat naar week 35: het *eind* van hetzelfde collegejaar. Bij een terugblik op een afgerond jaar keek de feature dan in de toekomst. In een live run bestaat die week nog niet en was de feature 0.

**Vroege weken zonder data in elk jaar vallen weg.** Een weekkolom aan het begin van het academisch jaar gaat alleen als feature mee als *elk* collegejaar (training én voorspeljaar) vanaf die week data heeft. Mist één jaar zijn eerste weken, door een gemist telbestand of doordat de aanmeldcyclus later opende, dan staan daar nullen die in de trainingsjaren nooit voorkomen. Het model ziet het jaar dan als iets wat het niet kent, en de voorspelling kan een factor naast zitten. Die vroege weken bevatten nauwelijks informatie over de uiteindelijke instroom, dus weglaten kost weinig. De startweek (`final_academic_week + 1`) is in elk jaar 0 en valt om dezelfde reden altijd weg.

**Wanneer vertrouw je het niet?** Als er veel vroege weken wegvallen, bijvoorbeeld omdat één trainingsjaar pas laat begint. Het model kijkt dan alleen naar latere weken. De pipeline [waarschuwt](../validatie.md#weekdekking-en-eindweek) als het voorspeljaar weken mist die eerdere jaren wel hebben.

De weekkolommen van het voorspeljaar zijn tot en met de voorspelweek echte waarnemingen; de weken daarna zijn de SARIMA-prognose. Een fout in die prognose werkt dus direct door in `SARIMA_cumulative`. Een week die binnen een jaar tussen twee waarnemingen ontbreekt, wordt in de weekkolommen geïnterpoleerd in plaats van op 0 gezet, in training én in het voorspeljaar (zie [SARIMA → Ontbrekende weken](sarima.md#ontbrekende-weken-afkappen-op-de-voorspelweek)).

**Als SARIMA geen prognose oplevert.** Voor sommige combinaties lukt de SARIMA-fit niet, bijvoorbeeld bij een te korte of vlakke reeks. De weken na de voorspelweek krijgen dan de **laatst bekende waarde** (de stand in de voorspelweek): "er komt niets meer bij". De regressor rekent daarmee en de console meldt welke combinaties het betreft. Dat is een bewust voorzichtige aanname. Bij een opleiding die nog flink groeit valt `SARIMA_cumulative` daardoor te laag uit. De output bevat voor die opleiding geen `Voorspelde vooraanmelders` voor de latere weken, want er is geen prognose. Zie je de melding voor een opleiding die je belangrijk vindt, beoordeel die voorspelling dan kritisch.

### Terugblik op een afgerond jaar

Voorspel je een jaar dat al voorbij is (een backtest, of terugblikjaren in dezelfde run), dan bevat de input ook de weken ná de voorspelweek. Die lagen op het voorspelmoment in de toekomst. Het model ziet ze niet: van het voorspeljaar gebruikt de pipeline alleen de weken tot en met de voorspelweek. Een terugblik geeft daardoor exact dezelfde voorspelling als een live run in die week, en de fout die je meet is een eerlijke maat voor de live nauwkeurigheid. Het dashboard toont wel de volledige, echte curve.

Tot [#318](https://github.com/cedanl/studentprognose/issues/318) gold dat niet. Lukte SARIMA niet voor een opleiding, dan bleven de echte latere weken in de features staan en "wist" de regressor de eindstand al. Terugblikken en daarop gebaseerde betrouwbaarheidsinschattingen konden daardoor te rooskleurig uitvallen. Vergelijk je met resultaten van vóór die fix, houd daar dan rekening mee.

### Waarom XGBoost hier?

Om dezelfde reden als bij de classifier (zie [Waarom XGBoost hier?](#waarom-xgboost-hier) hierboven): de relatie tussen het cumulatieve vooraanmeldpatroon en het uiteindelijke inschrijvingsaantal is niet-lineair en varieert per opleiding, herkomst en jaar. XGBoost leert dat patroon zonder dat de wekelijkse curve expliciet gemodelleerd hoeft te worden.

### Wanneer vertrouw je het niet?

- Cohorten waarbij de samenstelling van vooraanmelders sterk verschilt van historisch (bijv. na een nieuwe marketingcampagne of gewijzigd instroombeleid).
- Opleidingen met weinig historische observaties in de trainingsdata.

---

## Relatie tot andere modellen

De twee XGBoost-modellen staan elk aan het begin van een eigen spoor:

- De **classifier** levert per student een inschrijfkans; die kansen worden gesommeerd tot een cohortcurve die vervolgens met [SARIMA](sarima.md) wordt geëxtrapoleerd naar het einde van het jaar (zie [Individueel model](individueel.md)).
- De **regressor** vormt stap 2 van het cumulatieve spoor: hij vertaalt het (door SARIMA geëxtrapoleerde) vooraanmeldpatroon naar een verwacht aantal inschrijvingen.

De uitkomsten van beide sporen komen samen in het [ensemble](ensemble.md), waar ze gewogen worden gecombineerd met het [ratio-model](ratio-model.md).

---

## Feature importance

Na het trainen van elk XGBoost-model wordt de **feature importance** geëxtraheerd en gegroepeerd per oorspronkelijke feature. One-hot geëncodeerde categorieën worden teruggegroepeerd naar hun oorspronkelijke kolom (bijv. alle `Herkomst_NL`, `Herkomst_EER`, … worden samengevoegd tot `Herkomst`). Weeknummers worden weergegeven als `Week 1`, `Week 2`, etc.

Het dashboard toont de importances niet: dat richt zich op hoe goed het model voorspelt. Via de Python-API zijn ze na een run beschikbaar als `strategy.xgboost_importance`. De grafieken hieronder zijn voorbeelden op demodata.

<iframe src="../../assets/plots/xgb_classifier_importance.html" width="100%" height="450" frameborder="0" style="border-radius: 8px;"></iframe>

*Feature importance — XGBoost classifier · individueel spoor: welke kenmerken zijn het meest bepalend voor de individuele inschrijfkans? (demodata)*

<iframe src="../../assets/plots/xgb_regressor_importance.html" width="100%" height="450" frameborder="0" style="border-radius: 8px;"></iframe>

*Feature importance — XGBoost regressor · cumulatief spoor: welke features dragen het meest bij aan de voorspelling van het totale cohortaantal? (demodata)*

## Alternatieve regressiemodellen (cumulatief spoor)

Naast XGBoost zijn twee alternatieve regressiemodellen beschikbaar, configureerbaar via `model_config.cumulative_regressor` in `configuration.json`:

| Model | Config-waarde | Beschrijving |
|-------|---------------|-------------|
| **Ridge Regression** | `ridge` | L2-geregulariseerde lineaire regressie. Stabiel bij weinig trainingsdata en bij multicollineariteit (de weekkolommen zijn onderling sterk gecorreleerd). Interpreteerbare coëfficiënten. |
| **Random Forest** | `random_forest` | Ensemble van beslisbomen. Robuust bij kleine datasets, ingebouwde feature importance, minder gevoelig voor outliers dan gradient boosting. |

Alle regressors delen dezelfde preprocessing-pipeline (OneHotEncoding voor categorische features, passthrough voor numerieke features). Dit garandeert een eerlijke vergelijking.

Gebruik `studentprognose benchmark -w <week>` om te vergelijken welk model het best presteert op jouw data.

## Hyperparameter tuning

Dit betreft **stap 2** van het cumulatieve spoor — de regressor. De tijdreeks-trap (stap 1, SARIMA) heeft zijn eigen [orde-selectie](sarima.md#orde-selectie-tuning); beide trappen delen dezelfde tijd-bewuste cross-validatie en kun je samen tunen met `--tune-target both` / `tune="both"`.

### Wat doet het?

Tuning zoekt de hyperparameters van de cumulatieve regressor (leersnelheid, aantal bomen, boomdiepte, …) die het beste presteren op de **eigen historie** van een instelling — in plaats van te vertrouwen op vaste standaardwaarden. De zoektocht draait een kleine grid search en kiest de parameterset met de laagste gemiddelde fout.

```bash
studentprognose tune -d c -w 12      # regressor (stap 2); --tune-target both voor beide trappen
```

Het commando print een overzicht van alle geteste parametersets met hun MAPE — de best presterende set is gemarkeerd met `✓` — en een kant-en-klaar config-snippet voor `model_config.regressor_params`. Via de Python-API toont `run_pipeline_from_dataframes(..., tune=True)` (of `tune="both"`) exact hetzelfde overzicht, voorspelt vervolgens direct met de gevonden parameters, en geeft het voorspellings-DataFrame terug (het tuning-overzicht gaat naar de console).

### Hoe wordt geëvalueerd?

Identiek aan de [benchmark](benchmarks.md): **tijd-bewuste cross-validatie** (train op jaren tot en met N−1, valideer op jaar N, minimaal drie trainingsjaren) met MAPE als selectiemetriek. Elke kandidaat-parameterset wordt over alle folds beoordeeld; de laagste gemiddelde MAPE wint. Random k-fold wordt **niet** gebruikt: dat zou toekomstige jaren in de trainingsset lekken en de scores te optimistisch maken.

Tune je binnen een voorspelrun (`tune=True` in de API), dan gebruikt de tuning alleen jaren **vóór** het voorspeljaar. Bij een terugblik op een afgerond jaar zou dat jaar anders zelf een testfold zijn, en kiest de tuning parameters die de uitkomst al kennen. Er wordt één keer getuned, bij het eerste jaar van de run. Geef bij meerdere jaren die jaren daarom in oplopende volgorde op. Doe je dat niet, dan volgt een waarschuwing.

### Waarom een kleine grid en geen zware optimizer?

De trainingsset is klein — één rij per opleiding × jaar × herkomst × examentype, over een handvol jaren. Bij die omvang kiest een brede zoektocht sneller ruis dan signaal, en voegt een zwaardere optimizer (zoals Optuna) overfitting-risico toe zonder betrouwbare winst. De ingebouwde grid is daarom bewust compact en gericht op **regularisatie** (lagere boomdiepte, meer/minder bomen, lagere leersnelheid), de knoppen die op kleine data het meeste verschil maken.

### Aannames

- Er zijn genoeg historische collegejaren voor minstens één tijdreeks-split (≥ 4 jaren met data). Bij minder valt tuning terug op de standaardparameters en volgt een waarschuwing.
- De jaar-op-jaar-relatie tussen vooraanmeldpatroon en inschrijvingen is stabiel genoeg dat parameters die op het verleden goed scoren, ook voor het komende jaar gelden.

### Wanneer vertrouw je het niet?

- **Marginale verschillen.** Liggen de MAPE-waarden van de kandidaten dicht bij elkaar, dan is de "winnaar" grotendeels toeval. Wijk dan niet zonder reden af van de robuuste defaults.
- **Weinig validatiejaren.** Met slechts één of twee testjaren is de score een steekproef van bijna niets; behandel het resultaat als indicatief, niet als bewijs.
- **Tuning op het productiepad.** Tuning hoort een bewuste, periodieke stap te zijn waarvan je het resultaat *vastlegt* in `regressor_params`. Standaard tunen bij elke voorspelrun zou de uitkomst traag en niet-reproduceerbaar maken; daarom staat `tune` standaard uit.

### Relatie tot het ensemble

Tuning raakt uitsluitend stap 2 van het cumulatieve spoor (vooraanmelders → inschrijvingen). De tijdreeks-extrapolatie (SARIMA), het individuele spoor en de [ensemble-weging](ensemble.md) blijven ongewijzigd. Betere regressorparameters verbeteren dus alleen de cumulatieve component van de uiteindelijke ensemble-voorspelling.

## Implementatie

Zie `src/studentprognose/models/xgboost_classifier.py`, `src/studentprognose/models/xgboost_regressor.py`, `src/studentprognose/models/regressors.py`, `src/studentprognose/models/tuning.py` (hyperparameter tuning) en `src/studentprognose/models/importance.py`.
