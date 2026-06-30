# XGBoost

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

De inschrijfkans hangt af van meerdere interacterende factoren. XGBoost vangt niet-lineaire interacties op zonder expliciete feature engineering, en is robuust bij scheve klasseverdeling (de meeste vooraanmelders schrijven zich wel in — de klassen zijn ongelijk verdeeld).

### Features

| Type | Features |
|------|----------|
| Numeriek | Collegejaar, Sleutel_count, is_numerus_fixus |
| Categorisch | Examentype, Faculteit, Croho groepeernaam, Deadlineweek, Herkomst, Weeknummer, Opleiding, Type vooropleiding, Nationaliteit, EER, Geslacht, Plaats code eerste vooropleiding, School code/naam eerste vooropleiding, Land code eerste vooropleiding, Studieadres postcode/land/plaats |
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
| Afgeleid — dynamiek | `Gewogen_acceleration`: tweede afgeleide van de aanmeldstroom — `(huidig − t-2) − (t-2 − t-5)`. Positief = instroom versnelt, negatief = instroom vlakt af. |
| Afgeleid — commitment | `exclusivity_ratio`: aandeel aanmelders dat exclusief voor deze opleiding kiest — `Aantal aanmelders met 1 aanmelding / (Ongewogen vooraanmelders + ε)`. Hogere waarde = grotere commitment. |
| Categorisch | Examentype, Faculteit, Croho groepeernaam, Herkomst |

De lagged en afgeleide features worden per (opleiding × herkomst × examentype × jaar) berekend ten opzichte van de voorspelweek. Ontbreekt een referentieweek dan wordt als fallback week 1 gebruikt; is ook die afwezig dan wordt 0 ingevuld.

### Waarom XGBoost hier?

De relatie tussen het cumulatieve vooraanmeldpatroon en het uiteindelijke inschrijvingsaantal is niet-lineair en varieert per opleiding, herkomst en jaar. XGBoost kan dit patroon leren zonder dat de wekelijkse curve expliciet gemodelleerd hoeft te worden.

### Wanneer vertrouw je het niet?

- Cohorten waarbij de samenstelling van vooraanmelders sterk verschilt van historisch (bijv. na een nieuwe marketingcampagne of gewijzigd instroombeleid).
- Opleidingen met weinig historische observaties in de trainingsdata.

---

## Feature importance

Na het trainen van elk XGBoost-model wordt de **feature importance** geëxtraheerd en gegroepeerd per oorspronkelijke feature. One-hot geëncodeerde categorieën worden teruggegroepeerd naar hun oorspronkelijke kolom (bijv. alle `Herkomst_NL`, `Herkomst_EER`, … worden samengevoegd tot `Herkomst`). Weeknummers worden weergegeven als `Week 1`, `Week 2`, etc.

De gegroepeerde importances worden getoond in het interactieve dashboard (zie [Output begrijpen](../output-begrijpen.md#interactief-dashboard)).

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

### Wat doet het?

Tuning zoekt de hyperparameters van de cumulatieve regressor (leersnelheid, aantal bomen, boomdiepte, …) die het beste presteren op de **eigen historie** van een instelling — in plaats van te vertrouwen op vaste standaardwaarden. De zoektocht draait een kleine grid search en kiest de parameterset met de laagste gemiddelde fout.

```bash
studentprognose tune -d c -w 12
```

Het commando print een overzicht van alle geteste parametersets met hun MAPE — de best presterende set is gemarkeerd met `✓` — en een kant-en-klaar config-snippet voor `model_config.regressor_params`. Via de Python-API toont `run_pipeline_from_dataframes(..., tune=True)` exact hetzelfde overzicht, voorspelt vervolgens direct met de gevonden parameters, en geeft het voorspellings-DataFrame terug (het tuning-overzicht gaat naar de console).

### Hoe wordt geëvalueerd?

Identiek aan de [benchmark](benchmarks.md): **tijd-bewuste cross-validatie** (train op jaren tot en met N−1, valideer op jaar N, minimaal drie trainingsjaren) met MAPE als selectiemetriek. Elke kandidaat-parameterset wordt over alle folds beoordeeld; de laagste gemiddelde MAPE wint. Random k-fold wordt **niet** gebruikt: dat zou toekomstige jaren in de trainingsset lekken en de scores te optimistisch maken.

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
