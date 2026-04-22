# XGBoost

XGBoost wordt op twee plekken ingezet in de pipeline, met fundamenteel verschillende rollen: als **classifier** (individueel spoor) en als **regressor** (cumulatief spoor).

## XGBoost Classifier (individueel spoor)

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

### Hyperparameters en tuning

Alle hyperparameters zijn configureerbaar via `configuration.json` onder `model_config.hyperparameters.xgboost_classifier`. Standaardwaarden zijn geoptimaliseerd voor typische instroom-datasets.

**Early stopping** voorkomt overfitting: het model traint maximaal `n_estimators` bomen, maar stopt eerder als de validatiefout `early_stopping_rounds` rondes niet verbetert. De validatieset wordt gevormd door het meest recente trainingsjaar af te splitsen (temporele validatie, geen data leakage).

**Class balancing** via `scale_pos_weight` corrigeert voor de scheefverdeling in de data (de meeste vooraanmelders schrijven zich wel in). Standaard wordt dit automatisch berekend uit de trainingsdata.

Met `--tune` wordt een **randomized search met expanding-window cross-validatie** uitgevoerd over een parametergrid (max_depth, learning_rate, subsample, etc.). Standaard worden 50 willekeurige combinaties gesampled uit de 2.187 mogelijke combinaties — configureerbaar via `tuning_n_iter` in `configuration.json`. De gevonden optimale parameters worden gecacht en automatisch hergebruikt.

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
| Categorisch | Examentype, Faculteit, Croho groepeernaam, Herkomst |

### Waarom XGBoost hier?

De relatie tussen het cumulatieve vooraanmeldpatroon en het uiteindelijke inschrijvingsaantal is niet-lineair en varieert per opleiding, herkomst en jaar. XGBoost kan dit patroon leren zonder dat de wekelijkse curve expliciet gemodelleerd hoeft te worden.

### Hyperparameters en tuning

Net als de classifier zijn alle hyperparameters configureerbaar via `model_config.hyperparameters.xgboost_regressor`. Early stopping en temporele validatie werken identiek.

### Wanneer vertrouw je het niet?

- Cohorten waarbij de samenstelling van vooraanmelders sterk verschilt van historisch (bijv. na een nieuwe marketingcampagne of gewijzigd instroombeleid).
- Opleidingen met weinig historische observaties in de trainingsdata.

---

## Hyperparameter tuning

Met `--tune` draait de pipeline een **randomized search met expanding-window cross-validatie** voor beide XGBoost-modellen. Standaard worden 50 willekeurige parametercombinaties geëvalueerd (instelbaar via `tuning_n_iter`). Dit betekent:

- Fold 1: train op 2016–2020, valideer op 2021
- Fold 2: train op 2016–2021, valideer op 2022
- Fold 3: train op 2016–2022, valideer op 2023
- enz.

Deze aanpak respecteert de temporele volgorde en voorkomt data leakage. De classifier wordt geoptimaliseerd op cohort-level MAPE; de regressor op MAE.

Gevonden parameters worden opgeslagen in `data/output/tuning_cache.json` en hebben prioriteit boven de waarden in `configuration.json`. Verwijder het cachebestand om terug te vallen op de configuratiewaarden.

## Implementatie

Zie `src/studentprognose/models/xgboost_classifier.py` en `src/studentprognose/models/xgboost_regressor.py`.
