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

## Implementatie

Zie `src/studentprognose/models/xgboost_classifier.py`, `src/studentprognose/models/xgboost_regressor.py` en `src/studentprognose/models/importance.py`.
