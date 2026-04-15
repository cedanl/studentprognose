# XGBoost

XGBoost wordt op twee plekken ingezet in de pipeline, met fundamenteel verschillende rollen: als **classifier** (individueel spoor) en als **regressor** (cumulatief spoor).

## XGBoost Classifier (individueel spoor)

### Wat doet het?

De classifier voorspelt per individuele student de **kans dat deze zich uiteindelijk inschrijft**, gegeven persoonskenmerken en het moment van vooraanmelding.

### Waarom XGBoost hier?

De kans dat een student zich inschrijft hangt af van meerdere interacterende factoren (herkomst, opleiding, aanmeldmoment, examenjaar). XGBoost vangt niet-lineaire interacties goed op zonder expliciete feature engineering, en is robuust bij scheve klasse-verdeling (de meeste vooraanmelders schrijven zich wel in).

### Features

Typische features zijn o.a.:

- Herkomstland / nationaliteit
- Opleiding + examentype
- Week van aanmelding
- Aantal dagen tot inschrijfdeadline

### Aannames

- Historische inschrijfpatronen zijn representatief voor het huidige cohort.
- De beschikbare persoonskenmerken zijn voldoende om inschrij fkans te differentiëren.

---

## XGBoost Regressor (cumulatief spoor)

### Wat doet het?

De regressor voorspelt het **totaal aantal inschrijvingen** op basis van het tot nu toe geaggregeerde aantal vooraanmelders en historische conversiepatronen.

### Waarom XGBoost hier?

De relatie tussen vooraanmelders en inschrijvingen is niet-lineair en varieert per opleiding, herkomst en week. Een regressiemodel dat niet-lineaire patronen kan leren is hier sterker dan een eenvoudige ratio.

### Wanneer vertrouw je het niet?

- Cohorten waarbij de samenstelling van vooraanmelders sterk verschilt van historisch (bijv. na een nieuwe marketingcampagne).
- Opleidingen met zeer weinig observaties in de trainingsdata.

---

## Implementatie

Zie `src/studentprognose/models/xgboost_classifier.py` en `src/studentprognose/models/xgboost_regressor.py`.
