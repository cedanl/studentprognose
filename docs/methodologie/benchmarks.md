# Benchmarks

!!! abstract "In het kort"
    - **Wat het doet:** vergelijkt alle beschikbare modellen op jouw eigen data.
    - **Wanneer gebruik je het:** voordat je in de configuratie van model wisselt — zodat die keuze onderbouwd is in plaats van een gok.
    - **Wat betekent de uitkomst:** het model met de laagste fout op jouw historie is de beste kandidaat. Vuistregels: bij kleine datasets (numerus fixus) presteert Ridge vaak beter dan XGBoost, en bij korte tijdreeksen zijn ETS of Theta stabieler dan SARIMA.

Het benchmarkcommando vergelijkt alle beschikbare modellen op jouw eigen data, zodat je een onderbouwde keuze kunt maken vóór het wijzigen van de configuratie.

## Draaien

De `-d` flag is verplicht bij benchmark: kies `-d c` (cumulatief) of `-d i` (individueel). `-d both` is niet toegestaan.

```bash
# Cumulatief spoor
studentprognose benchmark -d c -w 12

# Individueel spoor
studentprognose benchmark -d i -w 12
```

### Cumulatief spoor

De cumulatieve benchmark evalueert alle combinaties van tijdreeksmodellen (SARIMA, ETS, Theta, AutoARIMA) en regressiemodellen (XGBoost, Ridge, Random Forest). Elk model wordt daarbij getraind op oudere jaren en getest op een recenter jaar — dat heet **time-series cross-validatie** (zie [Methodologie](#methodologie) hieronder).

### Individueel spoor

De individuele benchmark evalueert classificatiemodellen (XGBoost, Random Forest, Logistic Regression) die voorspellen of een aanmelder zich daadwerkelijk inschrijft.

## Methodologie

### Cross-validatie

De benchmark toetst elk model door het te trainen op de oudere jaren en het te testen op het eerstvolgende jaar — een aanpak die **leave-last-year-out** heet (letterlijk: "laat het laatste jaar eruit"):

- **Train:** alle beschikbare jaren tot en met jaar N−1
- **Test:** jaar N
- **Minimaal 3 trainingsjaren** per fold

Dit bootst de productiescenario na: het model traint op historische data en voorspelt het lopende jaar. De benchmark gebruikt dezelfde academische-jaargrens (`model_config.final_academic_week`) en dezelfde uit-de-data-afgeleide seizoenslengte als productie (zie [SARIMA → Seizoenslengte: afgeleid uit de data](sarima.md#seizoenslengte-afgeleid-uit-de-data)), zodat de gemeten modellen overeenkomen met wat productie draait. De regressor-evaluatie gebruikt bovendien dezelfde week-feature-set als productie: alleen de weekkolommen die de wide-pivot daadwerkelijk bevat (bij `final_academic_week` 36 zijn dat er ~43 in plaats van 52). Zo wordt geen enkele fold stil overgeslagen door ontbrekende kolommen.

!!! info "Tuning bouwt hierop voort"
    Het commando `studentprognose tune -d c` hergebruikt exact deze tijd-bewuste cross-validatie en MAPE-metriek voor **beide** trappen: `evaluate_regressor_model` voor de regressor-hyperparameters (stap 2) en `evaluate_timeseries_model` voor de SARIMA-ordes (stap 1, via `--tune-target sarima`/`both`). Waar de benchmark *modeltypes* vergelijkt, stemt tuning de *parameters/ordes* van het gekozen model af. Zie [XGBoost → Hyperparameter tuning](xgboost.md#hyperparameter-tuning) en [SARIMA → Orde-selectie](sarima.md#orde-selectie-tuning).

### Leakage-preventie

- Strikte temporale scheiding: traindata bevat nooit het testjaar
- Preprocessing (OneHotEncoding) wordt alleen gefit op traindata
- De tool controleert automatisch dat geen toekomstige data (target-waarden van het testjaar) in de training lekt

### Metrics

| Metric | Omschrijving | Wat betekent het voor mij? |
|--------|-------------|----------------------------|
| **MAPE** | Mean Absolute Percentage Error — relatieve fout, vergelijkbaar tussen opleidingen van verschillende grootte (cumulatief) | Lager is beter: hoeveel procent zit de voorspelling er gemiddeld naast. |
| **MAE** | Mean Absolute Error — absolute fout in studentaantallen (cumulatief + individueel geaggregeerd) | Lager is beter: hoeveel studenten zit de voorspelling er gemiddeld naast. |
| **RMSE** | Root Mean Squared Error — penaliseert grote fouten zwaarder (cumulatief) | Lager is beter, net als MAE, maar grote missers tellen extra zwaar. Ligt RMSE veel hoger dan MAE, dan zitten er een paar forse uitschieters in. |
| **Accuracy** | Fractie correct geclassificeerde aanmelders (individueel) | Hoger is beter: aandeel aanmelders dat correct is ingedeeld als (niet-)inschrijver. |
| **AUC-ROC** | Area Under the ROC Curve — hoe goed scheidt het model inschrijvers van niet-inschrijvers (individueel) | Hoger is beter: 1 is perfect onderscheid, 0,5 is niet beter dan gokken. |
| **F1** | Harmonisch gemiddelde van precision en recall (individueel) | Hoger is beter: balans tussen "hoe vaak klopt een 'wel'-voorspelling" en "hoeveel echte inschrijvers vindt het model". |
| **Aggregate MAE** | MAE op geaggregeerd niveau per opleiding/herkomst/examentype — de uiteindelijke business-metric (individueel) | Lager is beter: de fout op het niveau dat er echt toe doet. Kijk hier het eerst naar. |
| **Trainingstijd** | Gemiddelde fittijd per model in seconden | Geen kwaliteitsmaat, maar praktische kost: lager = sneller draaien. |

### Twee stappen apart gemeten

Het cumulatieve spoor heeft twee stappen. De benchmark meet beide apart:

1. **Stap 1 (tijdreeks):** hoe goed extrapoleert het model de vooraanmelderscurve? Gemeten als MAPE/MAE/RMSE op de curve zelf.
2. **Stap 2 (regressie):** hoe goed vertaalt het model vooraanmelders naar inschrijvingen? Gemeten op het eindaantal studenten.

Door apart te meten kun je zien waar de fout zit: in de extrapolatie of in de vertaling.

## Output

De benchmark slaat resultaten op in:

**Cumulatief spoor (`-d c`):**

- `data/output/benchmark_timeseries.csv` — resultaten per tijdreeksmodel
- `data/output/benchmark_regressor.csv` — resultaten per regressiemodel

**Individueel spoor (`-d i`):**

- `data/output/benchmark_classifier.csv` — resultaten per classificatiemodel

Kolommen bevatten model, testjaar, metrics, trainingssetgrootte en (bij cumulatief) convergentie-informatie.

## Interpretatie

- **Kleine datasets** (< 10 trainingsrijen, typisch bij numerus fixus): Ridge regression presteert hier vaak beter dan XGBoost door de L2-regularisatie.
- **Convergentie-failures**: als SARIMA voor veel combinaties faalt, overweeg ETS of Theta — deze zijn stabieler bij korte reeksen.
- **Trainingstijd**: AutoARIMA is aanzienlijk trager dan de andere modellen door de automatische orde-selectie.
