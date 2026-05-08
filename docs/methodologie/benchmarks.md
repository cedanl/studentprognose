# Benchmarks

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

De cumulatieve benchmark evalueert alle combinaties van tijdreeksmodellen (SARIMA, ETS, Theta, AutoARIMA) en regressiemodellen (XGBoost, Ridge, Random Forest) via time-series cross-validatie.

### Individueel spoor

De individuele benchmark evalueert classificatiemodellen (XGBoost, Random Forest, Logistic Regression) die voorspellen of een aanmelder zich daadwerkelijk inschrijft.

## Methodologie

### Cross-validatie

De benchmark gebruikt **leave-last-year-out** splits:

- **Train:** alle beschikbare jaren tot en met jaar N−1
- **Test:** jaar N
- **Minimaal 3 trainingsjaren** per fold

Dit bootst de productiescenario na: het model traint op historische data en voorspelt het lopende jaar.

### Leakage-preventie

- Strikte temporale scheiding: traindata bevat nooit het testjaar
- Preprocessing (OneHotEncoding) wordt alleen gefit op traindata
- Assertions in de code controleren dat geen target-waarden van het testjaar in de trainset terechtkomen

### Metrics

| Metric | Omschrijving |
|--------|-------------|
| **MAPE** | Mean Absolute Percentage Error — relatieve fout, vergelijkbaar tussen opleidingen van verschillende grootte (cumulatief) |
| **MAE** | Mean Absolute Error — absolute fout in studentaantallen (cumulatief + individueel geaggregeerd) |
| **RMSE** | Root Mean Squared Error — penaliseert grote fouten zwaarder (cumulatief) |
| **Accuracy** | Fractie correct geclassificeerde aanmelders (individueel) |
| **AUC-ROC** | Area Under the ROC Curve — hoe goed scheidt het model inschrijvers van niet-inschrijvers (individueel) |
| **F1** | Harmonisch gemiddelde van precision en recall (individueel) |
| **Aggregate MAE** | MAE op geaggregeerd niveau per opleiding/herkomst/examentype — de uiteindelijke business-metric (individueel) |
| **Trainingstijd** | Gemiddelde fittijd per model in seconden |

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
