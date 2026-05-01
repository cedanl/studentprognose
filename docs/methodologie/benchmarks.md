# Benchmarks

Het benchmarkcommando vergelijkt alle beschikbare modellen op jouw eigen data, zodat je een onderbouwde keuze kunt maken vóór het wijzigen van de configuratie.

## Draaien

```bash
studentprognose benchmark -w 12
```

De benchmark evalueert alle combinaties van tijdreeksmodellen (SARIMA, ETS, Theta, AutoARIMA) en regressiemodellen (XGBoost, Ridge, Random Forest) via time-series cross-validatie.

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
| **MAPE** | Mean Absolute Percentage Error — relatieve fout, vergelijkbaar tussen opleidingen van verschillende grootte |
| **MAE** | Mean Absolute Error — absolute fout in studentaantallen |
| **RMSE** | Root Mean Squared Error — penaliseert grote fouten zwaarder |
| **Trainingstijd** | Gemiddelde fittijd per model in seconden |

### Twee stappen apart gemeten

Het cumulatieve spoor heeft twee stappen. De benchmark meet beide apart:

1. **Stap 1 (tijdreeks):** hoe goed extrapoleert het model de vooraanmelderscurve? Gemeten als MAPE/MAE/RMSE op de curve zelf.
2. **Stap 2 (regressie):** hoe goed vertaalt het model vooraanmelders naar inschrijvingen? Gemeten op het eindaantal studenten.

Door apart te meten kun je zien waar de fout zit: in de extrapolatie of in de vertaling.

## Output

De benchmark slaat resultaten op in:

- `data/output/benchmark_timeseries.csv` — resultaten per tijdreeksmodel
- `data/output/benchmark_regressor.csv` — resultaten per regressiemodel

Kolommen bevatten model, programme, herkomst, examentype, testjaar, metrics, trainingssetgrootte en convergentie-informatie.

## Interpretatie

- **Kleine datasets** (< 10 trainingsrijen, typisch bij numerus fixus): Ridge regression presteert hier vaak beter dan XGBoost door de L2-regularisatie.
- **Convergentie-failures**: als SARIMA voor veel combinaties faalt, overweeg ETS of Theta — deze zijn stabieler bij korte reeksen.
- **Trainingstijd**: AutoARIMA is aanzienlijk trager dan de andere modellen door de automatische orde-selectie.
