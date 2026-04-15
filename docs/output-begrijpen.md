# Output begrijpen

TODO: outputbestanden, kolomdefinities, MAE/MAPE uitgelegd, wanneer is een prognose betrouwbaar.

## Outputbestanden

| Bestand | Beschrijving |
|---------|-------------|
| `output_first-years_*.xlsx` | Eerstejaars voorspellingen (eindresultaat) |
| `output_higher-years_*.xlsx` | Hogerjaars voorspellingen |
| `output_volume_*.xlsx` | Volume-voorspellingen (totaal) |
| `output_prelim_*.xlsx` | Tussenresultaat vóór ratio/ensemble/foutmaten |

## Kolomdefinities

| Kolom | Beschrijving |
|-------|-------------|
| `SARIMA_individual` | Voorspelling SARIMA individueel spoor |
| `SARIMA_cumulative` | Voorspelling SARIMA cumulatief spoor |
| `Prognose_ratio` | Voorspelling ratio-model |
| `Ensemble_prediction` | Gewogen ensemble-voorspelling |
| `MAE_*` | Gemiddelde absolute afwijking (historisch) |
| `MAPE_*` | Gemiddelde procentuele afwijking (historisch) |
