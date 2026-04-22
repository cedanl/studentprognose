# Ensemble

Het ensemble combineert de voorspellingen van SARIMA, XGBoost en het ratio-model tot één eindvoorspelling. Dit is alleen beschikbaar in de `-d b` (both) modus.

## Waarom een ensemble?

Geen enkel model presteert in alle situaties het beste. SARIMA is sterk bij stabiele seizoenspatronen; XGBoost vangt individuele kenmerken beter op; het ratio-model is robuust bij weinig data. Door modellen te combineren wordt de variantie in de eindvoorspelling verlaagd — mits de modellen niet allemaal op dezelfde manier fout gaan.

## Hoe worden de gewichten bepaald?

De ensemble-gewichten worden bepaald via een **grid search** over gewichtscombinaties (percentageparen). Voor elke combinatie van gewichten wordt de historische fout (MAE en MAPE) berekend over meerdere voorgaande jaren. De combinatie met de laagste historische fout wordt gekozen als optimale weging.

Dit is bewust anders dan een eenvoudige inverse-MAE weging: de grid search optimaliseert de gewichten direct op de ensemble-uitkomst, niet op de individuele modelfouten. Hierdoor wordt rekening gehouden met correlatie tussen modellen.

De optimale gewichten worden opgeslagen in `ensemble_weights.xlsx` en worden per opleiding, examentype en herkomst bepaald.

!!! warning "Gewichten zijn datumgevoelig"
    De gewichten zijn gebaseerd op historische prestaties. Als de situatie verandert (bijv. na een beleidsingreep of een uitzonderlijk jaar), kunnen verouderde gewichten een slecht presterend model te veel vertrouwen geven. Herbereken de gewichten na elk studiejaar via `archive/calculate_ensemble_weights.py`.

## Bij een eerste run

Als `ensemble_weights.xlsx` nog niet bestaat, gebruikt het model gelijke gewichten voor alle modellen. Dit is een redelijke standaard, maar niet optimaal. Draai de ensemble-optimalisatie pas nadat je één volledig studiejaar aan output hebt.

## Ensemble vs. individuele modellen

In de output staan zowel de individuele modelvoorspellingen (`SARIMA_individual`, `SARIMA_cumulative`, `Prognose_ratio`) als de ensemble-voorspelling (`Ensemble_prediction`). Dit stelt je in staat om:

- Te controleren of de modellen het eens zijn — hoge overeenkomst geeft meer vertrouwen in de uitkomst
- Afwijkende modellen te signaleren als early warning
- De toegevoegde waarde van het ensemble te beoordelen t.o.v. het ratio-model als simpele baseline

Zie [Output begrijpen](../output-begrijpen.md) voor uitleg over de outputkolommen.

## Configureerbare ensemble-gewichten

De gewichten die bepalen hoe zwaar SARIMA-individueel en SARIMA-cumulatief meewegen zijn instelbaar via `configuration.json`:

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

De gewichten gelden voor specifieke weekperiodes en examentypes:

| Sleutel | Situatie |
|---------|----------|
| `master_week_17_23` | Master-opleidingen in weken 17–23 (vlak voor HBO-inschrijfdeadline) |
| `week_30_34` | Weken 30–34 (zomerperiode, cumulatief minder betrouwbaar) |
| `week_35_37` | Weken 35–37 (vlak voor einddeadline) |
| `default` | Alle overige situaties |

Week 38 (einddeadline) gebruikt altijd 100% het individuele SARIMA-model en is niet configureerbaar.

Pas de gewichten aan op basis van validatieresultaten voor jouw instelling. Zie [Configuratie](../configuratie.md) voor de volledige optiedocumentatie.

## Getuned modellen in het ensemble

Wanneer `--tune` is gebruikt, trainen de onderliggende modellen (XGBoost classifier, regressor en SARIMA) met geoptimaliseerde hyperparameters. Dit verbetert de kwaliteit van de individuele modelvoorspellingen die het ensemble combineert. De ensemble-gewichten zelf worden niet gewijzigd door tuning — deze worden nog steeds bepaald via `ensemble_weights.xlsx` of de configureerbare weekgewichten.
