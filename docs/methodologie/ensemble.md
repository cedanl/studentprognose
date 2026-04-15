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
