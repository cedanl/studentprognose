# Ensemble

Het ensemble combineert de voorspellingen van SARIMA, XGBoost en het ratio-model tot één eindvoorspelling. Dit is alleen beschikbaar in de `-d b` (both) modus.

## Waarom een ensemble?

Geen enkel model presteert in alle situaties het beste. SARIMA is sterk bij stabiele seizoenspatronen; XGBoost vangt individuele kenmerken beter op; het ratio-model is robuust bij weinig data. Door modellen te combineren wordt de variantie in de eindvoorspelling verlaagd — mits de modellen niet allemaal op dezelfde manier fout gaan.

## Gewichten

Ensemble-gewichten worden bepaald op basis van historische modelfouten (MAE per opleiding per model). Een model dat het afgelopen jaar beter heeft gepresteerd krijgt een hoger gewicht.

$$w_m = \frac{1 / MAE_m}{\sum_{m'} 1 / MAE_{m'}}$$

De gewichten worden opgeslagen in `ensemble_weights.xlsx` en kunnen worden bijgewerkt via het post-processing script `archive/calculate_ensemble_weights.py`.

!!! warning "Gewichten zijn datumgevoelig"
    De gewichten zijn gebaseerd op historische prestaties. Als de situatie verandert (bijv. na een beleidsingreep), kunnen verouderde gewichten een slecht presterend model te veel vertrouwen geven. Herbereken de gewichten na elk studiejaar.

## Bij een eerste run

Als `ensemble_weights.xlsx` nog niet bestaat, gebruikt het model gelijke gewichten voor alle modellen. Dit is een redelijke standaard, maar niet optimaal. Draai het ensemble pas met aangepaste gewichten nadat je één volledig studiejaar aan output hebt.

## Ensemble vs. individuele modellen

In de output staan zowel de individuele modelvoorspellingen (`SARIMA_individual`, `SARIMA_cumulative`, `Prognose_ratio`) als de ensemble-voorspelling (`Ensemble_prediction`). Dit stelt je in staat om:

- Te controleren of de modellen het eens zijn (hoge overeenkomst = meer vertrouwen)
- Afwijkende modellen te signaleren als early warning
- De toegevoegde waarde van het ensemble te beoordelen t.o.v. individuele modellen

Zie [Output begrijpen](../output-begrijpen.md) voor uitleg over de outputkolommen.
