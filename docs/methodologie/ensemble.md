# Ensemble

!!! abstract "In het kort"
    - **Wat het doet:** combineert de voorspellingen van SARIMA, XGBoost en het ratio-model tot één gewogen eindvoorspelling.
    - **Vertrouw het als:** de losse modellen het onderling grotendeels eens zijn — hoge overeenkomst geeft meer vertrouwen.
    - **Wees voorzichtig als:** de gewichten verouderd zijn (na een beleidsingreep of uitzonderlijk jaar), of bij een eerste run zonder geoptimaliseerde gewichten.
    - **In het ensemble:** dit *is* het ensemble — alleen actief in de `-d b` (both) modus.

Het ensemble combineert de voorspellingen van SARIMA, XGBoost en het ratio-model tot één eindvoorspelling. Dit is alleen beschikbaar in de `-d b` (both) modus.

!!! tip "Uitgebreide versie — Jupyter notebook"
    Een uitvoerbare versie die laat zien hoe de gewichten per peilweek doorwerken in een
    voorspelling staat in [`notebooks/05_ensemble.ipynb`](https://github.com/cedanl/studentprognose/blob/main/notebooks/05_ensemble.ipynb).

## Waarom een ensemble?

Geen enkel model presteert in alle situaties het beste. SARIMA is sterk bij stabiele seizoenspatronen; XGBoost vangt individuele kenmerken beter op; het ratio-model is robuust bij weinig data. Door modellen te combineren wordt de variantie in de eindvoorspelling verlaagd — mits de modellen niet allemaal op dezelfde manier fout gaan.

## Ensemble vs. individuele modellen

In de output staan zowel de individuele modelvoorspellingen (`SARIMA_individual`, `SARIMA_cumulative`, `Prognose_ratio`) als de ensemble-voorspelling (`Ensemble_prediction`). Dit stelt je in staat om:

- Te controleren of de modellen het eens zijn — hoge overeenkomst geeft meer vertrouwen in de uitkomst
- Afwijkende modellen te signaleren als early warning
- De toegevoegde waarde van het ensemble te beoordelen t.o.v. het ratio-model als simpele baseline

Zie [Output lezen](../output-begrijpen.md) voor uitleg over de outputkolommen.

<iframe src="../../assets/plots/ensemble_comparison.html" width="100%" height="500" frameborder="0" style="border-radius: 8px;"></iframe>

*Vergelijking individuele modellen vs ensemble voor drie voorbeeldopleidingen (demodata)*

## Hoe worden de gewichten bepaald?

De gewichten in de pipeline komen uit **twee verschillende bronnen**, die elk iets anders wegen:

1. **Grid-search-gewichten** (`ensemble_weights.xlsx`) — bepalen per opleiding, examentype en herkomst hoe zwaar de losse modellen (SARIMA, XGBoost, ratio) meewegen. Deze worden geleerd uit historische fouten (hieronder beschreven).
2. **Configuratiegewichten** (`ensemble_weights` in `configuration.json`) — bepalen per weekperiode hoe zwaar SARIMA-individueel versus SARIMA-cumulatief meeweegt (zie [Configureerbare ensemble-gewichten](#configureerbare-ensemble-gewichten)).

!!! warning "Twee bronnen — controleer de voorrang in de configuratie-docs"
    Deze pagina beschrijft beide gewichtbronnen los van elkaar. Welke bron voorrang krijgt wanneer ze allebei aanwezig zijn, staat niet op deze pagina vast; raadpleeg de [Configuratie](../configuratie.md)-documentatie voor de precieze voorrangsregel.

De **grid-search-gewichten** worden bepaald via een grid search over gewichtscombinaties (percentageparen). Voor elke combinatie van gewichten wordt de historische fout (MAE en MAPE) berekend over meerdere voorgaande jaren. De combinatie met de laagste historische fout wordt gekozen als optimale weging.

Dit is bewust anders dan een eenvoudige inverse-MAE weging (waarbij elk model automatisch gewicht krijgt omgekeerd evenredig aan zijn eigen fout — een nauwkeuriger model weegt dan zwaarder). De grid search optimaliseert de gewichten namelijk direct op de ensemble-uitkomst, niet op de individuele modelfouten. Hierdoor wordt rekening gehouden met correlatie tussen modellen.

De optimale gewichten worden opgeslagen in `ensemble_weights.xlsx` en worden per opleiding, examentype en herkomst bepaald.

!!! warning "Gewichten zijn datumgevoelig"
    De gewichten zijn gebaseerd op historische prestaties. Als de situatie verandert (bijv. na een beleidsingreep of een uitzonderlijk jaar), kunnen verouderde gewichten een slecht presterend model te veel vertrouwen geven. Herbereken de gewichten na elk studiejaar via `scripts/calculate_ensemble_weights.py`.

## Bij een eerste run

Als `ensemble_weights.xlsx` nog niet bestaat, gebruikt het model gelijke gewichten voor alle modellen. Dit is een redelijke standaard, maar niet optimaal. Draai de ensemble-optimalisatie pas nadat je één volledig studiejaar aan output hebt.

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
| `week_35_37` | Weken 35 t/m `final_academic_week - 1` (vlak voor de einddeadline) |
| `default` | Alle overige situaties |

De **einddeadline** gebruikt altijd 100% het individuele SARIMA-model. De weekgrenzen (waaronder deze eindweek) volgen [`model_config.final_academic_week`](../configuratie-referentie.md#final_academic_week); de gewichten zelf zijn configureerbaar.

Pas de gewichten aan op basis van validatieresultaten voor jouw instelling. Zie [Configuratie](../configuratie.md) voor de volledige optiedocumentatie.
