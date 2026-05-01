# Methodologie

Deze sectie legt per model uit **hoe het werkt**, **waarom deze keuze is gemaakt** en **wanneer je de output kritisch moet beoordelen**.

## Modellen in het ensemble

| Model | Pagina | Rol in de pipeline |
|-------|--------|--------------------|
| SARIMA / ETS / Theta / AutoARIMA | [Tijdreeksmodellen](sarima.md) | Tijdreeksextrapolatie op basis van historische aanmeldpatronen |
| XGBoost classifier | [XGBoost](xgboost.md) | Kans per individuele student dat deze zich inschrijft |
| XGBoost / Ridge / Random Forest | [Regressiemodellen](xgboost.md) | Vertaling van vooraanmelders naar verwachte inschrijvingen |
| Ratio-model | [Ratio-model](ratio-model.md) | Eenvoudige historische ratio als referentiemodel |
| Ensemble | [Ensemble](ensemble.md) | Gewogen combinatie van bovenstaande modellen |
| Benchmarks | [Benchmarks](benchmarks.md) | Vergelijking van alternatieve modellen |

Het cumulatieve spoor is configureerbaar: via `model_config.cumulative_timeseries` en `model_config.cumulative_regressor` in de [configuratie](../configuratie.md#cumulative_timeseries) kies je welk tijdreeksmodel en welk regressiemodel wordt gebruikt.

## Datasporen

```mermaid
flowchart LR
    SL["Studielink\ntelbestanden"] --> CUM["Cumulatief spoor\n(-d c)"]
    SIS["Osiris / Usis\nper-student"] --> IND["Individueel spoor\n(-d i)"]
    CUM & IND --> ENS["Ensemble\n(-d b)"]
```

De twee sporen zijn bewust onafhankelijk van elkaar ontworpen zodat instellingen die geen toegang hebben tot individuele aanmelddata toch een voorspelling kunnen maken via het cumulatieve spoor.

## Aannames en beperkingen

- Het model extrapoleert op basis van historische patronen. **Structurele breuken** (bijv. nieuwe opleiding, COVID-jaar) worden niet automatisch gedetecteerd.
- Ensemble-gewichten worden bepaald op historische fouten; een model dat in het verleden goed presteerde krijgt meer gewicht, ook al is de situatie veranderd.
- De SARIMA-parameters zijn per opleiding gefixed. Bij opleidingen met weinig historische data is de modelfit minder betrouwbaar.

## Dashboard-visualisatie

Na het opslaan van de resultaten genereert de pipeline een interactief Plotly-dashboard per modus. Het dashboard biedt grafieken per opleiding (voorspellingen, foutmaten, feature importance) en wordt opgeslagen als zelfstandig HTML-bestand onder `data/output/visualisaties/`. Zie [Output begrijpen](../output-begrijpen.md#interactief-dashboard) voor details.
