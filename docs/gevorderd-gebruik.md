# Gevorderd gebruik

Deze pagina bundelt de gevorderde onderwerpen die je **niet** nodig hebt voor een gewone prognose: de tool aanroepen vanuit Python (notebooks, cloud), hyperparameters afstemmen, en het model evalueren met scalaire metrieken. Voor de dagelijkse workflow — zie de [Snelstart](snelstart.md) en [Draaien & CLI](aan-de-slag.md).

!!! tip "Heb je dit nodig?"
    Wil je gewoon een prognose draaien via de command line? Dan kun je deze pagina overslaan. Ze is bedoeld voor gebruikers die de tool in een eigen (cloud)pipeline inbouwen, of die de modellen willen afstemmen en evalueren.

## Gebruik als Python-package

Naast de CLI kun je `studentprognose` ook direct vanuit Python-scripts importeren. Dit is handig voor geautomatiseerde pipelines, notebooks, of cloudworkflows waarbij de data al in-memory beschikbaar is.

!!! tip "Uitvoerbaar voorbeeld — implementatienotebook"
    Voor een direct inpasbaar startpunt voor MS Fabric, Databricks of Azure: zie [`notebooks/implementatie.ipynb`](https://github.com/cedanl/studentprognose/blob/main/notebooks/implementatie.ipynb). Vereist alleen `pip install studentprognose` — geen repo-clone nodig.

### Beschikbare bouwstenen

```python
from studentprognose import (
    load_configuration,            # laad configuration.json (of package defaults)
    load_filtering,                # laad filtering JSON (of package defaults)
    load_data,                     # laad data vanaf schijf als DataFrames
    run_pipeline_cli,              # volledige CLI-pipeline (accepteert argv-lijst)
    run_pipeline_from_dataframes,  # pipeline met DataFrames in-memory
    build_dashboard_from_dataframes,  # interactieve HTML-dashboards uit DataFrames
    evaluate_predictions,          # output → scalaire evaluatiemetrieken per model
    pivot_metrics,                 # metrieken → model × jaar/week-matrix (backtest)
    to_mlflow_metrics,             # metrieken afvlakken voor mlflow.log_metrics (Fabric)
    PipelineConfig,                # configuratie-dataclass voor de pipeline
    DataOption,                    # enum: INDIVIDUAL / CUMULATIVE / BOTH_DATASETS
    StudentYearPrediction,         # enum: FIRST_YEARS / VOLUME
)
```

!!! tip "Het model evalueren"
    `evaluate_predictions` zet de pipeline-output om in scalaire metrieken (MAE, MAPE,
    WAPE, RMSE, bias, R²) per model; `pivot_metrics` vat een backtest over meerdere jaren
    samen tot een model × jaar-matrix; en `to_mlflow_metrics` levert ze klaar voor MLflow
    in MS Fabric / Databricks. Zie [Het model evalueren](#het-model-evalueren).

### Minimaal werkend voorbeeld (bestandsgebaseerd)

```python
from studentprognose import run_pipeline_cli

# Zelfde als de CLI — start de volledige pipeline inclusief ETL
run_pipeline_cli(["studentprognose", "-d", "c", "-y", "2025", "-w", "10"])

# Of sla ETL over als de data al verwerkt is
run_pipeline_cli(["studentprognose", "--noetl", "-d", "c", "-y", "2025", "-w", "10"])
```

### Cloud-gebruik (data al in-memory)

Gebruik `run_pipeline_from_dataframes` als je de data al geladen hebt, bijvoorbeeld vanuit Azure Blob Storage of Amazon S3. De ETL-stap wordt volledig overgeslagen.

```python
import pandas as pd
from studentprognose import run_pipeline_from_dataframes, DataOption

# Laad data uit cloud-opslag (voorbeeld met Azure SDK)
# from azure.storage.blob import BlobServiceClient
# blob_data = blob_client.download_blob().readall()
# df_cum = pd.read_csv(io.BytesIO(blob_data), sep=";", skiprows=[1])

# Of lokaal voor testen
df_cum = pd.read_csv("vooraanmeldingen_cumulatief.csv", sep=";", skiprows=[1])

result = run_pipeline_from_dataframes(
    year=2025,
    week=10,
    data_cumulative=df_cum,
    dataset=DataOption.CUMULATIVE,
    save_output=False,  # geen lokale uitvoerbestanden aanmaken (ook geen tussenresultaat)
)

if result is not None:
    print(result[["Croho groepeernaam", "Weighted_ensemble_prediction"]].head())
```

De functie accepteert ook een eigen configuratiedict, zodat je geen bestandssysteem nodig hebt:

```python
from studentprognose import run_pipeline_from_dataframes, DataOption
from studentprognose.config import load_defaults

# Begin met package-defaults en pas aan
config = load_defaults()
config["numerus_fixus"] = ["B Geneeskunde", "B Tandheelkunde"]

result = run_pipeline_from_dataframes(
    year=2025,
    week=10,
    data_cumulative=df_cum,
    configuration=config,
    dataset=DataOption.CUMULATIVE,
)
```

!!! warning "`year`/`week` moeten binnen je trainingsdata vallen"
    `run_pipeline_from_dataframes` controleert — net als de CLI — of `year` en `week`
    voorkomen in de meegegeven DataFrames. Valt een van beide buiten bereik, dan krijg je
    een heldere `ValueError` met de wél beschikbare range, in plaats van een stille `None`
    of een onverwachte kernel-crash in een notebook.

    ```python
    # Stel: je data loopt t/m 2025
    run_pipeline_from_dataframes(
        year=2030,            # buiten bereik
        week=10,
        data_cumulative=df_cum,
        dataset=DataOption.CUMULATIVE,
    )
    # ValueError: year=2030 valt buiten de beschikbare trainingsdata.
    #   Beschikbare data: jaren 2018-2025, weken 1-52.
    #   Pas year/week aan binnen deze range, of voeg trainingsdata toe.
    ```

    Pas `year`/`week` aan binnen de range, of voeg aanvullende trainingsdata toe.

### Dashboards genereren vanuit DataFrames

De interactieve Plotly-dashboards (die de CLI met `--dashboard` maakt) kun je vanuit een
notebook of cloud-run genereren met `build_dashboard_from_dataframes`. Dit is een
**aparte functie** die je ná (of in plaats van) `run_pipeline_from_dataframes` aanroept,
met dezelfde in-memory DataFrames. Ze draait de pipeline (zonder iets naar schijf te
schrijven) en schrijft daarna de dashboards weg naar `<cwd>/data/output/visualisations/`.

```python
from studentprognose import build_dashboard_from_dataframes, DataOption

output_dir = build_dashboard_from_dataframes(
    year=2025,
    week=10,
    data_cumulative=df_cum,
    data_student_numbers=sc,   # nodig voor de realisatie-/conversiegrafieken
    dataset=DataOption.CUMULATIVE,
    configuration=config,
)
print(f"Dashboards geschreven naar {output_dir}")
# -> <cwd>/data/output/visualisations/{individual,cumulative,final}/dashboard.html
```

!!! tip "Losse functie, geen extra parameter"
    Het dashboard is bewust een aparte functie en géén parameter op
    `run_pipeline_from_dataframes`: zo blijft de voorspelfunctie ongewijzigd en is de
    dashboard-stap los te draaien en te testen. Gebruik `cwd` om de uitvoermap te sturen.
    De `year`/`week`-rangecontrole is identiek aan die van `run_pipeline_from_dataframes`;
    komt geen enkele rij door de filters, dan krijg je een heldere `ValueError` in plaats
    van een leeg dashboard.

## Hyperparameter tuning

Het cumulatieve spoor is twee-traps en je kunt **beide trappen** afstemmen op je
eigen data:

| Trap | Wat | Wat tuning kiest | Vastgelegd in |
|------|-----|------------------|---------------|
| **Stap 1 — SARIMA** | extrapoleert de vooraanmeldcurve | ARIMA-ordes `(p,d,q)(P,D,Q,s)` | `forecaster_params.sarima` |
| **Stap 2 — regressor** | vertaalt de curve naar inschrijvingen | hyperparameters (default XGBoost) | `regressor_params.<naam>` |

Beide gebruiken een **tijd-bewuste** zoektocht (geen random k-fold — dat lekt
toekomst) met dezelfde MAPE-metriek als de [benchmark](methodologie/benchmarks.md),
en laten het operationele voorspelpad standaard onaangeroerd. Zie
[XGBoost → Hyperparameter tuning](methodologie/xgboost.md#hyperparameter-tuning)
(regressor) en [SARIMA → Orde-selectie](methodologie/sarima.md#orde-selectie-tuning)
voor de methodologie.

### Via de CLI — zoeken en vastleggen

```bash
studentprognose tune -d c -w 12                    # default: regressor (stap 2)
studentprognose tune -d c -w 12 --tune-target sarima   # alleen SARIMA (stap 1)
studentprognose tune -d c -w 12 --tune-target both     # beide trappen
```

Dit draait de zoektocht op je geladen data, print per trap een overzicht van alle
geteste sets met hun MAPE (de best presterende set gemarkeerd met `✓`), en geeft
een kant-en-klaar config-snippet terug. Plak dat snippet in `configuration.json`
om de gevonden waarden vast te leggen (zie
[Configuratie → vastleggen](configuratie-referentie.md#waarden-vastleggen)).
Vastgelegde waarden zijn reproduceerbaar: ze worden bij elke run gebruikt zonder
opnieuw te tunen.

### Via de Python-API — tunen en voorspellen in één keer

`run_pipeline_from_dataframes` heeft een `tune`-parameter (standaard `False`) waarmee
je kiest **welke trap(pen)** getuned worden:

```python
result = run_pipeline_from_dataframes(
    year=2025,
    week=10,
    data_cumulative=df_cum,
    data_student_numbers=data_studentcount,
    dataset=DataOption.CUMULATIVE,
    tune="both",  # tune stap 1 (SARIMA) én stap 2 (regressor), daarna voorspellen
)
```

De ondersteunde waarden:

| `tune=` | Wat wordt getuned |
|---------|-------------------|
| `False` *(default)* | niets — gebruikt vastgelegde/default-waarden |
| `True` of `"regressor"` | alleen de regressor (stap 2) |
| `"sarima"` | alleen SARIMA-ordes (stap 1) |
| `"both"` | beide trappen, elk apart gelogd |
| `{"regressor": {...}, "sarima": {...}}` | beide/één trap met **eigen zoekruimte** (waarde weglaten of `None` = ingebouwde grid) |

Elke gekozen trap draait de zoektocht éénmaal, voorspelt met de beste waarden en
koppelt ze terug in de configuratie zodat je ze kunt vastleggen. Het overzicht (met
`✓` op de winnaar) gaat naar de console; de functie **retourneert het
voorspellings-DataFrame**, niet het tuning-resultaat.

Bij `tune="both"` zie je twee aparte tabellen — één per trap, elk met een eigen `✓`:

```
           MAPE   Folds  Parameters
  ─────────────────────────────────
  ✓      0.0912       6  {"order": [1, 1, 1], "seasonal_order": [1, 1, 0, 52]}
         0.1041       6  {"order": [1, 0, 1], "seasonal_order": [1, 1, 1, 52]}

Beste parameters voor 'sarima' (MAPE=0.0912, 6 kandidaten):
Plak dit in je configuration.json om de ordes vast te leggen:
{ "model_config": { "forecaster_params": { "sarima": { ... } } } }

           MAPE   Folds  Parameters
  ─────────────────────────────────
  ✓      0.1424      12  {"learning_rate": 0.25, "n_estimators": 200, "max_depth": 5}
         0.1487      12  {"learning_rate": 0.1, "n_estimators": 200, "max_depth": 3}

Beste parameters voor 'xgboost' (MAPE=0.1424, 12 kandidaten):
Plak dit in je configuration.json om de parameters vast te leggen:
{ "model_config": { "regressor_params": { "xgboost": { ... } } } }
```

!!! note "Tuning is opt-in"
    Standaard (`tune=False`) gebruikt de pipeline de waarden uit
    `model_config.regressor_params` / `forecaster_params` of de modeldefaults —
    snel en reproduceerbaar. Zet `tune` alleen aan wanneer je bewust wilt afstemmen;
    het maakt de run langzamer en, zonder de uitkomst vast te leggen,
    niet-deterministisch.

## Het model evalueren

De per-rij `MAE_*`/`MAPE_*`-kolommen in de [output](output-begrijpen.md#foutmaatkolommen) zijn handig om één rij te lezen, maar voor **modelevaluatie** wil je geaggregeerde, scalaire metrieken (één getal per model). Daarvoor levert het pakket `evaluate_predictions`:

```python
from studentprognose import run_pipeline_from_dataframes, evaluate_predictions, DataOption

result = run_pipeline_from_dataframes(
    year=2023, week=12,
    data_cumulative=df_cum, data_student_numbers=df_sc,
    dataset=DataOption.CUMULATIVE, save_output=False,
)

metrics = evaluate_predictions(result, week=12)
print(metrics)
```

De functie vergelijkt elke voorspelkolom met de gerealiseerde `Aantal_studenten` en geeft per model één rij terug met:

| Metriek | Betekenis | Interpretatie |
|---------|-----------|---------------|
| `n` | aantal meegetelde (opleiding × herkomst)-rijen | klein `n` → metriek is ruisgevoelig |
| `mae` | Mean Absolute Error | gemiddelde afwijking in studenten |
| `mape` | Mean Absolute Percentage Error (fractie) | elke opleiding telt even zwaar — kleine opleidingen domineren |
| `wape` | Weighted APE = `Σ\|fout\| / Σ\|werkelijk\|` (fractie) | instellingsbrede procentuele fout, robuust tegen kleine opleidingen |
| `rmse` | Root Mean Squared Error | straft grote uitschieters zwaarder |
| `bias` | gemiddelde `voorspeld − werkelijk` | positief = structurele overschatting |
| `r2` | determinatiecoëfficiënt | aandeel verklaarde variantie |

`mape` en `wape` zijn fracties (`0.08` betekent 8%). De metrieken gebruiken dezelfde primitieven en numerus-fixus-uitsluiting als de `MAE_*`-kolommen, dus ze zijn consistent met de rest van de output.

!!! warning "Evalueren kan alleen via een backtest"
    Een model evalueren betekent voorspellingen vergelijken met de **gerealiseerde** instroom. Voor het lopende, nog niet afgeronde collegejaar bestaat die realisatie nog niet — `Aantal_studenten` is dan leeg en `evaluate_predictions` geeft een foutmelding. Evalueer daarom een **afgerond** collegejaar (bijv. voorspel `year=2023` terwijl je de realisatie van 2023 als `data_student_numbers` meegeeft). De modellen trainen sowieso alleen op jaren vóór het voorspeljaar, dus zo'n backtest lekt geen toekomstinformatie.

!!! tip "Geef altijd `week` mee bij cumulatieve output"
    In het cumulatieve spoor draagt alleen de **peilweekrij** de `SARIMA_cumulative`-voorspelling; latere weken bevatten enkel de vooraanmeldcurve (met `NaN` als voorspelling). Zonder `week=`-filter mengt `Prognose_ratio` bovendien meerdere weken. Geef de gebruikte peilweek mee (`evaluate_predictions(result, week=12)`) voor een zuivere, één-rij-per-opleiding vergelijking tussen modellen.

Segmenteer met `group_by` (bijv. `group_by="Examentype"` of `group_by=["Examentype", "Herkomst"]`) om te zien waar een model structureel afwijkt.

### Backtesten over meerdere jaren (`pivot_metrics`)

Eén afgerond jaar is **één datapunt**: de instroom schommelt jaar-op-jaar om redenen buiten het model. Voor een eerlijk beeld backtest je daarom meerdere afgeronde jaren op dezelfde peilweek, evalueer je per jaar met `group_by="Collegejaar"`, en draai je het resultaat met `pivot_metrics` tot een model × jaar-matrix:

```python
import pandas as pd
from studentprognose import run_pipeline_from_dataframes, evaluate_predictions, pivot_metrics, DataOption

WEEK = 10
frames = [
    run_pipeline_from_dataframes(
        year=jaar, week=WEEK, data_cumulative=df_cum,
        data_student_numbers=df_sc, dataset=DataOption.CUMULATIVE, save_output=False,
    )
    for jaar in (2021, 2022, 2023)            # afgeronde jaren met realisatie
]

metrics = evaluate_predictions(pd.concat(frames, ignore_index=True),
                               week=WEEK, group_by="Collegejaar")
print(pivot_metrics(metrics, value="wape", over="Collegejaar").round(3))
```

```text
                    2021   2022   2023   mean    min    max  n_groups
prediction
Prognose_ratio     0.078  0.071  0.069  0.073  0.069  0.078         3
SARIMA_cumulative  0.063  0.058  0.066  0.062  0.058  0.066         3
```

De `mean`-kolom is het gemiddelde over de jaren; `min`/`max` tonen de spreiding. Ligt de jaar-op-jaar spreiding in dezelfde orde als het verschil tussen modellen, trek dan geen harde conclusie uit één jaar. Kies met `value=` een andere metriek (`"mae"`, `"mape"`, …) en met `over=` een andere as (bijv. `over="Weeknummer"` voor de accuraatheid-over-tijd-curve bij één jaar).

### Metrieken loggen in MS Fabric / Databricks (MLflow)

Beide platforms hebben **MLflow** als ingebouwde experiment-tracker. `to_mlflow_metrics` vlakt het resultaat af tot een dict die je direct kunt loggen, zodat de metrieken in de Fabric/Databricks ML-experiment-UI verschijnen en over runs (jaar/peilweek) vergelijkbaar zijn:

```python
import mlflow
from studentprognose import evaluate_predictions, to_mlflow_metrics

metrics = evaluate_predictions(result, week=WEEK)

with mlflow.start_run(run_name=f"prognose_{YEAR}_wk{WEEK}"):
    mlflow.log_params({"year": YEAR, "week": WEEK, "dataset": "cumulatief"})
    mlflow.log_metrics(to_mlflow_metrics(metrics))
    mlflow.log_table(metrics, "metrics.json")  # volledige tabel als artifact
```

`mlflow` zit standaard in de Fabric- en Databricks-runtime; in een Fabric-notebook worden runs automatisch aan het gekoppelde experiment gehangen. Wil je geen MLflow gebruiken, dan kun je het `metrics`-DataFrame ook gewoon naar een Lakehouse/Delta-tabel wegschrijven voor je eigen monitoring.
