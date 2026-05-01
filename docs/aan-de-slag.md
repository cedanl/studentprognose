# Aan de slag

## Installatie

```bash
pip install studentprognose
```

Of met uv in een project:

```bash
uv add studentprognose
```

Vereisten: Python 3.12+

## Eerste keer: mapstructuur aanmaken

Na installatie maak je in een lege werkmap de benodigde mappen en een startconfiguratie aan:

```bash
studentprognose init
```

Dit schrijft:

- `configuration/configuration.json` — aanpasbare configuratie met alle standaardwaarden
- `configuration/filtering/base.json` — lege filtering (geen opleiding- of herkomstfilter)
- `data/input_raw/telbestanden/` — map voor je Studielink-telbestanden
- `data/input_raw/README.md` — beschrijving van welke bestanden hier horen

Bestaat een bestand al, dan wordt het overgeslagen. Je kunt `init` dus veilig opnieuw uitvoeren.

## Je data neerzetten

Zet je inputbestanden in de juiste mappen voordat je de pipeline start:

```
data/
├── input/                          ← verwerkte inputbestanden (na ETL)
│   ├── vooraanmeldingen_cumulatief.csv
│   ├── vooraanmeldingen_individueel.csv
│   ├── student_count_first-years.xlsx
│   ├── student_count_higher-years.xlsx
│   └── student_volume.xlsx
└── input_raw/                      ← ruwe bronbestanden (voor ETL)
    ├── telbestanden/               ← Studielink telbestanden
    │   ├── telbestandY2024W01.csv
    │   └── ...
    ├── individuele_aanmelddata.csv ← Osiris/Usis export
    └── oktober_bestand.xlsx        ← DUO oktober-bestand
```

Zie [Je data voorbereiden](je-data-voorbereiden.md) voor kolomspecificaties per bestand.

## Eerste run

Als je `-w` en `-y` weglaat, kiest de pipeline automatisch de **laatste beschikbare week en het laatste jaar uit je inputdata**. Je ziet dan een melding zoals:

```
Geen week/jaar opgegeven — automatisch gekozen op basis van beschikbare data: jaar 2024, week 38.
Geef -w en -y mee om een andere combinatie te gebruiken.
```

Dit voorkomt dat de tool faalt omdat de huidige systeemweek/-jaar nog niet in je trainingsdata zit.

```bash
# Beide sporen + ensemble (standaard) — automatisch laatste beschikbare week/jaar
uv run studentprognose

# Specifieke week en jaar
uv run studentprognose -w 10 -y 2025

# Alleen cumulatief spoor (geen individuele data nodig)
uv run studentprognose -d c

# ETL overslaan (data al eerder verwerkt)
uv run studentprognose --noetl
```

De output verschijnt in `data/output/`.

## CLI-referentie

Alle subcommando's en vlaggen zijn te bekijken via:

```bash
studentprognose --help
```

### Subcommando's

| Commando | Beschrijving |
|----------|-------------|
| `init` | Maak een nieuwe projectmap aan met configuratie en mappenstructuur |

### Vlaggen

| Vlag | Waarden | Standaard | Beschrijving |
|------|---------|-----------|-------------|
| `-w` | weeknummer(s) | laatste week in data | Voorspelweek(en), bijv. `-w 10` of `-w 8:12` |
| `-y` | jaar(en) | laatste jaar in data | Voorspeljaar(en), bijv. `-y 2025` of `-y 2024 2025` |
| `-d` | `b` / `c` / `i` | `b` | Dataset: `both`, `cumulative`, `individual` |
| `-sy` | `f` / `h` / `v` | `f` | Studentjaar: `first-years`, `higher-years`, `volume` |
| `-c` | pad | `configuration/configuration.json` | Configuratiebestand |
| `-f` | pad | `configuration/filtering/base.json` | Filterbestand |
| `-sk` | getal | `0` | Skip N jaren (backtesting) |
| `--noetl` | — | uit | Sla ETL én validatie over |
| `--yes` | — | uit | Sla validatieprompts over (voor CI/CD) |
| `--ci test N` | getal | — | Testmodus: beperkt tot N opleidingen |

Weekbereiken zijn mogelijk: `-w 8:12` is gelijk aan `-w 8 9 10 11 12`.

## Gebruik als Python-package

Naast de CLI kun je `studentprognose` ook direct vanuit Python-scripts importeren. Dit is handig voor geautomatiseerde pipelines, notebooks, of cloudworkflows waarbij de data al in-memory beschikbaar is.

### Beschikbare bouwstenen

```python
from studentprognose import (
    load_configuration,            # laad configuration.json (of package defaults)
    load_filtering,                # laad filtering JSON (of package defaults)
    load_data,                     # laad data vanaf schijf als DataFrames
    run_pipeline_cli,              # volledige CLI-pipeline (accepteert argv-lijst)
    run_pipeline_from_dataframes,  # pipeline met DataFrames in-memory
    PipelineConfig,                # configuratie-dataclass voor de pipeline
    DataOption,                    # enum: INDIVIDUAL / CUMULATIVE / BOTH_DATASETS
    StudentYearPrediction,         # enum: FIRST_YEARS / HIGHER_YEARS / VOLUME
)
```

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
    save_output=False,  # geen lokale uitvoerbestanden aanmaken
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

## Bekende valkuil: stille modus-downgrade

!!! warning "Let op bij `-d b`"
    Als je `-d both` (standaard) gebruikt maar de individuele aanmelddata ontbreekt of
    bevat niet de verwachte jaren, **valt de pipeline terug op `-d cumulative`**.
    De tool toont dan een waarschuwing, maar de output bevat alleen cumulatieve
    voorspellingen — geen ensemble, geen `SARIMA_individual`.

    Controleer altijd of `SARIMA_individual` kolommen aanwezig zijn in je output als je
    het ensemble verwacht.
