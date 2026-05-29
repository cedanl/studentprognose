# Aan de slag

## Voordat je begint

Je hebt **Python 3.12** nodig. Controleer je versie:

```bash
python --version
```

!!! warning "Alleen Python 3.12 wordt ondersteund"
    De pipeline wordt op één Python-versie ontwikkeld en getest. Nieuwere minor-versies (3.13, 3.14) zijn bewust uitgesloten — onder andere omdat `statsforecast` voor sommige minors nog geen Windows-wheels publiceert, wat tot een C-compiler-error tijdens `uv sync` leidt. Pin je Python-versie expliciet op 3.12:

    ```bash
    uv python pin 3.12
    ```

    `uv` downloadt zelf de juiste versie als die nog niet op je systeem staat — je hoeft niks van python.org te halen.

??? tip "Python niet gevonden of te oud?"

    Als je `python: command not found` ziet, of een versie lager dan 3.12:

    - **Windows**: Download Python via [python.org](https://www.python.org/downloads/). Vink bij installatie **"Add python.exe to PATH"** aan.
    - **macOS**: Installeer via [python.org](https://www.python.org/downloads/) of met Homebrew: `brew install python@3.12`
    - **Linux**: Gebruik je pakketbeheerder, bijv. `sudo apt install python3.12` (Ubuntu/Debian) of `sudo dnf install python3.12` (Fedora).

    Op sommige systemen heet het commando `python3` in plaats van `python`. Probeer `python3 --version` als `python --version` niet werkt.

## Installatie

We gebruiken [uv](https://docs.astral.sh/uv/) — een snelle Python-pakketbeheerder die virtual environments automatisch afhandelt.

**Stap 1 — Installeer uv** (eenmalig):

=== "macOS / Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

**Stap 2 — Installeer studentprognose:**

```bash
uv tool install studentprognose
```

**Updaten** naar een nieuwere versie:

```bash
uv tool upgrade studentprognose
```

## Eerste keer: mapstructuur aanmaken

Maak een werkmap aan voor je prognoseproject en draai daar `init`:

```bash
mkdir mijn-prognose
cd mijn-prognose
studentprognose init
```

Dit schrijft:

- `configuration/configuration.json` — aanpasbare configuratie met alle standaardwaarden
- `configuration/filtering/base.json` — lege filtering (geen opleiding- of herkomstfilter)
- `data/input_raw/telbestanden/` — map voor je Studielink-telbestanden
- `data/input_raw/README.md` — beschrijving van welke bestanden hier horen
- `data/output/` — map waarin de wekelijkse outputs én de doorlopende audittrail (`_totaal_*.xlsx`) worden opgeslagen

Bestaat een bestand al, dan wordt het overgeslagen. Je kunt `init` dus veilig opnieuw uitvoeren.

## Je data neerzetten

Zet je inputbestanden in de juiste mappen voordat je de pipeline start:

```text
data/
├── input/                          ← verwerkte inputbestanden (na ETL)
│   ├── vooraanmeldingen_cumulatief.csv
│   ├── vooraanmeldingen_individueel.csv
│   ├── student_count_first-years.xlsx
│   └── student_volume.xlsx
└── input_raw/                      ← ruwe bronbestanden (NIET nodig met --noetl)
    ├── telbestanden/               ← Studielink telbestanden
    │   ├── telbestandY2024W01.csv
    │   └── ...
    ├── individuele_aanmelddata.csv ← Osiris/Usis export
    └── oktober_bestand.xlsx        ← telbestand studenten
```

Zie [Je data voorbereiden](je-data-voorbereiden.md) voor kolomspecificaties per bestand.

## ETL overslaan met `--noetl`

**Heb je al verwerkte bestanden in `data/input/`** (bijv. uit een eigen ETL-proces, een eerdere run, of aangeleverd door een collega)? Dan hoef je de ruwe bronbestanden in `data/input_raw/` helemaal niet neer te zetten. Sla de ETL én de bijbehorende validatie over met `--noetl`:

```bash
studentprognose --noetl
```

Welke bestanden je dan precies in `data/input/` nodig hebt — en in welk formaat — lees je op [ETL overslaan](je-data-voorbereiden.md#etl-overslaan).

!!! warning "Wanneer je `--noetl` níét moet gebruiken"
    Heb je alleen ruwe bronbestanden (Studielink-telbestanden, Osiris/Usis-export, telbestand studenten)? Laat `--noetl` dan weg, zodat de ingebouwde ETL de verwerkte bestanden voor je aanmaakt.

## Wat voorspelt `-w 16 -y 2025`?

Voor `studentprognose` is `-w` de **peilweek** (kalenderweek 1–52) en `-y` het **collegejaar** waarvoor je een prognose wilt. Met `-w 16 -y 2025` zeg je dus: *"gebruik alle vooraanmelddata tot en met week 16 van collegejaar 2025 en geef me een prognose voor het uiteindelijke aantal inschrijvingen voor datzelfde collegejaar"*.

!!! example "Voorbeeld-run · B Bedrijfskunde · NL (demodata)"
    **178** vooraanmelders @ wk 16 (peilmoment) &nbsp;→&nbsp; **520** verwacht @ wk 38 &nbsp;→&nbsp; **400** ingeschreven (77% yield)

=== "Week voor week"

    <iframe src="../assets/plots/whatif_timeline.html" width="100%" height="540" frameborder="0" style="border-radius: 8px;"></iframe>

    De blauwe lijn toont de geobserveerde cumulatieve vooraanmelders tot peilmoment (wk 16). De oranje gestippelde lijn extrapoleert naar week 38 — met scherp knikpunt rond de 1-mei-deadline en een late-zomer-naloop. De gearceerde band groeit met de horizon: ±12 bij peilmoment, ±34 bij wk 38.

=== "In één oogopslag"

    <iframe src="../assets/plots/whatif_slope.html" width="100%" height="420" frameborder="0" style="border-radius: 8px;"></iframe>

    Drie kerngetallen op één schaal: peilmoment → SARIMA-extrapolatie naar wk 38 → verwachte inschrijvingen. De badges tussen de punten kwantificeren de overgangen — eerst de groei (×2.92 in vooraanmelders), daarna de yield (77% conversie naar ingeschreven).

!!! info "Waarom elke week opnieuw draaien?"
    Een prognose op één peilweek geeft je het beeld van *dat* moment. Door de pipeline wekelijks te draaien volg je de trends en fluctuaties in vooraanmeldingen mee — bijvoorbeeld het knikpunt rond de 1-mei-deadline of een achterblijvende naloop in de zomer. Zo zie je vroege signalen om je capaciteits- en marketingbeslissingen tijdig bij te sturen, in plaats van pas na het seizoen één eindprognose te hebben.

!!! tip "Andere peilweek of collegejaar?"
    Vervang `-w 16` door je eigen peilweek (1–52) en `-y 2025` door het collegejaar dat je wilt voorspellen. De pipeline pakt automatisch de data tot en met die week om het uiteindelijke aantal inschrijvingen voor datzelfde collegejaar te schatten.

## Eerste run

Als je `-w` en `-y` weglaat, kiest de pipeline automatisch de **laatste beschikbare week en het laatste jaar uit je inputdata**. Je ziet dan een melding zoals:

```
Geen week/jaar opgegeven — automatisch gekozen op basis van beschikbare data: jaar 2024, week 38.
Geef -w en -y mee om een andere combinatie te gebruiken.
```

Dit voorkomt dat de tool faalt omdat de huidige systeemweek/-jaar nog niet in je trainingsdata zit.

```bash
# Beide sporen + ensemble (standaard) — automatisch laatste beschikbare week/jaar
studentprognose

# Specifieke week en jaar
studentprognose -w 10 -y 2025

# Alleen cumulatief spoor (geen individuele data nodig)
studentprognose -d c

# ETL overslaan (data al eerder verwerkt)
studentprognose --noetl
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
| `benchmark` | Vergelijk alternatieve ML-modellen (`-d c` of `-d i` verplicht, zie [Benchmarks](methodologie/benchmarks.md)) |

### Vlaggen

| Vlag | Waarden | Standaard | Beschrijving |
|------|---------|-----------|-------------|
| `-w` | weeknummer(s) | laatste week in data | Voorspelweek(en), bijv. `-w 10` of `-w 8:12` |
| `-y` | jaar(en) | laatste jaar in data | Voorspeljaar(en), bijv. `-y 2025` of `-y 2024 2025` |
| `-d` | `b` / `c` / `i` | `b` | Dataset: `both`, `cumulative`, `individual` |
| `-sy` | `f` / `v` | `f` | Studentjaar: `first-years` (standaard), `volume` |
| `-c` | pad | `configuration/configuration.json` | Configuratiebestand |
| `-f` | pad | `configuration/filtering/base.json` | Filterbestand |
| `-sk` | getal | `0` | Skip N jaren (backtesting) |
| `--noetl` | — | uit | Sla ETL én validatie over |
| `--yes` | — | uit | Sla validatieprompts over (voor CI/CD) |
| `--dashboard` | — | uit | Genereer interactieve Plotly-dashboards (`data/output/visualisations/`) |
| `--ci test N` | getal | — | Testmodus: beperkt tot N opleidingen |

Weekbereiken zijn mogelijk: `-w 8:12` is gelijk aan `-w 8 9 10 11 12`.

!!! note "Eerstejaars als focus"
    De tool draait standaard op `-sy f` (eerstejaars). De modus `-sy v` (volume) is een vervolgstap — gebruik die pas als je eerstejaarsvoorspelling op orde is.

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
    PipelineConfig,                # configuratie-dataclass voor de pipeline
    DataOption,                    # enum: INDIVIDUAL / CUMULATIVE / BOTH_DATASETS
    StudentYearPrediction,         # enum: FIRST_YEARS / VOLUME
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

## Veelvoorkomende fouten

??? failure "`python: command not found` of versie te laag"

    Python is niet geïnstalleerd of niet beschikbaar in je PATH. Zie [Voordat je begint](#voordat-je-begint) voor installatie-instructies.

    Op sommige systemen heet het commando `python3` in plaats van `python`.

??? failure "`studentprognose: command not found` na installatie"

    Start je terminal opnieuw op zodat het `studentprognose`-commando beschikbaar wordt. Als het probleem aanhoudt, draai `uv tool update-shell` en open een nieuwe terminal.

??? failure "`ModuleNotFoundError: No module named 'studentprognose'`"

    Je draait Python buiten de omgeving waarin studentprognose is geïnstalleerd. Gebruik `uv run studentprognose ...` om de juiste Python-omgeving te kiezen.

??? failure "`TerminatedWorkerError` tijdens SARIMA (Windows)"

    Een worker-proces is op signaal-niveau gestopt — meestal door geheugendruk wanneer er veel parallelle workers tegelijk draaien op een laptop met beperkt RAM, soms door een crash in onderliggende C-code.

    De pipeline probeert vanaf nu automatisch opnieuw met 2 workers. Werkt dat ook niet, zet dan handmatig in `configuration/configuration.json`:

    ```json
    "runtime": { "cpu_count": 1 }
    ```

    `cpu_count: 1` schakelt parallelisatie helemaal uit. Trager, maar zonder worker-spawns en daarmee zonder deze klasse fouten.

## Andere installatiemethoden

??? note "Installeren met pip in een virtual environment"

    !!! warning "Installeer niet zonder virtual environment"
        Een `pip install` buiten een virtual environment plaatst packages in je systeemomgeving en kan conflicten veroorzaken met andere projecten.

    **Stap 1 — Maak een projectmap en virtual environment aan:**

    ```bash
    mkdir mijn-prognose
    cd mijn-prognose
    python -m venv .venv
    ```

    **Stap 2 — Activeer de virtual environment:**

    === "Windows (PowerShell)"

        ```powershell
        .venv\Scripts\Activate.ps1
        ```

    === "Windows (CMD)"

        ```cmd
        .venv\Scripts\activate.bat
        ```

    === "macOS / Linux"

        ```bash
        source .venv/bin/activate
        ```

    Je ziet nu `(.venv)` voor je prompt — dat bevestigt dat de omgeving actief is.

    **Stap 3 — Installeer studentprognose:**

    ```bash
    pip install studentprognose
    ```

    **Updaten:** `pip install --upgrade studentprognose`

    Elke nieuwe terminal vereist een nieuwe activatie (stap 2) vóór `studentprognose` werkt.

??? note "Bijdragen aan de broncode"

    ```bash
    git clone https://github.com/cedanl/studentprognose.git
    cd studentprognose
    uv run studentprognose --help
    ```

    `uv run` maakt automatisch een virtual environment aan op basis van `pyproject.toml`.
