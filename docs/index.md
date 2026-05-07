# Studentprognose

**Studentprognose** is een open-source tool van CEDA/Npuls voor het voorspellen van eerstejaars instroom in het hoger onderwijs. Het combineert tijdreeksmodellen (SARIMA), machine learning (XGBoost) en een ratio-model tot een ensemble-voorspelling.

## Voor wie is deze documentatie?

Deze documentatie richt zich op **data-analisten en onderzoekers** bij Nederlandse onderwijsinstellingen. De focus ligt niet alleen op *hoe* je de tool gebruikt, maar ook op *waarom* het model werkt zoals het werkt — inclusief aannames, beperkingen en situaties waarin je de output kritisch moet interpreteren.

## Snelstart

Vereisten: **Python 3.12+** ([installatie-instructies](aan-de-slag.md#voordat-je-begint))

=== "pipx (aanbevolen)"

    ```bash
    pipx install studentprognose
    ```

=== "pip (in een virtual environment)"

    ```bash
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
    pip install studentprognose
    ```

=== "uv"

    ```bash
    uv tool install studentprognose
    ```

Daarna:

```bash
studentprognose init      # mapstructuur + configuratie aanmaken
studentprognose --help    # alle opties bekijken
```

Zie [Aan de slag](aan-de-slag.md) voor een complete walkthrough inclusief data klaarzetten en je eerste run.

!!! tip "Heb je al verwerkte data?"
    Als je al beschikt over verwerkte bestanden (bijv. uit een eigen ETL-proces of van een collega), kun je de ingebouwde ETL overslaan met `--noetl`. Zie [ETL overslaan](je-data-voorbereiden.md#etl-overslaan) voor details.

## Architectuur op hoofdlijnen

Het model kent drie verwerkingssporen, afhankelijk van de beschikbare data:

| Modus | Vlag | Databron | Modellen |
|-------|------|----------|----------|
| Cumulatief | `-d c` | Studielink telbestanden | SARIMA + XGBoost regressor |
| Individueel | `-d i` | Osiris/Usis per-student | XGBoost classifier + SARIMA |
| Beide | `-d b` | Beide bronnen (standaard) | Volledig ensemble |

Zie [Methodologie](methodologie/index.md) voor een diepgaande uitleg per model.

## In de praktijk

Dit model is oorspronkelijk ontwikkeld door **Radboud Universiteit** en vervolgens samen met CEDA open source gemaakt zodat andere Nederlandse onderwijsinstellingen er ook van kunnen profiteren. Radboud is daarmee de grondlegger van dit project. VOX Nijmegen schreef hierover: [*De universiteit heeft nu haar eigen glazen bol*](https://www.voxweb.nl/nieuws/de-universiteit-heeft-nu-haar-eigen-glazen-bol-nieuw-model-voorspelt-toekomstige-instroom-van-studenten).
