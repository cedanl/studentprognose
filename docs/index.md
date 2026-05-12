# Studentprognose

**Studentprognose** is een open-source tool van CEDA/Npuls voor het voorspellen van eerstejaars instroom in het hoger onderwijs. Het combineert tijdreeksmodellen (SARIMA), machine learning (XGBoost) en een ratio-model tot een ensemble-voorspelling.

## Voor wie is deze documentatie?

Deze documentatie richt zich op **data-analisten en onderzoekers** bij Nederlandse onderwijsinstellingen. De focus ligt niet alleen op *hoe* je de tool gebruikt, maar ook op *waarom* het model werkt zoals het werkt — inclusief aannames, beperkingen en situaties waarin je de output kritisch moet interpreteren.

## Snelstart

Vereisten: **Python 3.12+** ([installatie-instructies](aan-de-slag.md#voordat-je-begint))

```bash
uv tool install studentprognose
```

Heb je [uv](https://docs.astral.sh/uv/) nog niet, of werk je liever met pip? Zie [Aan de slag → Installatie](aan-de-slag.md#installatie).

Daarna:

```bash
studentprognose init      # mapstructuur + configuratie aanmaken
studentprognose --help    # alle opties bekijken
```

Zie [Aan de slag](aan-de-slag.md) voor een complete walkthrough inclusief data klaarzetten en je eerste run.

!!! tip "Heb je al verwerkte data?"
    Als je verwerkte bestanden hebt die voldoen aan het [verwachte schema](je-data-voorbereiden.md#verwerkte-inputbestanden-data-input) (kolommen, types, namen), kun je de ETL overslaan met `--noetl`. Zie [ETL overslaan](je-data-voorbereiden.md#etl-overslaan) voor de exacte vereisten.

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

!!! info "Verhouding tot de Radboud-implementatie"
    De productie-implementatie van Radboud draait intern en is niet publiek toegankelijk. Deze CEDA-versie is de publieke, generieke variant: dezelfde methodologie, maar zonder Radboud-specifieke configuratie.

    - **Uitlegbaarheid** — methodologische keuzes (features, modelparameters, ensemble-gewichten) zijn onderbouwd vanuit een concrete instellingscontext en hier gedocumenteerd.
    - **Overdraagbaarheid** — Radboud-specifieke logica is in deze versie generiek gemaakt zodat andere instellingen er direct mee aan de slag kunnen.
