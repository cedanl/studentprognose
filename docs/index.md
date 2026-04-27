# Studentprognose

**Studentprognose** is een open-source tool van CEDA/Npuls voor het voorspellen van eerstejaars instroom in het hoger onderwijs. Het combineert tijdreeksmodellen (SARIMA), machine learning (XGBoost) en een ratio-model tot een ensemble-voorspelling.

## Voor wie is deze documentatie?

Deze documentatie richt zich op **data-analisten en onderzoekers** bij Nederlandse onderwijsinstellingen. De focus ligt niet alleen op *hoe* je de tool gebruikt, maar ook op *waarom* het model werkt zoals het werkt — inclusief aannames, beperkingen en situaties waarin je de output kritisch moet interpreteren.

## Snelstart

```bash
pip install studentprognose
studentprognose --help
```

Zie [Aan de slag](aan-de-slag.md) voor een complete walkthrough met demodata.

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
