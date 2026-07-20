# Studentprognose

**Studentprognose** voorspelt hoeveel eerstejaars zich uiteindelijk inschrijven — maanden vooruit, op basis van je eigen (voor)aanmelddata. Het is een open-source tool van CEDA/Npuls die tijdreeksmodellen (SARIMA), machine learning (XGBoost) en een ratio-model combineert tot één ensemble-voorspelling.

<iframe src="assets/plots/whatif_funnel.html" width="100%" height="420" frameborder="0" style="border-radius: 8px;"></iframe>

*Van vooraanmelders op een peilmoment naar het verwachte aantal inschrijvingen — voorbeeld op de meegeleverde demodata.*

## Snel beginnen

```bash
uv tool install studentprognose   # zie Installeren voor uv/pip/Windows
studentprognose init              # projectmap aanmaken
studentprognose                   # eerste prognose (op demodata)
```

Er zit **demodata** in de installatie, dus je kunt direct draaien zonder eigen data. Details in **[Installeren](installeren.md)** en de **[Snelstart](snelstart.md)** (5 minuten).

## Wat doet de tool?

Je kiest een **peilweek** en een **collegejaar**, en de tool geeft een prognose van het uiteindelijke aantal inschrijvingen. Onder water lopen tot drie sporen, afhankelijk van je data:

| Modus | Vlag | Databron | Wat het doet |
|-------|------|----------|--------------|
| Cumulatief | `-d c` | Studielink-telbestanden | Trekt de aanmeldcurve door naar het einde van het seizoen |
| Individueel | `-d i` | Osiris/Usis per student | Schat per aanmelder de inschrijfkans, telt die op |
| Beide | `-d b` | Beide bronnen (standaard) | Combineert beide sporen tot één ensemble-voorspelling |

Zie [Methodologie](methodologie/index.md) voor de uitleg per model en [Begrippen](begrippen.md) voor onbekende termen.

## Zo ziet de output eruit

Naast Excel-bestanden genereert de tool desgewenst interactieve dashboards (`--dashboard`):

<iframe src="assets/plots/output_cockpit.html" width="100%" height="400" frameborder="0" style="border-radius: 8px;"></iframe>

Zie [Output lezen](output-begrijpen.md) voor uitleg van elk cijfer en wanneer je een prognose wel of niet moet vertrouwen.

## Voor wie is deze documentatie?

Voor **data-analisten en onderzoekers** bij Nederlandse onderwijsinstellingen. Je hebt géén machine learning-expertise nodig. De focus ligt niet alleen op *hoe* je de tool gebruikt, maar ook op *waarom* het model werkt zoals het werkt — inclusief aannames, beperkingen en wanneer je de output kritisch moet interpreteren.

!!! tip "Liever experimenteren dan lezen?"
    In [`notebooks/`](https://github.com/cedanl/studentprognose/tree/main/notebooks) staan zeven **uitvoerbare** Jupyter-notebooks die de methodologie stap-voor-stap doorlopen op de demodata — van `00_overzicht.ipynb` tot `06_output_interpreteren.ipynb`. Zie de [notebooks-README](https://github.com/cedanl/studentprognose/blob/main/notebooks/README.md).

## In de praktijk

Dit model is oorspronkelijk ontwikkeld door **Radboud Universiteit** en vervolgens samen met CEDA open source gemaakt. VOX Nijmegen schreef hierover: [*De universiteit heeft nu haar eigen glazen bol*](https://www.voxweb.nl/nieuws/de-universiteit-heeft-nu-haar-eigen-glazen-bol-nieuw-model-voorspelt-toekomstige-instroom-van-studenten).

!!! info "Verhouding tot de Radboud-implementatie"
    De productie-implementatie van Radboud draait intern en is niet publiek. Deze CEDA-versie is de publieke, generieke variant: dezelfde methodologie, maar zonder Radboud-specifieke configuratie. Methodologische keuzes zijn hier gedocumenteerd, en Radboud-specifieke logica is generiek gemaakt zodat andere instellingen er direct mee aan de slag kunnen.
