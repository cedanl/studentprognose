# Begrippen

Een korte woordenlijst van termen die in deze documentatie terugkomen. Kom je een term tegen die hier niet staat? Meld het via [GitHub Issues](https://github.com/cedanl/studentprognose/issues).

## Kernbegrippen

| Term | In gewone taal |
|------|----------------|
| **Peilweek** | De kalenderweek (1–52) waarop je de aanmeldstand "bevriest" en waarvandaan je vooruit voorspelt. Vlag `-w`. |
| **Collegejaar** | Het jaar waarvoor je de instroom voorspelt. Vlag `-y`. |
| **Instroom / inschrijvingen** | Het uiteindelijke aantal studenten dat zich daadwerkelijk inschrijft — dat is wat de tool voorspelt. |
| **(Voor)aanmelding** | Een aanmelding via Studielink die (nog) geen inschrijving is. Niet elke aanmelder schrijft zich in. |
| **Conversie / yield** | Het aandeel aanmelders dat uiteindelijk inschrijft. |

## De twee sporen

De tool kan langs twee routes voorspellen, afhankelijk van je data:

| Term | In gewone taal |
|------|----------------|
| **Cumulatief spoor** (`-d c`) | Werkt met de opgetelde aanmeldaantallen per opleiding en week (Studielink-telbestanden). Trekt de aanmeldcurve door naar het einde van het seizoen. |
| **Individueel spoor** (`-d i`) | Werkt met één regel per aanmelder (Osiris/Usis). Schat per persoon de kans op inschrijving en telt die op. |
| **Beide** (`-d b`) | Draait beide sporen en combineert ze tot één [ensemble](#modellen)-voorspelling. Dit is de standaard. |

## Data

| Term | In gewone taal |
|------|----------------|
| **ETL** | *Extract, Transform, Load* — de stap die je ruwe bronbestanden inleest, opschoont en omzet naar het formaat dat het model verwacht. De tool doet dit automatisch; sla het over met `--noetl` als je data al verwerkt is. |
| **Telbestand** | Het Studielink-bestand met opgetelde aanmeldaantallen per opleiding, herkomst en week. |
| **Gewogen vooraanmelders** | Aanmeldingen gecorrigeerd voor studenten die zich voor meerdere opleidingen aanmelden (een aanmelding telt dan voor een deel mee). |
| **Ongewogen vooraanmelders** | Het "kale" aantal aanmeldingen, zonder die correctie. |
| **Brincode / instellingscode** | De landelijke code van een onderwijsinstelling (bijv. `21PC`). De teldata bevat álle instellingen; met `--institution` beperk je de run tot de jouwe. |
| **Kanonieke naam** | De vaste, interne kolomnaam die de tool verwacht. Heten je eigen kolommen anders, dan vertaal je ze via het `columns`-blok in de configuratie. |
| **Herkomst** | Onderscheid tussen bijv. Nederlandse en niet-Nederlandse studenten, gebruikt als voorspeldimensie. |

## Modellen

| Term | In gewone taal |
|------|----------------|
| **SARIMA** | Een tijdreeksmodel dat een curve doortrekt op basis van het patroon in eerdere jaren, inclusief het seizoenspatroon. Zie [SARIMA](methodologie/sarima.md). |
| **XGBoost** | Een machine-learning-model dat verbanden leert tussen kenmerken (features) en de uitkomst. Wordt zowel per student (kans op inschrijving) als op de curve gebruikt. Zie [XGBoost](methodologie/xgboost.md). |
| **Ratio-model** | Een eenvoudig model dat de historische verhouding aanmelding → inschrijving toepast op de huidige aanmeldingen. Zie [Ratio-model](methodologie/ratio-model.md). |
| **Ensemble** | De gewogen combinatie van de losse modellen tot één eindvoorspelling. Zie [Ensemble](methodologie/ensemble.md). |
| **Baseline** | Een bewust simpele referentievoorspelling (gelijk aan het ratio-model) om de complexere modellen tegen af te zetten. |
| **Numerus fixus** | Opleiding met een vast maximum aantal plaatsen; de voorspelling wordt afgekapt op dat maximum. |

## Evaluatie

| Term | In gewone taal |
|------|----------------|
| **MAE** | *Mean Absolute Error* — het gemiddelde aantal studenten waarmee een voorspelling ernaast zat. MAE 8 = gemiddeld 8 studenten afwijking. |
| **MAPE** | *Mean Absolute Percentage Error* — dezelfde afwijking als percentage. MAPE 0,12 = gemiddeld 12% ernaast. |
| **WAPE** | Procentuele fout over de hele instelling; grote opleidingen wegen zwaarder, waardoor kleine opleidingen de uitkomst niet domineren. |
| **Bias** | Of een model structureel te hoog (positief) of te laag (negatief) voorspelt. |
| **Backtest** | Een voorspelling maken voor een reeds **afgerond** jaar en die vergelijken met wat er echt gebeurde — de eerlijke manier om een model te beoordelen. |
| **Data leakage** | Wanneer een model per ongeluk informatie gebruikt die op het voorspelmoment nog niet bekend kon zijn. Dat maakt resultaten te mooi; de tool voorkomt dit bewust. |
