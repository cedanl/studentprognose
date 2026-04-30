<div align="center">
  <a href="https://github.com/cedanl/studentprognose">
    <img src="doc/header.svg" alt="Studentprognose" style="max-width: 100%;">
  </a>

  <h3>Voorspel je studentinstroom maanden vooruit — met je eigen data, op je eigen machine.</h3>

  <p>
    <a href="https://www.voxweb.nl/nieuws/de-universiteit-heeft-nu-haar-eigen-glazen-bol-nieuw-model-voorspelt-toekomstige-instroom-van-studenten"><img src="https://img.shields.io/badge/Ingezet_door-Radboud_Universiteit-darkred" alt="Radboud Universiteit"></a>
    <a href="https://github.com/cedanl"><img src="https://img.shields.io/badge/Onderhouden_door-CEDA-blue" alt="CEDA"></a>
    <img src="https://badgen.net/github/contributors/cedanl/studentprognose" alt="Contributors">
    <img src="https://img.shields.io/github/license/cedanl/studentprognose" alt="GitHub License">
    <a href="https://pypi.org/project/studentprognose/"><img src="https://img.shields.io/pypi/v/studentprognose" alt="PyPI"></a>
    <br>
    <a href="#"><img src="https://img.shields.io/badge/Python-≥3.12-3776AB?logo=python&logoColor=white" alt="Python"></a>
    <img src="https://badgen.net/github/last-commit/cedanl/studentprognose" alt="GitHub Last Commit">
    <a href="#"><img src="https://custom-icon-badges.demolab.com/badge/Windows-0078D6?logo=windows11&logoColor=white" alt="Windows"></a>
    <a href="#"><img src="https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=F0F0F0" alt="macOS"></a>
    <a href="#"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" alt="Linux"></a>
  </p>

</div>

> [!NOTE]
> Dit model is oorspronkelijk ontwikkeld door **Radboud Universiteit** en vervolgens samen met CEDA open source gemaakt zodat andere instellingen er ook van kunnen profiteren. Lees meer in het [VOX-artikel](https://www.voxweb.nl/nieuws/de-universiteit-heeft-nu-haar-eigen-glazen-bol-nieuw-model-voorspelt-toekomstige-instroom-van-studenten).

---

## 📦 Aan de slag

```bash
pip install studentprognose
studentprognose init
```

`init` maakt de benodigde mapstructuur aan en legt uit welke bestanden je moet aanleveren. Daarna:

```bash
studentprognose -w 6 -y 2024
```

Voor geautomatiseerde runs (cron, taakplanner) — sla de interactieve prompt over:

```bash
studentprognose -w 6 -y 2024 --yes
```

> [!NOTE]
> Heb je afwijkende kolomnamen in je Studielink-export? Voeg een `"columns"`-blok toe aan `configuration/configuration.json`. Zie [Configuratie](https://cedanl.github.io/studentprognose/configuratie/) voor uitleg en voorbeelden.

---

## 🛠️ Aan de slag voor ontwikkelaars

**Via de broncode** (met demodata):

```bash
# 1. Installeer uv (zie https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone de repository
git clone https://github.com/cedanl/studentprognose.git
cd studentprognose

# 3. Draai het model met demodata
uv run studentprognose -w 6 -y 2020
```

> [!NOTE]
> Demodata is meegeleverd in `data/input_raw/`, zodat je direct kunt starten. Gebruik `-y 2020` t/m `-y 2024` en `-w 1` t/m `-w 52`.

---

## 🗃️ Studielink Data

> [!IMPORTANT]
> Dit model werkt met **Studielink-telbestanden**. Je hebt deze data nodig om voorspellingen te maken voor jouw instelling. Demodata is meegeleverd zodat je het model eerst kunt uitproberen.

---

## Waarom dit model?

Dit model is gebouwd voor **data-analisten bij Nederlandse onderwijsinstellingen** die werken met Studielink-data. Je hebt geen machine learning-expertise nodig.

| | |
|---|---|
| **Bring Your Own Data** | Je levert je eigen data aan — er wordt niets extern gedeeld |
| **Privacy-vriendelijk** | Draait volledig lokaal op je eigen machine |
| **Open source** | Transparant, aanpasbaar en gratis te gebruiken |
| **Demo data inbegrepen** | Direct uitproberen zonder eigen data — demobestanden zitten in `data/input` |

---

## ✨ Gebruik

### Jaren en weken

Specificeer jaar en week met `-y` en `-w`:

```bash
uv run main.py -w 6 -y 2024
uv run main.py -W 1 2 3 -Y 2024
uv run main.py -year 2023 2024
uv run main.py -week 40 41
```

Gebruik slicing voor een reeks weken:

```bash
uv run main.py -w 10 : 20 -y 2023
```

### Datasets

Er zijn twee datasets beschikbaar: **individual** (per student) en **cumulative** (geaggregeerd per opleiding/herkomst/jaar/week). Standaard worden beide gebruikt.

```bash
uv run main.py -d individual
uv run main.py -D cumulative
uv run main.py -dataset both
```

### Datakwaliteitscontrole

Bij elke run wordt de ruwe inputdata automatisch gevalideerd **voordat de ETL start**. Zo weet je zeker dat je nooit 20 minuten wacht voor een prognose op basis van corrupte data.

Er zijn drie soorten bevindingen:

| Type | Voorbeeld | Wat er gebeurt |
|---|---|---|
| **Fout** | Ontbrekend bestand, ontbrekende kolom | Pipeline stopt direct |
| **Probleem** | Negatieve waarden, onbekende herkomstcode | Prompt: doorgaan of stoppen? |
| **Waarschuwing** | Klein percentage ontbrekende waarden, weekgaten | Gelogd, pipeline loopt door |

```
Validatie gevonden problemen die prognoses kunnen beïnvloeden:
  [PROBLEEM] telbestandY2025W10.csv: meercode_V = 0 voor 1 opleiding(en): B Opleiding X

Als je doorgaat, kunnen sommige opleidingen worden overgeslagen of
onbetrouwbare prognoses opleveren.

Wil je doorgaan met de pipeline? [j/N]
```

Gebruik `--yes` om de prompt te omzeilen in geautomatiseerde runs (bijv. CI/CD):

```bash
studentprognose -w 6 -y 2025 --yes
```

> [!NOTE]
> Met `--noetl` worden ETL én validatie overgeslagen. Gebruik dit alleen als de ruwe data al eerder succesvol verwerkt en gevalideerd is.

#### Validatie-instellingen aanpassen

De standaardwaarden (jaarbereiken, NaN-drempels, toegestane categorische waarden) staan in de broncode. Je kunt ze overschrijven door een `"validation"`-blok toe te voegen aan je `configuration.json`. Vermeld alleen de waarden die afwijken:

```json
{
    "validation": {
        "nan_error_threshold": 0.20,
        "telbestand": {
            "herkomst_allowed": ["N", "E", "R", "ONBEKEND"]
        }
    }
}
```

### Configuratie

Het standaard configuratiebestand is `configuration/configuration.json`. Dit kan worden overschreven:

```bash
uv run main.py -c pad/naar/configuration.json
uv run main.py -configuration langer/pad/naar/config.json
```

### Uitgebreid voorbeeld

Voorspel eerstejaars voor 2023 en 2024, weken 10 t/m 20, met beide datasets:

```bash
uv run main.py -y 2023 2024 -w 10 : 20 -d b
```

Voorspel eerstejaars voor collegejaar 2025/2026, week 5, alleen cumulatief:

```bash
uv run main.py -y 2025 -w 5 -d c
```

### Syntax overzicht

| Instelling              | Korte notatie  | Lange notatie    | Opties                                      |
|-------------------------|----------------|------------------|---------------------------------------------|
| Voorspellingsjaren      | `-y` of `-Y`   | `-year`          | Eén of meer jaren, bijv. `2023 2024`        |
| Voorspellingsweken      | `-w` of `-W`   | `-week`          | Eén of meer weken, bijv. `10 11 12`         |
| Slicing                 |                |                  | Gebruik `:` voor reeksen, bijv. `10 : 20`   |
| Dataset                 | `-d` of `-D`   | `-dataset`       | `i`/`individual`, `c`/`cumulative`, `b`/`both` |
| Configuratie            | `-c` of `-C`   | `-configuration` | Pad naar configuratiebestand                |
---

## 📁 Beschrijving van bestanden

### Input

| Bestand | Beschrijving |
|---------|-------------|
| **individual** | Individuele (voor)aanmeldingen per student. Wordt gebruikt voor de SARIMA_individual voorspelling. |
| **cumulative** | Aantal aanmeldingen per opleiding, herkomst, jaar, week en herinschrijving/hogerejaars. Wordt gebruikt voor de SARIMA_cumulative voorspelling. Verkregen via Studielink. |
| **latest** | Per opleiding, herkomst, jaar en week: aanmeldingen, voorspellingen en foutwaarden (MAE/MAPE). Wordt gebruikt voor volume- en hogerjaarsvoorspellingen. |
| **student_count_first-years** | Werkelijk aantal eerstejaars studenten per jaar, opleiding en herkomst. |
| **student_count_higher-years** | Werkelijk aantal hogerjaars studenten per jaar, opleiding en herkomst. |
| **student_volume** | Werkelijk totaal aantal studenten (eerstejaars + hogerjaars) per jaar, opleiding en herkomst. |
| **weighted_ensemble** | Gewichten per model voor de ensemble-voorspelling. |

### Output

| Bestand | Beschrijving |
|---------|-------------|
| **output_prelim.xlsx** | Voorlopige output met alle voorspellingen van de huidige run. |
| **output_first-years.xlsx** | Volledige output met voorspellingen voor eerstejaars studenten. |
| **output_higher-years.xlsx** | Volledige output met voorspellingen voor hogerjaars studenten. |
| **output_volume.xlsx** | Volledige output met volume-voorspellingen (totaal). |

---

## 🏗️ Architectuur

### Pipeline executievolgorde

**Gedeelde stappen (alle modi):**

| Stap | Fase | Bestand |
|------|------|---------|
| 1 | CLI parsing | `cli.py` |
| 2 | Validatie ruwe data (skip met `--noetl`) | `data/validation` |
| 3 | ETL (skip met `--noetl`) | `data/etl` |
| 4 | Configuratie laden | `config.py` |
| 5 | Data laden | `loader` → `preprocessing/add_zero_weeks` |
| 6 | CI subset (indien `--ci`) | `utils/ci_subset` |

**Modus-specifieke stappen:**

| Stap | Fase | Individual (`-d i`) | Cumulative (`-d c`) | Both (`-d b`) |
|------|------|---------------------|---------------------|---------------|
| 6 | Preprocessing | `strategies/individual` | `strategies/cumulative` | individual → cumulative |
| 7 | Filtering | `strategies/base` | `strategies/base` | `strategies/base` |
| 8 | Classificatie | `xgboost_classifier` | — | `xgboost_classifier` |
| 9 | Transformatie | `transforms` | — | `transforms` |
| 10 | SARIMA | `sarima` (individual) | `sarima` → `transforms` | `sarima` (both) |
| 11 | XGBoost regressor | — | `xgboost_regressor` | `xgboost_regressor` |
| 12 | Ratio model | — | `ratio` | `ratio` |
| 13 | Postprocessing + Opslaan | `postprocessor` | `postprocessor` | `postprocessor` |

Zie de [Technische README](doc/TECHNICAL_README.md) voor meer details over de architectuur.

---

## 🤝 Bijdragen

Dit project wordt actief onderhouden door [CEDA](https://github.com/cedanl). Wil je bijdragen of meedenken? Sluit je aan bij de [werkgroep](https://edu.nl/6d69d).

## 🆘 Ondersteuning

Voor vragen of problemen:
- **GitHub Issues**: [Probleem melden](https://github.com/cedanl/studentprognose/issues)

---

<div align="center">
  <sub>Gebouwd met ❤️ door de <a href="https://github.com/cedanl">CEDANL</a> community</sub>
</div>


