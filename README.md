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
    <a href="#"><img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python"></a>
    <img src="https://badgen.net/github/last-commit/cedanl/studentprognose" alt="GitHub Last Commit">
    <a href="#"><img src="https://custom-icon-badges.demolab.com/badge/Windows-0078D6?logo=windows11&logoColor=white" alt="Windows"></a>
    <a href="#"><img src="https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=F0F0F0" alt="macOS"></a>
    <a href="#"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" alt="Linux"></a>
  </p>

</div>

> [!NOTE]
> Dit model is oorspronkelijk ontwikkeld door **Radboud Universiteit** en vervolgens samen met CEDA open source gemaakt zodat andere instellingen er ook van kunnen profiteren. Lees meer in het [VOX-artikel](https://www.voxweb.nl/nieuws/de-universiteit-heeft-nu-haar-eigen-glazen-bol-nieuw-model-voorspelt-toekomstige-instroom-van-studenten).

> [!TIP]
> **Zie het in actie:** bekijk een [voorbeeld-dashboard en een uitgewerkte prognose](https://cedanl.github.io/studentprognose/) op de documentatiesite — van vooraanmelders op een peilmoment naar het verwachte aantal inschrijvingen, met demodata.

---

## 📦 Aan de slag

Vereisten: **Python 3.12**

Installeer met [uv](https://docs.astral.sh/uv/):

```bash
uv tool install studentprognose
```

> Heb je uv nog niet? Eenmalig installeren met `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux) of `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` (Windows). Voor pip-instructies, zie de [documentatie](https://cedanl.github.io/studentprognose/installeren/).

Na installatie:

```bash
studentprognose init           # mapstructuur aanmaken + demodata downloaden (optioneel)
studentprognose -d c -y 2024   # eerste prognose op demodata
```

`init` vraagt of je demodata wilt downloaden (4 MB) zodat je direct kunt starten. Kies `n` als je meteen je eigen data wilt gebruiken.

Voor geautomatiseerde runs (cron, taakplanner) — sla de interactieve prompt over:

```bash
studentprognose -w 6 -y 2024 --yes
```

> [!NOTE]
> Heb je afwijkende kolomnamen in je Studielink-export? Voeg een `"columns"`-blok toe aan `configuration/configuration.json`. Zie [Configuratie](https://cedanl.github.io/studentprognose/configuratie/) voor uitleg en voorbeelden.

Zie de [documentatie](https://cedanl.github.io/studentprognose/aan-de-slag/) voor een complete walkthrough met uitleg over Python-installatie, data klaarzetten en veelvoorkomende fouten.

---

## Waarom dit model?

Dit model is gebouwd voor **data-analisten bij Nederlandse onderwijsinstellingen** die werken met Studielink-data. Je hebt geen machine learning-expertise nodig.

| | |
|---|---|
| **Bring Your Own Data** | Je levert je eigen data aan — er wordt niets extern gedeeld |
| **Privacy-vriendelijk** | Draait volledig lokaal op je eigen machine |
| **Open source** | Transparant, aanpasbaar en gratis te gebruiken |
| **Demodata via init** | Direct uitproberen zonder eigen data — `studentprognose init` downloadt 4 MB demodata op aanvraag |

---

## 📊 Presentatie

Een introductie tot Studentprognose (HOlink 2026): waarom instroomprognose lastig is, hoe het model werkt en hoe je ermee aan de slag gaat.

- 📥 [Download de presentatie (PowerPoint)](doc/presentation/HOlink_studentprognose.pptx)
- 🌐 [Bekijk online in de documentatie](https://cedanl.github.io/studentprognose/presentatie/)

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

## ✨ Gebruik

> [!IMPORTANT]
> Dit model werkt met **Studielink-telbestanden**. Je hebt deze data nodig om voorspellingen te maken voor jouw instelling; de meegeleverde demodata laat je het model eerst uitproberen. Zie [Je data voorbereiden](https://cedanl.github.io/studentprognose/je-data-voorbereiden/).


```bash
studentprognose -w 6 -y 2024                  # specifieke week en jaar
studentprognose -w 10 : 20 -y 2023            # weekbereik
studentprognose -d c                           # alleen cumulatief spoor
studentprognose -d c --institution 21PC        # scoop de teldata op je eigen instelling
studentprognose -y 2023 2024 -w 10 : 20 -d b  # meerdere jaren, beide sporen

studentprognose benchmark -d c -w 12           # vergelijk alternatieve modellen
studentprognose tune -d c -w 12                # stem hyperparameters af (cumulatieve regressor)
```

| Vlag | Beschrijving | Opties |
|------|-------------|--------|
| `-w` | Voorspelweek(en) | weeknummers of bereik, bijv. `10 : 20` |
| `-y` | Voorspeljaar(en) | bijv. `2024` of `2023 2024` |
| `-d` | Dataset | `i`ndividual, `c`umulative, `b`oth (standaard) |
| `--institution` | Beperk teldata tot instelling(en) | Brincode(s), bijv. `21PC`; standaard alle |
| `--noetl` | Sla ETL over | als je al verwerkte data in `data/input/` hebt |
| `--yes` | Sla interactieve prompts over | voor CI/CD en cron |
| `--no-warnings` | Onderdruk UserWarning-meldingen | als bekende warnings de uitvoer onoverzichtelijk maken |

Naast voorspellen kun je met het `benchmark`-subcommando alternatieve ML-modellen vergelijken op je eigen data:

```bash
studentprognose benchmark -d c                 # vergelijk cumulatieve modellen
```

Zie de [documentatie](https://cedanl.github.io/studentprognose/aan-de-slag/#cli-referentie) voor alle vlaggen, configuratie, validatie-instellingen en uitgebreide voorbeelden, en [Benchmarks](https://cedanl.github.io/studentprognose/methodologie/benchmarks/) voor de modelvergelijking.

---

## 📁 Beschrijving van bestanden

### Input

| Bestand | Beschrijving |
|---------|-------------|
| **individual** | Individuele (voor)aanmeldingen per student. Voedt het [individueel model](https://cedanl.github.io/studentprognose/methodologie/individueel/) (XGBoost classifier + SARIMA) en levert de `SARIMA_individual` voorspelling. |
| **cumulative** | Aantal aanmeldingen per opleiding, herkomst, jaar, week en herinschrijving. Wordt gebruikt voor de SARIMA_cumulative voorspelling. Verkregen via Studielink. |
| **latest** | Per opleiding, herkomst, jaar en week: aanmeldingen, voorspellingen en foutwaarden (MAE/MAPE). |
| **student_count_first-years** | Werkelijk aantal eerstejaars studenten per jaar, opleiding en herkomst. |
| **student_volume** | Werkelijk totaal aantal ingeschreven studenten per jaar, opleiding en herkomst (alleen nodig bij `-sy v`). |
| **weighted_ensemble** | Gewichten per model voor de ensemble-voorspelling. |

### Output

| Bestand | Beschrijving |
|---------|-------------|
| **output_prelim.xlsx** | Voorlopige output met alle voorspellingen van de huidige run. |
| **output_first-years.xlsx** | Volledige output met voorspellingen voor eerstejaars studenten. |
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
| 5 | Data laden | `data/loader` → `data/preprocessing/add_zero_weeks` |
| 6 | CI subset (indien `--ci`) | `utils/ci_subset` |

**Modus-specifieke stappen** (vervolgen op de gedeelde stappen):

| Stap | Fase | Individual (`-d i`) | Cumulative (`-d c`) | Both (`-d b`) |
|------|------|---------------------|---------------------|---------------|
| 7 | Preprocessing | `strategies/individual` | `strategies/cumulative` | individual → cumulative |
| 8 | Filtering | `strategies/base` | `strategies/base` | `strategies/base` |
| 9 | Classificatie | `models/xgboost_classifier` | — | `models/xgboost_classifier` |
| 10 | Transformatie | `data/transforms` | — | `data/transforms` |
| 11 | SARIMA | `models/sarima` (individual) | `models/sarima` → `data/transforms` | `models/sarima` (both) |
| 12 | XGBoost regressor | — | `models/xgboost_regressor` | `models/xgboost_regressor` |
| 13 | Ratio model | — | `models/ratio` | `models/ratio` |
| 14 | Postprocessing + Opslaan | `output/postprocessor` | `output/postprocessor` | `output/postprocessor` |

Zie [`doc/PIPELINE.md`](doc/PIPELINE.md) voor de gedetailleerde end-to-end pipeline (inclusief Mermaid-diagrammen). Voor een uitleg van het individueel spoor, zie [Individueel model](https://cedanl.github.io/studentprognose/methodologie/individueel/) in de methodologische documentatie.

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


