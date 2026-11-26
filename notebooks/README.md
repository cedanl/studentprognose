# Studentprognose — Jupyter notebooks

Deze map bevat **uitvoerbare aanvullingen** op de [mkdocs-documentatie](../docs/). Waar de docs
de methodologie *beschrijven*, laten deze notebooks zien hoe het er in de praktijk uitziet —
met echte demodata, plots en tussenresultaten.

## Twee soorten notebooks

| Notebook | Doel | Vereiste |
|----------|------|----------|
| [`implementatie.ipynb`](implementatie.ipynb) | Startpunt voor instellingen die de pipeline inpassen in hun eigen data-infra (MS Fabric, Databricks, Azure). Draait standaard het CWF-model; het cumulatieve spoor staat er als alternatief bij | `pip install studentprognose` — geen repo-clone nodig |
| `00` t/m `07` | Uitvoerbare verdieping bij de methodologie-docs | Lokale repo-clone via `uv sync --group dev` |

## Voor wie?

Data-analisten en onderzoekers bij Nederlandse hogeronderwijsinstellingen die:

- de pipeline willen **inpassen** in hun eigen cloud-omgeving → begin bij `implementatie.ipynb`
- de modellen willen *begrijpen* en *experimenteren* → begin bij `00_overzicht.ipynb`

## Volgorde

De notebooks draaien om het **CWF-model**: het aanbevolen model, omdat het doorgaans
het nauwkeurigst is en er een interval en betrouwbaarheidslabel bij geeft. Op de demodata
heeft het in een backtest over vier jaar een duidelijk lagere fout dan "evenveel als vorig
jaar" (`02`) en dan het cumulatieve spoor (`05`).

| # | Notebook | Hoort bij docs-pagina | Tijd |
|---|----------|----------------------|------|
| 00 | [`00_overzicht.ipynb`](00_overzicht.ipynb) | [Methodologie · index](../docs/methodologie/index.md) | 5 min |
| 01 | [`01_data_voorbereiden.ipynb`](01_data_voorbereiden.ipynb) | [Je data voorbereiden](../docs/je-data-voorbereiden.md) + [Validatie](../docs/validatie.md) | 10 min |
| 02 | [`02_cwf_model.ipynb`](02_cwf_model.ipynb) | [Methodologie · CWF-model](../docs/methodologie/cwf.md) | 20 min |
| 03 | [`03_output_interpreteren.ipynb`](03_output_interpreteren.ipynb) | [Output begrijpen](../docs/output-begrijpen.md) | 10 min |

**Verdieping: de andere sporen.** Nog steeds te gebruiken (`-d c`, `-d i`, `-d b`), bijvoorbeeld om te vergelijken.

| # | Notebook | Hoort bij docs-pagina | Tijd |
|---|----------|----------------------|------|
| 04 | [`04_sarima.ipynb`](04_sarima.ipynb) | [Methodologie · SARIMA](../docs/methodologie/sarima.md) | 15 min |
| 05 | [`05_xgboost.ipynb`](05_xgboost.ipynb) | [Methodologie · XGBoost](../docs/methodologie/xgboost.md) | 20 min |
| 06 | [`06_ratio_model.ipynb`](06_ratio_model.ipynb) | [Methodologie · Ratio-model](../docs/methodologie/ratio-model.md) | 10 min |
| 07 | [`07_ensemble.ipynb`](07_ensemble.ipynb) | [Methodologie · Ensemble](../docs/methodologie/ensemble.md) | 10 min |

Je kunt elke notebook losstaand draaien — `00_overzicht.ipynb` is het beste startpunt als
je nieuw bent. De cumulatieve pipeline is zwaar; `05` en `07` draaien hem daarom op de
15 grootste opleidingen (zet `SELECTIE = []` voor alles).

## Installatie

Vanuit de projectroot:

```bash
uv sync --group dev      # installeert jupyter + alle dependencies
uv run jupyter notebook  # opent jupyter in de browser
```

Of, als je Jupyter Lab prefereert:

```bash
uv run jupyter lab
```

## Wat heb je nodig?

De notebooks draaien op de demodata in `data/input/` die met de repo wordt meegeleverd:

- `data/input/vooraanmeldingen_cumulatief.csv` — Studielink-telbestanden (verwerkt)
- `data/input/student_count_first-years.xlsx` — historische realisatie

De demodata is synthetisch: inschrijvingen zijn altijd 0 en de aanmelders zakken rond
week 23 kunstmatig in. Daarom gebruiken de notebooks peilweek 12.

**Alleen deel A van `05_xgboost.ipynb`** (de classifier) heeft ook individuele aanmelddata
nodig. Die zit niet in de repo; haal hem op met `studentprognose init` in de projectroot
(kies "j" bij de demodata). Hij komt dan in `data/input_raw/individuele_aanmelddata.csv`.

## Voor jouw eigen data

Vervang de bestanden in `data/input/` door je eigen aanbod (zelfde schema — zie
[`docs/je-data-voorbereiden.md`](../docs/je-data-voorbereiden.md)), en pas de voorbeelden
in de notebooks aan:

```python
PROGRAMMA = 56604             # Isatcode → kies een opleiding uit jouw data
HERKOMST = "NL"               # of "EER" / "Niet-EER"
PREDICT_YEAR = 2024           # collegejaar
PREDICT_WEEK = 12             # peilweek (1–37)
```

## De notebooks regenereren

Wijzigingen aan de notebooks worden gebouwd via een Python-script in plaats van handmatig
geredigeerd. Dit maakt diffs leesbaar en houdt de notebooks consistent.

```bash
uv run python notebooks/_build_notebooks.py
```

Voer daarna de notebooks uit om de output te verversen:

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/00_overzicht.ipynb --inplace
# herhalen voor 01..07
```

## CI-overweging

De notebooks worden **niet** automatisch in CI uitgevoerd (geen `nbmake` of
`jupyter nbconvert --execute` in `.github/workflows/`). Reden:

- de smoke test (`scripts/test_package.sh`) valideert dezelfde end-to-end code-paden
- toevoegen van een `jupyter`-dependency in CI is niet kosteloos qua runtime
- handmatig regenereren bij wijziging is laagdrempelig genoeg

Bij toekomstige iteraties kan dit heroverwogen worden zodra de notebooks structureler
onderhoud vragen.
