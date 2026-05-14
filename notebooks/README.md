# Studentprognose — Jupyter notebooks

Deze map bevat **uitvoerbare aanvullingen** op de [mkdocs-documentatie](../docs/). Waar de docs
de methodologie *beschrijven*, laten deze notebooks zien hoe het er in de praktijk uitziet —
met echte demodata, plots en tussenresultaten.

## Voor wie?

Data-analisten en onderzoekers bij Nederlandse hogeronderwijsinstellingen die:

- de modellen willen *begrijpen*, niet alleen toepassen
- willen *experimenteren* met parameters (peilweek, opleiding, herkomst)
- een *startpunt* zoeken voor hun eigen analyses op hun eigen data

## Volgorde

| # | Notebook | Hoort bij docs-pagina | Tijd |
|---|----------|----------------------|------|
| 00 | [`00_overzicht.ipynb`](00_overzicht.ipynb) | [Methodologie · index](../docs/methodologie/index.md) | 5 min |
| 01 | [`01_data_voorbereiden.ipynb`](01_data_voorbereiden.ipynb) | [Je data voorbereiden](../docs/je-data-voorbereiden.md) + [Validatie](../docs/validatie.md) | 15 min |
| 02 | [`02_sarima.ipynb`](02_sarima.ipynb) | [Methodologie · SARIMA](../docs/methodologie/sarima.md) | 15 min |
| 03 | [`03_xgboost.ipynb`](03_xgboost.ipynb) | [Methodologie · XGBoost](../docs/methodologie/xgboost.md) | 20 min |
| 04 | [`04_ratio_model.ipynb`](04_ratio_model.ipynb) | [Methodologie · Ratio-model](../docs/methodologie/ratio-model.md) | 10 min |
| 05 | [`05_ensemble.ipynb`](05_ensemble.ipynb) | [Methodologie · Ensemble](../docs/methodologie/ensemble.md) | 10 min |
| 06 | [`06_output_interpreteren.ipynb`](06_output_interpreteren.ipynb) | [Output begrijpen](../docs/output-begrijpen.md) | 15 min |

Je kunt elke notebook losstaand draaien — `00_overzicht.ipynb` is het beste startpunt als
je nieuw bent.

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

Alle notebooks draaien op de demodata in `data/input/` die met de repo wordt meegeleverd:

- `data/input/vooraanmeldingen_cumulatief.csv` — Studielink-telbestand (geaggregeerd)
- `data/input/student_count_first-years.xlsx` — historische realisatie

**Geen ruwe data** in `data/input_raw/` nodig — de notebooks slaan de ETL over en lezen
de verwerkte bestanden direct.

## Voor jouw eigen data

Vervang de bestanden in `data/input/` door je eigen aanbod (zelfde schema — zie
[`docs/je-data-voorbereiden.md`](../docs/je-data-voorbereiden.md)), en pas de voorbeelden
in de notebooks aan:

```python
PROGRAMMA = "B Psychologie"   # → kies een opleiding uit jouw data
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
# herhalen voor 01..06
```

## CI-overweging

De notebooks worden **niet** automatisch in CI uitgevoerd (geen `nbmake` of
`jupyter nbconvert --execute` in `.github/workflows/`). Reden:

- de smoke test (`scripts/test_package.sh`) valideert dezelfde end-to-end code-paden
- toevoegen van een `jupyter`-dependency in CI is niet kosteloos qua runtime
- handmatig regenereren bij wijziging is laagdrempelig genoeg

Bij toekomstige iteraties kan dit heroverwogen worden zodra de notebooks structureler
onderhoud vragen.
