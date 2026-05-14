# Plan van Aanpak — Issue #168

**Issue:** Voeg Jupyter notebook toe als uitgebreide aanvulling op code-secties in mkdocs.
**Branch:** `168-jupyter-notebook-mkdocs-aanvulling`
**Doel:** Een set runnable Jupyter-notebooks die de mkdocs-pagina's diepgaander, uitvoerbaar en met visualisaties illustreert voor data-analisten en onderzoekers.

## 1. Doelgroep en doel

**Doelgroep:** data-analisten en onderzoekers bij Nederlandse hogeronderwijsinstellingen.

**Doel van de notebooks:**

1. **Begrijpen:** waarom is het model zo opgebouwd, niet alleen wat het doet
2. **Reproduceren:** elke stap uit de mkdocs uitvoerbaar maken op de demodata in `data/input/`
3. **Experimenteren:** parameters draaien (peilweek, opleiding, herkomst) en het effect zien
4. **Inspecteren:** tussenresultaten — DataFrames, plots, foutmaten — bij elke stap zichtbaar

## 2. Structuur

```
notebooks/
├── README.md                       — overzicht, installatie-instructies, leesvolgorde
├── PVA.md                          — dit document
├── 00_overzicht.ipynb              — end-to-end mini-pipeline (eenvoudig)
├── 01_data_voorbereiden.ipynb      — data laden, schema, validatie
├── 02_sarima.ipynb                 — SARIMA-model in detail
├── 03_xgboost.ipynb                — XGBoost classifier + regressor
├── 04_ratio_model.ipynb            — ratio-model, baseline
├── 05_ensemble.ipynb               — ensemble samenstellen en interpreteren
└── 06_output_interpreteren.ipynb   — outputkolommen, foutmaten, dashboard
```

### Mapping notebook → mkdocs-pagina

| Notebook | Primaire docs-pagina | Aanvullend |
|----------|----------------------|------------|
| `00_overzicht.ipynb` | `methodologie/index.md` | `aan-de-slag.md` |
| `01_data_voorbereiden.ipynb` | `je-data-voorbereiden.md` | `validatie.md`, `configuratie.md` |
| `02_sarima.ipynb` | `methodologie/sarima.md` | — |
| `03_xgboost.ipynb` | `methodologie/xgboost.md` | — |
| `04_ratio_model.ipynb` | `methodologie/ratio-model.md` | — |
| `05_ensemble.ipynb` | `methodologie/ensemble.md` | — |
| `06_output_interpreteren.ipynb` | `output-begrijpen.md` | — |

## 3. Per notebook — inhoud en leerdoelen

### 00_overzicht.ipynb — Pipeline in 5 minuten

**Leerdoel:** snel begrijpen hoe de drie modellen samenkomen.

**Cellen:**

1. Markdown — Wat is studentprognose, drie datasporen (uit `methodologie/index.md`)
2. Code — Configuratie + data laden via `load_configuration` + `load_data`
3. Code — Pipeline draaien via `run_pipeline_from_dataframes(year=2024, week=12, dataset=BOTH)`
4. Markdown — Wat zit er in de output?
5. Code — Top-rijen tonen + één plot (Ensemble vs Baseline)
6. Markdown — Verwijzing naar de detail-notebooks

### 01_data_voorbereiden.ipynb — Data en validatie

**Leerdoel:** snappen welke datasets de pipeline verwacht, hoe ze eruitzien, en welke validatieregels gelden.

**Cellen:**

1. Cumulatief CSV (Studielink) — kolommen, schema-tabel, top-rijen
2. Inspectie: aantal opleidingen, jaren, weken, herkomstgroepen
3. Wide vs. long format (uit `data/transforms.py`) — visualiseer pivot
4. Validatieregels (uit `validatie.md`) — toon hard/soft/warning categorieën
5. Plot — wekelijkse aanmeldcurve voor één opleiding × herkomst over meerdere jaren
6. Plot — heatmap aanmelders per week × jaar

### 02_sarima.ipynb — Tijdreeksextrapolatie

**Leerdoel:** SARIMA-orde en seizoenscomponent begrijpen, model zien fitten en forecasten.

**Cellen:**

1. Markdown — Wat is SARIMA, waarom seizoenslengte 52, vaste ordes (`sarima.md`)
2. Code — Eén opleiding + herkomst selecteren, training-tijdreeks bouwen
3. Code — `SARIMAForecaster` fitten + forecasten
4. Plot — observed vs. forecast vs. (waar mogelijk) realisatie
5. Markdown — Aannames: herhaalbaar seizoen, ≥ 3 jaar data, stationariteit, CSS-ML
6. Code — Vergelijk twee opleidingen met verschillende historie-lengtes
7. Markdown — Wanneer vertrouw je het niet (uit docs)

### 03_xgboost.ipynb — Classifier + Regressor

**Leerdoel:** twee rollen van XGBoost onderscheiden, features begrijpen, feature importance lezen.

**Cellen:**

1. Markdown — Twee gebruiken (`xgboost.md` § Classifier / § Regressor)
2. Code — Classifier-features tonen op een sample uit individuele data
3. Code — Voor een vereenvoudigd voorbeeld: train classifier op 2 jaar data, voorspel kans op inschrijving
4. Plot — Feature importance (gegroepeerd)
5. Code — Regressor: lagged features (`Gewogen_t-2`, `Gewogen_t-5`), `Gewogen_acceleration`, `exclusivity_ratio`
6. Plot — Feature importance regressor
7. Markdown — Wanneer vertrouw je het niet

### 04_ratio_model.ipynb — De transparante baseline

**Leerdoel:** waarom een simpele ratio cruciaal is als baseline.

**Cellen:**

1. Markdown — Definitie + 3-jaars venster (`ratio-model.md`)
2. Code — Ratio handmatig herberekenen voor één opleiding/week
3. Plot — Ratio per week, gemiddeld over 3 jaar, voor enkele opleidingen
4. Code — Voorspelling uitrekenen `Aanmelding / Average_Ratio`
5. Markdown — Numerus fixus-cap toelichting
6. Markdown — Beperkingen (geen trend, gevoelig voor outliers)

### 05_ensemble.ipynb — Modellen combineren

**Leerdoel:** snappen hoe gewichten worden gekozen en wanneer het ensemble het ratio-model verslaat.

**Cellen:**

1. Markdown — Waarom ensemble (`ensemble.md`)
2. Code — `ensemble_weights` uit `configuration.json` ophalen en tonen
3. Code — Voor één opleiding: SARIMA_cumulative + SARIMA_individual + Prognose_ratio combineren met de configureerbare gewichten
4. Plot — Modellen naast elkaar, ensemble eroverheen, week 38 = 100% individueel
5. Markdown — Wanneer is de Baseline betrouwbaarder (uit `output-begrijpen.md`)
6. Code — `ensemble_weights.xlsx` interpretatie (indien aanwezig)

### 06_output_interpreteren.ipynb — Output lezen

**Leerdoel:** kolommen, foutmaten en dashboard correct interpreteren.

**Cellen:**

1. Markdown — Outputbestanden uit `output-begrijpen.md`
2. Code — Output-DataFrame inspecteren (kolomgroepen: voorspellingen, actuelen, foutmaten)
3. Markdown — MAE vs MAPE — voorbeeld met cijfers
4. Plot — Conversie-vergelijking vooraanmelders → inschrijvingen
5. Plot — Vergelijking ensemble vs werkelijke realisatie vorig jaar
6. Markdown — Wanneer is een prognose betrouwbaar (signalen uit docs)

## 4. Implementatieprincipes

### Reproduceerbaarheid
- Alle notebooks draaien op de demodata in `data/input/` (geen ruwe data nodig)
- Default peilmoment: `year=2024, week=12` — consistent over notebooks heen
- Default voorbeeldopleiding: `B Psychologie` (groot, NL-herkomst, stabiel)

### Markdown-cellen
- **Voor elke code-cel** een markdown-cel die het *waarom* uitlegt — geen herhaling van wat de code doet
- Verwijs expliciet naar de docs-pagina ("Zie [SARIMA → Aannames](../docs/methodologie/sarima.md#aannames)")
- Gebruik admonition-stijl waarschuwingen waar relevant

### Visualisaties
- Matplotlib voor inline plots (al dependency)
- Geen zware Plotly-dashboards in notebooks — die zitten al in het normale dashboard
- Elke plot heeft titel + assen-labels + bron-vermelding ("Bron: demodata")

### Pakketgebruik
- Importeer altijd via het public API: `from studentprognose import ...`
- Bij detail-cellen: importeer direct uit submodules (`studentprognose.models.sarima`) en motiveer dit als interne demonstratie

## 5. Acceptatiecriteria (uit issue)

- [x] Notebook(s) in `notebooks/` map
- [x] Draaien end-to-end op demodata
- [x] Iedere code-cel heeft een markdown-cel ervoor met het *waarom*
- [x] Visualisaties van tussenresultaten
- [x] Mkdocs-pagina's krijgen verwijzing onder elk methodologie-blok
- [x] `docs/index.md` krijgt sectie "Uitgebreide voorbeelden"
- [x] CI-beslissing gedocumenteerd in PR-tekst — voorstel: **geen automatische executie via `nbmake`** in de eerste iteratie omdat het toevoegen van een `jupyter`-dependency in CI niet kosteloos is en de smoke test dezelfde end-to-end paden valideert. Re-evalueren bij volgende iteratie.

## 6. Verplichte docs-update (CLAUDE.md-regel)

Bij deze PR hoort ook:

- `docs/index.md` — vermelding van de notebooks
- Verwijzing onder de relevante mkdocs-pagina's (`methodologie/sarima.md`, `xgboost.md`, `ratio-model.md`, `ensemble.md`, `output-begrijpen.md`, `je-data-voorbereiden.md`) — als kort admonition-blok

## 7. Iteratie naar 10/10

Na de eerste implementatie review op:

1. **Volledig draaien** — elk notebook van begin tot eind zonder errors
2. **Begrijpelijkheid** — staat elke markdown-cel sterk genoeg op zichzelf voor de doelgroep?
3. **Visualisatie-kwaliteit** — titels, labels, bron, duidelijkheid
4. **Mapping naar docs** — kan een lezer altijd terugvinden welke docs-paragraaf erbij hoort?
5. **Tijdsduur** — kan een lezer ieder notebook in ~15 min doorlopen?

Iteratie pas afgerond als alle 5 punten 10/10 scoren bij self-review.
