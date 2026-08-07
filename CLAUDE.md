# Claude Configuration — Studentprognose

## Project Overview

Studentprognose is een open-source CEDA/Npuls tool voor het voorspellen van eerstejaars instroom in het hoger onderwijs. Het combineert tijdreeksmodellen (SARIMA), machine learning (XGBoost) en een ratio-model tot een ensemble-voorspelling, draaiend als installeerbaar Python-package met CLI.

**Doelgroep van de tool:** data-analisten en onderzoekers bij Nederlandse hogeronderwijsinstellingen.

## Tech Stack

- **Python 3.12+** met **uv** voor dependency management
- **pandas / numpy** voor dataverwerking
- **statsforecast** voor SARIMA (`statsforecast.models.ARIMA`, vaste parameters)
- **scikit-learn** voor preprocessing (one-hot encoding, column transforms) en benchmark-baselines
- **xgboost** voor classificatie en regressie
- **argparse** voor de CLI
- **MkDocs + Material** voor documentatie (gehost via GitHub Pages)

## Project Structure

> Top-level `main.py` (repo-root) is een dunne shim die delegeert naar
> `studentprognose.main:main`; het echte entry point is de console-script
> `studentprognose` (zie `pyproject.toml`).

```
src/studentprognose/
├── cli.py               # CLI-definitie (argparse)
├── config.py            # Configuratie laden (+ deep-merge van overrides)
├── main.py              # Pipeline-orchestratie (cli/main entry point)
├── init.py              # `studentprognose init` — projectmap scaffolden
├── configuration/       # Ingebakken default-config + filtering/base.json
├── models/
│   ├── base.py          # BaseForecaster ABC
│   ├── sarima.py        # SARIMA-model + SARIMA-predictie-orchestratie
│   ├── forecasters.py   # ETS/Theta/AutoARIMA (benchmark-alternatieven)
│   ├── classifiers.py   # OOP-classifiers (benchmark-registry)
│   ├── xgboost_classifier.py  # predict_applicant() — individueel spoor
│   ├── regressors.py    # OOP-regressors + build_preprocessor (benchmark)
│   ├── xgboost_regressor.py   # predict_with_xgboost() — cumulatief spoor
│   ├── importance.py    # Feature-importance-aggregatie
│   └── ratio.py         # Ratio-model
├── strategies/
│   ├── individual.py    # Individueel spoor preprocessing
│   ├── cumulative.py    # Cumulatief spoor preprocessing
│   └── combined.py      # Both-modus
├── data/                # ETL, loader, validatie, transforms, preprocessing/
├── output/              # PostProcessor, dashboard, evaluation, validator
├── benchmark/           # `studentprognose benchmark` — alternatieve modellen vergelijken
└── utils/               # Week-/parallel-/key-hulpfuncties + constants

gui/                     # Optionele NiceGUI-webapp (schil rond de CLI)
├── app.py               # Entry point + routing (`python -m gui`)
├── theme.py             # Design-tokens (kleuren/statussen)
├── state.py             # Gedeelde sessiestate (actief project)
├── pages/               # Paginamodules (home, wizard, config, run, output, benchmark)
└── components/          # Herbruikbaar (layout, bestandskiezer, log-streaming)

docs/                    # MkDocs bronbestanden
├── index.md
├── aan-de-slag.md
├── je-data-voorbereiden.md
├── configuratie.md
├── validatie.md
├── output-begrijpen.md
└── methodologie/
    ├── index.md
    ├── sarima.md
    ├── xgboost.md
    ├── ratio-model.md
    └── ensemble.md
```

## Development Commands

```bash
# Applicatie draaien
uv run studentprognose --help

# Grafische interface (optioneel, NiceGUI) — draait op http://localhost:8080
uv run --extra gui python -m gui

# Tests
uv run pytest

# Linting
uv run ruff check .
uv run ruff format .

# Documentatie lokaal bekijken
uv run mkdocs serve

# Documentatie bouwen
uv run mkdocs build
```

## Teststrategie

### Niveaus

| Niveau | Bestand | Wat het bewaakt |
|--------|---------|-----------------|
| Unit | `tests/studentprognose/test_cli.py` | Argument-parsing, vlagcombinaties, exitcodes |
| Unit | `tests/studentprognose/test_validation.py` | Validatielogica, `_handle_result` exitcodes |
| Unit | `tests/studentprognose/test_strategies.py` | Strategie-contracten (preprocess return types) |
| Integratie | `scripts/test_package.sh` | Volledige pipeline met demodata, exitcode 0 |

CI draait alle unit tests + smoke test bij elke push en PR naar `main` (zie `.github/workflows/`).

### Wanneer schrijf je een test?

- **Nieuwe CLI-vlag** → voeg een `parse_args`-test toe in `test_cli.py` die de vlag en de default bewaakt.
- **Nieuwe validatiecheck** → voeg een test toe in `test_validation.py`. Test minimaal: correcte bevinding bij slechte data, geen bevinding bij geldige data, en het juiste type (hard/soft/warning).
- **Exitcode-gedrag** → gebruik altijd `pytest.raises(SystemExit) as exc` en assert `exc.value.code` expliciet — `pytest.raises(SystemExit)` zonder codeverificatie vangt ook `sys.exit(0)`.
- **Nieuwe strategie of model** → voeg een contracttest toe die de return-types van `preprocess()` en `predict_nr_of_students()` bewaakt.

### Wat de unit tests niet doen

De unit tests draaien zonder echte inputdata en zonder de volledige pipeline. Correctheid van voorspellingen en end-to-end uitvoer wordt bewaakt door de smoke test (`scripts/test_package.sh`), niet door pytest.

## Documentatie-regel (verplicht bij elke PR)

De documentatie in `docs/` is gericht op **analisten en onderzoekers**: niet alleen hoe de tool werkt, maar het **waarom** achter methodologische keuzes — aannames, beperkingen, en wanneer je output kritisch moet interpreteren.

### Bij elke PR: controleer welke docs-pagina's geraakt zijn

Gebruik de onderstaande mapping om te bepalen welke pagina('s) bijgewerkt moeten worden. Als een wijziging een pagina raakt, **moet die pagina worden meegenomen in de PR**. Een PR zonder bijbehorende docs-update is onvolledig.

**Deze check is verplicht voor elke PR, ook bij refactors en interne fixes.** "Geen gebruikersgerichte wijziging" is een conclusie die je trekt *nadat* je de tabel hebt doorlopen — niet een reden om de check over te slaan.

| Als je dit wijzigt... | ...update dan deze docs-pagina |
|-----------------------|-------------------------------|
| CLI-vlaggen / `cli.py` / `main.py` | `docs/aan-de-slag.md` |
| `configuration.json` / `config.py` | `docs/configuratie.md` |
| Inputformaten / ETL / `etl.py` | `docs/je-data-voorbereiden.md` |
| Datakwaliteitscontroles / validatie | `docs/validatie.md` |
| Outputbestanden / kolomnamen / outputschema | `docs/output-begrijpen.md` |
| `models/sarima.py` / SARIMA-parameters | `docs/methodologie/sarima.md` |
| `models/xgboost_*.py` / XGBoost-features of -parameters | `docs/methodologie/xgboost.md` |
| `models/ratio.py` / ratio-logica | `docs/methodologie/ratio-model.md` |
| `benchmark/` / `models/{classifiers,regressors,forecasters}.py` | `docs/methodologie/benchmarks.md` |
| `strategies/individual.py` / individueel-pipeline / filterregels | `docs/methodologie/individueel.md` |
| Ensemble-logica / gewichten / postprocessor | `docs/methodologie/ensemble.md` |
| Architectuur / pipeline-volgorde | `docs/methodologie/index.md` |
| `gui/` / GUI-pagina's of -gedrag | `docs/gui.md` |

### Wat hoort in een methodologie-pagina?

Methodologie-pagina's (`docs/methodologie/`) moeten antwoord geven op:

1. **Wat doet het?** — functionele beschrijving in begrijpelijke taal
2. **Waarom deze aanpak?** — motivatie voor de modelkeuze, alternatieven overwogen
3. **Aannames** — wat moet waar zijn opdat het model goed werkt
4. **Wanneer vertrouw je het niet?** — concrete situaties waarin de output kritisch beoordeeld moet worden
5. **Relatie tot andere modellen** — hoe past dit in het ensemble

Voeg geen implementatiedetails toe die direct uit de code leesbaar zijn. Schrijf voor iemand die het model wil begrijpen, niet voor iemand die de code wil debuggen.

## Code Conventions

- Gebruik type annotations in nieuwe functies
- Docstrings in Google-stijl (worden opgepikt door mkdocstrings)
- Geen hardgecodeerde instellingsnamen of paden in gedeelde code
- Mutable default arguments vermijden (gebruik `None` + check in body)
