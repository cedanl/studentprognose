# Claude Configuration — Studentprognose

## Project Overview

Studentprognose is een open-source CEDA/Npuls tool voor het voorspellen van eerstejaars instroom in het hoger onderwijs. Het combineert tijdreeksmodellen (SARIMA), machine learning (XGBoost) en een ratio-model tot een ensemble-voorspelling, draaiend als installeerbaar Python-package met CLI.

**Doelgroep van de tool:** data-analisten en onderzoekers bij Nederlandse hogeronderwijsinstellingen.

## Tech Stack

- **Python 3.12+** met **uv** voor dependency management
- **pandas / numpy** voor dataverwerking
- **statsmodels** voor SARIMA (SARIMAX met vaste parameters)
- **xgboost** voor classificatie en regressie
- **Rich** voor CLI-output
- **MkDocs + Material** voor documentatie (gehost via GitHub Pages)

## Project Structure

```
src/studentprognose/
├── cli.py               # CLI-definitie (argparse)
├── config.py            # Configuratie laden
├── main.py              # Pipeline-entry
├── models/
│   ├── base.py          # BaseForecaster ABC
│   ├── sarima.py        # SARIMA model
│   ├── xgboost_classifier.py
│   ├── xgboost_regressor.py
│   └── ratio.py         # Ratio-model
├── strategies/
│   ├── individual.py    # Individueel spoor preprocessing
│   ├── cumulative.py    # Cumulatief spoor preprocessing
│   └── combined.py      # Both-modus
├── output/              # Output-schrijvers
└── utils/               # Hulpfuncties

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
| Ensemble-logica / gewichten / postprocessor | `docs/methodologie/ensemble.md` |
| Architectuur / pipeline-volgorde | `docs/methodologie/index.md` |

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
