# Aan de slag

## Installatie

```bash
pip install studentprognose
```

Of met uv in een project:

```bash
uv add studentprognose
```

Vereisten: Python 3.12+

## Je data neerzetten

Zet je inputbestanden in de juiste mappen voordat je de pipeline start:

```
data/
├── input/                          ← verwerkte inputbestanden (na ETL)
│   ├── vooraanmeldingen_cumulatief.csv
│   ├── vooraanmeldingen_individueel.csv
│   ├── student_count_first-years.xlsx
│   ├── student_count_higher-years.xlsx
│   └── student_volume.xlsx
└── input_raw/                      ← ruwe bronbestanden (voor ETL)
    ├── telbestanden/               ← Studielink telbestanden
    │   ├── telbestandY2024W01.csv
    │   └── ...
    ├── individuele_aanmelddata.csv ← Osiris/Usis export
    └── oktober_bestand.xlsx        ← DUO oktober-bestand
```

Zie [Je data voorbereiden](je-data-voorbereiden.md) voor kolomspecificaties per bestand.

## Eerste run

```bash
# Beide sporen + ensemble (standaard) — voorspelling voor de huidige week
uv run studentprognose

# Specifieke week en jaar
uv run studentprognose -w 10 -y 2025

# Alleen cumulatief spoor (geen individuele data nodig)
uv run studentprognose -d c

# ETL overslaan (data al eerder verwerkt)
uv run studentprognose --noetl

# Hyperparameter tuning draaien (resultaten gecacht)
uv run studentprognose --tune -w 12 -y 2025 -d b --noetl
```

De output verschijnt in `data/output/`.

### Hyperparameter tuning

Met `--tune` draait de pipeline een grid search over XGBoost-hyperparameters en SARIMA-ordes vóór de reguliere voorspelling. De gevonden parameters worden opgeslagen in `data/output/tuning_cache.json` en automatisch hergebruikt bij volgende runs (ook zonder `--tune`).

Tuning duurt 5–60 minuten afhankelijk van de hoeveelheid data. Je hoeft het niet bij elke run te doen — typisch eenmaal per studiejaar of na significante datawijzigingen. Zie [Configuratie](configuratie.md#hyperparameters) voor de configureerbare grid-search-parameters.

## CLI-referentie

| Vlag | Waarden | Standaard | Beschrijving |
|------|---------|-----------|-------------|
| `-w` | weeknummer(s) | huidige week | Voorspelweek(en), bijv. `-w 10` of `-w 8:12` |
| `-y` | jaar(en) | huidig jaar | Voorspeljaar(en), bijv. `-y 2025` of `-y 2024 2025` |
| `-d` | `b` / `c` / `i` | `b` | Dataset: `both`, `cumulative`, `individual` |
| `-sy` | `f` / `h` / `v` | `f` | Studentjaar: `first-years`, `higher-years`, `volume` |
| `-c` | pad | `configuration/configuration.json` | Configuratiebestand |
| `-f` | pad | `configuration/filtering/base.json` | Filterbestand |
| `-sk` | getal | `0` | Skip N jaren (backtesting) |
| `--noetl` | — | uit | Sla ETL én validatie over |
| `--yes` | — | uit | Sla validatieprompts over (voor CI/CD) |
| `--tune` | — | uit | Draai hyperparameter tuning vóór de voorspelling |
| `--ci test N` | getal | — | Testmodus: beperkt tot N opleidingen |

Weekbereiken zijn mogelijk: `-w 8:12` is gelijk aan `-w 8 9 10 11 12`.

## Bekende valkuil: stille modus-downgrade

!!! warning "Let op bij `-d b`"
    Als je `-d both` (standaard) gebruikt maar de individuele aanmelddata ontbreekt of
    bevat niet de verwachte jaren, **valt de pipeline terug op `-d cumulative`**.
    De tool toont dan een waarschuwing, maar de output bevat alleen cumulatieve
    voorspellingen — geen ensemble, geen `SARIMA_individual`.

    Controleer altijd of `SARIMA_individual` kolommen aanwezig zijn in je output als je
    het ensemble verwacht.
