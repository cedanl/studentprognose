# Aan de slag

## Installatie

```bash
pip install studentprognose
```

Of met uv:

```bash
uv add studentprognose
```

## Eerste run met demodata

TODO: demodata toevoegen en walkthrough uitschrijven.

## Vereisten

- Python 3.12+
- Inputbestanden in het juiste formaat (zie [Je data voorbereiden](je-data-voorbereiden.md))

## CLI-overzicht

```bash
studentprognose --help
```

| Vlag | Beschrijving |
|------|-------------|
| `-d c` | Cumulatief spoor (Studielink-data) |
| `-d i` | Individueel spoor (Osiris/Usis-data) |
| `-d b` | Beide sporen + ensemble (standaard) |
| `--noetl` | ETL-stap overslaan |
| `--ci test N` | CI-modus met subset van N opleidingen |
