# Externe brondata (raw input)

Deze map bevat de **ruwe externe bestanden** die als input dienen voor het prognosemodel. Ze worden verwerkt door de ETL-stap (`src/data/etl.py`) en de resultaten komen in `data/input/`.

De meegeleverde bestanden bevatten **synthetische demodata**.

## Bestanden

| Bestand | Bron | Beschrijving |
|---------|------|-------------|
| `telbestanden/telbestandY{jaar}W{week}.csv` | Studielink | Wekelijkse telbestanden met cumulatieve vooraanmeldingen per opleiding |
| `oktober_bestand.xlsx` | DUO (1-cijfer HO) | Werkelijke inschrijvingen na 1 oktober (ground truth) |
| `individuele_aanmelddata.csv` | SIS / datawarehouse instelling | Per-student aanmeldingen met persoonskenmerken |

## Dataflow

```
telbestanden/*.csv          ──► rowbind + reformat ──► interpolate ──► data/input/vooraanmeldingen_cumulatief.csv
oktober_bestand.xlsx        ──► calculate_student_count ────────────► data/input/student_count_*.xlsx
individuele_aanmelddata.csv ──► copy ──────────────────────────────► data/input/vooraanmeldingen_individueel.csv
```

De ETL draait standaard bij het starten van de pipeline. Overslaan met `--noetl`.
