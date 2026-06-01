# Externe brondata (raw input)

Deze map bevat de **ruwe externe bestanden** die als input dienen voor het prognosemodel. Ze worden verwerkt door de ETL-stap (`src/data/etl.py`) en de resultaten komen in `data/input/`.

De meegeleverde bestanden bevatten **synthetische demodata**.

## Bestanden

| Bestand | Bron | Beschrijving |
|---------|------|-------------|
| `telbestanden/telbestandY{jaar}W{week}.csv` | Studielink | Wekelijkse telbestanden met cumulatieve vooraanmeldingen per opleiding |
| `oktober_bestand.xlsx` | Eigen levering instelling (telbestand studenten) | Werkelijke inschrijvingen per opleiding/herkomst/jaar (ground truth) |
| `individuele_aanmelddata.csv` | SIS / datawarehouse instelling | Per-student aanmeldingen met persoonskenmerken |

## Kolomspecificaties

- **Volledige kolomdefinities per bestand** — zie [`data/metadata/data_dictionary.csv`](../metadata/data_dictionary.csv).
- **Studielink telbestanden** volgen het officiële *Programma van Levering Telbestand Studielink* (Stichting Studielink, versie Definitief 2.8, 3 november 2022). PDF: <https://www.tignl.eu/downloads/studielink/pvl%20telbestand%20studielink.pdf>. Alle veldverwijzingen (`§5.x`) in de data dictionary verwijzen naar paragrafen in dit document.
- **Korte versie voor analisten** — zie [`docs/je-data-voorbereiden.md`](../../docs/je-data-voorbereiden.md).

## Dataflow

```
telbestanden/*.csv          ──► rowbind + reformat ──► interpolate ──► data/input/vooraanmeldingen_cumulatief.csv
oktober_bestand.xlsx        ──► calculate_student_count ────────────► data/input/student_count_*.xlsx
   (telbestand studenten)
individuele_aanmelddata.csv ──► copy ──────────────────────────────► data/input/vooraanmeldingen_individueel.csv
```

De ETL draait standaard bij het starten van de pipeline. Overslaan met `--noetl`.
