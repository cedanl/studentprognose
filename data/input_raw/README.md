# Externe brondata (raw input)

Deze map bevat de **ruwe externe bestanden** die als input dienen voor het prognosemodel. Ze worden eerst verwerkt door pre-processing scripts voordat ze in `data/input/` terechtkomen.

De meegeleverde bestanden bevatten **synthetische demodata**.

## Bestanden

| Bestand | Bron | Beschrijving | Pre-processing |
|---------|------|-------------|----------------|
| `telbestanden/telbestandY2024WXX.csv` | Studielink | Wekelijkse telbestanden met vooraanmeldingen per opleiding | `tools/rowbind_and_reformat_studielink_data.py` + `tools/interpolate.py` |
| `oktober_bestand.xlsx` | Studielink (1-cijfer HO) | Werkelijke inschrijvingen na 1 oktober (ground truth) | `tools/calculate_student_count.py` |
| `individuele_aanmelddata.csv` | SIS / datawarehouse instelling | Per-student aanmeldingen met persoonskenmerken | Geen — direct kopieren naar `data/input/vooraanmeldingen_individueel.csv` |
| `afstanden.xlsx` | Instelling | Afstand woonplaats tot universiteit per plaats | Geen — direct kopieren naar `data/input/afstanden.xlsx` |

## Dataflow

```
telbestanden/*.csv  ──► rowbind_and_reformat ──► interpolate ──► data/input/vooraanmeldingen_cumulatief.csv
oktober_bestand.xlsx ──► calculate_student_count ──────────────► data/input/student_count_*.xlsx
individuele_aanmelddata.csv ───────────────────────────────────► data/input/vooraanmeldingen_individueel.csv
afstanden.xlsx ────────────────────────────────────────────────► data/input/afstanden.xlsx
```
