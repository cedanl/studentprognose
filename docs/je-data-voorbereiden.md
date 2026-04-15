# Je data voorbereiden

TODO: uitwerken op basis van PIPELINE.md en datakwaliteitscontroles.

Zie ook `doc/PIPELINE.md` voor de volledige dataflow inclusief ETL-stappen.

## Verplichte inputbestanden

| Bestand | Beschrijving | Bron |
|---------|-------------|------|
| `vooraanmeldingen_cumulatief.csv` | Gewogen/ongewogen vooraanmelders per opleiding, herkomst, week, jaar | Studielink → ETL |
| `vooraanmeldingen_individueel.csv` | Per-student aanmelddata met persoonskenmerken | Osiris/Usis → ETL |
| `student_count_first-years.xlsx` | Werkelijk aantal eerstejaars per opleiding/herkomst/jaar | DUO oktober-bestand |
| `student_count_higher-years.xlsx` | Werkelijk aantal hogerjaars | DUO oktober-bestand |
| `student_volume.xlsx` | Totaal studentvolume | DUO oktober-bestand |
