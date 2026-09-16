# Snelstart (5 min)

Van niets naar je eerste prognose — met **demodata**, dus je hebt nog geen eigen data nodig.

!!! tip "Eerst installeren"
    Nog niet geïnstalleerd? Zie [Installeren](installeren.md) (drie commando's).

## 1. Projectmap aanmaken en demodata ophalen

```bash
mkdir mijn-prognose && cd mijn-prognose
studentprognose init
```

`init` maakt de mapstructuur en een `configuration/configuration.json` met alle standaardwaarden. Aan het einde vraagt het of je demodata wilt downloaden (4 MB):

```
Wil je demodata downloaden om direct te starten? (4 MB, ~10 sec) [j/n]: j
  Downloaden...
  demo-data.zip: 4.30MB [00:08, 512kB/s]
  Uitpakken...
  Demodata geïnstalleerd in data/input_raw/ ✓
```

Kies `n` als je meteen je eigen data wilt gebruiken — zie dan [Je data klaarzetten](je-data-voorbereiden.md).

## 2. Draaien

```bash
studentprognose -d c -y 2024
```

Wat er op het scherm verschijnt:

```
==== Valideren van ruwe inputdata ====

  Bestand                                     Status    Nodig voor
  ──────────────────────────────────────────────────────────────────
  data/input_raw/telbestanden                 ✓         -d cumulative, -d both
  data/input_raw/individuele_aanmelddata.csv  ✗         -d individual, -d both
  data/input_raw/oktober_bestand.xlsx         ✓         studentaantallen (optioneel)

  Beschikbare modi:
    -d cumulative      ✓
    -d individual      ✗  data/input_raw/individuele_aanmelddata.csv ontbreekt
    -d both            ✗  data/input_raw/individuele_aanmelddata.csv ontbreekt

==== Processing raw input data ====
[1/4] Rowbinding telbestanden...        → data/input/vooraanmeldingen_cumulatief.csv
[2/4] Interpolating missing weeks...    → data/input/vooraanmeldingen_cumulatief.csv
[3/4] Calculating student counts...     → data/input/student_count_*.xlsx
==== ETL complete ====

Loading data...
Preprocessing...

========================================
  Dataset:       cumulatief
  Trainingsdata: jaren 2020-2024
  Voorspelling:  jaar [2024], vanaf week [20]
  Voorspelt:     week 21 t/m 36 (16 weken vooruit)
========================================

Predicting first-years: 2024-20...
Start parallel predicting...
Postprocessing...
Saving output...
```

!!! note "Waarschuwingen zijn normaal bij de demodata"
    - **Individuele data ontbreekt** — de demodata bevat alleen telbestanden. `-d c` werkt prima; voor `-d b` of `-d i` heb je eigen individuele aanmelddata nodig.
    - **Zomergaten** — week 35–40 ontbreken in de telbestanden (Studielink levert in die periode niet). Ook deze waarschuwingen kun je negeren.

Een run duurt op de demodata typisch **2–4 minuten**, afhankelijk van je machine.

## 3. Resultaat bekijken

De output staat in `data/output/`. Open **`output_first-years_cumulatief.xlsx`**; de kolom `SARIMA_cumulative` bevat de prognose per opleiding.

Wil je het visueel? Voeg `--dashboard` toe:

```bash
studentprognose -d c -y 2024 --dashboard
```

<iframe src="../assets/plots/output_cockpit.html" width="100%" height="400" frameborder="0" style="border-radius: 8px;"></iframe>

*Voorbeelddashboard op de demodata: prognose, conversie en betrouwbaarheid per opleiding.*

## Wat nu?

| Ik wil… | Ga naar |
|---------|---------|
| Mijn eigen data gebruiken | [Je data klaarzetten](je-data-voorbereiden.md) |
| Begrijpen wat `-w`, `-y` en `-d` doen | [Draaien & CLI](aan-de-slag.md) |
| De output-cijfers interpreteren | [Output lezen](output-begrijpen.md) |
| Mijn eigen instelling instellen | [Configuratie](configuratie.md) |
| Weten hoe de modellen werken | [Methodologie](methodologie/index.md) |

!!! note "Kom je een term niet tegen?"
    De [Begrippenlijst](begrippen.md) legt peilweek, spoor, ETL, ensemble en de rest in gewone taal uit.
