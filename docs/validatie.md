# Validatie

De pipeline voert vóór de ETL automatisch een datakwaliteitscontrole uit op alle ruwe inputbestanden. Dit voorkomt dat fouten in de brondata pas later in de pipeline of in de output zichtbaar worden.

Gebruik `--noetl` om zowel de ETL als de validatie over te slaan (alleen als de data eerder al gevalideerd is).

## Drie typen bevindingen

| Type | Gedrag | Wanneer gebruiken |
|------|--------|------------------|
| **Hard error** | Pipeline stopt direct | Data is structureel onbruikbaar (ontbrekende kolommen, onleesbaar bestand) |
| **Soft error** | Pipeline vraagt om bevestiging | Data is twijfelachtig maar niet per se fout (onverwachte waarden, mogelijk verkeerd jaar) |
| **Waarschuwing** | Pipeline loopt door, melding in console | Automatisch gecorrigeerd, of niet-kritiek |

In geautomatiseerde runs (CI/CD) gebruik je `--yes` om de soft-error prompt te omzeilen.

## Gevalideerde bestanden

### Telbestanden (`data/input_raw/telbestanden/`)

| Controle | Type | Wat wordt gecheckt |
|----------|------|-------------------|
| Map bestaat | Hard error | `data/input_raw/telbestanden/` moet bestaan |
| Bestanden aanwezig | Hard error | Minimaal één bestand met patroon `telbestandY{jaar}W{week}.csv` |
| Verplichte kolommen | Hard error | `Studiejaar`, `Isatcode`, `Groepeernaam`, `Aantal`, `meercode_V`, `Herinschrijving`, `Hogerejaars`, `Herkomst` |
| Weeknummer in bestandsnaam | Hard error | Weeknummer moet tussen 1 en 53 liggen |
| Collegejaar bereik | Soft error | `Studiejaar` buiten `[huidig jaar − 15, huidig jaar + 2]` |
| Herkomst geldige waarden | Soft error | Elke waarde in `Herkomst` moet `N`, `E` of `R` zijn |
| Herinschrijving geldige waarden | Soft error | Elke waarde moet `J` of `N` zijn |
| Hogerejaars geldige waarden | Soft error | Elke waarde moet `J` of `N` zijn |
| meercode_V = 0 | Soft error | Leidt tot deling door nul in ETL |
| Aantal < 0 | Soft error | Negatieve aantallen zijn inhoudelijk onjuist |
| Ontbrekende waarden `Aantal` | Waarschuwing / Soft error | > 5% ontbrekend → waarschuwing; > 30% → soft error |
| Gaten tussen weken | Waarschuwing | Gat van > 2 weken binnen een jaar |
| Witruimte in categorische waarden | Waarschuwing | Automatisch gestript (`"J "` → `"J"`) |

### Individuele aanmelddata (`data/input_raw/individuele_aanmelddata.csv`)

| Controle | Type | Wat wordt gecheckt |
|----------|------|-------------------|
| Bestand bestaat | Waarschuwing | Niet aanwezig → overgeslagen (individueel spoor werkt niet) |
| Verplichte kolommen | Hard error | `Collegejaar`, `Croho`, `Inschrijfstatus`, `Datum Verzoek Inschr` (via kolomnamen-mapping) |
| Ontbrekende waarden | Waarschuwing / Soft error | Per verplichte kolom, zelfde drempels als telbestanden |

### Oktober-bestand (`data/input_raw/oktober_bestand.xlsx`)

| Controle | Type | Wat wordt gecheckt |
|----------|------|-------------------|
| Bestand bestaat | Waarschuwing | Niet aanwezig → studentaantallen worden niet berekend |
| Verplichte kolommen | Hard error | `Collegejaar`, `Groepeernaam Croho`, `Aantal eerstejaars croho`, `EER-NL-nietEER`, `Examentype code`, `Aantal Hoofdinschrijvingen` |
| Collegejaar bereik | Soft error | Zelfde bereikcontrole als telbestanden |
| Ontbrekende waarden | Waarschuwing / Soft error | Per verplichte kolom |

## Pre-prediction checks

Vóór elke modelrun voert de pipeline drie aanvullende checks uit op de **cumulatieve vooraanmelddata**. Ze draaien per `(jaar, week)`-combinatie, na ETL maar vóór de modellen.

| Check | Type | Wat wordt gecheckt |
|-------|------|-------------------|
| Decimaalintegriteit | Hard stop | `Gewogen vooraanmelders` bevat strings met komma's of niet-numerieke waarden |
| Lege dataset | Hard stop | Geen rijen aanwezig voor het gevraagde jaar+week |
| Historisch realisme | Hard stop / waarschuwing | Afwijking t.o.v. dezelfde week vorig jaar per opleiding/herkomst/examentype |

### Historisch realisme — drempelwaarden

Vergelijking per `(Croho groepeernaam, Herkomst, Examentype)`:

| Situatie | Gedrag |
|----------|--------|
| Afwijking > `max(25, 70% van vorig jaar)` | Hard stop — pipeline stopt |
| Afwijking > `max(15, 30% van vorig jaar)` | Waarschuwing — pipeline loopt door |

De absolute vloer (`max(…)`) voorkomt vals-positieven bij kleine opleidingen: een programma met 10 studenten vorig jaar en 18 dit jaar (80% relatief, 8 absoluut) triggert geen hard stop omdat de absolute drempel (25) niet gehaald wordt.

Numerus-fixus-opleidingen (examentype Bachelor) worden overgeslagen — hun aanmeldpatroon is beleidsmatig bepaald en niet vergelijkbaar met het historische patroon.

Als er geen vorig-jaar-data beschikbaar is (nieuwe opleiding), wordt de check stilzwijgend overgeslagen.

### Hard stop omzeilen met `--yes`

De decimaalcheck en lege-dataset-check zijn nooit te omzeilen: corrupte of afwezige data heeft geen veilige fallback. De historisch-realismecheck wél — gebruik `--yes` om een extreme afwijking te accepteren en door te gaan:

```bash
uv run studentprognose --yes -y 2024 -w 10
```

Met `--yes` verschijnt een waarschuwing in de console maar stopt de pipeline niet. Gebruik dit bewust: een extreme afwijking kan duiden op een Studielink-probleem dat je niet wilt meenemen in de modeltraining.

## Post-prediction checks

Nadat het ensemble zijn voorspellingen heeft opgeleverd, voert de pipeline twee informatieve checks uit. Ze stoppen de pipeline **nooit** — ze printen alleen waarschuwingen in de console.

| Check | Wat wordt gecheckt |
|-------|-------------------|
| Trend-realisme YoY | `Ensemble_prediction` wijkt > 50% én > 20 absoluut af van `Gewogen vooraanmelders` dezelfde week vorig jaar |
| Trend-realisme WoW | `Ensemble_prediction` wijkt > 30% én > 15 absoluut af van de voorspelling van de vorige week |
| NF-cap overschrijding | Gesommeerde `Ensemble_prediction` per numerus-fixus-opleiding overschrijdt het geconfigureerde plafond |

Pre-master-rijen worden uitgesloten van de NF-cap-check: ze tellen niet mee als nieuwe eerstejaars.

De week-op-week-check slaat de eerste week van elke run over (geen vorige week beschikbaar) en week 39 (eerste week van het nieuwe aanmeldseizoen — de vorige week, 38, is het einde van het vorige seizoen en geen zinvolle referentie).

!!! note "Alleen beschikbaar in combinatiemodus"
    De post-prediction checks zijn alleen actief als de pipeline wordt gestart met `-d both` of `-d b`. In cumulatief-enkel (`-d c`) of individueel-enkel (`-d i`) modus bestaat de `Ensemble_prediction`-kolom niet en worden de checks stilzwijgend overgeslagen.

## Drempels aanpassen

De standaarddrempels voor NaN-percentages en jaarbereiken zijn instelbaar via `configuration.json`. Zie [Configuratie — validation](configuratie.md#validation--validatiedrempels-overschrijven).

## Een validatiefout oplossen

**Hard error — ontbrekende kolommen:**
De kolomnaam in jouw bestand wijkt af van de kanonieke naam. Voeg een kolomnamen-mapping toe in `configuration.json` onder `columns.individual` of `columns.oktober`.

**Soft error — onverwacht collegejaar:**
Controleer of het bestand het juiste studiejaar bevat. Als de afwijking verwacht is (bijv. historische data), kun je `collegejaar_min_offset` verhogen of met `--yes` doorgaan.

**Soft error — ongeldige herkomstwaarden:**
Jouw instelling gebruikt mogelijk `"ONBEKEND"` of een andere waarde. Voeg die toe aan `validation.telbestand.herkomst_allowed` in je configuratie.

**Waarschuwing — witruimte gestript:**
De data wordt automatisch gecorrigeerd. Overweeg de brondata te corrigeren om dit te voorkomen.
