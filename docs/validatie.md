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
