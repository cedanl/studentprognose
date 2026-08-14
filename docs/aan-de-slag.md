# Draaien & CLI

Deze pagina legt uit wat de CLI-vlaggen doen en hoe je gerichte runs draait. Wil je eerst gewoon íéts zien draaien? Doe de [Snelstart](snelstart.md) met demodata.

!!! abstract "In het kort"
    - `studentprognose` (zonder vlaggen) draait beide sporen op de laatste beschikbare week/jaar.
    - `-w` = **peilweek**, `-y` = **collegejaar**, `-d` = welk(e) **spoor/sporen**.
    - Output staat in `data/output/`. Zie [Output lezen](output-begrijpen.md).

## Projectmap aanmaken

Draai eenmalig in je werkmap:

```bash
studentprognose init
```

Dit schrijft de configuratie en mapstructuur:

- `configuration/configuration.json` — configuratie met alle standaardwaarden
- `configuration/filtering/base.json` — leeg filter (geen opleiding-/herkomstfilter)
- `data/input_raw/telbestanden/` — map voor je Studielink-telbestanden
- `data/input_raw/README.md` — beschrijving van welke bestanden hier horen
- `data/output/` — map voor de output én de doorlopende audittrail (`_totaal_*.xlsx`)

Daarna vraagt `init` of je **demodata wilt downloaden** (4 MB telbestanden + studentaantallen) zodat je direct kunt draaien zonder eigen data. Kies `n` als je meteen je eigen data wilt gebruiken.

Bestaat een bestand al, dan wordt het overgeslagen; je kunt `init` veilig opnieuw draaien. Eigen data klaarzetten doe je op [Je data klaarzetten](je-data-voorbereiden.md).

## Wat voorspelt `-w 16 -y 2024`?

`-w` is de **peilweek** (kalenderweek 1–52) en `-y` het **collegejaar** waarvoor je een prognose wilt. `-w 16 -y 2024` betekent: *"gebruik alle vooraanmelddata tot en met week 16 van collegejaar 2024 en geef me het verwachte aantal inschrijvingen voor datzelfde jaar"*.

!!! example "Voorbeeld · B Bedrijfskunde · NL (demodata)"
    **178** vooraanmelders @ wk 16 &nbsp;→&nbsp; **520** verwacht @ wk 38 &nbsp;→&nbsp; **400** ingeschreven (77% yield)

    <iframe src="../assets/plots/whatif_timeline.html" width="100%" height="540" frameborder="0" style="border-radius: 8px;"></iframe>

    De blauwe lijn toont de geobserveerde cumulatieve vooraanmelders tot peilmoment (wk 16); de oranje lijn extrapoleert naar week 38. De band groeit met de horizon: ±12 bij peilmoment, ±34 bij wk 38.

!!! info "Waarom elke week opnieuw draaien?"
    Eén peilweek geeft het beeld van *dat* moment. Draai je wekelijks, dan volg je de trends mee — het knikpunt rond de 1-mei-deadline, een achterblijvende zomernaloop — en zie je vroege signalen om capaciteit en marketing tijdig bij te sturen.

## Eerste run

Laat je `-w` en `-y` weg, dan kiest de pipeline automatisch de **laatste beschikbare week en jaar uit je inputdata**. Dat voorkomt dat de tool faalt omdat de huidige systeemweek nog niet in je trainingsdata zit.

```bash
studentprognose                    # beide sporen, laatste week/jaar (standaard)
studentprognose -w 10 -y 2024      # specifieke week en jaar
studentprognose -d c               # alleen cumulatief spoor
studentprognose --noetl            # data al eerder verwerkt — sla ETL over
```

## CLI-referentie

Alle opties: `studentprognose --help`.

### Subcommando's

| Commando | Beschrijving |
|----------|-------------|
| `init` | Maak een nieuwe projectmap aan met configuratie en mappenstructuur |
| `benchmark` | Vergelijk alternatieve modellen (`-d c` of `-d i` verplicht, zie [Benchmarks](methodologie/benchmarks.md)) |
| `tune` | Stem het cumulatieve spoor af (`-d c` verplicht, zie [Hyperparameter tuning](gevorderd-gebruik.md#hyperparameter-tuning)) |

### Vlaggen

| Vlag | Waarden | Standaard | Beschrijving |
|------|---------|-----------|-------------|
| `-w` | weeknummer(s) | laatste week in data | Voorspelweek(en), bijv. `-w 10` of `-w 8:12` |
| `-y` | jaar(en) | laatste jaar in data | Voorspeljaar(en), bijv. `-y 2024` of `-y 2023 2024` |
| `-d` | `b` / `c` / `i` | `b` | Dataset: `both`, `cumulative`, `individual` |
| `-sy` | `f` / `h` / `v` | `f` | Studentjaar: `first-years`, `higher-years`, `volume` |
| `-c` | pad | `configuration/configuration.json` | Configuratiebestand |
| `-f` | pad | `configuration/filtering/base.json` | Filterbestand |
| `--institution` | instellingscode(s) | alle instellingen | Beperk de teldata tot één of meer instellingen — zie [hieronder](#-institution) |
| `-sk` | getal | `0` | Skip N jaren (backtesting) |
| `--noetl` | — | uit | Sla ETL én validatie over |
| `--yes` | — | uit | Sla validatieprompts over (voor CI/CD) |
| `--no-warnings` | — | uit | Onderdruk UserWarning-meldingen (historisch realisme, ontbrekende lag-fallback) |
| `--dashboard` | — | uit | Genereer interactieve dashboards (`data/output/visualisations/`) |
| `--tune-target` | `regressor` / `sarima` / `both` | `regressor` | Welke trap `tune` afstemt |
| `--ci test N` | getal | — | Testmodus: beperkt tot N opleidingen |

Weekbereiken mogen: `-w 8:12` = `-w 8 9 10 11 12`.

!!! note "Eerstejaars als focus"
    De tool draait standaard op `-sy f` (eerstejaars). Volume (`-sy v`) is een vervolgstap — gebruik die pas als je eerstejaarsvoorspelling op orde is.

### `--institution`

De landelijke Studielink-teldata bevat rijen van álle instellingen. Met `--institution` scoop je de run op je eigen instelling(en):

```bash
studentprognose -d c --institution 21PC          # één instelling
studentprognose -d c --institution 21PC 00IC      # meerdere
```

De vlag overschrijft de config-key [`institution_filter`](configuratie.md#institution_filter-beperk-de-teldata-tot-een-of-meer-instellingen); laat je hem weg, dan geldt de configuratiewaarde. Een onbekende code stopt de run met een duidelijke fout. Zet je je eigen Brincode vast in de config, dan hoef je de vlag niet elke keer mee te geven.

!!! warning "Standaard reken je over álle instellingen"
    Zonder filter rekent het cumulatieve spoor over de héle landelijke teldata — zelden wat je wilt. De run print bij elke start één regel met de scope, zodat je nooit ongemerkt over de verkeerde (of alle) instellingen rekent. Zet daarom je eigen Brincode vast via [`institution_filter`](configuratie.md#institution_filter-beperk-de-teldata-tot-een-of-meer-instellingen).

## De tool vanuit Python of de cloud aansturen

Wil je `studentprognose` aanroepen vanuit een notebook, MS Fabric, Databricks of cloud-opslag, hyperparameters afstemmen, of het model evalueren met scalaire metrieken? Zie [Gevorderd gebruik](gevorderd-gebruik.md). Voor een gewone prognose via de command line heb je dat niet nodig.

## Bekende valkuil: stille modus-downgrade

!!! warning "Let op bij `-d b`"
    Gebruik je `-d both` (standaard) maar ontbreekt de individuele aanmelddata (of mist die de verwachte jaren), dan **valt de pipeline terug op `-d cumulative`**. Je krijgt een waarschuwing, maar de output bevat dan alleen cumulatieve voorspellingen — geen ensemble, geen `SARIMA_individual`.

    Verwacht je het ensemble? Controleer of de `SARIMA_individual`-kolommen in je output staan.

## Veelvoorkomende fouten

Installatiefouten (`command not found`, `ModuleNotFoundError`) staan bij [Installeren → Installatie mislukt?](installeren.md#installatie-mislukt).

??? failure "`TerminatedWorkerError` tijdens SARIMA (Windows)"
    Een worker-proces is op signaal-niveau gestopt — meestal geheugendruk bij veel parallelle workers op een laptop met beperkt RAM. De pipeline probeert automatisch opnieuw met 2 workers. Werkt dat niet, zet dan in `configuration/configuration.json`:

    ```json
    "runtime": { "cpu_count": 1 }
    ```

    `cpu_count: 1` schakelt parallelisatie uit. Trager, maar zonder worker-spawns en daarmee zonder deze klasse fouten.
