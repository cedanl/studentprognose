# Output begrijpen

De pipeline schrijft de resultaten naar `data/output/`. Er zijn twee typen uitvoerbestanden: een tussenresultaat en drie eindresultaten.

!!! tip "Uitgebreide versie — Jupyter notebook"
    Een uitvoerbare versie die een in-memory pipeline-run uitvoert en de outputkolommen stap
    voor stap interpreteert (inclusief MAE/MAPE-voorbeelden en modelvergelijking) staat in
    [`notebooks/06_output_interpreteren.ipynb`](https://github.com/cedanl/studentprognose/blob/main/notebooks/06_output_interpreteren.ipynb).

## Outputbestanden

| Bestand | Fase | Beschrijving |
|---------|------|-------------|
| `output_prelim_{modus}.xlsx` | Tussenresultaat | Voorlopige voorspellingen vóór ratio-model, ensemble en foutmaten. Nuttig voor debugging. |
| `output_first-years_{modus}.xlsx` | Eindresultaat | Eerstejaars voorspellingen per opleiding/herkomst/week |
| `output_volume_{modus}.xlsx` | Eindresultaat | Totaal studentvolume-voorspellingen (alleen bij `-sy v`) |
| `_totaal_{studentjaar}_{modus}.xlsx` | Audittrail | Doorlopend bestand waar elke run zijn rijen aan toevoegt. Wordt nooit als input ingelezen. Zie [Audittrail](#audittrail-_totaalxlsx). |

`{modus}` is `cumulatief`, `individueel` of `beide`, afhankelijk van de gebruikte `-d` vlag.
`{studentjaar}` is `first-years`, `higher-years` of `volume`, afhankelijk van `-sy`.

## Audittrail (`_totaal_*.xlsx`)

Naast de week-specifieke `output_*.xlsx`-bestanden — die elke run worden overschreven — onderhoudt de pipeline een doorlopend audittrail per modus. Elke run voegt zijn rijen idempotent toe aan `data/output/_totaal_{studentjaar}_{modus}.xlsx`:

- **Bij eerste run** ontstaat het bestand met de kolommen van de huidige run.
- **Bij elke vervolgrun** worden de rijen toegevoegd. Bestaande rijen met dezelfde sleutel (jaar, week, opleiding, herkomst, examentype — de exacte kolomnamen volgen uit `column_roles`, in de standaardconfig: `Collegejaar`, `Weeknummer`, `Croho groepeernaam`, `Herkomst`, `Examentype`) worden **overschreven** in plaats van gedupliceerd — opnieuw draaien voor dezelfde week is dus veilig.
- **Run_date-kolom**: elke geschreven rij krijgt de datum van de run. Handig om te zien wanneer een voorspelling voor een (jaar, week)-combo is gegenereerd, bv. bij modelwijzigingen.

Het bestand wordt **nooit** door de pipeline als input ingelezen — de data loader leest enkel de paden uit `configuration.json`. Daarmee is een circulaire afhankelijkheid structureel uitgesloten.

### Wanneer is de audittrail nuttig?

- **Trends over weken heen** — open één bestand om te zien hoe de voorspelling voor een opleiding zich gedurende het inschrijfseizoen ontwikkelde, zonder dat je zelf wekelijkse exports hoeft te koppelen.
- **Achteraf reconstrueren** — `Run_date` toont wanneer een voorspelling is gemaakt, wat helpt bij retrospectief onderzoek of audits.

### Bekende limieten

- **Filterwijzigingen tussen runs** worden niet gedetecteerd. Als je in week 10 met `-f base.json` draait en in week 11 met een ander filter, leven beide rijensets naast elkaar in dezelfde audittrail. Geen filterhash; documenteer welke filterconfig hoort bij welke run zelf.
- **Modelversie-drift**: rijen uit oudere runs kunnen door een andere modelvariant zijn gegenereerd. `Run_date` maakt dit traceerbaar maar is geen vervanging voor versiebeheer van de pipeline zelf.
- **Numerus-fixusvoorspellingen** worden net als in de week-output afgekapt; de audittrail erft die waarden ongewijzigd.
- **CI-modus** (`--ci test N`) schrijft géén audittrail — alleen reguliere runs.

## Kolomdefinities

Elke rij in de output beschrijft een combinatie van **opleiding × herkomst × examentype × week × jaar**.

### Voorspelkolommen

| Kolom | Beschikbaar bij | Omschrijving |
|-------|-----------------|-------------|
| `SARIMA_cumulative` | `-d c` of `-d b` | SARIMA-voorspelling op basis van Studielink telbestanden |
| `SARIMA_individual` | `-d i` of `-d b` | SARIMA-voorspelling op basis van individuele aanmelddata |
| `Prognose_ratio` | `-d c` of `-d b` | Ratio-modelvoorspelling (3-jaars historisch gemiddelde) |
| `Ensemble_prediction` | `-d b` | Gewogen combinatie van bovenstaande modellen |
| `Baseline` | `-d c` of `-d b` | Naïeve referentie: `vorig_jaar_inschrijvingen / vorig_jaar_aanmeldingen × huidige_aanmeldingen`. Identiek aan `Prognose_ratio`, maar expliciet benoemd voor gebruik als planningsreferentie. |

Als `-d b` is gebruikt maar individuele data ontbreekt, zijn `SARIMA_individual` en `Ensemble_prediction` leeg — zie [bekende valkuil](aan-de-slag.md#bekende-valkuil-stille-modus-downgrade).

### Actuele aanmeldcijfers in de output

De output bevat naast voorspellingen ook de actuele Studielink-cijfers voor het voorspelmoment:

| Kolom | Omschrijving |
|-------|-------------|
| `Gewogen vooraanmelders` | Gewogen aanmeldingen op predict_week (actueel) |
| `Ongewogen vooraanmelders` | Ongewogen aanmeldingen op predict_week |
| `Aantal aanmelders met 1 aanmelding` | Aanmelders die exclusief voor deze opleiding kozen |
| `Inschrijvingen` | Reeds ingeschreven studenten op predict_week |

Deze kolommen worden direct uit de cumulatieve Studielink-snapshot gevuld voor de rijen die overeenkomen met het voorspeljaar en de voorspelweek. Zo staan de voorspelling en de actuele stand altijd op dezelfde rij.

### Wanneer is de Baseline betrouwbaarder dan het ensemble?

De `Baseline` is in bepaalde situaties een betrouwbaardere leidraad dan `Ensemble_prediction`:

- **Stabiele, grote opleidingen** — als een opleiding jaar op jaar een vaste conversieverhouding (aanmelding → inschrijving) heeft, geeft de naïeve ratio een scherpe schatting met weinig ruis.
- **Weinig trainingsdata** — SARIMA en XGBoost hebben meerdere jaren nodig om betrouwbare patronen te leren. Bij een jonge opleiding (< 4 jaar data) is het ensemble onzeker; de ratio is dan vaak stabieler.
- **Grote afwijking tussen Baseline en Ensemble** — als de twee ver uit elkaar liggen (> 15–20%), is dat een signaal om de invoerdata te controleren. Het ensemble kan reageren op een anomalie in de aanmelddata; de ratio weerspiegelt puur het huidige aanmeldvolume.

Omgekeerd is het ensemble betrouwbaarder als de conversieverhouding snel verandert (nieuw instroombeleid, deadlineverschuiving) of als de opleiding een sterk niet-lineair aanmeldpatroon heeft dat de ratio niet kan volgen.

### Foutmaatkolommen

Foutmaten zijn gebaseerd op **historische modelfouten** — hoe goed presteerde elk model in voorgaande jaren op dezelfde opleiding/herkomst/week? Ze zijn dus geen maat voor de nauwkeurigheid van de huidige voorspelling.

| Kolom | Omschrijving |
|-------|-------------|
| `MAE_Ensemble_prediction` | Gemiddelde absolute fout van het ensemble in voorgaande jaren |
| `MAE_Prognose_ratio` | Gemiddelde absolute fout van het ratio-model |
| `MAE_SARIMA_cumulative` | Gemiddelde absolute fout van SARIMA cumulatief |
| `MAE_SARIMA_individual` | Gemiddelde absolute fout van SARIMA individueel |
| `MAPE_Prognose_ratio` | Gemiddelde procentuele fout van het ratio-model |
| `MAPE_SARIMA_cumulative` | Gemiddelde procentuele fout van SARIMA cumulatief |
| `MAPE_SARIMA_individual` | Gemiddelde procentuele fout van SARIMA individueel |

**MAE** (Mean Absolute Error): gemiddeld aantal studenten waarmee het model afweek.
Een MAE van 8 betekent: het model zat in het verleden gemiddeld 8 studenten naast de werkelijkheid.

**MAPE** (Mean Absolute Percentage Error): gemiddelde procentuele afwijking.
Een MAPE van 0.12 betekent: het model zat gemiddeld 12% naast de werkelijkheid.

!!! note "Foutmaten zijn alleen beschikbaar als er historische data is"
    Bij een eerste run of bij opleidingen zonder historische modeloutput zijn de MAE/MAPE-kolommen leeg.

## Wanneer is een prognose betrouwbaar?

Er is geen harde drempel, maar de volgende signalen helpen:

**Meer vertrouwen:**

- De individuele modellen (`SARIMA_cumulative`, `SARIMA_individual`, `Prognose_ratio`) komen dicht bij elkaar uit — consensus tussen modellen is een goed teken
- De historische MAE is klein ten opzichte van het voorspelde aantal
- De voorspelling is gemaakt op of na week 10 (meer aanmelddata beschikbaar)

**Minder vertrouwen:**

- Grote spreiding tussen de modellen — overweeg elk model afzonderlijk te beoordelen
- Hoge historische MAE of MAPE
- Opleiding met weinig historische data (nieuw, of klein aantal inschrijvingen per jaar)
- Vroeg in het jaar (vóór week 6) — de tijdreeks is dan erg kort
- Het jaar na een uitzonderlijk jaar (COVID, beleidswijziging)

## Meerdere modellen vergelijken

De individuele modelkolommen naast `Ensemble_prediction` zijn bewust in de output opgenomen. Ze stellen je in staat om:

- Te controleren of de modellen het eens zijn
- Te signaleren welk model structureel afwijkt voor een specifieke opleiding
- De baseline (ratio-model) te vergelijken met de complexere modellen — als het ensemble niet beter is dan de ratio, is er reden om kritischer te kijken

Zie [Ensemble](methodologie/ensemble.md) voor uitleg over hoe de gewichten tot stand komen.

## Foutmaten en numerus-fixusopleidingen

MAE en MAPE worden berekend **exclusief numerus-fixusopleidingen**. Voor deze opleidingen worden de foutkolommen op `NaN` gezet. De reden: bij numerus-fixusopleidingen wordt het voorspelde aantal afgekapt op de capaciteitslimiet, waardoor de modelfouten niet vergelijkbaar zijn met reguliere opleidingen.

## Interactief dashboard

Naast de Excel-bestanden kan de pipeline interactieve HTML-dashboards genereren onder `data/output/visualisaties/`. Per modus (`-d i`, `-d c`, `-d b`) wordt een apart dashboard aangemaakt met daarin:

- **Individueel dashboard**: XGBoost-voorspellingen per opleiding, SARIMA-trajecten, feature importance (classifier).
- **Cumulatief dashboard**: SARIMA-voorspellingen op cumulatieve teldata, XGBoost-regressorresultaten, feature importance (regressor).
- **Eindoverzicht**: ensemble-voorspellingen per opleiding, foutmaten, vergelijking met vorige jaren.

De dashboards zijn zelfstandige HTML-bestanden (geen server nodig) en kunnen in elke browser geopend worden.

!!! info "Dashboard is opt-in"
    Sinds deze versie wordt het dashboard alleen gegenereerd als je expliciet `--dashboard` meegeeft. Een voorbeeld:

    ```bash
    studentprognose --dashboard -d both -w 10 -y 2025
    ```

    Mocht dashboard-generatie onverhoopt falen, dan loopt de rest van de pipeline gewoon door en wordt de stack trace weggeschreven naar `data/output/dashboard_error.log`. De Excel-output blijft in dat geval beschikbaar.

!!! note "Dashboard toont alleen de laatste week"
    Bij een multi-week run (bijv. `-w 10:20`) toont het dashboard alleen de prognose van de **laatste week** in de reeks. De Excel-output bevat wel alle weken.

Hieronder staan voorbeelden van de belangrijkste grafieken per dashboard, gegenereerd met demodata.

### Eindoverzicht (`final/dashboard.html`)

Altijd beschikbaar, ongeacht de gekozen modus. Toont het totaalplaatje: prognose per opleiding, verwachte groei/krimp, en betrouwbaarheid.

<iframe src="../assets/plots/output_cockpit.html" width="100%" height="400" frameborder="0" style="border-radius: 8px;"></iframe>

*Prognose per opleiding met realisatie vorig jaar, verschil en betrouwbaarheid. Betrouwbaarheid is gebaseerd op historische modelfouten: groen = hoog, geel = midden, rood = laag (demodata).*

<iframe src="../assets/plots/output_growth.html" width="100%" height="420" frameborder="0" style="border-radius: 8px;"></iframe>

*Verwachte groei (groen) en krimp (rood) t.o.v. vorig jaar. Het getal toont het absolute verschil in studenten (demodata).*

### Cumulatief dashboard (`cumulative/dashboard.html`)

Beschikbaar bij `-d c` of `-d b`. Toont analyses op basis van Studielink-telbestanden: wekelijkse aanmeldcurves, SARIMA-extrapolatie, en conversie van vooraanmelders naar inschrijvingen.

<iframe src="../assets/plots/output_conversion.html" width="100%" height="480" frameborder="0" style="border-radius: 8px;"></iframe>

*Vooraanmelders (week 38) naast werkelijke inschrijvingen per opleiding. Het percentage toont de conversieratio (demodata).*

<iframe src="../assets/plots/output_accuracy_heatmap.html" width="100%" height="420" frameborder="0" style="border-radius: 8px;"></iframe>

*MAPE per opleiding × model. Groen ≤ 10%, geel 10–25%, rood > 25%. De ★ markeert het best presterende model per opleiding (demodata).*

### Individueel dashboard (`individual/dashboard.html`)

Beschikbaar bij `-d i` of `-d b`. Toont analyses op basis van per-student aanmelddata: XGBoost-classificatie, SARIMA-trajecten, en nauwkeurigheid per opleiding.

<iframe src="../assets/plots/output_individual_cockpit.html" width="100%" height="400" frameborder="0" style="border-radius: 8px;"></iframe>

*Prognose per opleiding op basis van het individuele model, vergeleken met de realisatie van vorig jaar. Kleur in de Δ%-kolom toont de afwijking: groen ≤ 5%, geel 5–15%, rood > 15% (demodata).*

<iframe src="../assets/plots/output_scatter.html" width="100%" height="520" frameborder="0" style="border-radius: 8px;"></iframe>

*Elke bol is een opleiding in een bepaald jaar. Hoe dichter bij de diagonaal, hoe beter de voorspelling. Bolgrootte toont het werkelijke aantal studenten (demodata).*
