# CWF-model (confidence-weighted forecast)

!!! abstract "In het kort"
    - **Wat het doet:** voorspelt de eindinstroom per opleiding × herkomst × examentype rechtstreeks uit de cumulatieve vooraanmeldingen, met een **predictie-interval** en een **betrouwbaarheidslabel** (Groen/Oranje/Rood).
    - **Vertrouw het als:** de opleiding een paar jaar historie heeft, de aanmeldcyclus ver gevorderd is en het label Groen of Oranje is.
    - **Wees voorzichtig als:** het label Rood is, het nog vroeg in de cyclus is, de opleiding nieuw is of numerus fixus heeft.
    - **Ten opzichte van de andere sporen:** een zelfstandig derde model (`-d cwf`), niet onderdeel van het ensemble. Het heeft alleen telbestanden en studentaantallen nodig.

Het CWF-model is gebaseerd op het model van Radboud Institutional Research ([radboudir/studentprognose](https://github.com/radboudir/studentprognose), MIT-licentie). Studentprognose neemt de methode over en maakt hem configureerbaar voor elke instelling.

## Wat doet het?

Het model beantwoordt per combinatie één vraag: *welk deel van de mensen die op dit moment in de pijplijn zitten, schrijft zich straks echt in?* De pijplijn is het aantal gewogen vooraanmelders plus de inschrijvingen in de voorspelweek.

De voorspelling combineert twee schattingen:

1. **Het ratiosignaal.** De huidige pijplijn maal de *ratiofactor*. Die factor is de verhouding *instroom / pijplijn* in dezelfde week van eerdere jaren, met recente jaren zwaarder gewogen.
2. **De prior.** Een schatting die alleen naar de gerealiseerde instroom kijkt. Meestal is dat de instroom van vorig jaar; bij een duidelijke trend wordt een deel van die trend doorgetrokken, en bij een uitschieter wordt een gewogen gemiddelde genomen.

Hoeveel gewicht het ratiosignaal krijgt, hangt af van de **vulgraad**: welk deel van zijn uiteindelijke aanmeldpiek een opleiding in eerdere jaren op dit moment al had bereikt. Vroeg in de cyclus is de pijplijn nog klein en leunt de voorspelling op de prior. Laat in de cyclus bepaalt het ratiosignaal de voorspelling. Is de ratiofactor zelf onrustig (grote spreiding tussen jaren, weinig jaren), dan krijgt hij minder gewicht, ook laat in de cyclus.

??? note "De formules"
    Met $P$ de pijplijn (gewogen vooraanmelders + inschrijvingen) in de voorspelweek en $N_j$ de gerealiseerde instroom in jaar $j$:

    **Ratiofactor** (recency-gewogen geometrisch gemiddelde):

    $$R = \exp\Big(\sum_j w_j \log \frac{N_j}{P_j}\Big), \qquad w_j \propto e^{-\alpha\,(j_{\max} - j)}$$

    **Voorspelling:**

    $$\hat{y} = s \cdot (P \cdot R) + (1 - s) \cdot \text{prior}$$

    waarbij $s$ = vulgraad × $\min(1, \text{betrouwbaarheid}_R / 0.3)$.

    Bij weinig eigen jaren wordt $R$ in log-ruimte naar de ratio van vergelijkbare opleidingen getrokken, met eigen gewicht $n/(n+k)$ ($k$ = `pool_shrinkage_k`, standaard 3).

### Vergelijkbare opleidingen ("de pool")

Heeft een opleiding weinig eigen historie, dan leent het model van opleidingen met dezelfde herkomst en hetzelfde examentype. Dat gebeurt alleen met opleidingen die qua ratio (binnen 15%) en qua grootte (binnen een factor 2) vergelijkbaar zijn. Numerus-fixusbachelors lenen niet van andere opleidingen. Gewone bachelors lenen niet van numerus-fixusopleidingen.

### Predictie-interval

Het 95%-interval is een mengsel van twee schattingen:

- **Jaar-op-jaar:** hoe sterk de gerealiseerde instroom van deze combinatie van jaar tot jaar schommelt, gemeten als percentage en als aantal studenten. Hiervoor gebruikt het model alle beschikbare jaren met studentaantallen.
- **Trackrecord:** hoe ver het model er in eerdere jaren in dezelfde week naast zat. Dat wordt gemeten met *leave-one-out*: elk eerder jaar wordt voorspeld met alleen de andere jaren. Fouten van vergelijkbare opleidingen tellen mee, maar minder zwaar dan de eigen fouten.

Vroeg in de cyclus domineert het jaar-op-jaar-interval, laat in de cyclus het trackrecord. Daarnaast gelden drie regels:

- Het interval ligt aan elke kant minstens twee studenten van de prognose.
- Bij een vrijwel voltooide cyclus ligt de bovengrens nooit boven *ongewogen aanmelders + inschrijvingen*: meer mensen zijn er niet.
- Is er te weinig historie voor beide schattingen, dan valt het model terug op een interval dat alleen de toevalsvariatie van het verwachte aantal weergeeft (Poisson). Dat interval is een ondergrens voor de werkelijke onzekerheid.

### Betrouwbaarheidslabel

| Label | Betekenis | Wanneer |
|-------|-----------|---------|
| **Groen** | Zeer waarschijnlijk solide | Vulgraad ≥ 0,9, en in alle eerdere jaren zat het model op dit moment binnen ±16% (gemiddeld binnen ±8%) |
| **Oranje** | Bruikbare indicatie | Alles tussen Groen en Rood |
| **Rood** | Alleen een indicatie | Geen eigen historie, minder dan 3 eigen backtestjaren, vulgraad < 0,2, of een eerder jaar waarin het model er meer dan een factor 2 naast zat |

Een label houdt zich iets langer vast als het vorige week ook al zo was (hysterese). Zo springt een prognose niet elke week heen en weer tussen twee kleuren door een kleine verandering. De harde voorwaarden voor Rood en de drempels voor de vulgraad gelden altijd.

## Waarom deze aanpak?

1. **Eén getal plus een onzekerheidsmaat.** Beleidsmakers vragen niet alleen "hoeveel?", maar ook "hoe zeker?". Het interval en het label maken dat expliciet per prognose.
2. **Robuust bij weinig data.** Ratio's worden gemiddeld in log-ruimte (geometrisch gemiddelde), zodat één afwijkend jaar de schatting niet domineert. Dunne historie wordt aangevuld met vergelijkbare opleidingen.
3. **Geen look-ahead.** De vulgraad van een jaar is alleen berekend uit eerdere jaren. Een backtest gebruikt dus dezelfde informatie die het model live zou hebben gehad.
4. **Weinig data nodig.** Alleen de telbestanden en de oktobertelling; geen individuele aanmelddata.

Alternatieven die zijn overwogen staan in de andere sporen: een tijdreeks + regressie (cumulatief spoor) leert complexere patronen maar heeft meer historie nodig en levert geen interval op. Het eenvoudige [ratio-model](ratio-model.md) is transparant maar weegt jaren gelijk, kent geen prior en geen onzekerheid.

## Aannames

- **De conversie van pijplijn naar instroom is per opleiding redelijk stabiel** in dezelfde week van verschillende jaren.
- **De aanmeldcyclus verloopt elk jaar ongeveer even snel.** De vulgraad gaat ervan uit dat een opleiding in week *w* ongeveer even ver is als in eerdere jaren.
- **Inschrijvingen nemen de plaats in van aanmelders.** Na de aanmelddeadline worden vooraanmelders inschrijvingen; de pijplijn (gewogen + inschrijvingen) blijft dan ongeveer gelijk. Data waarin inschrijvingen ontbreken, maakt late weken onbetrouwbaar.
- **Herkomst *Onbekend* is gelijk verdeeld** over NL, EER en Niet-EER (uit te zetten met `redistribute_unknown_origin`).

## Wanneer vertrouw je het niet?

- **Vroeg in de cyclus** (lage vulgraad): de prognose is dan vooral de instroom van vorig jaar. Het label is dan Rood.
- **Nieuwe opleidingen zonder historie:** het model leunt volledig op vergelijkbare opleidingen. Stel indien mogelijk een verwachte instroom in via `cwf_params.predefined_priors`; dan zoekt het model opleidingen die op die verwachting lijken.
- **Numerus fixus:** de instroom wordt door de capaciteit bepaald. Het model kapt de som van de prognoses en bovengrenzen af op het plafond, maar weet niets van loting of selectie.
- **Systematische verschuivingen in de aanmelddata,** zoals een jaarlijkse opschoning van onafgeronde aanmeldingen of een ander moment waarop inschrijvingen worden geboekt. Het model vergelijkt dezelfde kalenderweek tussen jaren. Valt zo'n verschuiving niet elk jaar in dezelfde week, dan vergelijkt het appels met peren. Wees daarom extra kritisch in de weken waarin dat bij je instelling speelt.
- **Uitzonderlijke jaren in de historie** (bijv. een coronajaar). Geef die een lager gewicht met `cwf_params.year_weight_overrides`, bijvoorbeeld `{"2021": 0.5}`.

!!! warning "Demodata"
    De meegeleverde demodata is synthetisch: inschrijvingen zijn daar altijd 0 en vooraanmelders lopen na week 20 kunstmatig terug. Late weken geven op de demodata daardoor slechte prognoses. Op echte telbestanden doet dit zich niet voor.

## Relatie tot andere modellen

Het CWF-model is een **zelfstandig derde spoor**, naast het [cumulatieve](sarima.md) en het [individuele](individueel.md) spoor. Het draait met `-d cwf` en maakt geen deel uit van het [ensemble](ensemble.md) (`-d b` blijft ongewijzigd). Vergelijk de sporen op je eigen data met `studentprognose benchmark -d cwf` (zie [Benchmarks](benchmarks.md)) en het dashboard. Het CWF-model is vooral sterk als je naast een prognose ook wilt weten hoe zeker die is.

## Implementatie

- `src/studentprognose/models/cwf.py`: puntvoorspelling, parameters (`CWFParams`) en het betrouwbaarheidslabel.
- `src/studentprognose/models/cwf_interval.py`: het predictie-interval.
- `src/studentprognose/strategies/cwf.py`: de datavoorbereiding (vulgraad, herkomst *Onbekend*), de selectie van vergelijkbare opleidingen en de numerus-fixuscorrectie.

Alle parameters staan in `model_config.cwf_params`; zie de [configuratiereferentie](../configuratie-referentie.md#cwf_params).
