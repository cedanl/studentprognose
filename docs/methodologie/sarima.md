# SARIMA

SARIMA staat voor **Seasonal AutoRegressive Integrated Moving Average**. Het is het kerntijdreeksmodel in de pipeline en wordt ingezet in zowel het cumulatieve als het individuele spoor.

!!! tip "Uitgebreide versie — Jupyter notebook"
    Wil je het model zien fitten en forecasten op de demodata, inclusief aannames op de testbank?
    Zie [`notebooks/02_sarima.ipynb`](https://github.com/cedanl/studentprognose/blob/main/notebooks/02_sarima.ipynb).

## Wat doet het?

SARIMA extrapoleert het wekelijkse aanmeldpatroon van een opleiding naar het verwachte eindaantal inschrijvingen op week 38. Het model leert van historische seizoenspatronen: in welke weken stromen de meeste aanmeldingen binnen, hoe ziet de curve eruit vlak voor de inschrijfdeadline?

<iframe src="../../assets/plots/sarima_extrapolation.html" width="100%" height="500" frameborder="0" style="border-radius: 8px;"></iframe>

*Cumulatieve aanmeldcurve met SARIMA-extrapolatie (demodata)*

## Waarom SARIMA?

Aanmelddata heeft een sterk seizoenspatroon (jaarlijkse cyclus, wekelijkse granulariteit). SARIMA is specifiek ontworpen voor tijdreeksen met zulke structuur. Alternatieven zoals lineaire regressie op weeknummer negeren autokorrelatie in de data; neurale netwerken vereisen veel meer historische data dan de meeste instellingen beschikbaar hebben.

## Modelstructuur

$$SARIMA(p, d, q)(P, D, Q)_{52}$$

De nominale seizoenslengte is **52 weken** (één jaar). In het cumulatieve spoor wordt deze automatisch verkleind naar het werkelijke aantal gevulde weken per jaar wanneer dat lager ligt — zie [Seizoenslengte: afgeleid uit de data](#seizoenslengte-afgeleid-uit-de-data). De overige parameters zijn standaard **vaste waarden** (voor het cumulatieve spoor optioneel af te stemmen — zie [Orde-selectie (tuning)](#orde-selectie-tuning)). De keuze hangt af van het spoor en het moment in het jaar:

| Situatie | Order `(p,d,q)` | Seizoensorder `(P,D,Q)` |
|----------|----------------|------------------------|
| Cumulatief spoor | `(1, 0, 1)` | `(1, 1, 1)` |
| Individueel — Bachelor, weken 17–21 | `(1, 0, 1)` | `(1, 1, 1)` |
| Individueel — overige gevallen | `(1, 1, 1)` | `(1, 1, 0)` |

De implementatie gebruikt `statsforecast.models.ARIMA` (Nixtla) als backend. Dit vervangt de eerdere `statsmodels.tsa.statespace.SARIMAX`-backend. De modelspecificatie is identiek; het verschil zit in de schattingsmethode: **CSS-ML** (conditional sum of squares, gevolgd door maximum likelihood) in plaats van MLE via een Kalman-filter. Dit levert een snelheidswinst van ~10–15× per reeks op bij gelijkwaardige nauwkeurigheid.

Zie `src/studentprognose/models/sarima.py` voor de `SARIMAForecaster`-klasse.

## Orde-selectie (tuning)

De vaste ordes hierboven zijn onderbouwde standaardwaarden, maar je kunt ze voor het **cumulatieve spoor** afstemmen op je eigen historie.

### Wat doet het?

Tuning zoekt de ARIMA-ordes `(p,d,q)(P,D,Q,s)` die de vooraanmeldcurve het beste extrapoleren. Een kleine grid van orde-combinaties wordt geëvalueerd; de set met de laagste gemiddelde fout wint.

```bash
studentprognose tune -d c --tune-target sarima      # of: --tune-target both
```

Het commando print een overzicht met de MAPE per orde-combinatie (winnaar gemarkeerd met `✓`) en een `forecaster_params`-snippet om vast te leggen. Via de Python-API doet `run_pipeline_from_dataframes(..., tune="sarima")` (of `"both"`) hetzelfde en voorspelt direct met de gevonden ordes.

### Hoe wordt geëvalueerd?

Identiek aan de [benchmark](benchmarks.md): **tijd-bewuste cross-validatie** (train op jaren tot en met N−1, valideer op jaar N) met MAPE op de curve zelf als selectiemetriek — exact dezelfde `evaluate_timeseries_model` die de benchmark gebruikt. Random k-fold wordt **niet** gebruikt: dat zou toekomstige jaren in de trainingsset lekken.

### Waarom een kleine vaste grid en geen volledige `auto_arima`?

Een brede orde-zoektocht per serie (zoals `auto_arima` per opleiding × herkomst) is bij deze reekslengtes — een handvol collegejaren — duur en overfit-gevoelig. De ingebouwde grid is daarom bewust compact: een paar niet-seizoens-ordes × twee seizoens-ordes rond de waarden die de pipeline historisch gebruikte. Wil je toch breder zoeken? Overschrijf de grid met `model_config.sarima_tuning_grid`. Een volwaardige automatische selectie is los beschikbaar als het [AutoARIMA-model](#alternatieve-tijdreeksmodellen).

### Aannames

- Er zijn genoeg historische collegejaren voor minstens één tijdreeks-split (≥ 4 jaren met data). Bij minder valt tuning terug op de standaardordes met een waarschuwing.
- De seizoensvorm van de curve is stabiel genoeg dat ordes die op het verleden goed scoren, ook voor het komende jaar gelden.

### Wanneer vertrouw je het niet?

- **Marginale verschillen.** Liggen de MAPE-waarden dicht bij elkaar, dan is de winnaar grotendeels toeval — wijk dan niet zonder reden af van de robuuste defaults.
- **Korte/onregelmatige reeksen.** Bij weinig of sterk wisselende historie is de gekozen orde een steekproef van bijna niets; behandel het resultaat als indicatief.
- **Tuning op het productiepad.** Net als bij de regressor hoort tuning een bewuste, periodieke stap te zijn waarvan je het resultaat *vastlegt* in `forecaster_params`, niet iets dat elke run opnieuw draait.

### Relatie tot het ensemble

Orde-tuning raakt uitsluitend stap 1 van het cumulatieve spoor (de curve-extrapolatie). Stap 2 (de [regressor](xgboost.md#hyperparameter-tuning)), het individuele spoor en de [ensemble-weging](ensemble.md) blijven ongewijzigd.

## Exogene variabelen (individueel spoor)

In het individuele spoor kan een deadlineweek-variabele als exogene regresssor worden meegegeven. Dit markeert weken 16–17 (Bachelor niet-numerus fixus) of weken 1–2 (numerus fixus) als deadlineweek, zodat het model de aanmeldpiek rond deadlines beter kan modelleren.

## Aannames

1. **Herhaalbaar seizoenspatroon** — het model gaat ervan uit dat het patroon van dit jaar op het patroon van voorgaande jaren lijkt. Dit geldt niet bij structurele veranderingen (nieuwe opleiding, beleidsingreep, COVID).
2. **Voldoende historische data** — trainingsdata start vanaf het jaar ingesteld via `min_training_year` in `configuration.json` (standaard 2016). Bij opleidingen met minder dan ~3 jaar data zijn de seizoensschattingen onbetrouwbaar.
3. **Stationariteit** — het model past één seizoensdifferentiatie toe (`D=1`). Als de trend structureel niet-stationair is, kan de differentiatie onvoldoende zijn.
4. **CSS-ML schattingsmethode** — de backend gebruikt conditional sum of squares als startpunt voor maximum likelihood. Dit is sneller dan exact MLE via het Kalman-filter, maar kan bij zeer korte reeksen of reeksen met ontbrekende waarden iets afwijkende parameterschattingen opleveren.

## Wanneer vertrouw je het niet?

- Nieuwe opleiding (< 3 jaar historische data beschikbaar)
- Jaar na een uitzonderlijk jaar (bijv. post-COVID)
- Opleiding met sterke interjaarse variatie in aanmeldpatroon
- Vroeg in het jaar (weinig datapunten beschikbaar; de voorspelling is dan een lange extrapolatie)

## Academische-jaargrens (`final_academic_week`)

Het cumulatieve spoor modelleert één academisch jaar als een vaste week-volgorde die loopt van de reset-week (`final_academic_week + 1`) via week 52 door naar `final_academic_week` van het jaar erop. Deze grens bepaalt de voorspelhorizon (`pred_len`), de seizoensvolgorde van de kolommen en de reset-week die als nulpunt wordt geïnjecteerd.

Standaard is dit **week 38** (eind september, de inschrijfdeadline van het legacy Studielink-instellingsformaat). De grens is configureerbaar via `model_config.final_academic_week` omdat niet elke leverancier dezelfde kalender hanteert:

- **Legacy instellingsformaat:** leveringen lopen tot week 38 → `final_academic_week = 38`.
- **UvA SQL-telbestand:** de aanmeldfase eindigt rond week 36; er zijn geen leveringen in de weken 37–39 (de Studielink-cyclus reset daar). Met de standaard 38 zou het model proberen te voorspellen tot een week die niet in de data bestaat. Daarom `final_academic_week = 36`.

**Aanname:** de waarde moet overeenkomen met de week tot waar de aanmeldfase daadwerkelijk gegevens levert. Een te hoge waarde laat de voorspelling crashen of extrapoleert naar niet-bestaande weken; een te lage waarde kapt de voorspelhorizon onnodig af. De grens geldt voor het cumulatieve spoor; het individuele spoor gebruikt de vaste Studielink-kalender (week 38).

!!! warning "52-weeks model en ISO-week 53"
    Het seizoensvenster omvat maximaal **52 weken**. In lange ISO-jaren (bijv. een leverdatum eind december 2020 of 2026) levert de afleiding uit de leverdatum **ISO-week 53** op. Zulke weeksnapshots blijven wél in `vooraanmeldingen_cumulatief.csv` staan, maar vallen **buiten het seizoensvenster** dat het cumulatieve model gebruikt — ze worden niet meegetraind (de ETL geeft hierover een waarschuwing). Voor de jaaroverzichten betekent dit dat de allerlaatste decembersnapshot van een lang ISO-jaar niet apart wordt gemodelleerd; de eerstvolgende januarisnapshot (week 1) vangt de cumulatieve stand weer op. Volledige ondersteuning van week 53 zou een variabele seizoenslengte vereisen en is bewust niet in deze stap meegenomen.

## Seizoenslengte: afgeleid uit de data

De seizoenslag bepaalt naar welk punt in het verleden het model kijkt om "dezelfde week vorig jaar" te vinden. Nominaal is dat **52 weken**. Het cumulatieve spoor zet een reeks echter in elkaar door de trainingsjaren achter elkaar te plakken over precies de weekkolommen die ná het pivotteren bestaan — de **union van gevulde weken** over alle jaren en opleidingen, plus de geïnjecteerde reset-week. Lege start- en eindweken (vóór de openbaarmakingsdrempel van de telleverancier — bij UvA `Aantal ≥ 21` — en nadat de aanmeldingen weer onder die drempel zakken) bestaan niet als kolom. Die union telt daardoor **minder dan 52 weken** (op de huidige UvA-demodata empirisch ruwweg 43–45). Dat aantal is de werkelijke jaarblok-lengte ("stride") van de afgevlakte reeks.

Staat de seizoenslag vast op 52 terwijl de stride korter is, dan wijst de lag naar de **verkeerde week-in-het-jaar**: de jaarcyclus staat uit fase en het model kan de seizoensvorm niet reproduceren. Het netto-effect op de prognose is datagedreven — op de UvA-funneldata uit dit zich concreet als een prognose die ná de piek **omhoog drijft in plaats van mee te dalen** met de werkelijke (teruglopende) openstaande vooraanmeldingen.

Daarom stelt het cumulatieve spoor de seizoenslengte gelijk aan deze stride (`len(get_all_weeks_valid(...))`, inclusief de reset-week). De aanpassing **verkleint alleen**: een volledig gevulde cyclus (stride == 52) houdt `season_length = 52`, dus correct gevulde reeksen blijven ongemoeid. Dezelfde afleiding wordt toegepast in de [benchmark](benchmarks.md), zodat modelselectie hetzelfde model meet als productie.

**Aanname:** de set gevulde weken is over de jaren heen stabiel genoeg dat de union een goede benadering is van de cyclus per jaar. Wisselt de weekdekking sterk per jaar, dan wordt de seizoensuitlijning minder scherp.

**Wanneer vertrouw je het niet?** Bij leveranciers of opleidingen waar de set gevulde weken sterk wisselt tussen jaren, of bij een gemengde dataset waarin cohorten een uiteenlopende weekdekking hebben.

## Bekende hardgecodeerde uitzondering: 2021

In `utils/weeks.py` staat een expliciete uitzondering voor collegejaar 2021:

```python
if predict_year == 2021:
    max_week = 38
```

Dit forceert de maximale week voor dat jaar op 38 (de standaard inschrijfdeadline), in plaats van die te berekenen uit de data. De vermoedelijke reden is dat de aanmeldingscyclus in 2021 een afwijkend patroon had door COVID, waardoor de normale berekening geen betrouwbaar resultaat gaf. De achterliggende reden is nog niet formeel gedocumenteerd — zie issue [#84](https://github.com/cedanl/studentprognose/issues/84).

Dit illustreert een bredere aanname: **het model gaat ervan uit dat historische aanmeldpatronen representatief zijn**. Uitzonderingsjaren zoals 2021 moeten handmatig worden afgehandeld.

## Alternatieve tijdreeksmodellen

Naast SARIMA zijn drie alternatieve tijdreeksmodellen beschikbaar, configureerbaar via `model_config.cumulative_timeseries` in `configuration.json`:

| Model | Config-waarde | Beschrijving |
|-------|---------------|-------------|
| **ETS** (AutoETS) | `ets` | Exponential smoothing met automatische componentkeuze. Stabieler bij korte reeksen (<10 jaar) doordat er geen vaste ARIMA-ordes gekozen hoeven te worden. |
| **Theta** (AutoTheta) | `theta` | Zeer simpel model dat competitief is bij korte tijdreeksen (winnaar M3-competitie). Goede robuuste baseline. |
| **AutoARIMA** | `auto_arima` | Automatische selectie van ARIMA-ordes via AICc-minimalisatie. Flexibeler dan de vaste ordes van SARIMA, maar trager door de zoekprocedure. |

Alle modellen gebruiken dezelfde `BaseForecaster`-interface en dezelfde nominale seizoenslengte van 52 weken (in het cumulatieve spoor verkleind naar de werkelijke periode, zie [Seizoenslengte: afgeleid uit de data](#seizoenslengte-afgeleid-uit-de-data)). ETS en Theta ondersteunen geen exogene variabelen (relevant voor het individuele spoor).

Gebruik `studentprognose benchmark -w <week>` om te vergelijken welk model het best presteert op jouw data voordat je de keuze in de configuratie wijzigt.

## Relatie tot andere modellen

SARIMA levert een voorspelling per opleiding per week. In het `-d b` scenario wordt deze gecombineerd met de XGBoost-uitkomsten via het ensemble. Zie [Ensemble](ensemble.md).
