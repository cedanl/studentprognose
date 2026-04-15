# SARIMA

SARIMA staat voor **Seasonal AutoRegressive Integrated Moving Average**. Het is het kerntijdreeksmodel in de pipeline en wordt ingezet in zowel het cumulatieve als het individuele spoor.

## Wat doet het?

SARIMA extrapoleert het wekelijkse aanmeldpatroon van een opleiding naar het verwachte eindaantal inschrijvingen op week 38. Het model leert van historische seizoenspatronen: in welke weken stromen de meeste aanmeldingen binnen, hoe ziet de curve eruit vlak voor de inschrijfdeadline?

## Waarom SARIMA?

Aanmelddata heeft een sterk seizoenspatroon (jaarlijkse cyclus, wekelijkse granulariteit). SARIMA is specifiek ontworpen voor tijdreeksen met zulke structuur. Alternatieven zoals lineaire regressie op weeknummer negeren autokorrelatie in de data; neurale netwerken vereisen veel meer historische data dan de meeste instellingen beschikbaar hebben.

## Modelstructuur

$$SARIMA(p, d, q)(P, D, Q)_{52}$$

De seizoenslengte is altijd **52 weken**. De overige parameters zijn **vaste waarden** — er wordt geen automatische parameter-selectie toegepast. De keuze hangt af van het spoor en het moment in het jaar:

| Situatie | Order `(p,d,q)` | Seizoensorder `(P,D,Q)` |
|----------|----------------|------------------------|
| Cumulatief spoor | `(1, 0, 1)` | `(1, 1, 1)` |
| Individueel — Bachelor, weken 17–21 | `(1, 0, 1)` | `(1, 1, 1)` |
| Individueel — overige gevallen | `(1, 1, 1)` | `(1, 1, 0)` |

De implementatie gebruikt `statsmodels.tsa.statespace.SARIMAX`. Zie `src/studentprognose/models/sarima.py` voor de `SARIMAForecaster`-klasse.

## Exogene variabelen (individueel spoor)

In het individuele spoor kan een deadlineweek-variabele als exogene regresssor worden meegegeven. Dit markeert weken 16–17 (Bachelor niet-numerus fixus) of weken 1–2 (numerus fixus) als deadlineweek, zodat het model de aanmeldpiek rond deadlines beter kan modelleren.

## Aannames

1. **Herhaalbaar seizoenspatroon** — het model gaat ervan uit dat het patroon van dit jaar op het patroon van voorgaande jaren lijkt. Dit geldt niet bij structurele veranderingen (nieuwe opleiding, beleidsingreep, COVID).
2. **Voldoende historische data** — trainingsdata start vanaf 2016. Bij opleidingen met minder dan ~3 jaar data zijn de seizoensschattingen onbetrouwbaar.
3. **Stationariteit** — het model past één seizoensdifferentiatie toe (`D=1`). Als de trend structureel niet-stationair is, kan de differentiatie onvoldoende zijn.

## Wanneer vertrouw je het niet?

- Nieuwe opleiding (< 3 jaar historische data beschikbaar)
- Jaar na een uitzonderlijk jaar (bijv. post-COVID)
- Opleiding met sterke interjaarse variatie in aanmeldpatroon
- Vroeg in het jaar (weinig datapunten beschikbaar; de voorspelling is dan een lange extrapolatie)

## Relatie tot andere modellen

SARIMA levert een voorspelling per opleiding per week. In het `-d b` scenario wordt deze gecombineerd met de XGBoost-uitkomsten via het ensemble. Zie [Ensemble](ensemble.md).
