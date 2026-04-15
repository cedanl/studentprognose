# SARIMA

SARIMA staat voor **Seasonal AutoRegressive Integrated Moving Average**. Het is het kerntijdreeksmodel in de pipeline en wordt ingezet in zowel het cumulatieve als het individuele spoor.

## Wat doet het?

SARIMA extrapoleert het wekelijkse aanmeldpatroon van een opleiding naar het verwachte eindaantal inschrijvingen. Het model leert van historische seizoenspatronen: in welke weken stromen de meeste aanmeldingen binnen, hoe ziet de curve eruit vlak voor de inschrijfdeadline?

## Waarom SARIMA?

Aanmelddata heeft een sterk seizoenspatroon (jaarlijkse cyclus, wekelijkse granulariteit). SARIMA is specifiek ontworpen voor tijdreeksen met zulke structuur. Alternatieven zoals lineaire regressie op weeknummer negeren autokorrelatie in de data; neurale netwerken vereisen veel meer historische data dan de meeste instellingen beschikbaar hebben.

## Modelstructuur

$$SARIMA(p, d, q)(P, D, Q)_s$$

| Parameter | Betekenis |
|-----------|-----------|
| `p` | Orde van het autoregressieve deel |
| `d` | Differentiatie-orde (stationariseren) |
| `q` | Orde van het moving average-deel |
| `P, D, Q` | Seizoenscomponenten |
| `s` | Seizoenslengte (52 weken) |

De parameters worden automatisch bepaald via `pmdarima.auto_arima`. Zie `src/studentprognose/models/sarima.py` voor de implementatie.

## Aannames

1. **Stationariteit na differentiatie** — het model past differentiatie toe om trends te verwijderen. Als de trend in realiteit niet-stationair is (bijv. structurele groei), kan dit de voorspelling verstoren.
2. **Herhaalbaar seizoenspatroon** — het model gaat ervan uit dat het patroon van dit jaar op het patroon van voorgaande jaren lijkt. Dit geldt niet bij structurele veranderingen (nieuwe opleiding, beleidsingreep, COVID).
3. **Voldoende historische data** — bij minder dan ~3 jaar data zijn de seizoensschattingen onbetrouwbaar.

## Wanneer vertrouw je het niet?

- Nieuwe opleiding (< 3 jaar historische data)
- Jaar na een uitzonderlijk jaar (bijv. post-COVID)
- Opleiding met sterke interjaarse variatie in aanmeldpatroon
- Weken vroeg in het jaar (weinig datapunten nog beschikbaar)

## Relatie tot andere modellen

SARIMA levert een voorspelling per opleiding per week. In het `-d b` scenario wordt deze voorspelling gecombineerd met de XGBoost-uitkomsten via ensemble weging. Zie [Ensemble](ensemble.md).
