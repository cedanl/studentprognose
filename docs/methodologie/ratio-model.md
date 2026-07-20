# Ratio-model

!!! abstract "In het kort"
    - **Wat het doet:** past de historische verhouding tussen aanmeldvolume en inschrijvingen toe op het huidige aanmeldvolume.
    - **Vertrouw het als:** de instroom stabiel is en je een transparante, uitlegbare schatting nodig hebt.
    - **Wees voorzichtig als:** er sprake is van structurele groei of daling, een uitzonderlijk jaar (bijv. COVID) in de historie zit, of het nog vroeg in het jaar is.
    - **In het ensemble:** dient als eenvoudige baseline en sanity-check waartegen de complexere modellen worden afgezet.

Het ratio-model is het eenvoudigste model in de pipeline en dient als **referentiemodel**: volledig transparant, altijd beschikbaar, ook zonder betrouwbare trainingsdata voor de complexere modellen.

!!! tip "Uitgebreide versie — Jupyter notebook"
    Een uitvoerbare versie waarin we de ratio handmatig stap-voor-stap herberekenen en per week
    visualiseren staat in [`notebooks/04_ratio_model.ipynb`](https://github.com/cedanl/studentprognose/blob/main/notebooks/04_ratio_model.ipynb).

## Wat doet het?

Het ratio-model berekent per opleiding de historische verhouding tussen het totale aanmeldvolume op een bepaald moment in het jaar en het uiteindelijke aantal inschrijvingen. Die ratio wordt vervolgens toegepast op het huidige aanmeldvolume.

Kort gezegd: als in de afgelopen jaren rond deze week een vaste fractie van het aanmeldvolume uiteindelijk tot inschrijving leidde, dan gaat het model ervan uit dat die verhouding dit jaar weer opgaat.

??? note "De formules"
    **Definitie van aanmeldvolume:**

    $$\text{Aanmelding} = \text{Ongewogen vooraanmelders} + \text{Inschrijvingen}$$

    **Historische ratio (gemiddeld over 3 jaar):**

    $$\bar{R}_{t} = \frac{1}{3} \sum_{j=1}^{3} \frac{\text{Aanmelding}_{jaar-j,\, week=t}}{\text{Aantal\_studenten}_{jaar-j}}$$

    **Voorspelling:**

    $$\hat{y} = \frac{\text{Aanmelding}_{huidig,\, week=t}}{\bar{R}_{t}}$$

<iframe src="../../assets/plots/ratio_historical.html" width="100%" height="480" frameborder="0" style="border-radius: 8px;"></iframe>

*Ratio vooraanmelders / inschrijvingen per week voor een voorbeeldopleiding (demodata)*

Het venster van 3 jaar is vastgelegd in de constante `LOOKBACK_YEARS` (`src/studentprognose/utils/constants.py`) en geldt voor alle modellen die een historisch gemiddelde berekenen.

## Numerus fixus-correctie

Numerus fixus-opleidingen kennen een wettelijk vastgesteld maximum aantal plaatsen; meer studenten dan dat maximum kunnen niet instromen. Daarom past het model na de basisberekening een **cap** toe: als de gesommeerde voorspelling over de herkomstgroepen het maximum overschrijdt, wordt het overschot afgetrokken van de NL-herkomstgroep. Dat het overschot juist bij de NL-groep wordt weggehaald (en niet evenredig over alle groepen) is een beleidskeuze in de correctie.

## Waarom een ratio-model?

1. **Volledig transparant** — de output is direct te herleiden tot de historische data, zonder model-blackbox
2. **Altijd beschikbaar** — geen trainingsdata nodig buiten de 3-jaars historische ratio
3. **Baseline** — als het ensemble structureel slechter presteert dan de ratio, is er iets mis met de complexere modellen
4. **Auditeerbaar** — geschikt voor interne verantwoording richting management

## Wanneer vertrouw je het niet?

- Geen rekening met trends: bij structurele groei of daling is de 3-jaars ratio achterhaald.
- Gevoelig voor uitschieters: één abnormaal jaar (bijv. COVID) kan de gemiddelde ratio sterk beïnvloeden.
- Week-afhankelijk: de ratio varieert per week. Vroeg in het jaar (weinig aanmelders) is de ratio minder stabiel.

## Relatie tot andere modellen

Het ratio-model is de **baseline** van de pipeline: de eenvoudigste, volledig transparante schatting waartegen de complexere modellen worden afgezet. In het [ensemble](ensemble.md) fungeert het als sanity-check — presteert het ensemble structureel slechter dan de kale ratio, dan is dat een signaal dat er iets mis is met de zwaardere modellen ([SARIMA](sarima.md), [XGBoost](xgboost.md)) en niet dat de ratio het toevallig goed doet. Omdat het geen trainingsdata nodig heeft buiten de historische ratio, is het bovendien altijd beschikbaar, ook als de andere modellen door datagebrek afvallen.

## Implementatie

Zie `src/studentprognose/models/ratio.py` — de functie `predict_with_ratio`.
