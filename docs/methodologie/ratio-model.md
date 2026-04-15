# Ratio-model

Het ratio-model is het eenvoudigste model in de pipeline en dient als **referentiemodel**: een expliciete bodem die altijd beschikbaar is, ook zonder betrouwbare trainingsdata.

## Wat doet het?

Het ratio-model berekent per opleiding de historische verhouding tussen het aantal vooraanmelders op een bepaald moment in het jaar en het uiteindelijke aantal inschrijvingen. Die ratio wordt vervolgens toegepast op het huidige aantal vooraanmelders.

$$\hat{y}_{t} = \text{vooraanmelders}_t \times \frac{\bar{y}_{\text{hist}}}{\bar{x}_{\text{hist,t}}}$$

## Waarom een ratio-model?

Het ratio-model maakt geen aannames over seizoenspatronen of feature-importantie. Het is volledig transparant en auditeerbaar — de output is direct te herleiden tot de historische data. Dit maakt het geschikt als:

1. **Fallback** wanneer SARIMA of XGBoost onvoldoende data heeft
2. **Sanity check** naast de complexere modellen
3. **Baseline** voor het beoordelen van de toegevoegde waarde van de andere modellen (als het ensemble niet beter is dan de ratio, is er iets mis)

## Beperkingen

- Geen rekening met trends: als de instroom structureel groeit of daalt, is de historische ratio achterhaald.
- Gevoelig voor uitschieters in de historische data (één abnormaal jaar kan de ratio sterk beïnvloeden).

## Implementatie

Zie `src/studentprognose/models/ratio.py`.
