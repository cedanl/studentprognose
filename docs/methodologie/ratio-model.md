# Ratio-model

Het ratio-model is het eenvoudigste model in de pipeline en dient als **referentiemodel**: volledig transparant, altijd beschikbaar, ook zonder betrouwbare trainingsdata voor de complexere modellen.

## Wat doet het?

Het ratio-model berekent per opleiding de historische verhouding tussen het totale aanmeldvolume op een bepaald moment in het jaar en het uiteindelijke aantal inschrijvingen. Die ratio wordt vervolgens toegepast op het huidige aanmeldvolume.

**Definitie van aanmeldvolume:**

$$\text{Aanmelding} = \text{Ongewogen vooraanmelders} + \text{Inschrijvingen}$$

**Historische ratio (gemiddeld over 3 jaar):**

$$\bar{R}_{t} = \frac{1}{3} \sum_{j=1}^{3} \frac{\text{Aanmelding}_{jaar-j,\, week=t}}{\text{Aantal\_studenten}_{jaar-j}}$$

**Voorspelling:**

$$\hat{y} = \frac{\text{Aanmelding}_{huidig,\, week=t}}{\bar{R}_{t}}$$

Het venster van 3 jaar is hardgecodeerd (`predict_year - 3` t/m `predict_year - 1`).

## Numerus fixus-correctie

Na de basisberekening past het model een **cap** toe voor numerus fixus-opleidingen: als de gesommeerde voorspelling over herkomstgroepen het vastgestelde maximum overschrijdt, wordt het overschot afgetrokken van de NL-herkomstgroep.

## Waarom een ratio-model?

1. **Volledig transparant** — de output is direct te herleiden tot de historische data, zonder model-blackbox
2. **Altijd beschikbaar** — geen trainingsdata nodig buiten de 3-jaars historische ratio
3. **Baseline** — als het ensemble structureel slechter presteert dan de ratio, is er iets mis met de complexere modellen
4. **Auditeerbaar** — geschikt voor interne verantwoording richting management

## Beperkingen

- Geen rekening met trends: bij structurele groei of daling is de 3-jaars ratio achterhaald.
- Gevoelig voor uitschieters: één abnormaal jaar (bijv. COVID) kan de gemiddelde ratio sterk beïnvloeden.
- Week-afhankelijk: de ratio varieert per week. Vroeg in het jaar (weinig aanmelders) is de ratio minder stabiel.

## Implementatie

Zie `src/studentprognose/models/ratio.py` — de functie `predict_with_ratio`.
