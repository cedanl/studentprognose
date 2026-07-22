# Grafische interface

Naast de opdrachtregel (CLI) heeft studentprognose een optionele **grafische
interface**: een lokale webapp waarmee je een project opzet, de configuratie
instelt en voorspellingen draait zonder de terminal te hoeven gebruiken.

De GUI is een *schil* rond de CLI. Ze bevat geen eigen modellogica: elke actie
bouwt hetzelfde `studentprognose`-commando dat je ook zelf zou typen en voert dat
uit. Alles wat in de GUI kan, kan dus ook op de opdrachtregel — en omgekeerd.

## Installeren en starten

De interface gebruikt [NiceGUI](https://nicegui.io/), een optionele
afhankelijkheid. Installeer die met de `gui`-extra en start de app:

```bash
uv run --extra gui python -m gui
```

Open daarna [http://localhost:8080](http://localhost:8080) in je browser.

!!! note "Optioneel, draait vanuit de broncode"
    NiceGUI is niet nodig om de CLI te draaien; de `gui`-extra installeert het
    alleen wanneer je die expliciet meegeeft. De interface zelf draai je vanuit
    een clone van de repository (`git clone` + `uv run --extra gui python -m gui`)
    — ze wordt niet meegeleverd in het PyPI-pakket.

## Zo werkt de interface

De interface leidt je langs vijf stappen, in volgorde:

1. **Project** — kies of maak een projectmap met de juiste structuur. Je kunt
   hier optioneel de demodataset downloaden om het model direct te proberen.
2. **Configuratie** — stel de modelparameters en paden in.
3. **Filteren** — bepaal op welke opleidingen, herkomst en examentypes je draait.
4. **Uitvoeren** — start de voorspelling en volg de voortgang live.
5. **Resultaten** — bekijk een overzicht van de voorspellingen.

Nieuwe gebruikers volgen deze stappen van boven naar beneden; de stap-indicator
bovenaan toont waar je bent. Terugkerende gebruikers springen via de zijbalk
direct naar de gewenste pagina. Stappen die een project vereisen zijn
uitgeschakeld tot je er een hebt gekozen, zodat je nooit vastloopt.

## Wanneer gebruik je wat?

| Situatie | Aanbeveling |
|----------|-------------|
| Verkennen, eenmalige run, visueel overzicht | Grafische interface |
| Geautomatiseerde runs (cron, taakplanner) | CLI met `--yes` |
| Reproduceerbare scripts / CI | CLI |
| Cloud / notebooks (data in geheugen) | Python-API (`run_pipeline_from_dataframes`) |

## Verhouding tot de CLI

De GUI stelt exact dezelfde opties beschikbaar als de CLI (zie
[Draaien & CLI](aan-de-slag.md)). Waar de documentatie een CLI-vlag noemt,
correspondeert die met een veld in de interface. De onderliggende pipeline,
modellen en uitvoerbestanden zijn identiek.
