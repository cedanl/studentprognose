# Snelstart (5 min)

Van niets naar je eerste prognose — met de **meegeleverde demodata**, dus je hebt nog geen eigen data nodig.

!!! tip "Eerst installeren"
    Nog niet geïnstalleerd? Zie [Installeren](installeren.md) (drie commando's).

## 1. Projectmap aanmaken

```bash
mkdir mijn-prognose && cd mijn-prognose
studentprognose init
```

`init` maakt de mapstructuur en een `configuration/configuration.json` met alle standaardwaarden. Je kunt het veilig opnieuw draaien; bestaande bestanden worden overgeslagen.

## 2. Draaien

```bash
studentprognose
```

Zonder `-w`/`-y` kiest de tool automatisch de laatste beschikbare week en het laatste jaar uit de demodata. Je ziet een melding zoals:

```
Geen week/jaar opgegeven — automatisch gekozen op basis van beschikbare data: jaar 2024, week 38.
```

## 3. Resultaat bekijken

De output staat in `data/output/`. Open **`output_first-years_beide.xlsx`**; de kolom `Ensemble_prediction` bevat de prognose per opleiding.

Wil je het visueel? Voeg `--dashboard` toe voor interactieve HTML-grafieken:

```bash
studentprognose --dashboard
```

<iframe src="../assets/plots/output_cockpit.html" width="100%" height="400" frameborder="0" style="border-radius: 8px;"></iframe>

*Voorbeelddashboard op de demodata: prognose, conversie en betrouwbaarheid per opleiding.*

## Wat nu?

| Ik wil… | Ga naar |
|---------|---------|
| Mijn eigen data gebruiken | [Je data klaarzetten](je-data-voorbereiden.md) |
| Begrijpen wat `-w`, `-y` en `-d` doen | [Draaien & CLI](aan-de-slag.md) |
| De output-cijfers interpreteren | [Output lezen](output-begrijpen.md) |
| Mijn eigen instelling instellen | [Configuratie](configuratie.md) |
| Weten hoe de modellen werken | [Methodologie](methodologie/index.md) |

!!! note "Kom je een term niet tegen?"
    De [Begrippenlijst](begrippen.md) legt peilweek, spoor, ETL, ensemble en de rest in gewone taal uit.
