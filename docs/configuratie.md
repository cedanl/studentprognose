# Configuratie

!!! tip "Snelste route"
    De meeste gebruikers hoeven alleen hun eigen instelling (`institution_filter`) en eventueel een `numerus_fixus` in te stellen. Alle andere secties kunnen op hun standaardwaarde blijven staan.

De pipeline wordt geconfigureerd via `configuration/configuration.json`.

## Beginnen

Voer eenmalig `studentprognose init` uit in je werkmap. Dit schrijft een startconfiguratie met alle standaardwaarden. Je hoeft daarna alleen de velden aan te passen die bij jouw instelling afwijken.

## Voorbeeld minimale configuratie

In de praktijk pas je meestal maar 2-3 velden aan. Alle andere secties op deze pagina zijn referentie die je gerust kunt overslaan tot je ze nodig hebt.

Omdat de pipeline standaardwaarden inbouwt, hoef je alleen te specificeren wat afwijkt. Een typische startconfiguratie bevat alleen de instelling-specifieke velden:

```json
{
    "numerus_fixus": {
        "B Geneeskunde": 340
    },
    "ensemble_override_cumulative": [
        "B Geneeskunde"
    ]
}
```

Alle overige velden — paden, kolomnamen, modelparameters, ensemble-gewichten — vallen terug op de ingebakken standaardwaarden. Voer `studentprognose init` uit om een volledig ingevuld startbestand te krijgen dat je als referentie kunt gebruiken.

## Hoe configuratie laden werkt

De pipeline laadt altijd eerst de **ingebakken standaardwaarden** uit het package, en past jouw `configuration.json` daar bovenop als een *overschrijving*. Dat betekent:

- Je hoeft alleen te specificeren wat afwijkt van de standaard.
- Een ontbrekend veld in jouw bestand valt automatisch terug op de standaardwaarde.
- Als het configuratiebestand helemaal niet bestaat, worden de standaardwaarden gebruikt en verschijnt er een waarschuwing.

Het bestand heeft de onderstaande secties. De kolom **Pas je aan?** geeft aan hoe vaak je een sectie in de praktijk zelf wijzigt — de meeste kun je op de standaard laten staan.

| Sectie | Pas je aan? | Wat het doet |
|--------|-------------|--------------|
| `institution_filter` | Ja | Beperkt de teldata tot je eigen instelling(en). |
| `numerus_fixus` | Ja | Legt de capaciteit vast van opleidingen met een numerus fixus. |
| `ensemble_override_cumulative` | Ja | Radboud-specifieke lijst; vervangen of leegmaken voor je eigen instelling. |
| `exclude_from_combined` | Ja | Radboud-specifieke lijst; vervangen of leegmaken voor je eigen instelling. |
| `paths` | Zelden | Bestandspaden voor input- en outputbestanden. |
| `telbestand_filename_patterns` | Zelden | Herkent telbestanden aan hun bestandsnaam; alleen nodig bij afwijkende naamgeving. |
| `cumulative_input` | Zelden | Zet ruwe telbestanden om; staat standaard op het Studielink SQL-formaat. |
| `validation` | Zelden | Overschrijft validatiedrempels; alleen wat afwijkt van de standaard. |
| `runtime` | Zelden | Uitvoerparameters zoals het aantal CPU-cores. |
| `model_config` | Zelden | Modelparameters (statusmapping, trainingsjaren, modelkeuze). |
| `excluded_data_points` | Zelden | Sluit bekende probleemjaren uit de trainingsdata; standaard leeg. |
| `ensemble_weights` | Zelden | Weegt het individuele en cumulatieve model in het ensemble. |
| `model_features` | Zelden | Bepaalt welke kolommen als features in de modellen gaan. |
| `columns` | Zelden | Vertaalt eigen kolomnamen naar de namen die de pipeline verwacht. |
| `column_roles` | Nee | Interne koppeling van rollen aan kolomnamen; vrijwel nooit aanpassen. |

## `numerus_fixus`

Een object met **programmasleutels** als keys en het maximale inschrijvingsaantal (de NF-capaciteit) als waarde. Numerus-fixus-opleidingen krijgen een aparte behandeling: een eigen regressor (los van de gedeelde pool), een capaciteitsplafond op de voorspelling, en aparte foutrapportage.

```json
{
    "numerus_fixus": {
        "56604": 350,
        "50033": 260
    }
}
```

Als de gesommeerde voorspelling het maximum overschrijdt, wordt het overschot afgetrokken van de NL-herkomstgroep — zie [Ratio-model](methodologie/ratio-model.md) voor de achtergrond van die keuze. Opleidingen die hier niet staan worden niet gecapped.

!!! danger "De sleutel moet exact matchen met de programmakolom (`Croho groepeernaam`)"
    De NF-behandeling grijpt **alleen** aan als de sleutel exact overeenkomt met een waarde in de programmakolom van je data. Welk formaat dat is, hangt af van je instelling:

    - **Cumulatief spoor** — de programmakolom bevat sinds de Isatcode-migratie de numerieke **Isatcode** (CROHO-code, bijv. `56604`). Gebruik dan de Isatcode als sleutel.
    - **Individueel spoor / legacy** — de programmakolom bevat de **leesbare opleidingsnaam** (bijv. `"B Geneeskunde"`). Gebruik dan de naam als sleutel.

    Het dtype maakt niet uit: een numerieke sleutel wordt automatisch als getal geïnterpreteerd (`"56604"` en `56604` zijn equivalent). JSON kent geen inline-commentaar, dus noteer de leesbare naam bij voorkeur in je eigen documentatie of changelog naast de Isatcode.

!!! warning "Sleutelruimtes van de twee sporen overlappen niet (#238)"
    Het cumulatieve spoor keyt op Isatcodes, het individuele spoor op namen. Eén sleutel kan daardoor maar één van beide sporen matchen. Draai je `-d both`, dan grijpt een Isatcode-sleutel wél aan in het cumulatieve spoor maar niet in het individuele — je krijgt hierover een **waarschuwing**. Tot #238 is opgelost is er geen sleutel die beide sporen tegelijk bedient.

!!! info "Guard tegen stille misconfiguratie (#258)"
    Vóór de voorspelling controleert de pipeline of elke `numerus_fixus`-sleutel voorkomt in de programmakolom. Matcht een sleutel **geen enkel** geladen spoor (typefout of verkeerd formaat), dan **stopt de pipeline met een harde fout** — een niet-matchende sleutel wordt nooit meer stil genegeerd. Zie [validatie](validatie.md#numerus-fixus-sleutels).

## `institution_filter` — beperk de teldata tot één of meer instellingen

De landelijke Studielink-teldata bevat rijen van **álle** instellingen (elke Brincode staat in de kolom `Korte naam instelling`). Met `institution_filter` beperk je de pipeline tot je eigen instelling(en), zodat de modellen niet op andermans instroom trainen.

```json
{
    "institution_filter": ["21PC"]
}
```

- **Type:** lijst van instellingscodes (Brincodes, bijv. `"21PC"`) of leesbare instellingsnamen — precies zoals ze in de kolom `Korte naam instelling` staan.
- **Standaard (`[]`):** geen filter — **alle instellingen** in de data worden meegenomen. Dit is het backwards-compatibele default-gedrag.
- Meerdere instellingen: `["21PC", "00IC"]`.

Het filter grijpt aan op **load-tijd**, vóór preprocessing, zodat zowel de training als de voorspelling op de gekozen instelling(en) draaien.

!!! info "Welke sporen worden gefilterd?"
    Alleen sporen die een instellingskolom dragen — dat is het **cumulatieve spoor** (de landelijke teldata). Het **individuele spoor** is doorgaans de eigen aanmeldexport van één instelling en heeft geen instellingskolom; daar is het filter een no-op. Het studentaantallenbestand (`student_count`) wordt eveneens verondersteld al instelling-specifiek te zijn.

!!! warning "Onbekende instelling faalt hard"
    Komt een opgegeven code in het geheel niet voor in de data, dan stopt de pipeline met een duidelijke foutmelding (en de lijst van beschikbare instellingen) in plaats van stil een lege dataset te verwerken. Komt een code wél gedeeltelijk voor, dan zie je een waarschuwing voor de ontbrekende codes.

Je kunt deze config-waarde op de commandline overschrijven met de vlag [`--institution`](aan-de-slag.md#-institution) — handig om snel voor één instelling te draaien zonder de config aan te passen.

## `ensemble_override_cumulative` — ensemble-uitzondering per opleiding

Een lijst van opleidingsnamen (op `Croho groepeernaam`) waarvoor de ensemble-logica altijd het SARIMA-cumulatief model gebruikt, ongeacht weeknummer of examentype.

```json
{
    "ensemble_override_cumulative": [
        "B Geneeskunde",
        "B Biomedische Wetenschappen",
        "B Tandheelkunde"
    ]
}
```

Gebruik dit voor opleidingen met een numerus fixus of een sterk afwijkend aanmeldpatroon waarbij het cumulatieve SARIMA-model aantoonbaar beter presteert. Lege lijst (`[]`) schakelt de uitzondering uit voor alle opleidingen.

De waarden in de demo-configuratie zijn Radboud-specifiek. **Vervang of maak deze lijst leeg voor je eigen instelling.**

## `exclude_from_combined` — uitsluiting van combined-modus

Een lijst van opleidingsnamen (op `Croho groepeernaam`) die worden overgeslagen in de combined-modus (`-d both`). Opleidingen op deze lijst worden niet meegenomen in de combined-voorspelling.

```json
{
    "exclude_from_combined": [
        "M Educatie in de Mens- en Maatschappijwetenschappen"
    ]
}
```

Gebruik dit voor opleidingen waarvoor de combined-modus aantoonbaar slechter werkt dan het cumulatieve spoor alleen. Lege lijst schakelt de uitsluiting uit.

De waarde in de demo-configuratie is Radboud-specifiek. **Vervang of maak deze lijst leeg voor je eigen instelling.**

## Overige secties (referentie)

Alle secties met **Zelden** of **Nee** in de tabel hierboven — paden, kolomnamen, modelparameters, validatiedrempels, ensemble-gewichten — staan op de aparte pagina **[Configuratie — referentie](configuratie-referentie.md)**. Je hebt ze voor een gewone run niet nodig.

