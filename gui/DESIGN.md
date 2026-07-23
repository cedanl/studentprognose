# GUI-design system

Deze stijlgids is de referentie voor álle GUI-pagina's (#272). Elke feature-issue
(#266–#271) bouwt hiertegen. Doel: één coherent geheel dat aanvoelt als een
doordacht product, met duidelijke stappen en een gelikt eindoverzicht.

Wijzig tokens in `gui/theme.py`, niet los in pagina's. Gebruik de componenten uit
`gui/components/states.py` in plaats van states per pagina opnieuw te bouwen.

## 1. Navigatiestructuur

Drie modi naast elkaar, alle gevoed door één model in `gui/nav.py`:

**Pipeline-hub** (landingspagina `/`) — de volledige pipeline als klikbare keten,
de primaire navigatie-hub. Gerenderd door `components/pipeline_rail.py`:

```
[1: Project] → [2: Configuratie] → [3: Filteren] → [4: Modelleren] → [5: Resultaten]
                                              │ binnen stap 4:
                            [SARIMA ‖ XGBoost ‖ Ratio] → Ensemble        (zijspoor: Benchmark & tune)
```

De status per knoop weerspiegelt de echte projectstatus (`pipeline_rail.compute_states`):
afgerond (groen vinkje), nu (oranje rand + gevulde radio), te doen (grijs), of
vergrendeld (slot + tooltip "Kies eerst een project (stap 1)"). Betekenis wordt
altijd door kleur **én** icoon **én** tekst gedragen, nooit door kleur alleen. Elk
knooppunt navigeert naar de bijbehorende pagina; vergrendelde knopen niet.

**Wizard-flow** (nieuwe gebruikers) — lineaire progressie met stap-indicator:

```
[1: Project] → [2: Configuratie] → [3: Filteren] → [4: Uitvoeren] → [5: Resultaten]
```

De stepper (`layout._stepper`) toont afgeronde stappen met een groen vinkje, de
huidige stap gemarkeerd, en komende stappen grijs.

**Vrije navigatie** (terugkerende gebruikers) — de zijbalk (`layout._drawer`) blijft
op alle pagina's als snelnavigatie naar alle pagina's plus de losse tools
(Benchmark & tune).

**Toegankelijkheidsregels voor navigatie-items:**

| Situatie | Weergave |
|----------|----------|
| Pagina bestaat en project gekozen | Actief, klikbaar |
| Pagina nog niet gebouwd | Uitgeschakeld, tooltip "Nog in ontwikkeling" |
| Pagina vereist project, maar geen project gekozen | Uitgeschakeld, tooltip "Kies eerst een project (stap 1)" |

Zo ziet de gebruiker altijd de volledige flow, zonder ooit in een 404 te lopen.

## 2. Kleur & typografie — CEDA/Npuls-huisstijl

Bron van de tokens: `gui/theme.py`, in lijn met de gedeelde organisatie-
stylesheet [`cedanl/.github`](https://github.com/cedanl/.github) →
`docs/stylesheets/extra.css`. Het merk is **zwart primair, wit als achtergrond,
npuls-oranje als accent**, met de Npuls-secundaire kleuren voor reeksen/status.

| Token | Waarde | Gebruik |
|-------|--------|---------|
| `PRIMARY` | `#000000` (zwart) | App-bar, tekst, primaire knoppen |
| `ACCENT` | `#DD784B` (npuls-oranje) | Accenten, actieve navigatie/stepper/tabs, hero, links |
| `SECONDARY` | `#3D68EC` (npuls-blauw) | Secundaire highlights, KPI-iconen |
| `POSITIVE` | `#00AF81` (npuls-groen) | Succes, lage voorspelfout, winnend model |
| `WARNING` | `#E6A020` (amber) | Waarschuwing, matige fout |
| `NEGATIVE` | `#C0392B` (rood) | Fout, hoge voorspelfout |
| `INFO` | `#3D68EC` (npuls-blauw) | Bezig, neutrale info |

Grafieken gebruiken `CHART_SEQUENCE` (oranje → blauw → groen → geel → paars) en
de MAPE-buckets `MAPE_GOOD`/`MAPE_MEDIUM`/`MAPE_BAD` (groen/oranje/rood).

**Merkregel**: zwart draagt de structuur en primaire acties (hoog contrast,
premium); oranje is gereserveerd voor accenten en actieve staten. Meng deze
rollen niet — een oranje vlak signaleert "hier ben je / let hierop", niet een
standaardknop.

**Assets** (`gui/assets/`, geserveerd op `/gui-assets`): `logo.svg` (app-bar-
mark), `header.svg` (hero op de startpagina), `favicon.svg`. Allemaal zwart +
npuls-oranje.

- **Hiërarchie**: `text-3xl font-bold` (paginatitel) › `text-lg font-medium`
  (sectie) › `text-base` (body) › `text-sm opacity-70` (toelichting).
- **Spacing**: 8px-grid via Quasar/Tailwind-classes (`gap-2`, `p-6`, …).
- **Iconen**: Material Icons, consistent per concept (zie `nav.py`).

## 3. De vijf UI-states

Elke pagina die werk uitvoert, kent expliciet deze states. Gebruik
`states.status_badge(state)` voor de badge.

| State | Badge | Visuele behandeling |
|-------|-------|---------------------|
| `idle` | grijs | Nog niet gestart; start-knop actief zodra invoer geldig is |
| `ready` | groen | Invoer compleet en geldig |
| `running` | blauw | Spinner, stop-knop zichtbaar, start-knop uitgeschakeld |
| `success` | groen | Groene bevestiging + navigatie naar de volgende stap |
| `failed` | rood | `states.error_banner(...)` met herstelstap |

## 4. Leegsituaties

Elke pagina die data/config verwacht, toont bij leegte een
`states.empty_state(...)`: icoon + uitleg + één duidelijke call-to-action naar de
stap die het oplost. Nooit een kaal leeg scherm.

Voorbeeld (startpagina zonder project): "Nog geen project" → knop "Project
opzetten" → `/wizard`.

## 5. Foutafhandeling — actiegericht

Een gebruiker ziet **nooit** een kale traceback. Ruwe fouten gaan door
`errors.humanize_error(raw)`, dat een `FriendlyError(message, hint)` teruggeeft;
render die met `states.error_banner(message, hint)`.

| ❌ Niet tonen | ✅ Wel tonen |
|--------------|-------------|
| `FileNotFoundError: data/input/student_count_first-years.xlsx` | "Een benodigd bestand ontbreekt." + "Controleer de paden in Configuratie of draai eerst de ETL zonder --noetl." |

De volledige logregels blijven wél beschikbaar (uitklapbaar) voor wie ze wil zien.

## 6. Componentcontract (samengevat)

| Component | Functie | Bron |
|-----------|---------|------|
| Paginaschil | `page_shell(active, title, show_stepper=)` | `components/layout.py` |
| Pipeline-hub | `pipeline_rail.render(on_nav=)`, `compute_states()` | `components/pipeline_rail.py` |
| Statusbadge | `status_badge(state)` | `components/states.py` |
| Leegsituatie | `empty_state(icon, title, message, action_label, on_action)` | `components/states.py` |
| Foutbanner | `error_banner(message, hint)` | `components/states.py` |
| Infobanner | `info_banner(message)` | `components/states.py` |
| Sectiekop | `section_title(text, subtitle)` | `components/states.py` |
| Foutvertaling | `humanize_error(raw) -> FriendlyError` | `errors.py` |
| Navigatiemodel | `WIZARD_FLOW`, `TOOLS`, `register_route`, `is_available` | `nav.py` |
