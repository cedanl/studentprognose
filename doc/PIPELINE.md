# Dataflow: van Studielink naar Prognose

Dit document beschrijft hoe ruwe Studielink-data en instellingsdata worden getransformeerd naar de bestanden in `data/input/`, en hoe die vervolgens door het prognosemodel worden verwerkt tot output.

---

## Wat moet ik aanleveren?

### Verplichte bestanden

| Bestand | Beschrijving | Bron | Wanneer nodig |
|---------|-------------|------|---------------|
| `vooraanmeldingen_cumulatief.csv` | Gewogen/ongewogen vooraanmelders per opleiding, herkomst, week, jaar | Studielink telbestanden → ETL stap 1 + 2 | Bij `-d c` of `-d b` (default) |
| `vooraanmeldingen_individueel.csv` | Een rij per student-aanmelding met persoonskenmerken | Direct uit SIS/datawarehouse van de instelling | Bij `-d i` of `-d b` (default) |
| `student_count_first-years.xlsx` | Werkelijk aantal eerstejaars per opleiding/herkomst/jaar | Oktober-bestand (1-cijfer HO) → ETL stap 3 | Altijd |
| `student_count_higher-years.xlsx` | Werkelijk aantal hogerjaars per opleiding/herkomst/jaar | Oktober-bestand (1-cijfer HO) → ETL stap 3 | Altijd |
| `student_volume.xlsx` | Totaal studentvolume per opleiding/herkomst/jaar | Oktober-bestand (1-cijfer HO) → ETL stap 3 | Altijd |

### Optionele bestanden

| Bestand | Beschrijving | Bron |
|---------|-------------|------|
| `ensemble_weights.xlsx` | Gewichten per model voor de ensemble-voorspelling | Gegenereerd door post-processing stap A |
| `totaal_cumulatief.xlsx` / `totaal_individueel.xlsx` | Historische voorspellingen (eerdere model-runs) | Gegenereerd door post-processing stap B |

> Bij een eerste run zijn de optionele post-processing bestanden nog niet beschikbaar. Het model draait zonder — ze worden pas aangemaakt na de eerste model-run via de post-processing scripts.

---

## Overzicht

```mermaid
flowchart TD
    classDef bron fill:#3d3d3d,color:#fff,stroke:#555,stroke-width:2px
    classDef script fill:#1a6dd4,color:#fff,stroke:#0d4ea6,stroke-width:2px
    classDef verplicht fill:#1a8a3e,color:#fff,stroke:#136b2e,stroke-width:2px
    classDef optioneel fill:#c47415,color:#fff,stroke:#9a5c10,stroke-width:2px
    classDef model fill:#7c3aed,color:#fff,stroke:#5b21b6,stroke-width:3px
    classDef output fill:#dc2626,color:#fff,stroke:#991b1b,stroke-width:2px

    linkStyle default stroke:#888,stroke-width:1.5px

    %% ══════════════════════════════════════
    %% LAAG 1 — Externe bronnen
    %% ══════════════════════════════════════
    subgraph bronnen ["Externe bronnen"]
        direction LR
        SL["Studielink Telbestanden<br/><i>telbestandY2024WXX.csv</i>"]:::bron
        OKT["Oktober-bestand<br/><i>1-cijfer HO</i>"]:::bron
        SIS["Individuele aanmelddata<br/><i>SIS / datawarehouse</i>"]:::bron
    end

    %% ══════════════════════════════════════
    %% LAAG 2 — ETL (s01_etl.py)
    %% ══════════════════════════════════════
    subgraph etl ["ETL — s01_etl.py (--etl)"]
        direction LR
        S1["1 · Rowbind + reformat<br/><i>samenvoegen telbestanden</i>"]:::script
        S2["2 · Interpolate<br/><i>ontbrekende weken</i>"]:::script
        S3["3 · Calculate student count<br/><i>oktober → aantallen</i>"]:::script
        S4["4 · Copy bestanden<br/><i>individueel</i>"]:::script
    end

    %% ══════════════════════════════════════
    %% LAAG 3 — data/input/
    %% ══════════════════════════════════════
    subgraph input_verplicht ["data/input/ — verplicht"]
        direction LR
        VC["vooraanmeldingen_cumulatief.csv"]:::verplicht
        VI["vooraanmeldingen_individueel.csv"]:::verplicht
        SC["student_count_first-years.xlsx<br/>student_count_higher-years.xlsx<br/>student_volume.xlsx"]:::verplicht
    end

    subgraph input_optioneel ["data/input/ — optioneel"]
        direction LR
        EW["ensemble_weights.xlsx"]:::optioneel
        TC["totaal_cumulatief.xlsx<br/>totaal_individueel.xlsx"]:::optioneel
    end

    %% ══════════════════════════════════════
    %% LAAG 4 — Prognosemodel (main.py)
    %% ══════════════════════════════════════
    subgraph pipeline ["Prognosemodel — main.py"]
        direction TD

        LOAD["<b>s02_loader</b> → <b>s03_add_zero_weeks</b><br/><i>data laden + nulweken toevoegen</i>"]:::script
        CISUB["<b>s04_ci_subset</b><br/><i>optioneel: subset voor CI</i>"]:::script

        LOAD --> CISUB
        CISUB --> IND_PATH & CUM_PATH & BOTH_PATH

        subgraph IND_PATH ["-d individual"]
            direction TD
            I_PRE["strategies/individual<br/><i>preprocessing</i>"]
            I_S05["<b>s05</b>_xgboost_classifier<br/><i>kans per student</i>"]
            I_S06["<b>s06</b>_transforms<br/><i>pivot + cumsum</i>"]
            I_S07["<b>s07</b>_sarima<br/><i>SARIMA individual</i>"]
            I_PRE --> I_S05 --> I_S06 --> I_S07
        end

        subgraph CUM_PATH ["-d cumulative"]
            direction TD
            C_PRE["strategies/cumulative<br/><i>preprocessing</i>"]
            C_S07["<b>s07</b>_sarima → <b>s06</b><br/><i>SARIMA cumulative</i>"]
            C_S08["<b>s08</b>_xgboost_regressor<br/><i>vooraanmelders → inschrijvingen</i>"]
            C_S09["<b>s09</b>_ratio<br/><i>historische verhoudingen</i>"]
            C_PRE --> C_S07 --> C_S08 --> C_S09
        end

        subgraph BOTH_PATH ["-d both (standaard)"]
            direction TD
            B_PRE["strategies/combined<br/><i>preprocessing beide</i>"]
            B_S05["<b>s05</b>_xgboost_classifier"]
            B_S06["<b>s06</b>_transforms"]
            B_S07["<b>s07</b>_sarima<br/><i>individual + cumulative</i>"]
            B_S08["<b>s08</b>_xgboost_regressor"]
            B_S09["<b>s09</b>_ratio"]
            B_PRE --> B_S05 --> B_S06 --> B_S07 --> B_S08 --> B_S09
        end

        S10["<b>s10_postprocessor</b><br/><i>ensemble, foutmaten, opslaan</i>"]:::script
        I_S07 --> S10
        C_S09 --> S10
        B_S09 --> S10
    end

    %% ══════════════════════════════════════
    %% LAAG 5 — Output
    %% ══════════════════════════════════════
    OUT["<b>data/output/</b><br/><br/>output_prelim_*.xlsx<br/>output_first-years_*.xlsx<br/>output_higher-years_*.xlsx<br/>output_volume_*.xlsx"]:::output

    %% ══════════════════════════════════════
    %% LAAG 6 — Post-processing (archive/)
    %% ══════════════════════════════════════
    subgraph postproc ["Post-processing scripts (archive/)"]
        direction LR
        PA["A · calculate_ensemble_weights.py<br/><i>optimale gewichten</i>"]:::script
        PB["B · append_studentcount_and_compute_errors.py<br/><i>werkelijke aantallen + fouten</i>"]:::script
    end

    %% ── Bronnen → ETL ──
    SL --> S1 --> S2 --> VC
    OKT --> S3 --> SC
    SIS --> S4 --> VI
    S4 --> AF

    %% ── data/input → Model ──
    VC & VI & SC --> LOAD
    EW & TC -.-> LOAD

    %% ── Model → Output ──
    S10 ==> OUT

    %% ── Output → Post-processing ──
    OUT --> PA
    OUT --> PB
    SC --> PB

    %% ── Feedback loop ──
    PA -.-> EW
    PB -.-> TC

    %% ── Subgraph styling ──
    style bronnen fill:#f5f5f5,stroke:#bbb,stroke-width:1px,color:#333
    style etl fill:#eff6ff,stroke:#93c5fd,stroke-width:1px,color:#1e40af
    style input_verplicht fill:#f0fdf4,stroke:#86efac,stroke-width:1px,color:#166534
    style input_optioneel fill:#fffbeb,stroke:#fcd34d,stroke-width:1px,color:#92400e
    style pipeline fill:#faf5ff,stroke:#c4b5fd,stroke-width:2px,color:#5b21b6
    style IND_PATH fill:#dcfce7,stroke:#86efac,stroke-width:1px,color:#166534
    style CUM_PATH fill:#fef3c7,stroke:#fcd34d,stroke-width:1px,color:#92400e
    style BOTH_PATH fill:#ede9fe,stroke:#c4b5fd,stroke-width:1px,color:#5b21b6
    style postproc fill:#eff6ff,stroke:#93c5fd,stroke-width:1px,color:#1e40af
```

---

## ETL Scripts (s01_etl.py)

Het ETL-script (`uv run main.py --etl`) transformeert ruwe data in `data/input_raw/` naar verwerkte bestanden in `data/input/`.

| Stap | Actie | Input | Output |
|------|-------|-------|--------|
| 1 | Rowbind + reformat | `data/input_raw/telbestanden/*.csv` | Samengevoegd CSV |
| 2 | Interpolatie | Samengevoegd CSV | `vooraanmeldingen_cumulatief.csv` |
| 3 | Studentaantallen | Oktober-bestand (1-cijfer HO) | `student_count_*.xlsx`, `student_volume.xlsx` |
| 4 | Kopieren | Individuele data | `vooraanmeldingen_individueel.csv` |

---

## Pipeline Executievolgorde

**Gedeelde stappen (alle modi):** `cli.py` → `s01_etl`* → `config.py` → `s02_loader` → `s03_add_zero_weeks` → `s04_ci_subset`*

| Stap | Fase | Individual (`-d i`) | Cumulative (`-d c`) | Both (`-d b`) |
|------|------|---------------------|---------------------|---------------|
| 6 | Preprocessing | `strategies/individual` | `strategies/cumulative` | individual → cumulative |
| 7 | Filtering | `strategies/base` | `strategies/base` | `strategies/base` |
| 8 | Classificatie | `s05_xgboost_classifier` | — | `s05_xgboost_classifier` |
| 9 | Transformatie | `s06_transforms` | — | `s06_transforms` |
| 10 | SARIMA | `s07_sarima` (individual) | `s07_sarima` → `s06_transforms` | `s07_sarima` (both) |
| 11 | XGBoost regressor | — | `s08_xgboost_regressor` | `s08_xgboost_regressor` |
| 12 | Ratio model | — | `s09_ratio` | `s09_ratio` |
| 13 | Postprocessing | `s10_postprocessor` | `s10_postprocessor` | `s10_postprocessor` |

<small>* alleen met `--etl` resp. `--ci test N`</small>

---

## Twee databronnen, twee sporen

**Cumulatief spoor (Studielink → model)**
Studielink levert wekelijks telbestanden met geaggregeerde aanmeldcijfers per opleiding. Het ETL-script voegt deze samen (stap 1), interpoleert ontbrekende weken (stap 2), en schrijft `vooraanmeldingen_cumulatief.csv`. Dit vormt de basis voor de `SARIMA_cumulative` voorspelling.

**Individueel spoor (instelling → model)**
De instelling levert per-student aanmelddata uit het eigen SIS/datawarehouse. Dit bestand wordt direct aangeleverd als `vooraanmeldingen_individueel.csv`. Het vormt de basis voor de `SARIMA_individual` voorspelling.

**Studentaantallen (oktober-bestand → ground truth)**
Het oktober-bestand (1-cijfer HO) bevat de werkelijke inschrijvingen na 1 oktober. Het ETL-script leidt hieruit de ground truth af die het model als referentie gebruikt.

Het prognosemodel combineert beide sporen via ensemble weging om een voorspelling te maken van het verwachte aantal studenten.

---

## Post-processing (feedback loop)

Na een model-run kunnen de volgende scripts worden gedraaid om de input voor de volgende run te verbeteren:

| Stap | Script | Input | Output |
|------|--------|-------|--------|
| A | `archive/calculate_ensemble_weights.py` | `totaal_cumulatief.xlsx` + `ensemble_weights.xlsx` | `ensemble_weights.xlsx` (bijgewerkt) |
| B | `archive/append_studentcount_and_compute_errors.py` | `totaal_*.xlsx` + `student_count_*.xlsx` | `totaal_*.xlsx` (bijgewerkt met werkelijke aantallen + fouten) |

---

## Output bestanden

| Bestand | Beschrijving |
|---------|-------------|
| `output_prelim_*.xlsx` | Voorlopige voorspellingen (tussenresultaat) |
| `output_first-years_*.xlsx` | Eerstejaars voorspellingen (eindresultaat) |
| `output_higher-years_*.xlsx` | Hogerjaars voorspellingen (eindresultaat) |
| `output_volume_*.xlsx` | Volume-voorspellingen (totaal) |

### Kolommen in output

Per rij: **opleiding + herkomst + examentype + week + jaar**

Voorspellingen: `SARIMA_individual`, `SARIMA_cumulative`, `Prognose_ratio`, `Ensemble_prediction`

Foutmaten: `MAE_*` (gemiddelde absolute afwijking), `MAPE_*` (procentuele afwijking)
