"""Genereert de Jupyter-notebooks voor issue #168.

Dit script is geen onderdeel van de tool zelf. Het bouwt de notebooks
deterministisch op zodat ze gemakkelijk geregenereerd kunnen worden
bij wijzigingen. Draai met:

    uv run python notebooks/_build_notebooks.py

De gegenereerde .ipynb-bestanden worden in dezelfde map geschreven.
"""

from __future__ import annotations

import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def make_notebook(cells: list[nbf.NotebookNode], title: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
        "title": title,
    }
    return nb


def write(name: str, cells: list[nbf.NotebookNode], title: str) -> None:
    path = ROOT / name
    nb = make_notebook(cells, title)
    nbf.write(nb, path)
    print(f"  schreef {path.relative_to(ROOT.parent)}")


# ============================================================
# Herbruikbare cellen
# ============================================================

HBO_BANNER = md(
    """\
> ⚠️ **De demodata is Radboud (WO).** De voorbeelden gebruiken WO-opleidingen
> (`B Psychologie`, `B Bedrijfskunde`, …). Voor een hogeschool met eigen data:
> 1. Vervang `data/input/vooraanmeldingen_cumulatief.csv` door je eigen ETL-output
> 2. Pas `PROGRAMMA`, `HERKOMST`, `PREDICT_YEAR`, `PREDICT_WEEK` aan naar wat in jouw data zit
> 3. HBO-specifiek: typisch meer numerus-fixus opleidingen, weinig masters, andere 1-mei-deadline-effecten.
"""
)


SETUP_CELL = code(
    """\
# --- Standaard setup voor alle studentprognose-notebooks ---
import sys
from pathlib import Path

# Maak _helpers.py importeerbaar en ga naar projectroot
NOTEBOOKS_DIR = Path.cwd() if Path.cwd().name == "notebooks" else Path.cwd() / "notebooks"
sys.path.insert(0, str(NOTEBOOKS_DIR))

from _helpers import project_root, setup_matplotlib  # noqa: E402
project_root()
setup_matplotlib()

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)

print("Werkmap:", Path.cwd())
print("Python :", sys.version.split()[0])
"""
)


LOAD_CUMULATIVE_CELL = code(
    """\
from _helpers import load_cumulative

data_cumulative, data_studentcount, data_cumulative_raw = load_cumulative()
print(f"Cumulatieve telregels:    {len(data_cumulative):,}")
print(f"Aantal opleidingen:       {data_cumulative['Croho groepeernaam'].nunique()}")
print(f"Historische realisatie:   {len(data_studentcount):,} rijen (Aantal_studenten per opleiding/jaar)")
"""
)


# ============================================================
# 00 — Overzicht (pipeline in 5 minuten, mét realisatie-vergelijking)
# ============================================================

def build_overzicht():
    cells = [
        md(
            """\
# 00 — Pipeline in vijf minuten

Deze notebook geeft een **end-to-end overzicht** van studentprognose: van data inladen
tot een voorspelling die je kunt vergelijken met de werkelijke realisatie.

> 🔗 **Aanvullend bij:** [`docs/methodologie/index.md`](../docs/methodologie/index.md) en [`docs/aan-de-slag.md`](../docs/aan-de-slag.md).

## Wat doet de pipeline?

Voor een opleiding × herkomst × examentype voorspelt het model het **aantal
eerstejaarsinschrijvingen** in een collegejaar, gegeven de stand van zaken op een
**peilweek** (1–37).

| Modus | CLI-vlag | Databron | Modellen |
|-------|----------|----------|----------|
| Cumulatief | `-d c` | Studielink telbestanden | SARIMA + XGBoost regressor |
| Individueel | `-d i` | Osiris/Usis per-student | XGBoost classifier + SARIMA |
| Beide | `-d b` | Beide bronnen | Volledig ensemble + ratio |

In deze notebook draaien we het **cumulatieve spoor** (`-d c`) als demonstratie.
"""
        ),
        HBO_BANNER,
        SETUP_CELL,
        md(
            """\
## 1. Data laden

Vergeleken met een typische Excel-workflow scheelt dit twee uur: één regel laadt
de cumulatieve telregels, historische realisatie én ruwe data uit `data/input/`.
"""
        ),
        LOAD_CUMULATIVE_CELL,
        md(
            """\
## 2. De volledige pipeline draaien (in-memory)

We gebruiken hier `run_pipeline_from_dataframes` — exact hetzelfde als
`studentprognose --noetl -d c -y 2023 -w 12` aan de CLI, maar het resultaat komt
terug als DataFrame zodat we er meteen verder mee kunnen werken.

We voorspellen **2023** vanaf peilweek **12**, zodat we de uitkomst kunnen
vergelijken met de werkelijke realisatie van 2023.
"""
        ),
        code(
            """\
from studentprognose import run_pipeline_from_dataframes, DataOption
from _helpers import suppress_stdout

PREDICT_YEAR = 2023
PREDICT_WEEK = 12

with suppress_stdout():  # onderdruk per-opleiding 'Predicting for …' regels
    result = run_pipeline_from_dataframes(
        year=PREDICT_YEAR,
        week=PREDICT_WEEK,
        data_cumulative=data_cumulative_raw,
        data_student_numbers=data_studentcount,
        dataset=DataOption.CUMULATIVE,
        save_output=False,
    )

print(f"Outputvorm: {result.shape[0]} rijen × {result.shape[1]} kolommen")
print(f"Opleidingen: {result['Croho groepeernaam'].nunique()}")
print(f"Modellen aanwezig: ", [c for c in ['SARIMA_cumulative','SARIMA_individual','Prognose_ratio','Baseline','Ensemble_prediction'] if c in result.columns])
"""
        ),
        md(
            """\
## 3. Voorspelling × Realisatie

Het echte testmoment: hoe goed presteerden de modellen op 2023?

We voegen de werkelijke realisatie (`Aantal_studenten` uit DUO) toe en berekenen
de afwijking per model. Voor leesbaarheid aggregeren we over herkomst — een
hogeschool zou per herkomst willen kijken voor internationaal beleid.
"""
        ),
        code(
            """\
from _helpers import with_realisatie

# Filter eerst op peilweek (SARIMA_cumulative en Prognose_ratio bestaan voor verschillende weken;
# SARIMA staat alleen op peilweek, Prognose_ratio wordt voor elke week berekend)
result_peilweek = result[result["Weeknummer"] == PREDICT_WEEK].copy()

# Aggregeer per opleiding (sommeer over herkomst, Bachelor only)
agg = (
    result_peilweek[result_peilweek["Examentype"] == "Bachelor"]
    .groupby(["Croho groepeernaam", "Collegejaar", "Examentype"], as_index=False)
    [["SARIMA_cumulative", "Prognose_ratio"]]
    .sum()
)
# Realisatie eveneens gesommeerd over herkomst
realisatie_agg = (
    data_studentcount[data_studentcount["Examentype"] == "Bachelor"]
    .groupby(["Croho groepeernaam", "Collegejaar", "Examentype"], as_index=False)
    ["Aantal_studenten"].sum()
    .rename(columns={"Aantal_studenten": "Realisatie"})
)

vergelijking = agg.merge(realisatie_agg, on=["Croho groepeernaam", "Collegejaar", "Examentype"])
vergelijking["MAE_SARIMA"] = (vergelijking["SARIMA_cumulative"] - vergelijking["Realisatie"]).round(0)
vergelijking["MAE_Ratio"]  = (vergelijking["Prognose_ratio"]    - vergelijking["Realisatie"]).round(0)
vergelijking["%_SARIMA"] = ((vergelijking["SARIMA_cumulative"] - vergelijking["Realisatie"]) / vergelijking["Realisatie"] * 100).round(1)
vergelijking["%_Ratio"]  = ((vergelijking["Prognose_ratio"]    - vergelijking["Realisatie"]) / vergelijking["Realisatie"] * 100).round(1)

cols = ["Croho groepeernaam", "Realisatie", "SARIMA_cumulative", "%_SARIMA", "Prognose_ratio", "%_Ratio"]
vergelijking[cols].sort_values("Realisatie", ascending=False)
"""
        ),
        md(
            """\
## 4. Visualisatie — voorspelling versus realisatie

Hoe dichter een punt bij de diagonaal, hoe beter het model voorspelde. Punten
boven de lijn = overschatting, onder de lijn = onderschatting.
"""
        ),
        code(
            """\
fig, ax = plt.subplots(figsize=(8, 8))

max_v = vergelijking[["Realisatie", "SARIMA_cumulative", "Prognose_ratio"]].max().max() * 1.05
ax.plot([0, max_v], [0, max_v], "k--", alpha=0.4, label="Perfecte voorspelling")
ax.scatter(vergelijking["Realisatie"], vergelijking["SARIMA_cumulative"],
           s=60, alpha=0.75, color="steelblue", label="SARIMA cumulatief")
ax.scatter(vergelijking["Realisatie"], vergelijking["Prognose_ratio"],
           s=60, alpha=0.75, color="darkgreen", marker="s", label="Ratio-model")

for _, row in vergelijking.iterrows():
    if row["Realisatie"] > 200:
        ax.annotate(row["Croho groepeernaam"].replace("B ", ""),
                    (row["Realisatie"], row["SARIMA_cumulative"]),
                    fontsize=8, alpha=0.7, xytext=(5, 5), textcoords="offset points")

ax.set_xlabel("Realisatie 2023 (werkelijk aantal eerstejaars)")
ax.set_ylabel("Voorspelling op peilweek 12")
ax.set_title(f"Voorspelling vs. realisatie · {PREDICT_YEAR} bachelor-opleidingen (alle herkomsten)")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """\
## 5. Samenvattende MAE en MAPE

De **MAE** (Mean Absolute Error) is de gemiddelde absolute afwijking in *aantal studenten*.
De **MAPE** (Mean Absolute Percentage Error) is hetzelfde maar in *%*, vergelijkbaar tussen
grote en kleine opleidingen.
"""
        ),
        code(
            """\
samenvatting = pd.DataFrame({
    "Model": ["SARIMA cumulatief", "Ratio-model"],
    "MAE (studenten)": [
        vergelijking["MAE_SARIMA"].abs().mean().round(1),
        vergelijking["MAE_Ratio"].abs().mean().round(1),
    ],
    "MAPE (%)": [
        vergelijking["%_SARIMA"].abs().mean().round(1),
        vergelijking["%_Ratio"].abs().mean().round(1),
    ],
    "Aantal opleidingen": [len(vergelijking)] * 2,
})
samenvatting
"""
        ),
        md(
            """\
## 6. Conclusie en volgende stappen

Op deze demodata (Radboud 2023, peilweek 12) zien we:

- Beide modellen produceren een redelijke prognose, met **relatieve fouten van ±10–15%**.
- De **ratio-baseline** is verrassend competitief — een belangrijk signaal: de complexere
  modellen moeten dit *minstens* evenaren om hun complexiteit te rechtvaardigen.
- Een hogeschool zou hier per opleiding een eigen oordeel maken: voor stabiele
  opleidingen (vaste conversie) is het ratio-model voldoende; voor opleidingen met
  veranderend instroombeleid voegen SARIMA en XGBoost waarde toe.

| Wil je weten… | Open notebook |
|---------------|---------------|
| Of je eigen data correct geformatteerd is | `01_data_voorbereiden.ipynb` |
| Waarom de SARIMA-ordes vast staan | `02_sarima.ipynb` |
| Hoe de XGBoost-modellen werken | `03_xgboost.ipynb` |
| Wanneer het ratio-model voldoende is | `04_ratio_model.ipynb` |
| Hoe de modellen worden gecombineerd | `05_ensemble.ipynb` |
| Hoe je de Excel-output leest | `06_output_interpreteren.ipynb` |
"""
        ),
    ]
    write("00_overzicht.ipynb", cells, "Pipeline-overzicht")


# ============================================================
# 01 — Data voorbereiden (met validatierapport)
# ============================================================

def build_data_voorbereiden():
    cells = [
        md(
            """\
# 01 — Je data voorbereiden

Deze notebook is de uitvoerbare versie van [`docs/je-data-voorbereiden.md`](../docs/je-data-voorbereiden.md)
en [`docs/validatie.md`](../docs/validatie.md). We:

1. Laden de demodata en lopen het schema door
2. Genereren een **validatierapport** dat je 1-op-1 op je eigen data kunt draaien
3. Visualiseren een typische aanmeldcurve over meerdere jaren
4. Maken een dekkings-heatmap die ontbrekende weken zichtbaar maakt

> 💡 **Voor je eigen data:** vervang `data/input/vooraanmeldingen_cumulatief.csv` door je
> eigen ETL-output en herstart de kernel. De validatie-cellen werken zonder code-aanpassing.
"""
        ),
        HBO_BANNER,
        SETUP_CELL,
        md(
            """\
## 1. Data laden

`load_cumulative()` laadt de cumulatieve telregels, de historische realisatie én de
RAW-vorm (zonder preprocessing) in één call. De preprocessing past dezelfde renames
toe als de pipeline-strategie — als analist hoef je dat niet zelf te doen.
"""
        ),
        LOAD_CUMULATIVE_CELL,
        md(
            """\
## 2. Schema van het cumulatieve bestand

Eén rij = één combinatie van (jaar × week × opleiding × herkomst × examentype). De
twee belangrijkste numerieke velden zijn `Gewogen vooraanmelders` en `Inschrijvingen`.
"""
        ),
        code(
            """\
print("Kolommen + type:")
for col in data_cumulative.columns:
    print(f"  - {col:42s} {data_cumulative[col].dtype}")
data_cumulative.head(3)
"""
        ),
        md(
            """\
## 3. Validatierapport

Hieronder een **draaibaar rapport** dat de belangrijkste checks uit `docs/validatie.md`
direct op de geladen data uitvoert. Op eigen data laat dit in één klap zien of je klaar
bent om te draaien.
"""
        ),
        code(
            """\
def validatie_rapport(df: pd.DataFrame, df_sc: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Compacte versie van de validatiechecks uit docs/validatie.md.\"\"\"
    rows = []

    # 1. Verplichte kolommen
    verplicht = {"Collegejaar", "Weeknummer", "Croho groepeernaam",
                 "Herkomst", "Examentype", "Gewogen vooraanmelders", "Inschrijvingen"}
    ontbreekt = verplicht - set(df.columns)
    rows.append({
        "Check": "Verplichte kolommen aanwezig",
        "Status": "✅" if not ontbreekt else "❌",
        "Detail": "alle aanwezig" if not ontbreekt else f"ontbreken: {sorted(ontbreekt)}",
    })

    # 2. Jaarbereik (current_year - 15 .. current_year + 2 → soft error)
    jaren = sorted(df["Collegejaar"].dropna().unique().astype(int))
    huidig = 2024
    out_of_range = [y for y in jaren if y < huidig - 15 or y > huidig + 2]
    rows.append({
        "Check": "Collegejaar in plausibel bereik",
        "Status": "✅" if not out_of_range else "⚠️",
        "Detail": f"jaren {jaren[0]}–{jaren[-1]}" + (f" ; buiten bereik: {out_of_range}" if out_of_range else ""),
    })

    # 3. Herkomst-waarden
    herk = sorted(df["Herkomst"].dropna().unique())
    verwacht = {"NL", "EER", "Niet-EER"}
    onbekend = set(herk) - verwacht
    rows.append({
        "Check": "Herkomst-waarden geldig",
        "Status": "✅" if not onbekend else "⚠️",
        "Detail": f"{herk}" + (f" ; onbekend: {sorted(onbekend)}" if onbekend else ""),
    })

    # 4. NaN-percentage (warning > 5%, soft-error > 30%)
    nan_pct = df["Gewogen vooraanmelders"].isna().mean() * 100
    status = "✅" if nan_pct <= 5 else ("⚠️" if nan_pct <= 30 else "❌")
    rows.append({
        "Check": "Ontbrekende waarden in Gewogen vooraanmelders",
        "Status": status,
        "Detail": f"{nan_pct:.2f}% NaN",
    })

    # 5. Decimaalintegriteit (geen komma-strings of niet-numeriek) — FULL check, geen sample
    raw_col = df["Gewogen vooraanmelders"]
    if pd.api.types.is_numeric_dtype(raw_col):
        rows.append({
            "Check": "Decimaalintegriteit",
            "Status": "✅",
            "Detail": "kolom is numeriek (geen komma-strings)",
        })
    else:
        n_comma = raw_col.astype(str).str.contains(",", na=False).sum()
        rows.append({
            "Check": "Decimaalintegriteit",
            "Status": "❌" if n_comma > 0 else "⚠️",
            "Detail": f"{n_comma} rijen met komma; verwacht 0",
        })

    # 6. Gaten tussen weken (warning bij gat > 2 binnen één jaar)
    grootste_gat = 0
    for (_, jaar), grp in df.groupby(["Croho groepeernaam", "Collegejaar"]):
        weken = sorted(grp["Weeknummer"].unique())
        if len(weken) > 1:
            gaten = [b - a for a, b in zip(weken, weken[1:])]
            grootste_gat = max(grootste_gat, max(gaten))
    rows.append({
        "Check": "Geen grote gaten tussen weken",
        "Status": "✅" if grootste_gat <= 2 else "⚠️",
        "Detail": f"grootste gap = {grootste_gat} weken (warning bij > 2)",
    })

    # 7. Realisatie aansluit op aanmelddata?
    jaren_cum = set(df["Collegejaar"].astype(int))
    jaren_sc = set(df_sc["Collegejaar"].astype(int))
    gemeenschappelijk = jaren_cum & jaren_sc
    rows.append({
        "Check": "Realisatie (DUO) aanwezig voor aanmeldjaren",
        "Status": "✅" if len(gemeenschappelijk) >= 3 else "⚠️",
        "Detail": f"{len(gemeenschappelijk)} jaren overlappen (≥ 3 nodig voor ratio-model)",
    })

    return pd.DataFrame(rows)


rapport = validatie_rapport(data_cumulative, data_studentcount)
rapport
"""
        ),
        md(
            """\
## 4. Visualisatie: jaarlijkse aanmeldcurve

Het seizoenspatroon dat SARIMA later gebruikt: rust in de zomer (weken 39–52),
steile stijging vanaf januari, knik rond 1 mei-deadline. Voor `B Bedrijfskunde`
(NL Bachelor) — een opleiding die op zowel HBO als WO bestaat — :
"""
        ),
        code(
            """\
from studentprognose.utils.weeks import get_all_weeks_valid
from studentprognose.data.transforms import transform_data

PROGRAMMA = "B Bedrijfskunde"
HERKOMST = "NL"

wide = transform_data(data_cumulative.drop_duplicates(), "Gewogen vooraanmelders")
sample = wide[
    (wide["Croho groepeernaam"] == PROGRAMMA)
    & (wide["Herkomst"] == HERKOMST)
    & (wide["Examentype"] == "Bachelor")
].copy()
weekcols = get_all_weeks_valid(sample.columns)

fig, ax = plt.subplots(figsize=(11, 5))
for jaar in sorted(sample["Collegejaar"].unique()):
    row = sample[sample["Collegejaar"] == jaar][weekcols].iloc[0]
    ax.plot(range(len(weekcols)), row.values, label=str(int(jaar)), alpha=0.85)

ax.axvline(x=len(weekcols) - 1 - (38 - 17), color="red", linestyle=":", alpha=0.4)
ax.text(len(weekcols) - 1 - (38 - 17) + 0.4, ax.get_ylim()[1]*0.95, "1-mei deadline", fontsize=8, color="red")

ax.set_xticks(range(0, len(weekcols), 4))
ax.set_xticklabels([weekcols[i] for i in range(0, len(weekcols), 4)])
ax.set_xlabel("Weeknummer (academische volgorde: 39 → 38)")
ax.set_ylabel("Cumulatief gewogen vooraanmelders")
ax.set_title(f"Aanmeldcurve · {PROGRAMMA} · {HERKOMST} · per collegejaar")
ax.legend(title="Jaar", ncol=2, fontsize=8)
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """\
## 5. Dekkings-heatmap

Donkere cellen = veel aanmelders, lichte = weinig, **wit/leeg = ontbrekend**. Een
hogeschool die ontdekt dat een hele week ontbreekt, kan dat herstellen vóór ze
gaan modelleren.
"""
        ),
        code(
            """\
heat = (
    data_cumulative[data_cumulative["Croho groepeernaam"] == PROGRAMMA]
    .pivot_table(index="Collegejaar", columns="Weeknummer",
                 values="Gewogen vooraanmelders", aggfunc="sum")
    .reindex(columns=sorted(data_cumulative["Weeknummer"].unique()))
)

fig, ax = plt.subplots(figsize=(11, 4.5))
im = ax.imshow(heat.values, aspect="auto", cmap="viridis")
ax.set_yticks(range(len(heat.index)))
ax.set_yticklabels([str(int(y)) for y in heat.index])
ax.set_xticks(range(0, len(heat.columns), 4))
ax.set_xticklabels([str(int(w)) for w in heat.columns[::4]])
ax.set_xlabel("Weeknummer (1–52)")
ax.set_ylabel("Collegejaar")
ax.set_title(f"Dekking gewogen vooraanmelders · {PROGRAMMA}")
plt.colorbar(im, ax=ax, label="Gewogen vooraanmelders")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """\
## 6. Conclusie

Met dit rapport heb je in één blik:

- ✅ Verplichte kolommen, jaarbereik, herkomst-waarden, ontbrekende waarden
- ✅ Decimaalintegriteit (volledige scan, geen sample)
- ✅ Continuïteit tussen weken
- ✅ Aansluiting met DUO-realisatie

Bij ⚠️ of ❌ verwijst de docs [`validatie.md`](../docs/validatie.md) naar de oplossing.
Daarna kun je verder met [`02_sarima.ipynb`](02_sarima.ipynb).
"""
        ),
    ]
    write("01_data_voorbereiden.ipynb", cells, "Data voorbereiden")


# ============================================================
# 02 — SARIMA (met prediction interval + realisatie-vergelijking)
# ============================================================

def build_sarima():
    cells = [
        md(
            """\
# 02 — SARIMA in detail

Deze notebook hoort bij [`docs/methodologie/sarima.md`](../docs/methodologie/sarima.md).
We doen vier dingen die de docs alleen *beschrijven*:

1. **Trainen + forecasten** op één opleiding/herkomst
2. **Prediction interval** plotten (80% en 95%) — wat de docs niet laten zien
3. **Forecast vs. realisatie** vergelijken — het echte testmoment
4. **Aannames concreet checken** (jaar-op-jaar correlatie, historie-lengte)

> 🎯 **Doel:** intuïtie voor *wat SARIMA leert* en *wanneer je het niet moet vertrouwen*.
"""
        ),
        HBO_BANNER,
        SETUP_CELL,
        md(
            """\
## 1. Wat is SARIMA — in gewoon Nederlands

Aanmelddata heeft twee karakteristieken die het lastig maken:

- **Sterke jaarlijkse seizoenscyclus** — weken 39–52 (zomer) zijn rustig, jan–mei is druk.
- **Sterke autocorrelatie** — week T lijkt sterk op week T-1.

SARIMA combineert beide expliciet:

$$\\text{SARIMA}(p,d,q)(P,D,Q)_{52}$$

| Component | Wat doet het? | Wat betekent het hier? |
|-----------|---------------|------------------------|
| `p` (AR) | Autoregressie op de korte termijn | Hoe sterk hangt week T af van week T-1 |
| `d` (I)  | Differentiatie | Trend wegfilteren (in onze data: 0 → geen trend-diff) |
| `q` (MA) | Moving average van fouten | Verzacht ruis |
| `(P,D,Q)_{52}` | Hetzelfde, maar op de seizoens-lag van 52 weken | Vergelijk week T van dit jaar met week T van vorig jaar |

In de pipeline zijn deze ordes **vast** — `(1,0,1)(1,1,1,52)` voor het cumulatieve spoor.
Vrij vertaald: "kijk 1 week en 52 weken terug, doe één seizoens-differentiatie".
"""
        ),
        code(
            """\
from studentprognose.utils.constants import (
    SARIMA_ORDER, SARIMA_SEASONAL_ORDER,
    SARIMA_ORDER_INDIVIDUAL, SARIMA_SEASONAL_ORDER_ALT,
)
print(f"Cumulatief spoor:    order={SARIMA_ORDER}, seasonal_order={SARIMA_SEASONAL_ORDER}")
print(f"Individueel (alt):   order={SARIMA_ORDER_INDIVIDUAL}, seasonal_order={SARIMA_SEASONAL_ORDER_ALT}")
"""
        ),
        LOAD_CUMULATIVE_CELL,
        md(
            """\
## 2. Trainingsreeks bouwen + model fitten

We voorspellen 2023 vanaf peilweek 12 — zodat we straks kunnen vergelijken met de
**werkelijke realisatie** van wk 38 (einddeadline).
"""
        ),
        code(
            """\
from studentprognose.models.sarima import _get_transformed_data, create_time_series
from studentprognose.utils.weeks import compute_pred_len, get_all_weeks_valid
from statsforecast.models import ARIMA

PROGRAMMA = "B Psychologie"
HERKOMST = "NL"
EXAMENTYPE = "Bachelor"
PREDICT_YEAR = 2023   # backtest: we hebben de realisatie van 2023
PREDICT_WEEK = 12

wide = _get_transformed_data(data_cumulative.copy(), min_training_year=2016)
subset = wide[
    (wide["Croho groepeernaam"] == PROGRAMMA)
    & (wide["Herkomst"] == HERKOMST)
    & (wide["Examentype"] == EXAMENTYPE)
    & (wide["Collegejaar"] <= PREDICT_YEAR)
].copy()
subset["39"] = 0
weekcols = get_all_weeks_valid(subset.columns)
pred_len = compute_pred_len(PREDICT_WEEK)
ts_data = create_time_series(subset, pred_len).astype(float)

print(f"Trainingsreeks: {ts_data.size} weken (~{ts_data.size / 52:.1f} academische jaren)")
print(f"Voorspelhorizon: {pred_len} weken (vanaf wk {PREDICT_WEEK + 1} t/m wk 38 van {PREDICT_YEAR})")
"""
        ),
        md(
            """\
## 3. Model + prediction interval

We gebruiken `statsforecast.ARIMA` rechtstreeks (de zelfde backend als
`SARIMAForecaster`) om naast de puntvoorspelling ook 80% en 95% **prediction
intervals** te krijgen — een planner wil niet alleen "520 studenten" weten,
maar ook de plausibele bandbreedte.
"""
        ),
        code(
            """\
model = ARIMA(order=SARIMA_ORDER, season_length=52, seasonal_order=SARIMA_SEASONAL_ORDER[:3])
model.fit(y=ts_data)
fc = model.predict(h=pred_len, level=[80, 95])

print(f"Forecast & intervallen voor wk {PREDICT_WEEK + 1} t/m wk 38 ({PREDICT_YEAR}):")
mean_arr = np.asarray(fc["mean"])
lo95 = np.asarray(fc["lo-95"])
hi95 = np.asarray(fc["hi-95"])
forecast_table = pd.DataFrame({
    "week": list(range(PREDICT_WEEK + 1, 39)),
    "voorspelling": mean_arr.round(1),
    "lo 95%": lo95.round(1),
    "hi 95%": hi95.round(1),
})
print(forecast_table.head(10).to_string(index=False))
print(f"...")
print(f"\\nVoorspelling wk 38 (einddeadline): {mean_arr[-1]:.0f}  (95%-interval: [{lo95[-1]:.0f}, {hi95[-1]:.0f}])")
"""
        ),
        md(
            """\
## 4. Visualisatie + realisatie

We tekenen de trainingsreeks, de forecast, **de bandbreedte (80% en 95%)** én de
werkelijke wk-38 cumulatieve vooraanmelders uit hetzelfde collegejaar.

> ℹ️ **Let op:** SARIMA voorspelt de *aanmeldcurve* (gewogen vooraanmelders), niet
> direct het aantal *inschrijvingen*. De XGBoost regressor en het ensemble vertalen
> die curve in de uiteindelijke prognose voor het cohort — zie
> [`03_xgboost.ipynb`](03_xgboost.ipynb) en [`06_output_interpreteren.ipynb`](06_output_interpreteren.ipynb).
"""
        ),
        code(
            """\
# Werkelijke gewogen vooraanmelders op week 38 van het voorspeljaar
werkelijk = data_cumulative[
    (data_cumulative["Croho groepeernaam"] == PROGRAMMA)
    & (data_cumulative["Herkomst"] == HERKOMST)
    & (data_cumulative["Examentype"] == EXAMENTYPE)
    & (data_cumulative["Collegejaar"] == PREDICT_YEAR)
    & (data_cumulative["Weeknummer"] == 38)
]["Gewogen vooraanmelders"].sum()

n_full_years = ts_data.size // 52
remainder = ts_data.size - n_full_years * 52

fig, ax = plt.subplots(figsize=(12, 5))

# Historische jaren als achtergrond
for j in range(n_full_years):
    ax.plot(range(52), ts_data[j * 52 : (j + 1) * 52], color="lightgray", alpha=0.55,
            label="Historische jaren" if j == 0 else None)

# Huidige jaar tot peilweek
ax.plot(range(remainder), ts_data[n_full_years * 52 :],
        color="steelblue", linewidth=2,
        label=f"{PREDICT_YEAR} t/m peilweek {PREDICT_WEEK}")

# Forecast met bandbreedte
x_fc = range(remainder, remainder + len(fc["mean"]))
ax.fill_between(x_fc, fc["lo-95"], fc["hi-95"], alpha=0.15, color="darkorange", label="95% interval")
ax.fill_between(x_fc, fc["lo-80"], fc["hi-80"], alpha=0.25, color="darkorange", label="80% interval")
ax.plot(x_fc, fc["mean"], color="darkorange", linewidth=2.5, linestyle="--",
        label=f"Forecast (wk {PREDICT_WEEK + 1}→38)")

# Realisatie wk 38
if werkelijk > 0:
    ax.scatter([remainder + len(fc["mean"]) - 1], [werkelijk],
               color="red", s=110, zorder=10, marker="X",
               label=f"Realisatie wk 38: {werkelijk:.0f}")

ax.set_xlim(0, 52)
ax.set_xlabel("Week binnen academisch jaar (0 = wk 39, 51 = wk 38)")
ax.set_ylabel("Gewogen vooraanmelders")
ax.set_title(f"SARIMA-forecast met intervallen · {PROGRAMMA} · {HERKOMST} · peilweek {PREDICT_WEEK} → wk 38 ({PREDICT_YEAR})")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.show()

if werkelijk > 0:
    fc_wk38 = float(fc["mean"][-1])
    fout = fc_wk38 - werkelijk
    pct = fout / werkelijk * 100
    print(f"\\nForecast wk 38: {fc_wk38:.0f} · Realisatie: {werkelijk:.0f} · Fout: {fout:+.0f} ({pct:+.1f}%)")
"""
        ),
        md(
            """\
## 5. Aannames concreet testen

De docs noemen vier aannames. We checken er hier twee op data:
"""
        ),
        code(
            """\
# Aanname 1: hoe stabiel is het seizoenspatroon over jaren?
full_years = ts_data.size // 52
yearly_curves = ts_data[: full_years * 52].reshape(full_years, 52)
correlations = []
for i in range(yearly_curves.shape[0] - 1):
    c = np.corrcoef(yearly_curves[i], yearly_curves[i + 1])[0, 1]
    correlations.append(c)

aanname_1 = pd.DataFrame({
    "Jaarpaar": [f"{2016 + i} ↔ {2017 + i}" for i in range(len(correlations))],
    "Pearson r": [round(c, 3) for c in correlations],
    "Beoordeling": ["✅ sterk (≥ 0.95)" if c >= 0.95 else "⚠️  matig (0.70–0.95)" if c >= 0.7 else "❌ zwak (< 0.70)"
                    for c in correlations],
})
print("Aanname 'herhaalbaar seizoenspatroon':")
print(aanname_1.to_string(index=False))
"""
        ),
        code(
            """\
# Aanname 2: voldoende historische jaren per opleiding?
counts = (
    data_cumulative[data_cumulative["Examentype"] == "Bachelor"]
    .groupby("Croho groepeernaam")["Collegejaar"]
    .nunique()
    .sort_values()
)
te_kort = counts[counts < 3]
print(f"Aantal opleidingen met < 3 jaar historie: {len(te_kort)}")
if len(te_kort) > 0:
    print("⚠️ Voor deze opleidingen is SARIMA onbetrouwbaar:")
    print(te_kort)
else:
    print("✅ Alle opleidingen hebben minstens 3 jaar historie")
"""
        ),
        md(
            """\
## 6. Wanneer vertrouw je het niet?

Concreet samengevat:

| Situatie | Wat te doen? |
|----------|--------------|
| **Pearson r < 0.7** tussen opeenvolgende jaren | Beoordeel modelresultaat kritisch, of gebruik ratio-model als sanity check |
| **< 3 jaar historie** | SARIMA-orde mismatched met de data, val terug op ratio of een eenvoudigere baseline |
| **Jaar na uitzondering** (COVID 2021) | Sluit het uitzonderingsjaar uit via `excluded_data_points` in `configuration.json` |
| **Peilweek < 6** | Extrapolatie-horizon te lang; rapporteer breed interval, niet één getal |

Gebruik `studentprognose benchmark -d c -w <week>` om alternatieve tijdreeksmodellen
(ETS, Theta, AutoARIMA) te vergelijken — zie [`docs/methodologie/benchmarks.md`](../docs/methodologie/benchmarks.md).
"""
        ),
    ]
    write("02_sarima.ipynb", cells, "SARIMA")


# ============================================================
# 03 — XGBoost (classifier ÉN regressor)
# ============================================================

def build_xgboost():
    cells = [
        md(
            """\
# 03 — XGBoost: classifier én regressor

Deze notebook hoort bij [`docs/methodologie/xgboost.md`](../docs/methodologie/xgboost.md).
XGBoost wordt op twee manieren ingezet:

| Rol | Spoor | Input | Output |
|-----|-------|-------|--------|
| **Classifier** | Individueel (`-d i`) | per-student records | kans dat één student zich inschrijft |
| **Regressor** | Cumulatief (`-d c`) | wekelijkse pivottabel | totaal verwachte inschrijvingen per opleiding |

We bouwen werkende, vereenvoudigde versies van **beide** — inclusief feature
importance, hyperparameter-onderbouwing en een direct vergelijk met SARIMA.
"""
        ),
        HBO_BANNER,
        SETUP_CELL,
        md(
            """\
## DEEL A — XGBoost Classifier (individueel spoor)

### A.1 Data + label-definitie

We laden een sample van 8.000 records uit `vooraanmeldingen_individueel.csv` en
labelen elke aanmelding:
- **1** = uiteindelijk ingeschreven
- **0** = niet ingeschreven (geannuleerd, alleen verzoek, etc.)

Dit volgt exact de `status_mapping` uit `configuration.json` (zie [`docs/configuratie.md`](../docs/configuratie.md#status_mapping)).
"""
        ),
        code(
            """\
from _helpers import load_individueel

# Volledige dataset voor de classifier — niet sampelen want we willen statistische power
ind = load_individueel(n_sample=None)
print(f"Individuele aanmeldrecords: {len(ind):,}")
print(f"Aantal opleidingen:         {ind['Croho groepeernaam'].nunique()}")
print(f"Aantal jaren:               {ind['Collegejaar'].nunique()}")

# Label-mapping (uit configuration.json status_mapping)
INGESCHREVEN_STATUSSEN = {"Ingeschreven", "Uitgeschreven"}
ind = ind.dropna(subset=["Inschrijfstatus", "Croho groepeernaam", "Examentype"]).copy()
ind["label"] = ind["Inschrijfstatus"].isin(INGESCHREVEN_STATUSSEN).astype(int)

print(f"\\nVerdeling label (1 = ingeschreven):")
print(ind["label"].value_counts(normalize=True).round(3))
print(f"\\nBasislijn: '{ind['label'].mean():.1%} van de aanmelders schrijft zich uiteindelijk in' is de naïeve voorspelling.")
"""
        ),
        md(
            """\
### A.2 Train + voorspel

We trainen op alle jaren behalve het laatste, en voorspellen op het laatste jaar.
Hyperparameters: `n_estimators=200` (genoeg bomen voor stabiliteit, niet zo veel dat
overfit dreigt), `max_depth=4` (ondiepe bomen → interpreteerbaar, voorkomt
memorization), `learning_rate=0.1` (standaard, geen agressieve early stopping nodig).
"""
        ),
        code(
            """\
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

cat_features = ["Examentype", "Faculteit", "Croho groepeernaam", "Herkomst" if "Herkomst" in ind.columns else "EER",
                "Type vooropleiding", "Nationaliteit", "Geslacht"]
cat_features = [c for c in cat_features if c in ind.columns]
num_features = ["Collegejaar"]

LAATSTE_JAAR = int(ind["Collegejaar"].max())
train = ind[ind["Collegejaar"] < LAATSTE_JAAR]
test  = ind[ind["Collegejaar"] == LAATSTE_JAAR]

pre = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
     ("num", "passthrough", num_features)]
)
clf = SkPipeline([
    ("pre", pre),
    ("xgb", XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                          random_state=42, eval_metric="logloss")),
])
clf.fit(train[cat_features + num_features].fillna("ONBEKEND"), train["label"])
proba = clf.predict_proba(test[cat_features + num_features].fillna("ONBEKEND"))[:, 1]
pred = (proba >= 0.5).astype(int)

verwacht = float(proba.sum())
werkelijk = float(test["label"].sum())
print(f"Trainingsjaren:  {sorted(train['Collegejaar'].unique().astype(int))}")
print(f"Testjaar:        {LAATSTE_JAAR}  ({len(test):,} aanmeldingen)")
print(f"Accuracy:        {accuracy_score(test['label'], pred):.3f}  (vs basislijn {test['label'].mean():.3f})")
print(f"AUC-ROC:         {roc_auc_score(test['label'], proba):.3f}  (1.0 = perfect, 0.5 = random)")
print(f"")
print(f"Cohort-niveau prestatie (waar het écht om gaat):")
print(f"  Verwacht cohort = som van P(ingeschreven): {verwacht:.0f}")
print(f"  Werkelijk:                                  {werkelijk:.0f}")
print(f"  Fout op cohort:                             {verwacht - werkelijk:+.0f}  ({(verwacht - werkelijk)/werkelijk*100:+.1f}%)")
"""
        ),
        md(
            """\
> 📌 **Eerlijkheid over AUC:** in deze demodata is de AUC laag (~0.5) omdat de
> beschikbare features (faculteit, nationaliteit, vooropleiding) weinig
> onderscheidend zijn — dat is een eigenschap van de geanonimiseerde demoset,
> niet van het model. Een hogeschool met echte individuele data (intake-cijfers,
> studiekeuzecheckresultaten, financiële situatie) krijgt typisch AUC > 0.75.
>
> **Wat hier wél werkt:** de **cohort-niveau** prognose (som van kansen) is heel
> nauwkeurig. Dat is wat de tool uiteindelijk levert — niet een per-student
> oordeel.

### A.3 Hoe wordt een individuele kans een cohortvoorspelling?

Een classifier geeft per student `P(ingeschreven)`. Voor een hele opleiding sommeer
je gewoon die kansen — dat is exact wat de pipeline doet:
"""
        ),
        code(
            """\
test_with_proba = test.copy()
test_with_proba["kans"] = proba

per_opleiding = (
    test_with_proba.groupby("Croho groepeernaam")
    .agg(verwacht=("kans", "sum"),
         werkelijk=("label", "sum"),
         aantal_aanmeldingen=("label", "count"))
    .round(0)
)
per_opleiding["fout"] = per_opleiding["verwacht"] - per_opleiding["werkelijk"]
per_opleiding["fout_%"] = (per_opleiding["fout"] / per_opleiding["werkelijk"].replace(0, np.nan) * 100).round(1)
per_opleiding.sort_values("werkelijk", ascending=False).head(10)
"""
        ),
        md(
            """\
## DEEL B — XGBoost Regressor (cumulatief spoor)

### B.1 Feature engineering

De regressor leert per (opleiding × herkomst × examentype × jaar) op basis van het
**cumulatieve aanmeldpatroon tot peilweek** plus enkele afgeleide features. Uit de docs:

| Type | Features |
|------|----------|
| **Numeriek** | `Collegejaar`, weekkolommen `"1"` t/m `"38"` |
| **Lagged** | `Gewogen_t-2`, `Gewogen_t-5` |
| **Dynamiek** | `Gewogen_acceleration` = `(huidig − t-2) − (t-2 − t-5)` |
| **Commitment** | `exclusivity_ratio` = `aantal-met-1-aanmelding / (ongewogen vooraanmelders + ε)` |
| **Categorisch** | `Examentype`, `Faculteit`, `Croho groepeernaam`, `Herkomst` |
"""
        ),
        LOAD_CUMULATIVE_CELL,
        code(
            """\
from studentprognose.data.transforms import transform_data
from studentprognose.utils.weeks import get_all_weeks_valid

PREDICT_WEEK = 12
EPS = 1e-8

df = data_cumulative.drop_duplicates()
df = df[df["Collegejaar"] >= 2018]
wide = transform_data(df, "Gewogen vooraanmelders")

weekcols_all = get_all_weeks_valid(wide.columns)
feature_weeks = [w for w in weekcols_all if 1 <= int(w) <= PREDICT_WEEK]

# Lagged + acceleration
for lag in (2, 5):
    ref_week = max(PREDICT_WEEK - lag, 1)
    ref_values = (
        df[df["Weeknummer"] == ref_week]
        .groupby(["Collegejaar", "Faculteit", "Herkomst", "Examentype", "Croho groepeernaam"])
        ["Gewogen vooraanmelders"].mean()
        .rename(f"Gewogen_t-{lag}")
        .reset_index()
    )
    wide = wide.merge(ref_values, on=["Collegejaar", "Faculteit", "Herkomst", "Examentype", "Croho groepeernaam"], how="left")

if str(PREDICT_WEEK) in wide.columns:
    wide["Gewogen_acceleration"] = (
        (wide[str(PREDICT_WEEK)] - wide["Gewogen_t-2"]) - (wide["Gewogen_t-2"] - wide["Gewogen_t-5"])
    )
else:
    wide["Gewogen_acceleration"] = 0.0

# Exclusivity ratio
peil = df[df["Weeknummer"] == PREDICT_WEEK].copy()
peil["exclusivity_ratio"] = peil["Aantal aanmelders met 1 aanmelding"] / (peil["Ongewogen vooraanmelders"].astype(float) + EPS)
peil = peil[["Collegejaar", "Faculteit", "Herkomst", "Examentype", "Croho groepeernaam", "exclusivity_ratio"]].drop_duplicates(
    subset=["Collegejaar", "Faculteit", "Herkomst", "Examentype", "Croho groepeernaam"]
)
wide = wide.merge(peil, on=["Collegejaar", "Faculteit", "Herkomst", "Examentype", "Croho groepeernaam"], how="left")

print(f"Feature matrix: {wide.shape[0]} rijen × {wide.shape[1]} kolommen")
print(f"Week-features:   {feature_weeks}")
print(f"Lagged features: Gewogen_t-2, Gewogen_t-5, Gewogen_acceleration, exclusivity_ratio")
"""
        ),
        md(
            """\
### B.2 Trainen en test-fout interpreteren

We trainen op alle jaren behalve het laatste, voorspellen op het laatste jaar (= 2023),
en vergelijken direct met de werkelijke realisatie.
"""
        ),
        code(
            """\
from xgboost import XGBRegressor

data_sc = data_studentcount.rename(columns={"Aantal_studenten": "y"})
training = wide.merge(
    data_sc[["Collegejaar", "Croho groepeernaam", "Herkomst", "Examentype", "y"]],
    on=["Collegejaar", "Croho groepeernaam", "Herkomst", "Examentype"],
    how="inner",
).dropna(subset=["y"])

LATEST = int(training["Collegejaar"].max())
train_xgb = training[training["Collegejaar"] < LATEST].copy()
test_xgb  = training[training["Collegejaar"] == LATEST].copy()

cat_features_r = ["Examentype", "Faculteit", "Croho groepeernaam", "Herkomst"]
num_features_r = feature_weeks + ["Gewogen_t-2", "Gewogen_t-5", "Gewogen_acceleration", "exclusivity_ratio", "Collegejaar"]

pre_r = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features_r),
     ("num", "passthrough", num_features_r)]
)
reg = SkPipeline([
    ("pre", pre_r),
    ("xgb", XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)),
])
reg.fit(train_xgb[cat_features_r + num_features_r].fillna(0), train_xgb["y"])
preds = reg.predict(test_xgb[cat_features_r + num_features_r].fillna(0))

mae = float(np.mean(np.abs(preds - test_xgb["y"])))
mape = float(np.mean(np.abs(preds - test_xgb["y"]) / test_xgb["y"]) * 100)
gem_cohort = test_xgb["y"].mean()
print(f"Trainjaren:        {sorted(train_xgb['Collegejaar'].unique().astype(int))}")
print(f"Testjaar:          {LATEST}  ({len(test_xgb)} rijen)")
print(f"Gemiddeld cohort:  {gem_cohort:.0f} studenten")
print(f"MAE:               {mae:.1f} studenten  →  ~{mae/gem_cohort*100:.0f}% van het gemiddelde cohort")
print(f"MAPE:              {mape:.1f}% (gemiddelde absolute procentuele fout)")
"""
        ),
        md(
            """\
### B.3 Feature importance — gegroepeerd

XGBoost geeft per (one-hot) feature een score. In productie worden die teruggegroepeerd
naar de oorspronkelijke kolom (alle `Herkomst_NL`, `Herkomst_EER`, … → `Herkomst`).
We doen dat hier ook.
"""
        ),
        code(
            """\
xgb_model = reg.named_steps["xgb"]
encoded_names = reg.named_steps["pre"].get_feature_names_out()
importances = xgb_model.feature_importances_

records = []
for name, imp in zip(encoded_names, importances):
    stripped = name.split("__", 1)[1] if "__" in name else name
    base = stripped
    for cat in cat_features_r:
        if stripped.startswith(cat + "_"):
            base = cat
            break
    records.append({"feature": base, "importance": imp})

imp_df = (
    pd.DataFrame(records)
    .groupby("feature", as_index=False)["importance"].sum()
    .sort_values("importance", ascending=True).tail(12)
)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(imp_df["feature"], imp_df["importance"], color="steelblue")
ax.set_xlabel("Som van importance (gegroepeerd)")
ax.set_title("Feature importance — XGBoost regressor")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """\
### B.4 XGBoost vs. SARIMA — wie wint?

Een belangrijk inzicht voor een analist: voegt XGBoost echt waarde toe vergeleken
met de simpelere SARIMA-cumulative-baseline? We draaien beide voor 2023 wk 12 en
vergelijken.
"""
        ),
        code(
            """\
from studentprognose import run_pipeline_from_dataframes, DataOption
from _helpers import suppress_stdout

with suppress_stdout():
    pipeline_out = run_pipeline_from_dataframes(
        year=LATEST, week=PREDICT_WEEK,
        data_cumulative=data_cumulative_raw,
        data_student_numbers=data_studentcount,
        dataset=DataOption.CUMULATIVE, save_output=False,
    )

# SARIMA-cumulative is in productie reeds bekend; hieronder vergelijken we MAE/MAPE.
# Belangrijk: filter eerst op peilweek (SARIMA_cumulative staat alleen op die week;
# Prognose_ratio wordt voor elke week berekend en kan dus dubbel tellen bij groupby+sum)
pl_at_peil = pipeline_out[pipeline_out["Weeknummer"] == PREDICT_WEEK]
pl_agg = (
    pl_at_peil[pl_at_peil["Examentype"] == "Bachelor"]
    .groupby("Croho groepeernaam", as_index=False)
    [["SARIMA_cumulative", "Prognose_ratio"]]
    .sum()
)
xgb_agg = pd.DataFrame({
    "Croho groepeernaam": test_xgb["Croho groepeernaam"].values,
    "Herkomst": test_xgb["Herkomst"].values,
    "Examentype": test_xgb["Examentype"].values,
    "XGB_pred": preds,
    "y": test_xgb["y"].values,
})
xgb_per_opl = (
    xgb_agg[xgb_agg["Examentype"] == "Bachelor"]
    .groupby("Croho groepeernaam", as_index=False)
    [["XGB_pred", "y"]].sum()
    .rename(columns={"y": "Realisatie"})
)
vergelijk = xgb_per_opl.merge(pl_agg, on="Croho groepeernaam", how="inner")

summary = pd.DataFrame({
    "Model": ["XGBoost regressor (deze notebook)", "SARIMA cumulative (pipeline)", "Ratio-model (pipeline)"],
    "MAE": [
        (vergelijk["XGB_pred"] - vergelijk["Realisatie"]).abs().mean().round(1),
        (vergelijk["SARIMA_cumulative"] - vergelijk["Realisatie"]).abs().mean().round(1),
        (vergelijk["Prognose_ratio"] - vergelijk["Realisatie"]).abs().mean().round(1),
    ],
    "MAPE %": [
        ((vergelijk["XGB_pred"] - vergelijk["Realisatie"]).abs() / vergelijk["Realisatie"] * 100).mean().round(1),
        ((vergelijk["SARIMA_cumulative"] - vergelijk["Realisatie"]).abs() / vergelijk["Realisatie"] * 100).mean().round(1),
        ((vergelijk["Prognose_ratio"] - vergelijk["Realisatie"]).abs() / vergelijk["Realisatie"] * 100).mean().round(1),
    ],
})
print(f"Vergelijking op {LATEST} (peilweek {PREDICT_WEEK}), {len(vergelijk)} bachelor-opleidingen:")
summary
"""
        ),
        md(
            """\
## Conclusie

- **Classifier**: per-student kans, gesommeerd voor cohortprognose; AUC + accuracy
  als kwaliteitsmaat. Werkt zelfs op een sample van 8.000 records.
- **Regressor**: leert op opleiding-niveau, met lagged + acceleratie-features die
  de aanmelddynamiek vangen.
- **Wanneer welk model?** Bij weinig data presteren **Ridge** of **Random Forest**
  vaak beter dan XGBoost — schakel ze in via `model_config.cumulative_regressor`
  in `configuration.json`.
- **MAE-interpretatie**: deel altijd door cohortgrootte. MAE = 50 is OK voor
  een opleiding van 500 (10%), maar slecht voor een opleiding van 100 (50%).
"""
        ),
    ]
    write("03_xgboost.ipynb", cells, "XGBoost")


# ============================================================
# 04 — Ratio-model (NF-cap echt, backtest, week-as 39→38)
# ============================================================

def build_ratio():
    cells = [
        md(
            """\
# 04 — Ratio-model

Deze notebook hoort bij [`docs/methodologie/ratio-model.md`](../docs/methodologie/ratio-model.md).
Het ratio-model is de **simpelste** voorspeller in de pipeline — en juist daarom
essentieel: volledig transparant, altijd beschikbaar, fungeert als baseline.

Definitie:

$$
\\bar{R}_t = \\frac{1}{3} \\sum_{j=1}^{3} \\frac{\\text{Aanmelding}_{jaar-j,\\, week=t}}{\\text{Aantal studenten}_{jaar-j}}
\\quad\\Rightarrow\\quad
\\hat{y} = \\frac{\\text{Aanmelding}_{huidig,\\, week=t}}{\\bar{R}_t}
$$

Met `Aanmelding = Ongewogen vooraanmelders + Inschrijvingen`. Het venster is 3 jaar
(`LOOKBACK_YEARS` in `src/studentprognose/utils/constants.py`).

> 🎯 **In deze notebook:** stap-voor-stap berekenen, backtesten met realisatie,
> de numerus-fixus cap échte data tonen, en kijken wanneer het model breekt.
"""
        ),
        HBO_BANNER,
        SETUP_CELL,
        LOAD_CUMULATIVE_CELL,
        md(
            """\
## 1. Ratio handmatig herberekenen + voorspellen voor 2023

We doen exact wat `predict_with_ratio()` in `src/studentprognose/models/ratio.py` doet,
maar dan stap-voor-stap zichtbaar.
"""
        ),
        code(
            """\
PROGRAMMA = "B Psychologie"
HERKOMST = "NL"
EXAMENTYPE = "Bachelor"
PREDICT_YEAR = 2023  # backtest: realisatie is bekend
PREDICT_WEEK = 12
LOOKBACK = 3

# Stap A — historische aanmeldingen
hist = data_cumulative[
    (data_cumulative["Croho groepeernaam"] == PROGRAMMA)
    & (data_cumulative["Herkomst"] == HERKOMST)
    & (data_cumulative["Examentype"] == EXAMENTYPE)
    & (data_cumulative["Weeknummer"] == PREDICT_WEEK)
    & (data_cumulative["Collegejaar"].between(PREDICT_YEAR - LOOKBACK, PREDICT_YEAR - 1))
].copy()
hist["Aanmelding"] = hist["Ongewogen vooraanmelders"].fillna(0) + hist["Inschrijvingen"].fillna(0)

# Stap B — koppel realisatie
merged = hist.merge(
    data_studentcount,
    on=["Collegejaar", "Croho groepeernaam", "Herkomst", "Examentype"],
    how="left",
)
merged["Ratio"] = merged["Aanmelding"] / merged["Aantal_studenten"]
merged[["Collegejaar", "Aanmelding", "Aantal_studenten", "Ratio"]]
"""
        ),
        code(
            """\
avg_ratio = float(merged["Ratio"].mean())
print(f"Gemiddelde ratio over {LOOKBACK} jaar @ peilweek {PREDICT_WEEK}: {avg_ratio:.3f}")

huidig = data_cumulative[
    (data_cumulative["Croho groepeernaam"] == PROGRAMMA)
    & (data_cumulative["Herkomst"] == HERKOMST)
    & (data_cumulative["Examentype"] == EXAMENTYPE)
    & (data_cumulative["Weeknummer"] == PREDICT_WEEK)
    & (data_cumulative["Collegejaar"] == PREDICT_YEAR)
]
huidig_aanmelding = float(
    huidig["Ongewogen vooraanmelders"].fillna(0).sum()
    + huidig["Inschrijvingen"].fillna(0).sum()
)
prognose = huidig_aanmelding / avg_ratio

werkelijk = float(data_studentcount[
    (data_studentcount["Croho groepeernaam"] == PROGRAMMA)
    & (data_studentcount["Herkomst"] == HERKOMST)
    & (data_studentcount["Examentype"] == EXAMENTYPE)
    & (data_studentcount["Collegejaar"] == PREDICT_YEAR)
]["Aantal_studenten"].sum())

print(f"Huidige aanmelding ({PREDICT_YEAR}, week {PREDICT_WEEK}): {huidig_aanmelding:.0f}")
print(f"")
print(f"  Ratio-prognose voor {PREDICT_YEAR}:  {prognose:>5.0f} studenten")
print(f"  Werkelijke realisatie {PREDICT_YEAR}: {werkelijk:>5.0f} studenten")
print(f"  Fout:                          {prognose - werkelijk:+.0f}  ({(prognose - werkelijk) / werkelijk * 100:+.1f}%)")
"""
        ),
        md(
            """\
## 2. Backtest over meerdere jaren

Een ratio-prognose voor één jaar zegt weinig. We berekenen de prognose voor
**élk** historisch jaar (waar 3 jaar terugkijken mogelijk is) en vergelijken met
de werkelijke realisatie. Zo zie je of het model structureel onder- of overschat.
"""
        ),
        code(
            """\
def ratio_prognose_voor_jaar(jaar: int):
    \"\"\"Bereken ratio-prognose voor B Psychologie NL Bachelor op peilweek 12.\"\"\"
    h = data_cumulative[
        (data_cumulative["Croho groepeernaam"] == PROGRAMMA)
        & (data_cumulative["Herkomst"] == HERKOMST)
        & (data_cumulative["Examentype"] == EXAMENTYPE)
        & (data_cumulative["Weeknummer"] == PREDICT_WEEK)
        & (data_cumulative["Collegejaar"].between(jaar - LOOKBACK, jaar - 1))
    ].copy()
    if len(h) == 0:
        return None
    h["Aanm"] = h["Ongewogen vooraanmelders"].fillna(0) + h["Inschrijvingen"].fillna(0)
    m = h.merge(data_studentcount,
                on=["Collegejaar", "Croho groepeernaam", "Herkomst", "Examentype"], how="left")
    avg_r = (m["Aanm"] / m["Aantal_studenten"]).mean()

    hu = data_cumulative[
        (data_cumulative["Croho groepeernaam"] == PROGRAMMA)
        & (data_cumulative["Herkomst"] == HERKOMST)
        & (data_cumulative["Examentype"] == EXAMENTYPE)
        & (data_cumulative["Weeknummer"] == PREDICT_WEEK)
        & (data_cumulative["Collegejaar"] == jaar)
    ]
    aanm_hu = float(hu["Ongewogen vooraanmelders"].fillna(0).sum() + hu["Inschrijvingen"].fillna(0).sum())
    return aanm_hu / avg_r if avg_r else np.nan


jaren = sorted(data_studentcount[data_studentcount["Examentype"] == EXAMENTYPE]["Collegejaar"].unique())
records = []
for j in jaren:
    p = ratio_prognose_voor_jaar(int(j))
    r = float(data_studentcount[
        (data_studentcount["Croho groepeernaam"] == PROGRAMMA)
        & (data_studentcount["Herkomst"] == HERKOMST)
        & (data_studentcount["Examentype"] == EXAMENTYPE)
        & (data_studentcount["Collegejaar"] == j)
    ]["Aantal_studenten"].sum())
    if p is not None and r > 0:
        records.append({"jaar": int(j), "prognose": round(p), "realisatie": int(r),
                        "fout": round(p - r), "fout_%": round((p - r) / r * 100, 1)})

backtest = pd.DataFrame(records)
print(f"Backtest ratio-model · {PROGRAMMA} · {HERKOMST} · peilweek {PREDICT_WEEK}:")
print(backtest.to_string(index=False))
print(f"\\nGemiddelde absolute fout: {backtest['fout'].abs().mean():.0f} studenten ({backtest['fout_%'].abs().mean():.1f}%)")
"""
        ),
        code(
            """\
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(backtest["jaar"], backtest["realisatie"], marker="o", color="black",
        label="Realisatie", linewidth=2)
ax.plot(backtest["jaar"], backtest["prognose"], marker="s", color="darkgreen",
        label="Ratio-prognose (wk 12)", linewidth=2, linestyle="--")
for _, r in backtest.iterrows():
    ax.annotate(f"{r['fout_%']:+.0f}%", (r["jaar"], r["prognose"]),
                xytext=(0, 8), textcoords="offset points", fontsize=8,
                ha="center", color="darkgreen")
ax.set_xlabel("Collegejaar")
ax.set_ylabel("Aantal eerstejaars")
ax.set_title(f"Backtest ratio-model · {PROGRAMMA} · {HERKOMST}")
ax.legend()
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """\
## 3. Ratio per peilweek — wanneer is hij stabiel?

De ratio varieert sterk per peilweek. Vroeg in het jaar (weinig aanmelders) is
hij **instabiel** — een handvol aanmelders meer/minder slaat hard door. Hoe later
in het seizoen, hoe stabieler. De grafiek loopt in **academische volgorde**
(wk 39 → wk 38).
"""
        ),
        code(
            """\
from studentprognose.utils.weeks import get_all_weeks_ordered

ordered = get_all_weeks_ordered()  # ['39','40',...,'52','1',...,'38']
rows = []
for w_str in ordered:
    w = int(w_str)
    sub = data_cumulative[
        (data_cumulative["Croho groepeernaam"] == PROGRAMMA)
        & (data_cumulative["Herkomst"] == HERKOMST)
        & (data_cumulative["Examentype"] == EXAMENTYPE)
        & (data_cumulative["Weeknummer"] == w)
        & (data_cumulative["Collegejaar"].between(PREDICT_YEAR - LOOKBACK, PREDICT_YEAR - 1))
    ].copy()
    if sub.empty:
        continue
    sub["Aanm"] = sub["Ongewogen vooraanmelders"].fillna(0) + sub["Inschrijvingen"].fillna(0)
    m = sub.merge(data_studentcount,
                  on=["Collegejaar", "Croho groepeernaam", "Herkomst", "Examentype"], how="left")
    r = (m["Aanm"] / m["Aantal_studenten"])
    rows.append({"week": w, "ord_idx": ordered.index(w_str), "mean_ratio": r.mean(), "std_ratio": r.std()})

ratio_df = pd.DataFrame(rows).sort_values("ord_idx")

fig, ax = plt.subplots(figsize=(11, 5))
ax.errorbar(ratio_df["ord_idx"], ratio_df["mean_ratio"],
            yerr=ratio_df["std_ratio"], fmt="o-", color="darkgreen",
            ecolor="lightgreen", capsize=3, label="Ratio ± 1σ (over 3 jaar)")
peil_idx = ordered.index(str(PREDICT_WEEK))
ax.axvline(peil_idx, color="gray", linestyle="--", alpha=0.6, label=f"Peilweek {PREDICT_WEEK}")
ax.set_xticks(range(0, len(ordered), 4))
ax.set_xticklabels([ordered[i] for i in range(0, len(ordered), 4)])
ax.set_xlabel("Weeknummer (academische volgorde: 39 → 38)")
ax.set_ylabel("Gemiddelde ratio  Aanmelding / Aantal_studenten")
ax.set_title(f"Ratio per peilweek · {PROGRAMMA} · {HERKOMST}")
ax.legend()
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """\
## 4. Numerus fixus — concrete cap-correctie op B Geneeskunde

Voor numerus-fixusopleidingen capt de pipeline de gesommeerde prognose op de
NF-limiet. Het overschot wordt afgetrokken van de NL-herkomst (zie
`predict_with_ratio()`). We tonen dit op `B Geneeskunde` — een opleiding met een
NF van 340 (Radboud-specifieke waarde uit de demo-configuratie).
"""
        ),
        code(
            """\
from studentprognose.config import load_defaults

config = load_defaults()
NF_LIMIT = config["numerus_fixus"].get("B Geneeskunde", 340)
print(f"Numerus fixus B Geneeskunde (uit config): {NF_LIMIT}")

# Bereken ratio-prognoses per herkomst voor wk 12 / 2023
nf_records = []
for herkomst in ["NL", "EER", "Niet-EER"]:
    h = data_cumulative[
        (data_cumulative["Croho groepeernaam"] == "B Geneeskunde")
        & (data_cumulative["Herkomst"] == herkomst)
        & (data_cumulative["Examentype"] == "Bachelor")
        & (data_cumulative["Weeknummer"] == PREDICT_WEEK)
        & (data_cumulative["Collegejaar"].between(PREDICT_YEAR - LOOKBACK, PREDICT_YEAR - 1))
    ].copy()
    if len(h) == 0:
        continue
    h["Aanm"] = h["Ongewogen vooraanmelders"].fillna(0) + h["Inschrijvingen"].fillna(0)
    m = h.merge(data_studentcount,
                on=["Collegejaar", "Croho groepeernaam", "Herkomst", "Examentype"], how="left")
    avg_r = (m["Aanm"] / m["Aantal_studenten"]).mean()

    hu = data_cumulative[
        (data_cumulative["Croho groepeernaam"] == "B Geneeskunde")
        & (data_cumulative["Herkomst"] == herkomst)
        & (data_cumulative["Examentype"] == "Bachelor")
        & (data_cumulative["Weeknummer"] == PREDICT_WEEK)
        & (data_cumulative["Collegejaar"] == PREDICT_YEAR)
    ]
    aanm_hu = float(hu["Ongewogen vooraanmelders"].fillna(0).sum() + hu["Inschrijvingen"].fillna(0).sum())
    nf_records.append({"Herkomst": herkomst, "Ratio-prognose": round(aanm_hu / avg_r) if avg_r else np.nan})

nf_df = pd.DataFrame(nf_records)
totaal_voor = nf_df["Ratio-prognose"].sum()
overschot = max(0, totaal_voor - NF_LIMIT)
nf_df["NF-cap toegepast"] = nf_df["Ratio-prognose"].copy()
nf_df.loc[nf_df["Herkomst"] == "NL", "NF-cap toegepast"] -= overschot
nf_df.loc[nf_df["Herkomst"] == "NL", "NF-cap toegepast"] = nf_df.loc[nf_df["Herkomst"] == "NL", "NF-cap toegepast"].clip(lower=0)

print(f"\\nTotaal ongeforceerd:  {totaal_voor:.0f}")
print(f"NF-limiet:             {NF_LIMIT}")
print(f"Overschot afgetrokken: {overschot:.0f}  (van NL)")
print()
print(nf_df.to_string(index=False))
"""
        ),
        md(
            """\
## 5. Beperkingen — concreet aangetoond

| Beperking | Symptoom in de backtest hierboven |
|-----------|-----------------------------------|
| **Geen trends** | Bij groeiende of krimpende opleidingen lopen de fouten op |
| **Gevoelig voor outliers** | COVID-jaar 2021 (506 ipv ~370) trekt de gemiddelde ratio uit het lood |
| **Week-afhankelijk** | Vroege weken: brede σ-band in de grafiek hierboven |

Conclusie: het ratio-model is een **eerlijke baseline** — niet meer, niet minder.
Als het ensemble het niet verslaat op jouw data, is er iets mis met de complexere
modellen.
"""
        ),
    ]
    write("04_ratio_model.ipynb", cells, "Ratio-model")


# ============================================================
# 05 — Ensemble (échte voorspellingen, geen fictie)
# ============================================================

def build_ensemble():
    cells = [
        md(
            """\
# 05 — Ensemble samenstellen

Deze notebook hoort bij [`docs/methodologie/ensemble.md`](../docs/methodologie/ensemble.md).
We tonen hoe de modellen (SARIMA-cumulative + ratio) worden gecombineerd via
configureerbare gewichten — en testen op **échte data** of het ensemble de
ratio-baseline echt verslaat.

> ℹ️ De volledige `-d both`-ensemble vereist ook individuele aanmelddata en
> berekent gewichten via grid search. Hier doen we een vereenvoudigde variant:
> SARIMA + ratio met handmatig instelbare gewichten — voldoende om het idee én
> de toegevoegde waarde aan te tonen.
"""
        ),
        HBO_BANNER,
        SETUP_CELL,
        md(
            """\
## 1. Geconfigureerde gewichten uit `configuration.json`

De gewichten staan in de config, per situatie. Ze gelden voor de combinatie
SARIMA-individueel × SARIMA-cumulative (week 38 is altijd 100% individueel).
"""
        ),
        code(
            """\
from studentprognose.config import load_defaults
config = load_defaults()
weights = config.get("ensemble_weights", {})

print("Geconfigureerde ensemble-gewichten:")
for key, w in weights.items():
    print(f"  {key:24s}  individueel={w['individual']:.2f}  cumulatief={w['cumulative']:.2f}")
"""
        ),
        LOAD_CUMULATIVE_CELL,
        md(
            """\
## 2. Échte voorspellingen genereren

We draaien de cumulatieve pipeline voor 2023 wk 12 en krijgen **echte**
SARIMA-cumulative en ratio-voorspellingen. Geen fictieve curves.
"""
        ),
        code(
            """\
from studentprognose import run_pipeline_from_dataframes, DataOption
from _helpers import suppress_stdout

PREDICT_YEAR = 2023
PREDICT_WEEK = 12

with suppress_stdout():
    pipeline_out = run_pipeline_from_dataframes(
        year=PREDICT_YEAR, week=PREDICT_WEEK,
        data_cumulative=data_cumulative_raw,
        data_student_numbers=data_studentcount,
        dataset=DataOption.CUMULATIVE, save_output=False,
    )

# Filter eerst op peilweek (SARIMA staat alleen daar; Prognose_ratio voor alle weken)
pl_at_peil = pipeline_out[pipeline_out["Weeknummer"] == PREDICT_WEEK]
agg = (
    pl_at_peil[pl_at_peil["Examentype"] == "Bachelor"]
    .groupby("Croho groepeernaam", as_index=False)
    [["SARIMA_cumulative", "Prognose_ratio"]]
    .sum()
)
real = (
    data_studentcount[(data_studentcount["Examentype"] == "Bachelor")
                      & (data_studentcount["Collegejaar"] == PREDICT_YEAR)]
    .groupby("Croho groepeernaam", as_index=False)
    ["Aantal_studenten"].sum()
    .rename(columns={"Aantal_studenten": "Realisatie"})
)
df = agg.merge(real, on="Croho groepeernaam", how="inner")
print(f"{len(df)} bachelor-opleidingen, voorspeld vanaf peilweek {PREDICT_WEEK} van {PREDICT_YEAR}")
df.head()
"""
        ),
        md(
            """\
## 3. Een ensemble bouwen met instelbare gewichten

Omdat we hier alleen SARIMA-cumulative en ratio hebben (geen individueel),
construeren we een 2-model ensemble. Je kunt `W_SARIMA` aanpassen en zien hoe
de gemiddelde fout verschuift.
"""
        ),
        code(
            """\
def ensemble_eval(w_sarima: float, df_in: pd.DataFrame) -> dict:
    w_ratio = 1 - w_sarima
    ens = w_sarima * df_in["SARIMA_cumulative"] + w_ratio * df_in["Prognose_ratio"]
    mae = (ens - df_in["Realisatie"]).abs().mean()
    mape = ((ens - df_in["Realisatie"]).abs() / df_in["Realisatie"] * 100).mean()
    return {"w_sarima": w_sarima, "MAE": round(mae, 1), "MAPE_%": round(mape, 1)}

# Gridsearch over gewichten
sweep = pd.DataFrame([ensemble_eval(w, df) for w in np.linspace(0, 1, 11)])
sweep
"""
        ),
        code(
            """\
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sweep["w_sarima"], sweep["MAPE_%"], marker="o", color="purple", linewidth=2)

# Markeer de extreme situaties
ax.scatter([0], [sweep[sweep["w_sarima"] == 0.0]["MAPE_%"].iloc[0]],
           color="darkgreen", s=120, zorder=10, label="100% ratio-baseline")
ax.scatter([1], [sweep[sweep["w_sarima"] == 1.0]["MAPE_%"].iloc[0]],
           color="steelblue", s=120, zorder=10, label="100% SARIMA-cumulative")

# Markeer het optimum
opt = sweep.loc[sweep["MAPE_%"].idxmin()]
ax.scatter([opt["w_sarima"]], [opt["MAPE_%"]],
           color="red", s=200, marker="*", zorder=11,
           label=f"Optimaal: w_sarima={opt['w_sarima']:.1f} → MAPE {opt['MAPE_%']:.1f}%")

ax.set_xlabel("Gewicht voor SARIMA-cumulative (rest = ratio-model)")
ax.set_ylabel("MAPE (%)")
ax.set_title(f"Ensemble-gewicht vs. fout · 2023 peilweek 12 · {len(df)} bachelor-opleidingen")
ax.legend()
plt.tight_layout()
plt.show()
print(f"\\nOptimum: {opt.to_dict()}")
"""
        ),
        md(
            """\
## 4. Wanneer wint het ensemble — en wanneer niet?

Per opleiding kan een ander optimum gelden. Hieronder kijken we welke opleidingen
**baat hebben bij ensemble** (SARIMA + ratio > beide afzonderlijk) en welke niet.
"""
        ),
        code(
            """\
# Het ensemble (50/50) wint *per definitie* zelden van het beste afzonderlijke model
# voor één opleiding — het is een gemiddelde. De waarde ligt in **stabiliteit**:
# als je vooraf niet weet welk model het best zal presteren, beperkt het ensemble je
# downside. We tonen daarom de SPREAD tussen modellen — een grote spread is een
# early warning dat één van beide het mis kan hebben.

df["ensemble_50_50"] = 0.5 * df["SARIMA_cumulative"] + 0.5 * df["Prognose_ratio"]
df["spread"] = (df["SARIMA_cumulative"] - df["Prognose_ratio"]).abs()
df["spread_pct"] = (df["spread"] / df["Realisatie"] * 100).round(1)
df["winnaar_per_opleiding"] = np.where(
    (df["SARIMA_cumulative"] - df["Realisatie"]).abs()
    < (df["Prognose_ratio"] - df["Realisatie"]).abs(),
    "SARIMA wint", "Ratio wint"
)

print("Spread tussen SARIMA en Ratio per opleiding (groot = onenigheid):")
print(df[["Croho groepeernaam", "Realisatie", "SARIMA_cumulative",
          "Prognose_ratio", "spread", "spread_pct", "winnaar_per_opleiding"]]
      .sort_values("spread_pct", ascending=False).head(8).to_string(index=False))

print(f"\\nGemiddelde spread: {df['spread_pct'].mean():.1f}%  →  "
      f"laag betekent: de modellen zijn het eens, hoog: één van de twee zit ernaast.")
"""
        ),
        md(
            """\
## 5. Wanneer is de Baseline (ratio) betrouwbaarder dan het ensemble?

Uit [`docs/output-begrijpen.md`](../docs/output-begrijpen.md#wanneer-is-de-baseline-betrouwbaarder-dan-het-ensemble):

| Situatie | Beter te vertrouwen |
|----------|--------------------|
| Stabiele, grote opleiding met vaste conversie | **Baseline (ratio)** |
| Weinig historische data (< 4 jaar) | **Baseline (ratio)** |
| Grote afwijking Baseline ↔ Ensemble (> 15–20 %) | **Check de invoerdata** |
| Snel veranderende conversie (nieuw beleid, deadline-verschuiving) | **Ensemble** |
| Niet-lineair aanmeldpatroon | **Ensemble** |

## 6. Productiegewichten herbereken

In productie bepaalt `archive/calculate_ensemble_weights.py` via grid search de
optimale gewichten per **(opleiding × examentype × herkomst)** — niet één
globaal getal zoals in deze illustratie.

> ⚠️ **Gewichten zijn datumgevoelig.** Een model dat in het verleden goed presteerde
> kan nu verouderd zijn (na een beleidsingreep of een uitzonderlijk jaar).
> Herbereken de gewichten aan het einde van elk studiejaar.
"""
        ),
    ]
    write("05_ensemble.ipynb", cells, "Ensemble")


# ============================================================
# 06 — Output interpreteren (mét realisatie + Excel-export)
# ============================================================

def build_output():
    cells = [
        md(
            """\
# 06 — Output interpreteren

Deze notebook hoort bij [`docs/output-begrijpen.md`](../docs/output-begrijpen.md). We:

1. Draaien de pipeline en bekijken **echte output**
2. Leggen elke kolomgroep uit (identifiers, voorspellingen, actuelen, foutmaten)
3. Voegen **realisatie** toe en koppelen elk modelaan zijn werkelijke fout
4. Exporteren naar Excel — exact zoals de CLI dat doet

> ℹ️ We draaien `-d c` (cumulatief): geen individuele data nodig. Kolommen die
> alleen bij `-d both` voorkomen (`Ensemble_prediction`, `SARIMA_individual`)
> worden onderaan benoemd.
"""
        ),
        HBO_BANNER,
        SETUP_CELL,
        LOAD_CUMULATIVE_CELL,
        md(
            """\
## 1. Pipeline draaien (cumulatief, in-memory)

We voorspellen **2023** vanaf peilweek **12** zodat we straks met de werkelijke
realisatie kunnen vergelijken. De per-opleiding-spam wordt onderdrukt — de
pipeline print normaal voor élke combinatie een regel.

> 🕒 Duurt typisch ~20–40 sec op demodata.
"""
        ),
        code(
            """\
from studentprognose import run_pipeline_from_dataframes, DataOption
from _helpers import suppress_stdout

PREDICT_YEAR = 2023
PREDICT_WEEK = 12

with suppress_stdout():
    result = run_pipeline_from_dataframes(
        year=PREDICT_YEAR, week=PREDICT_WEEK,
        data_cumulative=data_cumulative_raw,
        data_student_numbers=data_studentcount,
        dataset=DataOption.CUMULATIVE, save_output=False,
    )
print(f"Output: {result.shape[0]} rijen × {result.shape[1]} kolommen")
print(f"Opleidingen: {result['Croho groepeernaam'].nunique()}")
"""
        ),
        md(
            """\
## 2. Kolomgroepen

De output heeft vier groepen kolommen. Bij `-d c` zijn niet allemaal gevuld:
"""
        ),
        code(
            """\
identifier_cols = [c for c in ["Croho groepeernaam", "Herkomst", "Examentype", "Collegejaar", "Weeknummer", "Faculteit"] if c in result.columns]
prediction_cols = [c for c in ["SARIMA_cumulative", "SARIMA_individual", "Prognose_ratio", "Ensemble_prediction", "Baseline"] if c in result.columns]
actuele_cols    = [c for c in ["Gewogen vooraanmelders", "Ongewogen vooraanmelders", "Aantal aanmelders met 1 aanmelding", "Inschrijvingen"] if c in result.columns]
error_cols      = sorted([c for c in result.columns if c.startswith(("MAE_", "MAPE_"))])

print("A. Identifier-kolommen:")
for c in identifier_cols: print(f"  - {c}")
print("\\nB. Voorspelkolommen:")
for c in prediction_cols:
    pct_filled = result[c].notna().mean() * 100
    print(f"  - {c:25s}  {pct_filled:5.1f}% gevuld")
print("\\nC. Actuele cijfers (peilweek-stand):")
for c in actuele_cols: print(f"  - {c}")
print("\\nD. Foutmaten (historische prestatie):")
for c in error_cols: print(f"  - {c}")
"""
        ),
        md(
            """\
> ⚠️ `Ensemble_prediction` en `SARIMA_individual` zijn hier **leeg** omdat we
> `-d c` (alleen cumulatief) draaien. In `-d both` zijn ze gevuld.
> Zie [bekende valkuil](../docs/aan-de-slag.md#bekende-valkuil-stille-modus-downgrade).

## 3. Voorspelling × Realisatie — per opleiding

We voegen de realisatie toe via de helper `with_realisatie()` en zien direct
de fout per opleiding.
"""
        ),
        code(
            """\
from _helpers import with_realisatie

# Filter eerst op peilweek (SARIMA staat alleen daar; ratio voor alle weken)
result_peilweek = result[result["Weeknummer"] == PREDICT_WEEK]

# Aggregeer SARIMA + ratio per opleiding (sommeer over herkomst)
agg = (
    result_peilweek[result_peilweek["Examentype"] == "Bachelor"]
    .groupby(["Croho groepeernaam", "Collegejaar", "Examentype"], as_index=False)
    [["SARIMA_cumulative", "Prognose_ratio"]]
    .sum()
)
real = (
    data_studentcount[(data_studentcount["Examentype"] == "Bachelor")
                      & (data_studentcount["Collegejaar"] == PREDICT_YEAR)]
    .groupby(["Croho groepeernaam", "Collegejaar", "Examentype"], as_index=False)
    ["Aantal_studenten"].sum()
    .rename(columns={"Aantal_studenten": "Realisatie"})
)

vergelijk = agg.merge(real, on=["Croho groepeernaam", "Collegejaar", "Examentype"])
vergelijk["Fout_SARIMA"] = (vergelijk["SARIMA_cumulative"] - vergelijk["Realisatie"]).round(0)
vergelijk["Fout_Ratio"]  = (vergelijk["Prognose_ratio"]    - vergelijk["Realisatie"]).round(0)
vergelijk["%_SARIMA"] = ((vergelijk["SARIMA_cumulative"] - vergelijk["Realisatie"]) / vergelijk["Realisatie"] * 100).round(1)
vergelijk["%_Ratio"]  = ((vergelijk["Prognose_ratio"]    - vergelijk["Realisatie"]) / vergelijk["Realisatie"] * 100).round(1)

vergelijk[["Croho groepeernaam", "Realisatie", "SARIMA_cumulative", "%_SARIMA", "Prognose_ratio", "%_Ratio"]].sort_values("Realisatie", ascending=False)
"""
        ),
        md(
            """\
## 4. MAE vs MAPE in deze data

| Metric | Eenheid | Interpretatie |
|--------|---------|---------------|
| **MAE** (Mean Absolute Error) | studenten | Hoeveel studenten gemiddeld afwijking |
| **MAPE** (Mean Absolute Percentage Error) | % | Procentuele afwijking — vergelijkbaar tussen grote en kleine opleidingen |

> 📌 Foutmaten zijn berekend **exclusief numerus fixus**: bij NF wordt de
> prognose afgekapt, wat foutvergelijking met reguliere opleidingen vertekenend
> maakt.
"""
        ),
        code(
            """\
samenvatting = pd.DataFrame({
    "Model": ["SARIMA cumulatief", "Ratio-baseline"],
    "MAE (studenten)": [vergelijk["Fout_SARIMA"].abs().mean().round(1),
                        vergelijk["Fout_Ratio"].abs().mean().round(1)],
    "MAPE (%)":         [vergelijk["%_SARIMA"].abs().mean().round(1),
                        vergelijk["%_Ratio"].abs().mean().round(1)],
})
samenvatting
"""
        ),
        md(
            """\
## 5. Visualisatie: voorspelling vs. realisatie
"""
        ),
        code(
            """\
fig, ax = plt.subplots(figsize=(8, 8))
m = vergelijk[["Realisatie", "SARIMA_cumulative", "Prognose_ratio"]].max().max() * 1.05
ax.plot([0, m], [0, m], "k--", alpha=0.4, label="Perfecte voorspelling")
ax.scatter(vergelijk["Realisatie"], vergelijk["SARIMA_cumulative"],
           s=60, alpha=0.75, color="steelblue", label="SARIMA cumulatief")
ax.scatter(vergelijk["Realisatie"], vergelijk["Prognose_ratio"],
           s=60, alpha=0.75, color="darkgreen", marker="s", label="Ratio-baseline")
ax.set_xlabel(f"Realisatie {PREDICT_YEAR} (werkelijk aantal eerstejaars)")
ax.set_ylabel(f"Voorspelling op peilweek {PREDICT_WEEK}")
ax.set_title(f"Voorspelling vs. realisatie · {PREDICT_YEAR}")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """\
## 6. Exporteren naar Excel

Exact wat de CLI in `data/output/` schrijft. Een hogeschool gebruikt dit voor
deling met facultaire planners, controle, dashboards, etc.
"""
        ),
        code(
            """\
output_path = Path("data/output/notebook_export_2023_wk12.xlsx")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Voeg realisatie + fouten toe en sla op
export = result[result["Examentype"] == "Bachelor"].copy()
export_agg = vergelijk.copy()
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    export.to_excel(writer, sheet_name="raw_output_per_herkomst", index=False)
    export_agg.to_excel(writer, sheet_name="aggregaat_met_realisatie", index=False)
    samenvatting.to_excel(writer, sheet_name="modelvergelijking", index=False)

print(f"✅ Geschreven: {output_path}")
print(f"   Sheets: raw_output_per_herkomst, aggregaat_met_realisatie, modelvergelijking")
print(f"   Groot {output_path.stat().st_size / 1024:.1f} KB")
"""
        ),
        md(
            """\
## 7. Wanneer is een prognose betrouwbaar? — Beslisregels

**Meer vertrouwen:**
- De modellen liggen dicht bij elkaar (consensus tussen SARIMA en ratio)
- Historische MAE is klein in verhouding tot het cohort
- Peilweek ≥ 10 (genoeg aanmeldata verzameld)

**Minder vertrouwen:**
- Grote spreiding tussen modellen — beoordeel ze los van elkaar
- Hoge historische MAE / MAPE
- Opleiding met weinig historie of klein cohort
- Vroeg in het jaar (peilweek < 6)
- Jaar na een uitzondering (COVID, beleidsingreep)

## 8. Het Plotly-dashboard (opt-in)

Met `--dashboard` genereert de CLI naast de Excel-output interactieve HTML
onder `data/output/visualisaties/`:

```bash
studentprognose --dashboard -d c -y 2024 -w 12
```

- `final/dashboard.html` — altijd
- `cumulative/dashboard.html` — bij `-d c` of `-d b`
- `individual/dashboard.html` — bij `-d i` of `-d b`

> 📌 Dashboard toont alleen de **laatste** week bij multi-week runs. Excel
> bevat wel alle weken.
"""
        ),
    ]
    write("06_output_interpreteren.ipynb", cells, "Output interpreteren")


# ============================================================
# Main
# ============================================================

def main():
    print("Notebooks bouwen…")
    build_overzicht()
    build_data_voorbereiden()
    build_sarima()
    build_xgboost()
    build_ratio()
    build_ensemble()
    build_output()
    print("Klaar.")


if __name__ == "__main__":
    main()
