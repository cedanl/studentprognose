"""Genereert de Jupyter-notebooks in deze map.

Dit script is geen onderdeel van de tool zelf. Het bouwt de notebooks
deterministisch op zodat ze gemakkelijk geregenereerd kunnen worden
bij wijzigingen. Draai met:

    uv run python notebooks/_build_notebooks.py

De gegenereerde .ipynb-bestanden worden in dezelfde map geschreven. Daarna
voer je ze uit om de output te verversen (zie ``notebooks/README.md``).

Indeling:

- 00–03: het CWF-model, het aanbevolen model (overzicht, data, model, output).
- 04–07: verdieping op de andere sporen (SARIMA, XGBoost, ratio, ensemble).
"""

from __future__ import annotations

import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Bestanden van een eerdere indeling die bij het bouwen worden opgeruimd.
VEROUDERD = [
    "02_sarima.ipynb",
    "03_xgboost.ipynb",
    "04_ratio_model.ipynb",
    "05_ensemble.ipynb",
    "06_output_interpreteren.ipynb",
]


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

DEMO_BANNER = md(
    """\
> ℹ️ **Over de demodata.** De notebooks draaien op de meegeleverde, synthetische
> demodata in `data/input/`. Daarin staan de aanmeldingen van alle instellingen
> per opleiding opgeteld, net zoals de pipeline dat doet zonder `institution_filter`.
> Opleidingen zijn **Isatcodes** (bijv. `56604` = B Psychologie); waar het helpt
> staat de naam erachter.
>
> Twee eigenaardigheden van de demodata: **inschrijvingen zijn altijd 0** en
> **de aanmelders zakken vanaf eind mei (rond week 23) kunstmatig in**. Daarom
> gebruiken de notebooks een vroege peilweek (12). Op echte telbestanden speelt dit niet.
>
> **Met eigen data:** vervang de bestanden in `data/input/` en pas `PROGRAMMA`,
> `PREDICT_YEAR` en `PREDICT_WEEK` aan.
"""
)

VERDIEPING_BANNER = md(
    """\
> 🧭 **Verdieping: een ander spoor.** Het aanbevolen model is het
> [CWF-model](02_cwf_model.ipynb): het is doorgaans nauwkeuriger en
> geeft er een onzekerheidsmarge en betrouwbaarheidslabel bij. Deze notebook legt
> een van de andere sporen uit. Die zijn nog steeds te gebruiken (`-d c`, `-d i`,
> `-d b`), bijvoorbeeld om te vergelijken.
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

print("Projectmap gevonden:", "ja" if Path("data/input").is_dir() else "nee — start Jupyter vanuit de repo")
print("Python:", sys.version.split()[0])
"""
)

LOAD_CUMULATIVE_CELL = code(
    """\
from _helpers import load_cumulative, opleidingsnamen, label

data_cumulative, data_studentcount, data_cumulative_raw = load_cumulative()
NAMEN = opleidingsnamen()
print(f"Telregels (opgeteld over instellingen): {len(data_cumulative):,}")
print(f"Aantal opleidingen:                     {data_cumulative['Croho groepeernaam'].nunique()}")
print(f"Collegejaren:                           {data_cumulative['Collegejaar'].min()}–{data_cumulative['Collegejaar'].max()}")
print(f"Realisatie (studentaantallen):          {len(data_studentcount):,} rijen")
"""
)

SELECTIE_CELL = code(
    """\
from _helpers import grootste_opleidingen

# De cumulatieve pipeline (SARIMA + XGBoost) is zwaar: hij fit een tijdreeks per
# opleiding × herkomst. Voor de snelheid draaien we hem hier op de grootste
# bachelors. Zet SELECTIE = [] om alle opleidingen te voorspellen.
SELECTIE = grootste_opleidingen(data_studentcount, n=15, examentype="Bachelor")
FILTERING = {"filtering": {"programme": SELECTIE, "herkomst": [], "examentype": []}}
print(f"Selectie: {len(SELECTIE)} opleidingen, o.a. " + ", ".join(label(c, NAMEN) for c in SELECTIE[:3]))
"""
)


# ============================================================
# 00 — Overzicht (CWF-model end-to-end, mét realisatie)
# ============================================================

def build_overzicht():
    cells = [
        md(
            """\
# 00 — Studentprognose in vijf minuten

Deze notebook laat van begin tot eind zien wat studentprognose doet: van data
inladen tot een prognose met onzekerheidsmarge, en hoe die prognose zich verhoudt
tot de werkelijke instroom.

> 🔗 **Aanvullend bij:** [`docs/methodologie/index.md`](../docs/methodologie/index.md),
> [`docs/methodologie/cwf.md`](../docs/methodologie/cwf.md) en [`docs/aan-de-slag.md`](../docs/aan-de-slag.md).

## Wat doet de tool?

Voor elke combinatie van opleiding × herkomst × examentype voorspelt de tool het
**aantal eerstejaars** in een collegejaar, gegeven de stand van de aanmeldingen in
een **peilweek**.

| Model | CLI-vlag | Databron | Levert |
|-------|----------|----------|--------|
| **CWF-model (aanbevolen)** | `-d cwf` | Studielink-telbestanden + oktobertelling | Prognose, 95%-interval en betrouwbaarheidslabel |
| Cumulatief spoor | `-d c` | Studielink-telbestanden | SARIMA + XGBoost + ratio |
| Individueel spoor | `-d i` | Aanmelddata per student | XGBoost-classifier + SARIMA |
| Beide sporen | `-d b` | Beide bronnen | Ensemble van de twee sporen |

We draaien hier het **CWF-model**. Het is doorgaans het nauwkeurigst
en zegt er ook bij hoe zeker de prognose is. De andere sporen komen aan bod in de
verdiepingsnotebooks 04–07.
"""
        ),
        DEMO_BANNER,
        SETUP_CELL,
        md(
            """\
## 1. Data laden

Eén regel laadt de cumulatieve telregels en de historische realisatie uit
`data/input/`.
"""
        ),
        LOAD_CUMULATIVE_CELL,
        md(
            """\
## 2. Het CWF-model draaien

`run_pipeline_from_dataframes` doet hetzelfde als
`studentprognose -d cwf -y 2023 -w 12` aan de opdrachtregel, maar geeft het
resultaat terug als DataFrame.

We voorspellen **2023** vanaf peilweek **12**. Dat jaar is afgerond, dus we kunnen
de prognose straks naast de werkelijke instroom leggen. Het model gebruikt
alleen eerdere jaren: het is een eerlijke terugblik.
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
        year=PREDICT_YEAR,
        week=PREDICT_WEEK,
        data_cumulative=data_cumulative_raw,
        data_student_numbers=data_studentcount,
        dataset=DataOption.CWF,
        save_output=False,
    )

print(f"Output: {result.shape[0]} rijen × {result.shape[1]} kolommen")
result[["Croho groepeernaam", "Herkomst", "Examentype", "Prognose_cwf",
        "cwf_ondergrens", "cwf_bovengrens", "cwf_betrouwbaarheid"]].head()
"""
        ),
        md(
            """\
Elke rij heeft drie dingen:

- **`Prognose_cwf`**: het verwachte aantal eerstejaars;
- **`cwf_ondergrens` / `cwf_bovengrens`**: het 95%-interval, de bandbreedte waarbinnen
  de instroom naar verwachting valt;
- **`cwf_betrouwbaarheid`**: Groen (solide), Oranje (bruikbare indicatie) of Rood
  (alleen een indicatie).

## 3. Prognose naast de realisatie

De output bevat de werkelijke instroom (`Aantal_studenten`) al, omdat 2023 afgerond
is. We kijken naar drie dingen: hoe groot de fout is, hoe vaak de realisatie binnen
het interval viel, en of het model beter is dan de naïeve voorspelling
"evenveel als vorig jaar".
"""
        ),
        code(
            """\
eval_df = result.dropna(subset=["Aantal_studenten", "Prognose_cwf"]).copy()

vorig_jaar = (
    data_studentcount[data_studentcount["Collegejaar"] == PREDICT_YEAR - 1]
    .groupby(["Croho groepeernaam", "Herkomst", "Examentype"], as_index=False)["Aantal_studenten"].sum()
    .rename(columns={"Aantal_studenten": "Vorig_jaar"})
)
eval_df = eval_df.merge(vorig_jaar, on=["Croho groepeernaam", "Herkomst", "Examentype"], how="left")


def wape(pred, actual):
    return float((pred - actual).abs().sum() / actual.abs().sum())


binnen = (eval_df["Aantal_studenten"] >= eval_df["cwf_ondergrens"]) & (
    eval_df["Aantal_studenten"] <= eval_df["cwf_bovengrens"]
)
naief = eval_df.dropna(subset=["Vorig_jaar"])

samenvatting = pd.DataFrame({
    "Maat": ["Aantal combinaties", "WAPE CWF-model", "WAPE 'evenveel als vorig jaar'",
             "Realisatie binnen 95%-interval"],
    "Waarde": [
        f"{len(eval_df)}",
        f"{wape(eval_df['Prognose_cwf'], eval_df['Aantal_studenten']):.1%}",
        f"{wape(naief['Vorig_jaar'], naief['Aantal_studenten']):.1%}",
        f"{binnen.mean():.0%}",
    ],
})
samenvatting
"""
        ),
        md(
            """\
**WAPE** is de totale absolute fout gedeeld door de totale instroom: 10% betekent
dat het model er over alle opleidingen samen 10% naast zat. Grote opleidingen
tellen daarin zwaarder mee dan kleine, wat past bij een planningsvraag.

## 4. Per betrouwbaarheidslabel

Het label moet iets zeggen: prognoses met label Rood zouden er vaker naast moeten
zitten dan prognoses met label Oranje of Groen.
"""
        ),
        code(
            """\
eval_df["binnen_interval"] = binnen
per_label = (
    eval_df.groupby("cwf_betrouwbaarheid")
    .apply(lambda g: pd.Series({
        "combinaties": len(g),
        "WAPE": round(wape(g["Prognose_cwf"], g["Aantal_studenten"]), 3),
        "binnen interval": round(g["binnen_interval"].mean(), 2),
    }), include_groups=False)
    .reindex(["Groen", "Oranje", "Rood"])
    .dropna(how="all")
)
per_label
"""
        ),
        md(
            """\
In week 12 is de aanmeldcyclus nog niet ver gevorderd. Groen vraagt een vulgraad
van minstens 0,9 (zie [02 — CWF-model](02_cwf_model.ipynb)); die wordt pas later in
het jaar bereikt. Daarom zie je hier alleen Oranje en Rood.

## 5. Visualisatie: de grootste opleidingen

Per combinatie de prognose met interval (lijn) en de realisatie (kruis). Valt het
kruis op de lijn, dan zat de realisatie binnen het interval.
"""
        ),
        code(
            """\
top = eval_df[eval_df["Herkomst"] == "NL"].nlargest(15, "Aantal_studenten").iloc[::-1]
kleur = {"Groen": "seagreen", "Oranje": "darkorange", "Rood": "firebrick"}

fig, ax = plt.subplots(figsize=(10, 7))
y = np.arange(len(top))
for yi, (_, r) in zip(y, top.iterrows()):
    c = kleur.get(r["cwf_betrouwbaarheid"], "gray")
    ax.plot([r["cwf_ondergrens"], r["cwf_bovengrens"]], [yi, yi], color=c, linewidth=6, alpha=0.35)
    ax.plot(r["Prognose_cwf"], yi, "o", color=c)
ax.scatter(top["Aantal_studenten"], y, marker="X", color="black", zorder=5, label="Realisatie")
for lab, c in kleur.items():
    if (top["cwf_betrouwbaarheid"] == lab).any():
        ax.plot([], [], "o-", color=c, linewidth=4, alpha=0.6, label=f"Prognose + interval ({lab})")
ax.set_yticks(y)
ax.set_yticklabels([f"{label(c, NAMEN, 32)} ({r})" for c, r in zip(top["Croho groepeernaam"], top["Examentype"])])
ax.set_xlabel("Eerstejaars")
ax.set_title(f"CWF-prognose met 95%-interval · NL · {PREDICT_YEAR} · peilweek {PREDICT_WEEK}")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """\
## 6. Volgende stappen

| Wil je weten… | Open notebook |
|---------------|---------------|
| Of je eigen data klaar is voor het model | `01_data_voorbereiden.ipynb` |
| Hoe het CWF-model tot een prognose, interval en label komt | `02_cwf_model.ipynb` |
| Hoe je de output leest en deelt | `03_output_interpreteren.ipynb` |
| Hoe de andere sporen werken (SARIMA, XGBoost, ratio, ensemble) | `04`–`07` |
"""
        ),
    ]
    write("00_overzicht.ipynb", cells, "Overzicht")


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
2. Draaien een **validatierapport** dat je 1-op-1 op je eigen data kunt gebruiken
3. Bekijken de **pijplijn**: de reeks waar het CWF-model op rekent
4. Maken een dekkings-heatmap die ontbrekende weken zichtbaar maakt

Het CWF-model heeft maar twee bronnen nodig: de **Studielink-telbestanden** en de
**oktobertelling** (de gerealiseerde instroom). Individuele aanmelddata is niet nodig.

> 💡 **Voor je eigen data:** vervang `data/input/vooraanmeldingen_cumulatief.csv` en
> `data/input/student_count_first-years.xlsx` door je eigen ETL-output en herstart de
> kernel. De validatiecellen werken zonder code-aanpassing.
"""
        ),
        DEMO_BANNER,
        SETUP_CELL,
        md(
            """\
## 1. Data laden

`load_cumulative()` laadt de telregels en bereidt ze voor met dezelfde functie als
de pipeline: kolommen hernoemen, alleen eerstejaars, en optellen over instellingen
tot één rij per opleiding × herkomst × examentype × jaar × week.
"""
        ),
        LOAD_CUMULATIVE_CELL,
        md(
            """\
## 2. Schema

Eén rij = één combinatie van jaar × week × opleiding × herkomst × examentype. De
belangrijkste velden zijn `Gewogen vooraanmelders` en `Inschrijvingen`; samen vormen
ze de **pijplijn** (`ts`).
"""
        ),
        code(
            """\
print("Kolommen + type:")
for col in data_cumulative.columns:
    print(f"  - {col:38s} {data_cumulative[col].dtype}")
data_cumulative.head(3)
"""
        ),
        md(
            """\
## 3. Validatierapport

Een **draaibaar rapport** met de belangrijkste checks uit `docs/validatie.md`. Op
eigen data laat dit in één keer zien of je klaar bent om te draaien.
"""
        ),
        code(
            """\
import datetime
from studentprognose.utils.weeks import get_all_weeks_ordered


def validatie_rapport(df: pd.DataFrame, df_sc: pd.DataFrame) -> pd.DataFrame:
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

    # 2. Jaarbereik (huidig jaar - 15 .. huidig jaar + 2)
    jaren = sorted(df["Collegejaar"].dropna().unique().astype(int))
    huidig = datetime.date.today().year
    buiten = [y for y in jaren if y < huidig - 15 or y > huidig + 2]
    rows.append({
        "Check": "Collegejaar in plausibel bereik",
        "Status": "✅" if not buiten else "⚠️",
        "Detail": f"jaren {jaren[0]}–{jaren[-1]}" + (f" ; buiten bereik: {buiten}" if buiten else ""),
    })

    # 3. Herkomst-waarden
    herk = sorted(df["Herkomst"].dropna().unique())
    onbekend = set(herk) - {"NL", "EER", "Niet-EER"}
    rows.append({
        "Check": "Herkomst-waarden geldig",
        "Status": "✅" if not onbekend else "⚠️",
        "Detail": f"{herk}" + (f" ; onbekend: {sorted(onbekend)}" if onbekend else ""),
    })

    # 4. Ontbrekende waarden (warning > 5%, soft error > 30%)
    nan_pct = df["Gewogen vooraanmelders"].isna().mean() * 100
    rows.append({
        "Check": "Ontbrekende waarden in Gewogen vooraanmelders",
        "Status": "✅" if nan_pct <= 5 else ("⚠️" if nan_pct <= 30 else "❌"),
        "Detail": f"{nan_pct:.2f}% leeg",
    })

    # 5. Programmasleutel is een Isatcode (getal)
    sleutel = df["Croho groepeernaam"]
    rows.append({
        "Check": "Opleidingen als Isatcode",
        "Status": "✅" if pd.api.types.is_integer_dtype(sleutel) else "⚠️",
        "Detail": f"type {sleutel.dtype}",
    })

    # 6. Gaten tussen weken binnen een collegejaar, in academische volgorde (39 → 38)
    positie = {int(w): i for i, w in enumerate(get_all_weeks_ordered())}
    df_pos = df.assign(positie=df["Weeknummer"].map(positie)).dropna(subset=["positie"])
    gaten = [
        int(np.diff(np.sort(p)).max())
        for p in df_pos.groupby(["Croho groepeernaam", "Collegejaar"])["positie"].unique()
        if len(p) > 1
    ]
    n_groot = sum(g > 2 for g in gaten)
    rows.append({
        "Check": "Geen grote gaten tussen weken",
        "Status": "✅" if n_groot == 0 else "⚠️",
        "Detail": f"{n_groot} van {len(gaten)} opleiding-jaren met een gat > 2 weken",
    })

    # 7. Realisatie sluit aan op aanmelddata (CWF heeft ≥ 3 eigen jaren nodig voor Groen/Oranje)
    gemeenschappelijk = set(df["Collegejaar"].astype(int)) & set(df_sc["Collegejaar"].astype(int))
    rows.append({
        "Check": "Realisatie aanwezig voor aanmeldjaren",
        "Status": "✅" if len(gemeenschappelijk) >= 3 else "⚠️",
        "Detail": f"{len(gemeenschappelijk)} jaren overlappen (≥ 3 nodig)",
    })

    return pd.DataFrame(rows)


validatie_rapport(data_cumulative, data_studentcount)
"""
        ),
        md(
            """\
Een gat tussen weken is geen ramp: de pipeline vult een ontbrekende week tussen twee
waarnemingen op door te interpoleren. Bij kleine opleidingen komt het vaak voor,
omdat er niet elke week een aanmelding bij komt. Ontbreken er bij grote opleidingen
veel weken, controleer dan of alle telbestanden zijn ingelezen.

## 4. De pijplijn per jaar

Het CWF-model rekent op de **pijplijn**: gewogen vooraanmelders plus inschrijvingen.
Hieronder het verloop per collegejaar voor één opleiding, in academische volgorde
(week 39 → week 38). Hoe meer de jaren op elkaar lijken, hoe beter het model werkt.
"""
        ),
        code(
            """\
from studentprognose.utils.weeks import get_all_weeks_valid
from studentprognose.data.transforms import transform_data

PROGRAMMA = 56604  # B Psychologie
HERKOMST = "NL"
EXAMENTYPE = "Bachelor"

wide = transform_data(data_cumulative, "ts", interpolate_interior_gaps=True)
sample = wide[
    (wide["Croho groepeernaam"] == PROGRAMMA)
    & (wide["Herkomst"] == HERKOMST)
    & (wide["Examentype"] == EXAMENTYPE)
]
weekcols = get_all_weeks_valid(sample.columns)

laatste_week = data_cumulative.groupby("Collegejaar")["Weeknummer"].apply(
    lambda w: max(weekcols.index(str(x)) for x in w if str(x) in weekcols)
)

fig, ax = plt.subplots(figsize=(11, 5))
for _, row in sample.sort_values("Collegejaar").iterrows():
    jaar = int(row["Collegejaar"])
    n = laatste_week[jaar] + 1  # het lopende jaar stopt bij de laatst bekende week
    ax.plot(range(n), row[weekcols].astype(float).values[:n], label=str(jaar), alpha=0.85)

ax.set_xticks(range(0, len(weekcols), 4))
ax.set_xticklabels([weekcols[i] for i in range(0, len(weekcols), 4)])
ax.set_xlabel("Weeknummer (academische volgorde: 39 → 38)")
ax.set_ylabel("Pijplijn (gewogen vooraanmelders + inschrijvingen)")
ax.set_title(f"Pijplijn · {label(PROGRAMMA, NAMEN)} · {HERKOMST} · per collegejaar")
ax.legend(title="Jaar", ncol=2, fontsize=8)
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """\
In de demodata zakken de curves rond week 23 in. Dat is een eigenschap van de
synthetische data (inschrijvingen zijn er altijd 0), geen echt aanmeldgedrag. Het
lopende jaar stopt bij de laatst bekende week.

## 5. Dekkings-heatmap

Donkere cellen = veel aanmelders, lichte = weinig, **wit = ontbrekend**. Een
instelling die ontdekt dat een hele week ontbreekt, kan dat herstellen vóór het
modelleren.
"""
        ),
        code(
            """\
heat = (
    data_cumulative[data_cumulative["Croho groepeernaam"] == PROGRAMMA]
    .pivot_table(index="Collegejaar", columns="Weeknummer",
                 values="Gewogen vooraanmelders", aggfunc="sum")
    .reindex(columns=range(1, 54))
)

fig, ax = plt.subplots(figsize=(11, 4.5))
im = ax.imshow(heat.values, aspect="auto", cmap="viridis")
ax.set_yticks(range(len(heat.index)))
ax.set_yticklabels([str(int(y)) for y in heat.index])
ax.set_xticks(range(0, len(heat.columns), 4))
ax.set_xticklabels([str(int(w)) for w in heat.columns[::4]])
ax.set_xlabel("Weeknummer (kalender)")
ax.set_ylabel("Collegejaar")
ax.set_title(f"Dekking gewogen vooraanmelders · {label(PROGRAMMA, NAMEN)}")
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
- ✅ Opleidingen als Isatcode
- ✅ Continuïteit tussen weken
- ✅ Aansluiting met de realisatie

Bij ⚠️ of ❌ verwijst [`validatie.md`](../docs/validatie.md) naar de oplossing.
Daarna kun je verder met [`02_cwf_model.ipynb`](02_cwf_model.ipynb).
"""
        ),
    ]
    write("01_data_voorbereiden.ipynb", cells, "Data voorbereiden")


# ============================================================
# 02 — CWF-model in detail
# ============================================================

def build_cwf():
    cells = [
        md(
            """\
# 02 — Het CWF-model in detail

Deze notebook hoort bij [`docs/methodologie/cwf.md`](../docs/methodologie/cwf.md).
Het CWF-model (confidence-weighted forecast) beantwoordt per combinatie één vraag:

> *Welk deel van de mensen die nu in de pijplijn zitten, schrijft zich straks echt in?*

We rekenen het stap voor stap na voor één opleiding:

1. **Pijplijn en ratio** per eerder jaar
2. **Vulgraad**: hoe ver is de aanmeldcyclus?
3. **Ratiofactor en prior**: de twee schattingen
4. **Samenvoegen** tot één prognose
5. **Interval en label**
6. **Backtest** over meerdere jaren, vergeleken met "evenveel als vorig jaar"
"""
        ),
        DEMO_BANNER,
        SETUP_CELL,
        LOAD_CUMULATIVE_CELL,
        md(
            """\
## 1. De werkdata van het model

Het model zet de data eerst om naar één rij per combinatie × jaar × week, met drie
afgeleide kolommen:

| Kolom | Betekenis |
|-------|-----------|
| `Pipeline` | Gewogen vooraanmelders + inschrijvingen in die week |
| `Ratio` | Gerealiseerde instroom / pijplijn: welk deel van de pijplijn uiteindelijk instroomde |
| `Vulgraad` | Welk deel van de uiteindelijke aanmeldpiek op dat moment al bereikt was |

We bouwen die werkdata met dezelfde functies als het model zelf.
"""
        ),
        code(
            """\
from studentprognose.config import load_defaults, get_final_academic_week
from studentprognose.models.cwf import CWFParams, ConfidenceWeightedForecast
from studentprognose.strategies.cumulative import prepare_cumulative_data
from studentprognose.strategies.cwf import build_cwf_history, redistribute_unknown_origin

config = load_defaults()
params = CWFParams.from_config(config)
final_week = get_final_academic_week(config)

data = redistribute_unknown_origin(prepare_cumulative_data(data_cumulative_raw.copy()))
studentcount = data_studentcount.copy()
studentcount["Croho groepeernaam"] = studentcount["Croho groepeernaam"].astype(data["Croho groepeernaam"].dtype)
history = build_cwf_history(data, studentcount, final_week, params.fill_rate_recency_alpha)

PROGRAMMA = 56604  # B Psychologie
HERKOMST = "NL"
EXAMENTYPE = "Bachelor"
PREDICT_YEAR = 2023
PREDICT_WEEK = 12

combo = history[
    (history["Croho groepeernaam"] == PROGRAMMA)
    & (history["Herkomst"] == HERKOMST)
    & (history["Examentype"] == EXAMENTYPE)
]
print(f"Combinatie: {label(PROGRAMMA, NAMEN)} · {HERKOMST} · {EXAMENTYPE}")
combo[combo["Weeknummer"] == PREDICT_WEEK][
    ["Collegejaar", "Pipeline", "Aantal_studenten", "Ratio", "Vulgraad"]
].set_index("Collegejaar").round(3)
"""
        ),
        md(
            """\
De ratio zegt hoeveel van de pijplijn in week 12 uiteindelijk instroomde. Hoe
stabieler die ratio over de jaren, hoe betrouwbaarder het ratiosignaal. Voor het
voorspeljaar zelf is `Aantal_studenten` (en dus de ratio) nog onbekend: dat is wat
we voorspellen.

## 2. Vulgraad: hoe ver is de cyclus?

Vroeg in het jaar is de pijplijn nog klein en zegt hij weinig over de eindstand. De
**vulgraad** meet welk deel van de aanmeldpiek een opleiding in eerdere jaren op dit
moment al had bereikt. Hij wordt alleen uit **eerdere** jaren berekend, zodat het
model niet in de toekomst kijkt.
"""
        ),
        code(
            """\
from studentprognose.utils.weeks import get_all_weeks_ordered

volgorde = get_all_weeks_ordered(final_week)
vj = combo[combo["Collegejaar"] == PREDICT_YEAR].copy()
vj["positie"] = vj["Weeknummer"].astype(str).map({w: i for i, w in enumerate(volgorde)})
vj = vj.dropna(subset=["positie"]).sort_values("positie")

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(vj["positie"], vj["Vulgraad"], color="purple", linewidth=2)
ax.axhline(params.reliability_orange_min_fill, color="darkorange", linestyle=":", label="Ondergrens Oranje")
ax.axhline(params.reliability_green_min_fill, color="seagreen", linestyle=":", label="Ondergrens Groen")
ax.axvline(volgorde.index(str(PREDICT_WEEK)), color="gray", linestyle="--", label=f"Peilweek {PREDICT_WEEK}")
ax.set_xticks(range(0, len(volgorde), 4))
ax.set_xticklabels([volgorde[i] for i in range(0, len(volgorde), 4)])
ax.set_ylim(0, 1.05)
ax.set_xlabel("Weeknummer (academische volgorde: 39 → 38)")
ax.set_ylabel("Vulgraad")
ax.set_title(f"Vulgraad · {label(PROGRAMMA, NAMEN)} · {HERKOMST} · {PREDICT_YEAR}")
ax.legend()
plt.tight_layout()
plt.show()

VULGRAAD = float(vj.loc[vj["Weeknummer"] == PREDICT_WEEK, "Vulgraad"].iloc[0])
print(f"Vulgraad in week {PREDICT_WEEK}: {VULGRAAD:.2f}")
"""
        ),
        md(
            """\
## 3. Ratiofactor en prior

Het model maakt twee schattingen:

1. **Het ratiosignaal**: de huidige pijplijn × de **ratiofactor**. De ratiofactor is
   een gemiddelde van de ratio's van eerdere jaren, met recente jaren zwaarder
   gewogen (meetkundig gemiddelde, zodat één uitschieter niet domineert). Bij weinig
   eigen jaren wordt hij getrokken naar de ratio van vergelijkbare opleidingen.
2. **De prior**: een schatting die alleen naar de gerealiseerde instroom kijkt.
   Meestal de instroom van vorig jaar; bij een duidelijke trend wordt een deel van
   die trend doorgetrokken.

We roepen het model aan met precies de historie die het in de pipeline krijgt.
"""
        ),
        code(
            """\
week_slice = history[(history["Weeknummer"] == PREDICT_WEEK) & (history["Collegejaar"] < PREDICT_YEAR)]
groep = week_slice[(week_slice["Examentype"] == EXAMENTYPE) & (week_slice["Herkomst"] == HERKOMST)]
groep = groep[groep["Aantal_studenten"].notna()]
own = groep[groep["Croho groepeernaam"] == PROGRAMMA]
pool = groep[groep["Croho groepeernaam"] != PROGRAMMA]
pipeline_nu = float(combo.loc[(combo["Collegejaar"] == PREDICT_YEAR) & (combo["Weeknummer"] == PREDICT_WEEK), "Pipeline"].iloc[0])

model = ConfidenceWeightedForecast(params)
prior = model.get_prior(own["Aantal_studenten"], own["Collegejaar"].to_numpy(dtype=float), np.nan, True)
uit = model.predict(own=own, current_student_pipeline=pipeline_nu, pre_defined_prior=np.nan,
                    pool=pool, has_own_history=True, programme=PROGRAMMA)

signaal = pipeline_nu * uit["ratio_factor"]
print(f"Pijplijn nu (week {PREDICT_WEEK}, {PREDICT_YEAR}): {pipeline_nu:8.1f}")
print(f"Ratiofactor:                     {uit['ratio_factor']:8.2f}")
print(f"Ratiosignaal = pijplijn × factor: {signaal:8.1f}")
print(f"Prior (op basis van instroom):    {prior:8.1f}")
print(f"Gewicht van het ratiosignaal:     {uit['signal_weight']:8.2f}")
"""
        ),
        md(
            """\
## 4. Samenvoegen tot één prognose

$$\\hat{y} = s \\cdot (\\text{pijplijn} \\times R) + (1 - s) \\cdot \\text{prior}$$

Het gewicht $s$ is de vulgraad (gemiddeld over eerdere jaren), verlaagd als de
ratiofactor zelf onrustig is. Vroeg in de cyclus leunt de prognose dus op de prior,
laat in de cyclus op het ratiosignaal.
"""
        ),
        code(
            """\
s = uit["signal_weight"]
handmatig = s * signaal + (1 - s) * prior
print(f"Handmatig: {s:.2f} × {signaal:.1f} + {1 - s:.2f} × {prior:.1f} = {handmatig:.1f}")
print(f"Model:     {uit['prediction']}")
print("(Klein verschil komt doordat de getoonde ratiofactor op twee decimalen is afgerond.)")
"""
        ),
        md(
            """\
## 5. Interval en label

Het model doet meer dan een puntschatting. Het **95%-interval** mengt twee bronnen
van onzekerheid:

- **Jaar-op-jaar**: hoe sterk de instroom van deze combinatie schommelt;
- **Trackrecord**: hoe ver het model er in eerdere jaren in dezelfde week naast zat
  (elk eerder jaar voorspeld met alleen de andere jaren).

Vroeg in de cyclus weegt de jaar-op-jaar-schommeling zwaarder, laat in de cyclus het
trackrecord. Het **label** vat samen hoe zeker de prognose is:

| Label | Wanneer |
|-------|---------|
| **Groen** | Vulgraad ≥ 0,9 en in alle eerdere jaren binnen ±16% (gemiddeld ±8%) |
| **Oranje** | Alles tussen Groen en Rood |
| **Rood** | Geen eigen historie, < 3 eigen backtestjaren, vulgraad < 0,2, of een eerder jaar met een fout van meer dan een factor 2 |

We draaien het model voor deze ene opleiding om prognose, interval en label te zien.
"""
        ),
        code(
            """\
import os
from studentprognose.strategies.cwf import CWFStrategy
from _helpers import suppress_stdout
from studentprognose.utils.weeks import DataOption

strategy = CWFStrategy(data_cumulative_raw, data_studentcount, config, None, None,
                       os.getcwd(), DataOption.CWF, None)
strategy.preprocess()
strategy.set_filtering([PROGRAMMA], [], [EXAMENTYPE])
with suppress_stdout():
    een = strategy.predict_nr_of_students(PREDICT_YEAR, PREDICT_WEEK)

een = een.merge(
    data_studentcount[data_studentcount["Collegejaar"] == PREDICT_YEAR],
    on=["Croho groepeernaam", "Herkomst", "Examentype", "Collegejaar"], how="left",
)
een[["Herkomst", "Prognose_cwf", "cwf_ondergrens", "cwf_bovengrens",
     "cwf_betrouwbaarheid", "cwf_vulgraad", "cwf_ratiofactor", "Aantal_studenten"]]
"""
        ),
        md(
            """\
De rij voor NL hoort dezelfde prognose te geven als de handmatige berekening
hierboven. `Aantal_studenten` is de werkelijke instroom van 2023, ter vergelijking.

## 6. Backtest: hoe goed is het model over meerdere jaren?

Eén jaar zegt weinig. `backtest_cwf` voorspelt elk afgerond jaar vanaf dezelfde
peilweek, alsof het model toen live draaide. Dit is wat
`studentprognose benchmark -d cwf -w 12` ook doet.

> 🕒 Duurt op de demodata ongeveer een minuut.
"""
        ),
        code(
            """\
from studentprognose.benchmark.evaluate_cwf import backtest_cwf, summarise_cwf

with suppress_stdout():
    backtest = backtest_cwf(data_cumulative_raw, data_studentcount, config, PREDICT_WEEK)
per_jaar, per_label = summarise_cwf(backtest)

kolommen = {"n": "combinaties", "wape": "WAPE CWF", "naive_wape": "WAPE vorig jaar",
            "coverage": "binnen interval", "median_rel_width": "mediane breedte interval"}
per_jaar.rename(columns=kolommen)[["test_year", *kolommen.values()]].round(3)
"""
        ),
        md(
            """\
Lees de tabel zo:

- **WAPE CWF lager dan WAPE vorig jaar**: het model voegt iets toe boven
  "evenveel als vorig jaar".
- **Binnen interval**: bij een goed 95%-interval ligt dit rond 0,95. Veel lager
  betekent dat het interval te smal is, veel hoger dat het te ruim is.
- **Mediane breedte**: de breedte van het interval als deel van de prognose.

Per label:
"""
        ),
        code(
            """\
per_label.rename(columns=kolommen)[["label", *kolommen.values()]].round(3)
"""
        ),
        md(
            """\
## 7. Wanneer vertrouw je het niet?

| Situatie | Wat te doen? |
|----------|--------------|
| **Label Rood** | Gebruik de prognose als indicatie; kijk naar het interval, niet naar één getal |
| **Vroeg in de cyclus** (lage vulgraad) | De prognose is vooral de instroom van vorig jaar |
| **Nieuwe opleiding** zonder historie | Stel een verwachte instroom in via `cwf_params.predefined_priors` |
| **Uitzonderlijk jaar** in de historie (bijv. corona) | Geef het minder gewicht via `cwf_params.year_weight_overrides`, bijv. `{"2021": 0.5}` |
| **Numerus fixus** | Het model kapt af op de capaciteit, maar weet niets van loting of selectie |

Alle parameters staan in `model_config.cwf_params`; zie de
[configuratiereferentie](../docs/configuratie-referentie.md#cwf_params).
"""
        ),
    ]
    write("02_cwf_model.ipynb", cells, "CWF-model")


# ============================================================
# 03 — Output interpreteren (CWF)
# ============================================================

def build_output():
    cells = [
        md(
            """\
# 03 — Output interpreteren

Deze notebook hoort bij [`docs/output-begrijpen.md`](../docs/output-begrijpen.md#cwf-model-d-cwf).
We:

1. Draaien het CWF-model en lopen de **kolommen** langs
2. Lezen één rij hardop
3. Bekijken hoe je **interval en label** gebruikt in een rapportage
4. Exporteren naar Excel, zoals de opdrachtregel dat doet
"""
        ),
        DEMO_BANNER,
        SETUP_CELL,
        LOAD_CUMULATIVE_CELL,
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
        dataset=DataOption.CWF, save_output=False,
    )
print(f"Output: {result.shape[0]} rijen × {result.shape[1]} kolommen")
"""
        ),
        md(
            """\
## 1. De kolommen

| Groep | Kolommen | Betekenis |
|-------|----------|-----------|
| Sleutel | `Croho groepeernaam`, `Herkomst`, `Examentype`, `Collegejaar`, `Weeknummer` | Waar de rij over gaat |
| Prognose | `Prognose_cwf` | Verwacht aantal eerstejaars |
| Onzekerheid | `cwf_ondergrens`, `cwf_bovengrens` | 95%-interval |
| | `cwf_betrouwbaarheid` | Groen / Oranje / Rood |
| Achtergrond | `cwf_vulgraad` | Hoe ver de cyclus is (0–1) |
| | `cwf_ratiofactor` | Welk deel van de pijplijn naar verwachting instroomt |
| Stand peilweek | `Gewogen vooraanmelders`, `Ongewogen vooraanmelders`, `Inschrijvingen`, … | De aanmeldingen op het peilmoment |
| Realisatie | `Aantal_studenten`, `MAE_Prognose_cwf`, `MAPE_Prognose_cwf` | Alleen gevuld als de instroom al bekend is (terugblik) |

In de Excel-uitvoer van de opdrachtregel staat bovendien `Opleidingsnaam` naast de code.
"""
        ),
        code(
            """\
for kolom in result.columns:
    gevuld = result[kolom].notna().mean()
    print(f"  {kolom:38s} {gevuld:6.0%} gevuld")
"""
        ),
        md(
            """\
## 2. Eén rij hardop gelezen
"""
        ),
        code(
            """\
rij = result[(result["Croho groepeernaam"] == 56604) & (result["Herkomst"] == "NL")
             & (result["Examentype"] == "Bachelor")].iloc[0]

print(f"Opleiding:  {label(rij['Croho groepeernaam'], NAMEN)} · {rij['Herkomst']} · {rij['Examentype']}")
print(f"Peilmoment: week {rij['Weeknummer']} van {rij['Collegejaar']}")
print(f"Pijplijn:   {rij['Gewogen vooraanmelders'] + rij['Inschrijvingen']:.0f} (gewogen vooraanmelders + inschrijvingen)")
print(f"Vulgraad:   {rij['cwf_vulgraad']:.2f} → de cyclus is voor zo'n {rij['cwf_vulgraad']:.0%} gevorderd")
print(f"Ratio:      {rij['cwf_ratiofactor']:.2f} → zo'n {rij['cwf_ratiofactor']:.0%} van de pijplijn stroomt naar verwachting in")
print()
print(f"Prognose:   {rij['Prognose_cwf']:.0f} eerstejaars, "
      f"waarschijnlijk tussen {rij['cwf_ondergrens']:.0f} en {rij['cwf_bovengrens']:.0f}")
print(f"Label:      {rij['cwf_betrouwbaarheid']}")
if pd.notna(rij["Aantal_studenten"]):
    print(f"Realisatie: {rij['Aantal_studenten']:.0f}")
"""
        ),
        md(
            """\
## 3. Interval en label in een rapportage

Een prognose zonder bandbreedte wekt de indruk van meer zekerheid dan er is. Twee
vuistregels:

- **Rapporteer het interval**, zeker bij Oranje en Rood. "Tussen 230 en 290" is
  eerlijker dan "260".
- **Tel prognoses op, intervallen niet.** De som van de ondergrenzen is geen
  ondergrens voor het totaal: niet alle opleidingen vallen tegelijk tegen.

Hieronder per label hoe breed de intervallen zijn, als deel van de prognose.
"""
        ),
        code(
            """\
r = result[result["Prognose_cwf"] > 0].copy()
r["breedte_rel"] = (r["cwf_bovengrens"] - r["cwf_ondergrens"]) / r["Prognose_cwf"]
(
    r.groupby("cwf_betrouwbaarheid")
    .agg(combinaties=("Prognose_cwf", "size"),
         mediane_prognose=("Prognose_cwf", "median"),
         mediane_breedte=("breedte_rel", "median"))
    .reindex(["Groen", "Oranje", "Rood"]).dropna(how="all").round(2)
)
"""
        ),
        code(
            """\
fig, ax = plt.subplots(figsize=(9, 5))
kleur = {"Groen": "seagreen", "Oranje": "darkorange", "Rood": "firebrick"}
for lab, g in r.groupby("cwf_betrouwbaarheid"):
    ax.scatter(g["Prognose_cwf"], g["breedte_rel"], s=18, alpha=0.6, color=kleur.get(lab, "gray"), label=lab)
ax.set_xscale("log")
ax.set_xlabel("Prognose (log-schaal)")
ax.set_ylabel("Breedte interval / prognose")
ax.set_title("Kleine opleidingen hebben relatief bredere intervallen")
ax.legend(title="Label")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """\
Kleine opleidingen schommelen relatief sterker van jaar tot jaar, dus hun interval is
relatief breder. Dat is geen fout van het model maar een eerlijke weergave van de
onzekerheid.

## 4. Exporteren naar Excel

De opdrachtregel schrijft `data/output/output_first-years_cwf.xlsx`. Hier maken we
een eigen export met een overzicht per opleiding (opgeteld over herkomst).
"""
        ),
        code(
            """\
output_path = Path("data/output/notebook_export_cwf.xlsx")
output_path.parent.mkdir(parents=True, exist_ok=True)

export = result.copy()
export.insert(1, "Opleidingsnaam", export["Croho groepeernaam"].map(NAMEN))
per_opleiding = (
    export.groupby(["Croho groepeernaam", "Opleidingsnaam", "Examentype"], as_index=False, dropna=False)
    [["Prognose_cwf", "Aantal_studenten"]].sum(min_count=1)
)
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    export.to_excel(writer, sheet_name="per_herkomst", index=False)
    per_opleiding.to_excel(writer, sheet_name="per_opleiding", index=False)

print(f"✅ Geschreven: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")
"""
        ),
        md(
            """\
## 5. Dashboard

Met `--dashboard` maakt de opdrachtregel er een interactief HTML-dashboard bij, met
prognose, interval en label per opleiding:

```bash
studentprognose -d cwf -y 2025 -w 12 --dashboard
```

Zie [Output begrijpen](../docs/output-begrijpen.md#interactief-dashboard).

## 6. Beslisregels

**Meer vertrouwen:**
- Label Groen of Oranje
- Smal interval in verhouding tot de prognose
- Later in de cyclus (hogere vulgraad)

**Minder vertrouwen:**
- Label Rood
- Nieuwe opleiding of weinig historie
- Numerus fixus (de capaciteit bepaalt de instroom, niet de aanmeldingen)
- Een jaar met een beleidswijziging of een ander aanmeldmoment
"""
        ),
    ]
    write("03_output_interpreteren.ipynb", cells, "Output interpreteren")


# ============================================================
# 04 — SARIMA (verdieping)
# ============================================================

def build_sarima():
    cells = [
        md(
            """\
# 04 — Verdieping: SARIMA

Deze notebook hoort bij [`docs/methodologie/sarima.md`](../docs/methodologie/sarima.md).
SARIMA is de tijdreeks in het cumulatieve en individuele spoor. Het voorspelt hoe de
aanmeldcurve na de peilweek verder loopt. We:

1. **Trainen en voorspellen** voor één opleiding/herkomst
2. Een **predictie-interval** (80% en 95%) plotten
3. De voorspelling naast de **werkelijke curve** leggen
4. De **aannames** op data checken
"""
        ),
        VERDIEPING_BANNER,
        DEMO_BANNER,
        SETUP_CELL,
        md(
            """\
## 1. Wat is SARIMA, in gewoon Nederlands

Aanmelddata heeft twee eigenschappen:

- **Een sterke jaarcyclus**: de zomer is rustig, januari–mei is druk.
- **Sterke samenhang van week tot week**: week T lijkt op week T-1.

SARIMA combineert beide:

$$\\text{SARIMA}(p,d,q)(P,D,Q)_{s}$$

| Component | Wat doet het? | Wat betekent het hier? |
|-----------|---------------|------------------------|
| `p` (AR) | Autoregressie op korte termijn | Hoe sterk hangt week T af van week T-1 |
| `d` (I)  | Differentiëren | Trend wegfilteren |
| `q` (MA) | Voortschrijdend gemiddelde van fouten | Dempt ruis |
| `(P,D,Q)_s` | Hetzelfde, op de seizoenslag | Vergelijk een week met dezelfde week vorig jaar |

De seizoenslengte `s` is het aantal weken per jaar in de data (meestal 52; minder
als er weken ontbreken). De ordes zijn **vast**:
"""
        ),
        code(
            """\
from studentprognose.utils.constants import (
    SARIMA_ORDER, SARIMA_SEASONAL_ORDER,
    SARIMA_ORDER_INDIVIDUAL, SARIMA_SEASONAL_ORDER_ALT,
)
print(f"Cumulatief spoor:  order={SARIMA_ORDER}, seasonal_order={SARIMA_SEASONAL_ORDER}")
print(f"Individueel spoor: order={SARIMA_ORDER_INDIVIDUAL}, seasonal_order={SARIMA_SEASONAL_ORDER_ALT}")
"""
        ),
        LOAD_CUMULATIVE_CELL,
        md(
            """\
## 2. Trainingsreeks bouwen

We plakken de jaren achter elkaar tot één lange reeks die eindigt op peilweek 12 van
2023. Weken ná de peilweek waren op dat moment nog niet bekend en gaan er vooraf uit.
Dit zijn dezelfde stappen als in de pipeline.
"""
        ),
        code(
            """\
from studentprognose.config import load_defaults
from studentprognose.data.transforms import drop_weeks_after_predict_week, ensure_start_week
from studentprognose.models.sarima import _get_transformed_data, split_time_series
from studentprognose.utils.weeks import get_all_weeks_valid
from statsforecast.models import ARIMA

PROGRAMMA = 56604  # B Psychologie
HERKOMST = "NL"
EXAMENTYPE = "Bachelor"
PREDICT_YEAR = 2023
PREDICT_WEEK = 12
MIN_TRAINING_YEAR = load_defaults()["model_config"].get("min_training_year", 2016)


def reeks_voor(df):
    wide = _get_transformed_data(df, min_training_year=MIN_TRAINING_YEAR)
    wide = wide[
        (wide["Croho groepeernaam"] == PROGRAMMA)
        & (wide["Herkomst"] == HERKOMST)
        & (wide["Examentype"] == EXAMENTYPE)
        & (wide["Collegejaar"] <= PREDICT_YEAR)
    ]
    return ensure_start_week(wide)


subset = reeks_voor(drop_weeks_after_predict_week(data_cumulative.copy(), PREDICT_YEAR, PREDICT_WEEK))
ts_data, target_weeks = split_time_series(subset, PREDICT_YEAR, PREDICT_WEEK)
PERIODE = len(get_all_weeks_valid(subset.columns))

print(f"Trainingsreeks: {ts_data.size} weken (~{ts_data.size / PERIODE:.1f} jaar van {PERIODE} weken)")
print(f"Voorspelhorizon: {len(target_weeks)} weken (week {target_weeks[0]} t/m {target_weeks[-1]})")
"""
        ),
        md(
            """\
## 3. Model en predictie-interval

We gebruiken `statsforecast.ARIMA` rechtstreeks (dezelfde basis als de pipeline), zodat
we naast de puntvoorspelling ook 80%- en 95%-intervallen krijgen.
"""
        ),
        code(
            """\
model = ARIMA(order=SARIMA_ORDER, season_length=PERIODE, seasonal_order=SARIMA_SEASONAL_ORDER[:3])
model.fit(y=ts_data)
fc = model.predict(h=len(target_weeks), level=[80, 95])

forecast_table = pd.DataFrame({
    "week": target_weeks,
    "voorspelling": np.asarray(fc["mean"]).round(1),
    "lo 95%": np.asarray(fc["lo-95"]).round(1),
    "hi 95%": np.asarray(fc["hi-95"]).round(1),
})
forecast_table.head(10)
"""
        ),
        md(
            """\
## 4. Voorspelling naast de werkelijke curve

We tekenen de eerdere jaren (grijs), 2023 tot de peilweek (blauw), de voorspelling met
intervallen (oranje) en de **werkelijke** curve van 2023 na de peilweek (zwart).

> ℹ️ SARIMA voorspelt de *aanmeldcurve*, niet direct het aantal *eerstejaars*. De
> XGBoost-regressor vertaalt de curve naar een prognose; zie
> [`05_xgboost.ipynb`](05_xgboost.ipynb).
"""
        ),
        code(
            """\
volledig = reeks_voor(data_cumulative.copy())
weekcols = get_all_weeks_valid(volledig.columns)
werkelijk_2023 = volledig[volledig["Collegejaar"] == PREDICT_YEAR][weekcols].iloc[0].astype(float)

n_jaren = ts_data.size // PERIODE
start_vj = n_jaren * PERIODE
x_fc = np.arange(PERIODE - len(target_weeks), PERIODE)

fig, ax = plt.subplots(figsize=(12, 5))
for j in range(n_jaren):
    ax.plot(range(PERIODE), ts_data[j * PERIODE:(j + 1) * PERIODE], color="lightgray",
            alpha=0.6, label="Eerdere jaren" if j == 0 else None)
ax.plot(range(ts_data.size - start_vj), ts_data[start_vj:], color="steelblue", linewidth=2,
        label=f"{PREDICT_YEAR} t/m peilweek {PREDICT_WEEK}")
ax.plot(x_fc, werkelijk_2023[target_weeks].values, color="black", linewidth=1.5,
        label=f"Werkelijk {PREDICT_YEAR}")
ax.fill_between(x_fc, fc["lo-95"], fc["hi-95"], alpha=0.15, color="darkorange", label="95% interval")
ax.fill_between(x_fc, fc["lo-80"], fc["hi-80"], alpha=0.25, color="darkorange", label="80% interval")
ax.plot(x_fc, fc["mean"], color="darkorange", linewidth=2.5, linestyle="--", label="Voorspelling")
ax.set_xticks(range(0, PERIODE, 4))
ax.set_xticklabels([weekcols[i] for i in range(0, PERIODE, 4)])
ax.set_xlabel("Weeknummer (academische volgorde)")
ax.set_ylabel("Pijplijn (gewogen vooraanmelders + inschrijvingen)")
ax.set_title(f"SARIMA · {label(PROGRAMMA, NAMEN)} · {HERKOMST} · peilweek {PREDICT_WEEK} ({PREDICT_YEAR})")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.show()

binnen = ((werkelijk_2023[target_weeks].values >= fc["lo-95"]) & (werkelijk_2023[target_weeks].values <= fc["hi-95"])).mean()
print(f"Deel van de weken waarin de werkelijke curve binnen het 95%-interval lag: {binnen:.0%}")
"""
        ),
        md(
            """\
De daling rond week 23 is een eigenschap van de demodata. Omdat hij elk jaar in
dezelfde week zit, neemt SARIMA hem over: dat is precies wat een seizoensmodel doet.
Het laat ook de zwakte zien: SARIMA herhaalt het patroon van eerdere jaren, ook als
dat patroon dit jaar zou veranderen.

## 5. Aannames op data testen
"""
        ),
        code(
            """\
# Aanname 1: het seizoenspatroon herhaalt zich van jaar tot jaar
jaren = sorted(subset["Collegejaar"].astype(int))
curves = ts_data[: n_jaren * PERIODE].reshape(n_jaren, PERIODE)
correlaties = [np.corrcoef(curves[i], curves[i + 1])[0, 1] for i in range(n_jaren - 1)]

pd.DataFrame({
    "Jaarpaar": [f"{jaren[i]} ↔ {jaren[i + 1]}" for i in range(len(correlaties))],
    "Pearson r": np.round(correlaties, 3),
    "Beoordeling": ["✅ sterk (≥ 0,95)" if c >= 0.95 else "⚠️ matig (0,70–0,95)" if c >= 0.7 else "❌ zwak (< 0,70)"
                    for c in correlaties],
})
"""
        ),
        code(
            """\
# Aanname 2: genoeg historische jaren per opleiding
counts = (
    data_cumulative[data_cumulative["Examentype"] == "Bachelor"]
    .groupby("Croho groepeernaam")["Collegejaar"].nunique()
)
te_kort = counts[counts < 3]
print(f"Bachelor-opleidingen met < 3 jaar historie: {len(te_kort)} van {len(counts)}")
"""
        ),
        md(
            """\
## 6. Wanneer vertrouw je het niet?

| Situatie | Wat te doen? |
|----------|--------------|
| **Pearson r < 0,7** tussen opeenvolgende jaren | Beoordeel de prognose kritisch; vergelijk met het CWF-model |
| **< 3 jaar historie** | SARIMA heeft te weinig seizoenen; gebruik het CWF-model, dat leent van vergelijkbare opleidingen |
| **Uitzonderingsjaar** (bijv. corona) | Sluit het uit via `excluded_data_points` in `configuration.json` |
| **Peilweek < 6** | Lange horizon; rapporteer een bandbreedte, niet één getal |

Vergelijk tijdreeksmodellen op je eigen data met `studentprognose benchmark -d c -w <week>`;
zie [`docs/methodologie/benchmarks.md`](../docs/methodologie/benchmarks.md).
"""
        ),
    ]
    write("04_sarima.ipynb", cells, "SARIMA")


# ============================================================
# 05 — XGBoost (verdieping)
# ============================================================

def build_xgboost():
    cells = [
        md(
            """\
# 05 — Verdieping: XGBoost (classifier én regressor)

Deze notebook hoort bij [`docs/methodologie/xgboost.md`](../docs/methodologie/xgboost.md)
en [`docs/methodologie/individueel.md`](../docs/methodologie/individueel.md).
XGBoost wordt op twee manieren ingezet:

| Rol | Spoor | Input | Output |
|-----|-------|-------|--------|
| **Classifier** | Individueel (`-d i`) | aanmeldingen per student | kans dat één student zich inschrijft |
| **Regressor** | Cumulatief (`-d c`) | aanmeldcurve per opleiding | verwacht aantal eerstejaars |

We bouwen vereenvoudigde, werkende versies van beide en vergelijken met de pipeline.
"""
        ),
        VERDIEPING_BANNER,
        DEMO_BANNER,
        SETUP_CELL,
        md(
            """\
## DEEL A — Classifier (individueel spoor)

### A.1 Data en label

Deel A gebruikt de **individuele aanmelddata**. Die zit niet in de repo; je krijgt hem
met `studentprognose init` (kies "j" bij de demodata). De helper zoekt in
`data/input/vooraanmeldingen_individueel.csv` en `data/input_raw/individuele_aanmelddata.csv`.

Elke aanmelding krijgt een label: **1** = ingeschreven, **0** = niet ingeschreven.
"""
        ),
        code(
            """\
from _helpers import load_individueel

ind = load_individueel(n_sample=None)
if ind is None:
    raise FileNotFoundError(
        "Geen individuele aanmelddata gevonden. Draai `studentprognose init` in de "
        "projectmap en kies 'j' bij de demodata, of sla deel A over."
    )
print(f"Aanmeldingen:  {len(ind):,}")
print(f"Opleidingen:   {ind['Croho groepeernaam'].nunique()}")
print(f"Collegejaren:  {ind['Collegejaar'].min()}–{ind['Collegejaar'].max()}")

INGESCHREVEN_STATUSSEN = {"Ingeschreven", "Uitgeschreven"}
ind = ind.dropna(subset=["Inschrijfstatus", "Croho groepeernaam", "Examentype"]).copy()
ind["label"] = ind["Inschrijfstatus"].isin(INGESCHREVEN_STATUSSEN).astype(int)
print(f"Deel dat zich inschrijft: {ind['label'].mean():.1%}  (de naïeve voorspelling)")
"""
        ),
        md(
            """\
### A.2 Trainen en voorspellen

We trainen op alle jaren behalve het laatste en voorspellen het laatste jaar.
`n_estimators=200`, `max_depth=4` (ondiepe bomen, minder overfitting),
`learning_rate=0.1`.
"""
        ),
        code(
            """\
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

cat_features = [c for c in ["Examentype", "Faculteit", "Croho groepeernaam", "EER",
                            "Type vooropleiding", "Nationaliteit", "Geslacht"] if c in ind.columns]
num_features = ["Collegejaar"]

LAATSTE_JAAR = int(ind["Collegejaar"].max())
train = ind[ind["Collegejaar"] < LAATSTE_JAAR]
test = ind[ind["Collegejaar"] == LAATSTE_JAAR]

clf = SkPipeline([
    ("pre", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ("num", "passthrough", num_features),
    ])),
    ("xgb", XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                          random_state=42, eval_metric="logloss", n_jobs=2)),
])
X_train = train[cat_features + num_features].astype({c: str for c in cat_features}).fillna("ONBEKEND")
X_test = test[cat_features + num_features].astype({c: str for c in cat_features}).fillna("ONBEKEND")
clf.fit(X_train, train["label"])
proba = clf.predict_proba(X_test)[:, 1]

verwacht, werkelijk = float(proba.sum()), float(test["label"].sum())
print(f"Testjaar:  {LAATSTE_JAAR} ({len(test):,} aanmeldingen)")
print(f"Accuracy:  {accuracy_score(test['label'], proba >= 0.5):.3f}  (naïef: {max(test['label'].mean(), 1 - test['label'].mean()):.3f})")
print(f"AUC-ROC:   {roc_auc_score(test['label'], proba):.3f}  (1,0 = perfect, 0,5 = gokken)")
print(f"Cohort:    verwacht {verwacht:.0f}, werkelijk {werkelijk:.0f} ({(verwacht - werkelijk) / werkelijk:+.1%})")
"""
        ),
        md(
            """\
> 📌 **Over de AUC:** de demodata is synthetisch en de beschikbare kenmerken zeggen
> weinig over wie zich inschrijft. Met echte data (vooropleiding, cijfers,
> studiekeuzecheck) ligt de AUC doorgaans hoger. Waar het om gaat is het
> **cohort**: de som van de kansen.

### A.3 Van kans per student naar cohort

Voor een opleiding tel je de kansen op. Dat is precies wat de pipeline doet.
"""
        ),
        code(
            """\
t = test.assign(kans=proba)
per_opleiding = (
    t.groupby("Croho groepeernaam")
    .agg(verwacht=("kans", "sum"), werkelijk=("label", "sum"), aanmeldingen=("label", "size"))
    .round(0)
)
per_opleiding["fout_%"] = ((per_opleiding["verwacht"] - per_opleiding["werkelijk"])
                           / per_opleiding["werkelijk"].replace(0, np.nan) * 100).round(1)
per_opleiding.sort_values("werkelijk", ascending=False).head(10)
"""
        ),
        md(
            """\
## DEEL B — Regressor (cumulatief spoor)

### B.1 Kenmerken

De regressor leert per opleiding × herkomst × examentype × jaar op basis van de
aanmeldcurve tot de peilweek, plus een paar afgeleide kenmerken:

| Type | Kenmerken |
|------|-----------|
| **Weken** | gewogen vooraanmelders per week tot de peilweek |
| **Vertraagd** | `Gewogen_t-2`, `Gewogen_t-5` |
| **Versnelling** | `(huidig − t-2) − (t-2 − t-5)` |
| **Exclusiviteit** | `exclusivity_ratio` = aanmelders met één aanmelding / alle aanmelders (in de demodata altijd 0) |
| **Categorisch** | `Examentype`, `Croho groepeernaam`, `Herkomst` |
"""
        ),
        LOAD_CUMULATIVE_CELL,
        code(
            """\
from studentprognose.data.transforms import transform_data
from studentprognose.utils.weeks import get_all_weeks_valid

PREDICT_WEEK = 12
SLEUTELS = ["Collegejaar", "Faculteit", "Herkomst", "Examentype", "Croho groepeernaam"]

df = data_cumulative[data_cumulative["Collegejaar"] >= 2018]
wide = transform_data(df, "Gewogen vooraanmelders")
feature_weeks = [w for w in get_all_weeks_valid(wide.columns) if 1 <= int(w) <= PREDICT_WEEK]

for lag in (2, 5):
    ref = (
        df[df["Weeknummer"] == max(PREDICT_WEEK - lag, 1)]
        .groupby(SLEUTELS)["Gewogen vooraanmelders"].sum()
        .rename(f"Gewogen_t-{lag}").reset_index()
    )
    wide = wide.merge(ref, on=SLEUTELS, how="left")
wide["Gewogen_acceleration"] = (
    (wide[str(PREDICT_WEEK)] - wide["Gewogen_t-2"]) - (wide["Gewogen_t-2"] - wide["Gewogen_t-5"])
)

# Exclusiviteit: welk deel van de aanmelders zich alleen voor deze opleiding aanmeldde
peil = df[df["Weeknummer"] == PREDICT_WEEK].groupby(SLEUTELS)[
    ["Aantal aanmelders met 1 aanmelding", "Ongewogen vooraanmelders"]].sum()
peil["exclusivity_ratio"] = peil["Aantal aanmelders met 1 aanmelding"] / (peil["Ongewogen vooraanmelders"] + 1e-8)
wide = wide.merge(peil[["exclusivity_ratio"]].reset_index(), on=SLEUTELS, how="left")
print(f"Kenmerkenmatrix: {wide.shape[0]} rijen × {wide.shape[1]} kolommen")
"""
        ),
        md(
            """\
### B.2 Trainen en testen

We trainen op alle jaren vóór het laatste jaar met realisatie en testen op dat jaar.
"""
        ),
        code(
            """\
from xgboost import XGBRegressor

training = wide.merge(
    data_studentcount.rename(columns={"Aantal_studenten": "y"})[
        ["Collegejaar", "Croho groepeernaam", "Herkomst", "Examentype", "y"]],
    on=["Collegejaar", "Croho groepeernaam", "Herkomst", "Examentype"], how="inner",
).dropna(subset=["y"])

LATEST = int(training["Collegejaar"].max())
train_xgb = training[training["Collegejaar"] < LATEST].copy()
test_xgb = training[training["Collegejaar"] == LATEST].copy()

cat_r = ["Examentype", "Croho groepeernaam", "Herkomst"]
num_r = feature_weeks + ["Gewogen_t-2", "Gewogen_t-5", "Gewogen_acceleration", "exclusivity_ratio", "Collegejaar"]
reg = SkPipeline([
    ("pre", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_r),
        ("num", "passthrough", num_r),
    ])),
    ("xgb", XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=2)),
])
X_tr = train_xgb[cat_r + num_r].astype({"Croho groepeernaam": str}).fillna(0)
X_te = test_xgb[cat_r + num_r].astype({"Croho groepeernaam": str}).fillna(0)
reg.fit(X_tr, train_xgb["y"])
preds = reg.predict(X_te)

wape = float(np.abs(preds - test_xgb["y"]).sum() / test_xgb["y"].sum())
print(f"Trainjaren: {sorted(int(j) for j in train_xgb['Collegejaar'].unique())}")
print(f"Testjaar:   {LATEST} ({len(test_xgb)} rijen)")
print(f"WAPE:       {wape:.1%}")
"""
        ),
        md(
            """\
### B.3 Welke kenmerken tellen?

XGBoost geeft per (one-hot) kenmerk een score. We tellen die per oorspronkelijke kolom
op, zoals de pipeline dat ook doet.
"""
        ),
        code(
            """\
namen_enc = reg.named_steps["pre"].get_feature_names_out()
records = []
for naam, imp in zip(namen_enc, reg.named_steps["xgb"].feature_importances_):
    kaal = naam.split("__", 1)[1] if "__" in naam else naam
    basis = next((c for c in cat_r if kaal.startswith(c + "_")), kaal)
    records.append({"kenmerk": basis, "importance": imp})

imp_df = (pd.DataFrame(records).groupby("kenmerk", as_index=False)["importance"].sum()
          .sort_values("importance").tail(12))
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(imp_df["kenmerk"], imp_df["importance"], color="steelblue")
ax.set_xlabel("Som van importance (gegroepeerd)")
ax.set_title("Kenmerkbelang — XGBoost-regressor")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """\
### B.4 Vergelijking met de pipeline en het CWF-model

We draaien de cumulatieve pipeline (SARIMA + XGBoost) en het CWF-model voor
hetzelfde jaar en dezelfde peilweek, op een selectie van de grootste opleidingen.
"""
        ),
        SELECTIE_CELL,
        code(
            """\
from studentprognose import run_pipeline_from_dataframes, DataOption
from _helpers import suppress_stdout

runs = {}
for naam, optie in [("cumulatief", DataOption.CUMULATIVE), ("cwf", DataOption.CWF)]:
    with suppress_stdout():
        runs[naam] = run_pipeline_from_dataframes(
            year=LATEST, week=PREDICT_WEEK,
            data_cumulative=data_cumulative_raw, data_student_numbers=data_studentcount,
            dataset=optie, filtering=FILTERING, save_output=False,
        )

sleutel = ["Croho groepeernaam", "Herkomst", "Examentype"]
cum = runs["cumulatief"]
cum = cum[cum["Weeknummer"] == PREDICT_WEEK][sleutel + ["SARIMA_cumulative"]]
cwf = runs["cwf"][sleutel + ["Prognose_cwf"]]
eigen = test_xgb[sleutel].assign(XGB_notebook=preds)
real = data_studentcount[data_studentcount["Collegejaar"] == LATEST][sleutel + ["Aantal_studenten"]]

vergelijk = (real.merge(cum, on=sleutel).merge(cwf, on=sleutel).merge(eigen, on=sleutel)
             .query("`Croho groepeernaam` in @SELECTIE"))


def wape_kolom(k):
    return float((vergelijk[k] - vergelijk["Aantal_studenten"]).abs().sum() / vergelijk["Aantal_studenten"].sum())


pd.DataFrame({
    "Model": ["XGBoost (deze notebook)", "Cumulatief spoor (SARIMA + XGBoost)", "CWF-model"],
    "WAPE": [f"{wape_kolom(k):.1%}" for k in ["XGB_notebook", "SARIMA_cumulative", "Prognose_cwf"]],
    "Combinaties": len(vergelijk),
})
"""
        ),
        md(
            """\
## Conclusie

- **Classifier**: kans per student, opgeteld tot een cohort. Beoordeel hem op het
  cohort, niet op losse studenten.
- **Regressor**: leert op opleidingsniveau uit de aanmeldcurve.
- **Welk regressiemodel?** Bij weinig data werken Ridge of Random Forest soms beter;
  kies via `model_config.cumulative_regressor` in `configuration.json` en vergelijk met
  `studentprognose benchmark -d c`.
- **Ten opzichte van CWF**: vergelijk de WAPE hierboven. Het CWF-model geeft er bovendien
  een interval en label bij.
"""
        ),
    ]
    write("05_xgboost.ipynb", cells, "XGBoost")


# ============================================================
# 06 — Ratio-model (verdieping)
# ============================================================

def build_ratio():
    cells = [
        md(
            """\
# 06 — Verdieping: het ratio-model

Deze notebook hoort bij [`docs/methodologie/ratio-model.md`](../docs/methodologie/ratio-model.md).
Het ratio-model is de **simpelste** voorspeller in het cumulatieve spoor: volledig
transparant, altijd beschikbaar, en daarom de baseline.

$$
\\bar{R}_t = \\frac{1}{3} \\sum_{j=1}^{3} \\frac{\\text{Aanmelding}_{jaar-j,\\, week=t}}{\\text{Aantal studenten}_{jaar-j}}
\\quad\\Rightarrow\\quad
\\hat{y} = \\frac{\\text{Aanmelding}_{huidig,\\, week=t}}{\\bar{R}_t}
$$

Met `Aanmelding = Ongewogen vooraanmelders + Inschrijvingen` en een venster van 3 jaar
(`LOOKBACK_YEARS`).

Het [CWF-model](02_cwf_model.ipynb) bouwt op hetzelfde idee voort, maar weegt recente
jaren zwaarder, combineert met een prior en geeft een interval.
"""
        ),
        VERDIEPING_BANNER,
        DEMO_BANNER,
        SETUP_CELL,
        LOAD_CUMULATIVE_CELL,
        md(
            """\
## 1. Ratio stap voor stap, voorspelling voor 2023
"""
        ),
        code(
            """\
PROGRAMMA = 56604  # B Psychologie
HERKOMST = "NL"
EXAMENTYPE = "Bachelor"
PREDICT_YEAR = 2023
PREDICT_WEEK = 12
LOOKBACK = 3

combo = data_cumulative[
    (data_cumulative["Croho groepeernaam"] == PROGRAMMA)
    & (data_cumulative["Herkomst"] == HERKOMST)
    & (data_cumulative["Examentype"] == EXAMENTYPE)
].copy()
combo["Aanmelding"] = combo["Ongewogen vooraanmelders"].fillna(0) + combo["Inschrijvingen"].fillna(0)
combo = combo.merge(data_studentcount, on=["Collegejaar", "Croho groepeernaam", "Herkomst", "Examentype"], how="left")
combo["Ratio"] = combo["Aanmelding"] / combo["Aantal_studenten"]


def ratio_prognose(jaar: int, week: int = PREDICT_WEEK):
    hist = combo[(combo["Weeknummer"] == week) & combo["Collegejaar"].between(jaar - LOOKBACK, jaar - 1)]
    nu = combo[(combo["Weeknummer"] == week) & (combo["Collegejaar"] == jaar)]
    gem_ratio = hist["Ratio"].mean()
    if hist.empty or nu.empty or not np.isfinite(gem_ratio) or gem_ratio == 0:
        return np.nan, hist
    return float(nu["Aanmelding"].sum() / gem_ratio), hist


prognose, hist = ratio_prognose(PREDICT_YEAR)
werkelijk = float(combo.loc[combo["Collegejaar"] == PREDICT_YEAR, "Aantal_studenten"].dropna().iloc[0])
print(f"{label(PROGRAMMA, NAMEN)} · {HERKOMST} · peilweek {PREDICT_WEEK}")
print(hist[["Collegejaar", "Aanmelding", "Aantal_studenten", "Ratio"]].round(3).to_string(index=False))
print(f"Gemiddelde ratio: {hist['Ratio'].mean():.3f}")
print(f"Prognose {PREDICT_YEAR}: {prognose:.0f} · realisatie: {werkelijk:.0f} · fout: {(prognose - werkelijk) / werkelijk:+.1%}")
"""
        ),
        md(
            """\
## 2. Backtest over meerdere jaren

Eén jaar zegt weinig. We voorspellen elk jaar waarvoor drie eerdere jaren bestaan.
"""
        ),
        code(
            """\
records = []
for jaar in sorted(combo["Collegejaar"].unique()):
    p, _ = ratio_prognose(int(jaar))
    r = combo.loc[combo["Collegejaar"] == jaar, "Aantal_studenten"].dropna()
    if np.isfinite(p) and len(r):
        r = float(r.iloc[0])
        records.append({"jaar": int(jaar), "prognose": round(p), "realisatie": int(r),
                        "fout_%": round((p - r) / r * 100, 1)})
backtest = pd.DataFrame(records)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(backtest["jaar"], backtest["realisatie"], "o-", color="black", label="Realisatie")
ax.plot(backtest["jaar"], backtest["prognose"], "s--", color="darkgreen", label=f"Ratio-prognose (wk {PREDICT_WEEK})")
for _, r in backtest.iterrows():
    ax.annotate(f"{r['fout_%']:+.0f}%", (r["jaar"], r["prognose"]), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=8, color="darkgreen")
ax.set_xlabel("Collegejaar")
ax.set_ylabel("Eerstejaars")
ax.set_title(f"Backtest ratio-model · {label(PROGRAMMA, NAMEN)} · {HERKOMST}")
ax.legend()
plt.tight_layout()
plt.show()
backtest
"""
        ),
        md(
            """\
## 3. Ratio per peilweek: wanneer is hij stabiel?

Vroeg in het jaar is de ratio **onrustig**: een handvol aanmelders meer of minder
slaat hard door. Later in het seizoen wordt hij stabieler.
"""
        ),
        code(
            """\
from studentprognose.utils.weeks import get_all_weeks_ordered

volgorde = get_all_weeks_ordered()
vensterjaren = combo[combo["Collegejaar"].between(PREDICT_YEAR - LOOKBACK, PREDICT_YEAR - 1)]
per_week = (
    vensterjaren.replace([np.inf, -np.inf], np.nan)
    .groupby("Weeknummer")["Ratio"].agg(["mean", "std"]).reset_index()
)
per_week["positie"] = per_week["Weeknummer"].astype(str).map({w: i for i, w in enumerate(volgorde)})
per_week = per_week.dropna(subset=["positie"]).sort_values("positie")

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.errorbar(per_week["positie"], per_week["mean"], yerr=per_week["std"], fmt="o-",
            color="darkgreen", ecolor="lightgreen", capsize=3, label="Ratio ± 1σ (over 3 jaar)")
ax.axvline(volgorde.index(str(PREDICT_WEEK)), color="gray", linestyle="--", label=f"Peilweek {PREDICT_WEEK}")
ax.set_xticks(range(0, len(volgorde), 4))
ax.set_xticklabels([volgorde[i] for i in range(0, len(volgorde), 4)])
ax.set_xlabel("Weeknummer (academische volgorde: 39 → 38)")
ax.set_ylabel("Aanmelding / Aantal_studenten")
ax.set_title(f"Ratio per peilweek · {label(PROGRAMMA, NAMEN)} · {HERKOMST}")
ax.legend()
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """\
## 4. Numerus fixus: afkappen op de capaciteit

Voor numerus-fixusopleidingen kapt het ratio-model de opgetelde prognose af op de
capaciteit; het overschot gaat van de NL-herkomst af. We laten dat zien op
B Geneeskunde (Isatcode `56551`) met een voorbeeldcapaciteit van 340. In je eigen
configuratie zet je `"numerus_fixus": {"56551": 340}`.
"""
        ),
        code(
            """\
NF_CODE = 56551  # B Geneeskunde
NF_CAPACITEIT = 340

nf = data_cumulative[(data_cumulative["Croho groepeernaam"] == NF_CODE)
                     & (data_cumulative["Examentype"] == "Bachelor")].copy()
nf["Aanmelding"] = nf["Ongewogen vooraanmelders"].fillna(0) + nf["Inschrijvingen"].fillna(0)
nf = nf.merge(data_studentcount, on=["Collegejaar", "Croho groepeernaam", "Herkomst", "Examentype"], how="left")

rijen = []
for herkomst in ["NL", "EER", "Niet-EER"]:
    h = nf[(nf["Herkomst"] == herkomst) & (nf["Weeknummer"] == PREDICT_WEEK)]
    hist_h = h[h["Collegejaar"].between(PREDICT_YEAR - LOOKBACK, PREDICT_YEAR - 1)]
    gem = (hist_h["Aanmelding"] / hist_h["Aantal_studenten"]).replace([np.inf, -np.inf], np.nan).mean()
    nu = h.loc[h["Collegejaar"] == PREDICT_YEAR, "Aanmelding"].sum()
    if np.isfinite(gem) and gem > 0:
        rijen.append({"Herkomst": herkomst, "Ratio-prognose": round(nu / gem)})

nf_df = pd.DataFrame(rijen)
overschot = max(0, nf_df["Ratio-prognose"].sum() - NF_CAPACITEIT)
nf_df["Na afkappen"] = nf_df["Ratio-prognose"]
nf_df.loc[nf_df["Herkomst"] == "NL", "Na afkappen"] = (nf_df.loc[nf_df["Herkomst"] == "NL", "Ratio-prognose"] - overschot).clip(lower=0)

print(f"{label(NF_CODE, NAMEN)} · capaciteit {NF_CAPACITEIT}")
print(f"Totaal vóór afkappen: {nf_df['Ratio-prognose'].sum():.0f} · overschot: {overschot:.0f} (van NL)")
nf_df
"""
        ),
        md(
            """\
## 5. Beperkingen

| Beperking | Waar zie je het? |
|-----------|------------------|
| **Geen trends** | Bij groeiende of krimpende opleidingen lopen de fouten in de backtest op |
| **Gevoelig voor uitschieters** | Eén afwijkend jaar in het venster trekt de gemiddelde ratio mee |
| **Week-afhankelijk** | Vroege weken: brede σ-band in de grafiek hierboven |

Het ratio-model is een **eerlijke baseline**. Het CWF-model pakt de eerste twee
beperkingen aan: recente jaren wegen zwaarder, een geometrisch gemiddelde dempt
uitschieters, en het interval maakt de onzekerheid zichtbaar.
"""
        ),
    ]
    write("06_ratio_model.ipynb", cells, "Ratio-model")


# ============================================================
# 07 — Ensemble (verdieping)
# ============================================================

def build_ensemble():
    cells = [
        md(
            """\
# 07 — Verdieping: het ensemble

Deze notebook hoort bij [`docs/methodologie/ensemble.md`](../docs/methodologie/ensemble.md).
In `-d b` combineert de pipeline het individuele en het cumulatieve spoor met
configureerbare gewichten. We tonen het idee met twee voorspellingen uit het
cumulatieve spoor (SARIMA en ratio), en testen of een combinatie beter is dan elk
afzonderlijk.

> ℹ️ Het volledige `-d b`-ensemble heeft ook individuele aanmelddata nodig en bepaalt
> de gewichten per opleiding via grid search. Hier doen we een vereenvoudigde variant
> met één globaal gewicht.
"""
        ),
        VERDIEPING_BANNER,
        DEMO_BANNER,
        SETUP_CELL,
        md(
            """\
## 1. Geconfigureerde gewichten

De gewichten staan in `configuration.json`, per situatie, voor de combinatie
individueel × cumulatief.
"""
        ),
        code(
            """\
from studentprognose.config import load_defaults

for key, w in load_defaults().get("ensemble_weights", {}).items():
    print(f"  {key:20s}  individueel={w['individual']:.2f}  cumulatief={w['cumulative']:.2f}")
"""
        ),
        LOAD_CUMULATIVE_CELL,
        md(
            """\
## 2. Voorspellingen genereren

We draaien het cumulatieve spoor voor 2023, peilweek 12, op een selectie van de
grootste opleidingen.
"""
        ),
        SELECTIE_CELL,
        code(
            """\
from studentprognose import run_pipeline_from_dataframes, DataOption
from _helpers import suppress_stdout

PREDICT_YEAR = 2023
PREDICT_WEEK = 12

with suppress_stdout():
    out = run_pipeline_from_dataframes(
        year=PREDICT_YEAR, week=PREDICT_WEEK,
        data_cumulative=data_cumulative_raw, data_student_numbers=data_studentcount,
        dataset=DataOption.CUMULATIVE, filtering=FILTERING, save_output=False,
    )

sleutel = ["Croho groepeernaam", "Herkomst", "Examentype"]
df = (
    out[out["Weeknummer"] == PREDICT_WEEK][sleutel + ["SARIMA_cumulative", "Prognose_ratio"]]
    .merge(data_studentcount[data_studentcount["Collegejaar"] == PREDICT_YEAR][sleutel + ["Aantal_studenten"]],
           on=sleutel)
    .rename(columns={"Aantal_studenten": "Realisatie"})
    .dropna()
)
print(f"{len(df)} combinaties met beide voorspellingen en realisatie")
df.head()
"""
        ),
        md(
            """\
## 3. Gewicht laten variëren

We combineren `w × SARIMA + (1 − w) × ratio` en kijken welke `w` de kleinste fout geeft.
"""
        ),
        code(
            """\
def wape(pred):
    return float((pred - df["Realisatie"]).abs().sum() / df["Realisatie"].sum())


sweep = pd.DataFrame([
    {"w_sarima": w, "WAPE": wape(w * df["SARIMA_cumulative"] + (1 - w) * df["Prognose_ratio"])}
    for w in np.linspace(0, 1, 11)
])
opt = sweep.loc[sweep["WAPE"].idxmin()]

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(sweep["w_sarima"], sweep["WAPE"], "o-", color="purple")
ax.scatter([opt["w_sarima"]], [opt["WAPE"]], s=200, marker="*", color="red", zorder=5,
           label=f"Beste gewicht: {opt['w_sarima']:.1f}")
ax.set_xlabel("Gewicht SARIMA (rest = ratio)")
ax.set_ylabel("WAPE")
ax.set_title(f"Ensemble-gewicht vs. fout · {PREDICT_YEAR} · peilweek {PREDICT_WEEK}")
ax.legend()
plt.tight_layout()
plt.show()
sweep.round(3)
"""
        ),
        md(
            """\
## 4. Onenigheid tussen modellen als waarschuwing

Een ensemble wint zelden van het beste losse model voor één opleiding: het is een
gemiddelde. De waarde zit in **stabiliteit**. Lopen de modellen ver uiteen, dan is dat
een signaal dat er één naast zit.
"""
        ),
        code(
            """\
df["spreiding_%"] = ((df["SARIMA_cumulative"] - df["Prognose_ratio"]).abs() / df["Realisatie"] * 100).round(1)
df["beste"] = np.where(
    (df["SARIMA_cumulative"] - df["Realisatie"]).abs() < (df["Prognose_ratio"] - df["Realisatie"]).abs(),
    "SARIMA", "ratio",
)
df.assign(opleiding=[label(c, NAMEN, 30) for c in df["Croho groepeernaam"]])[
    ["opleiding", "Herkomst", "Realisatie", "SARIMA_cumulative", "Prognose_ratio", "spreiding_%", "beste"]
].sort_values("spreiding_%", ascending=False).head(8)
"""
        ),
        md(
            """\
## 5. Gewichten in productie

In productie bepaalt `scripts/calculate_ensemble_weights.py` de gewichten per
**opleiding × examentype × herkomst**, niet één globaal getal.

> ⚠️ **Gewichten verouderen.** Een model dat eerder goed presteerde kan na een
> beleidswijziging of uitzonderlijk jaar achterlopen. Herbereken ze elk studiejaar.

Het [CWF-model](02_cwf_model.ipynb) heeft geen gewichten nodig: het weegt per
combinatie zelf af tussen ratiosignaal en prior, en geeft er een interval bij.
"""
        ),
    ]
    write("07_ensemble.ipynb", cells, "Ensemble")


# ============================================================
# Main
# ============================================================


def main():
    print("Notebooks bouwen…")
    for naam in VEROUDERD:
        pad = ROOT / naam
        if pad.exists():
            pad.unlink()
            print(f"  verwijderd {pad.relative_to(ROOT.parent)} (oude indeling)")
    build_overzicht()
    build_data_voorbereiden()
    build_cwf()
    build_output()
    build_sarima()
    build_xgboost()
    build_ratio()
    build_ensemble()
    print("Klaar.")


if __name__ == "__main__":
    main()
