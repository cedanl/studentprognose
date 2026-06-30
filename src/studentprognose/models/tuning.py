"""Tijd-bewuste hyperparameter tuning voor het cumulatieve regressiemodel.

Het cumulatieve spoor (stap 2) traint een regressor om ``Aantal_studenten`` te
voorspellen uit de cumulatieve vooraanmeldcurve. Dit module zoekt — via een
kleine, regularisatie-gerichte grid search — de hyperparameters die het beste
presteren op de eigen historie van een instelling.

Belangrijke ontwerpkeuzes:

* **Tijd-bewust, geen random k-fold.** De evaluatie gebruikt
  :func:`~studentprognose.benchmark.splitter.time_series_split`: train op jaren
  ``≤ N``, valideer op jaar ``N+1``. Random k-fold zou toekomstige jaren in de
  trainingsset lekken en de scores te optimistisch maken.
* **Hergebruik van de benchmark-evaluatie.** Elke kandidaat-parameterset wordt
  beoordeeld met :func:`~studentprognose.benchmark.evaluate_regressor.evaluate_regressor_model`,
  exact dezelfde per-fold-pipeline (preprocessing, NF-splitsing, metrics) als de
  ``benchmark``-stap. De geselecteerde MAPE is daardoor vergelijkbaar.
* **Klein, geen Optuna.** Bij de datagrootte van dit probleem (één rij per
  ``programma × jaar × herkomst × examentype``, over een handvol jaren) volstaat
  een kleine grid; een zwaardere optimizer voegt overfitting-risico toe zonder
  betrouwbare winst.
"""

import itertools
import json

import numpy as np
import pandas as pd

from studentprognose.benchmark.evaluate_regressor import evaluate_regressor_model
from studentprognose.models import REGRESSOR_REGISTRY


# Kleine, regularisatie-gerichte zoekruimtes per regressor. Bewust compact
# gehouden: met weinig trainingsjaren is elke validatie-split klein, dus een
# brede grid kiest sneller ruis dan signaal.
DEFAULT_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "xgboost": {
        "learning_rate": [0.05, 0.1, 0.25],
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
    },
    "ridge": {
        "alpha": [0.1, 1.0, 10.0],
    },
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
    },
    "gradient_boosting": {
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
    },
    "extra_trees": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
    },
}


def expand_grid(grid: dict[str, list]) -> list[dict]:
    """Vouw een parametergrid uit tot een lijst van concrete parameter-dicts.

    Een lege grid levert ``[{}]`` (één kandidaat met modeldefaults), zodat de
    aanroeper altijd minstens de defaults evalueert.

    Args:
        grid: Mapping van parameternaam naar een lijst van te proberen waarden.

    Returns:
        Lijst van parameter-dicts — het cartesisch product over alle waarden.
    """
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def tune_regressor(
    full_data: pd.DataFrame,
    data_studentcount: pd.DataFrame,
    config: dict,
    regressor_name: str | None = None,
    grid: dict[str, list] | None = None,
    min_training_year: int = 2016,
    min_train_years: int = 3,
    numerus_fixus_list: list[str] | None = None,
) -> dict:
    """Zoek de beste hyperparameters voor de cumulatieve regressor via tijd-bewuste CV.

    Elke kandidaat-parameterset wordt geëvalueerd met
    :func:`~studentprognose.benchmark.evaluate_regressor.evaluate_regressor_model`
    over alle tijdreeks-splits; de gemiddelde MAPE over de folds is de
    selectiemetriek (lager is beter). De kandidaat met de laagste eindige
    gemiddelde MAPE wint.

    Args:
        full_data: Wide-format data met SARIMA-voorspellingen en engineered
            features (zelfde vorm als de benchmark-/productie-feature-matrix).
        data_studentcount: Studentaantallen per
            (programma, jaar, herkomst, examentype).
        config: Configuratie-dict (column_roles, model_features, model_config).
        regressor_name: Te tunen regressor. Bij ``None`` wordt
            ``model_config.cumulative_regressor`` gebruikt.
        grid: Zoekruimte. Bij ``None`` wordt :data:`DEFAULT_PARAM_GRIDS` voor de
            betreffende regressor gebruikt.
        min_training_year: Vroegste trainingsjaar (zie ``time_series_split``).
        min_train_years: Minimaal aantal trainingsjaren per fold.
        numerus_fixus_list: Lijst van NF-programma's (apart geëvalueerd).

    Returns:
        Dict met:

        * ``regressor_name`` — de getunede regressor.
        * ``best_params`` — beste parameter-dict, of ``None`` als geen enkele
          kandidaat een eindige score opleverde (bijv. te weinig
          trainingsjaren voor een tijdreeks-split).
        * ``best_mape`` — gemiddelde MAPE van de winnaar (``nan`` als geen).
        * ``results`` — lijst van ``{"params", "mean_mape", "n_evals"}`` per
          kandidaat, oplopend gesorteerd op MAPE (eindige scores eerst).
        * ``n_candidates`` — aantal geëvalueerde parametersets.

    Raises:
        ValueError: Als ``regressor_name`` niet in de regressor-registry staat.
    """
    name = regressor_name or config.get("model_config", {}).get("cumulative_regressor", "xgboost")
    if name not in REGRESSOR_REGISTRY:
        raise ValueError(
            f"Onbekend regressiemodel: '{name}'. "
            f"Geldige opties: {', '.join(REGRESSOR_REGISTRY)}"
        )

    search_grid = grid if grid is not None else DEFAULT_PARAM_GRIDS.get(name, {})
    candidates = expand_grid(search_grid)

    results = []
    for params in candidates:
        # Bind params via een default-arg zodat de closure de huidige waarde
        # vastlegt (niet de laatste lus-waarde) en evaluate_regressor_model per
        # fold een verse regressor krijgt.
        def factory(p=params):
            return REGRESSOR_REGISTRY[name](**p)

        df = evaluate_regressor_model(
            full_data,
            data_studentcount,
            factory,
            config=config,
            min_training_year=min_training_year,
            min_train_years=min_train_years,
            numerus_fixus_list=numerus_fixus_list,
        )

        if not df.empty and df["mape"].notna().any():
            mean_mape = float(df["mape"].mean())
        else:
            mean_mape = float("nan")

        results.append({"params": params, "mean_mape": mean_mape, "n_evals": int(len(df))})

    results.sort(key=lambda r: (not np.isfinite(r["mean_mape"]), r["mean_mape"]))

    finite = [r for r in results if np.isfinite(r["mean_mape"])]
    best = finite[0] if finite else None

    return {
        "regressor_name": name,
        "best_params": best["params"] if best else None,
        "best_mape": best["mean_mape"] if best else float("nan"),
        "results": results,
        "n_candidates": len(candidates),
    }


def config_snippet(regressor_name: str, best_params: dict) -> str:
    """Bouw een config-snippet dat de gebruiker in configuration.json kan plakken.

    Args:
        regressor_name: Naam van de getunede regressor (sleutel in
            ``regressor_params``).
        best_params: De gevonden beste hyperparameters.

    Returns:
        Een ingesprongen JSON-snippet als string, klaar om in
        ``configuration.json`` te plakken.
    """
    snippet = {"model_config": {"regressor_params": {regressor_name: best_params}}}
    return (
        "\nPlak dit in je configuration.json om de parameters vast te leggen:\n"
        + json.dumps(snippet, indent=4)
    )


def format_tuning_results(result: dict) -> str:
    """Render het volledige kandidatenoverzicht + config-snippet als string.

    Bouwt de tabel (één rij per geteste parameterset, oplopend op MAPE) plus —
    bij een geldige winnaar — een config-snippet. Wordt gebruikt door zowel het
    ``tune``-CLI-commando als het API-pad (``run_pipeline_from_dataframes(...,
    tune=True)``), zodat beide identieke output tonen.

    Args:
        result: Het resultaat-dict van :func:`tune_regressor`.

    Returns:
        De geformatteerde, meerregelige tekst (zonder trailing newline).
    """
    rows = result["results"]
    if not rows:
        return "Geen kandidaten geëvalueerd."

    best_params = result["best_params"]
    header = f"{'':2} {'MAPE':>10} {'Folds':>7}  Parameters"
    lines = [f"  {header}", f"  {'─' * len(header)}"]
    best_marked = False
    for row in rows:
        mape = row["mean_mape"]
        mape_s = f"{mape:.4f}" if pd.notna(mape) else "—"
        params_s = json.dumps(row["params"]) if row["params"] else "(defaults)"
        # Eén ✓ voor de winnaar (laagste MAPE); best_marked voorkomt dubbele
        # markering als twee kandidaten identieke params zouden hebben.
        is_best = not best_marked and best_params is not None and row["params"] == best_params
        marker = "✓" if is_best else " "
        best_marked = best_marked or is_best
        lines.append(f"  {marker:2} {mape_s:>10} {row['n_evals']:>7}  {params_s}")

    if result["best_params"] is None:
        lines.append(
            "\nGeen geldige resultaten: waarschijnlijk te weinig trainingsjaren voor "
            "een tijdreeks-split. Voeg meer historische collegejaren toe."
        )
        return "\n".join(lines)

    lines.append(
        f"\nBeste parameters voor '{result['regressor_name']}' "
        f"(MAPE={result['best_mape']:.4f}, {result['n_candidates']} kandidaten):"
    )
    lines.append(config_snippet(result["regressor_name"], result["best_params"]))
    return "\n".join(lines)
