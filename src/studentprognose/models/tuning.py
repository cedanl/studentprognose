"""Tijd-bewuste hyperparameter tuning voor het cumulatieve spoor.

Het cumulatieve spoor is twee-traps: **stap 1** extrapoleert de cumulatieve
vooraanmeldcurve met een tijdreeksmodel (SARIMA), **stap 2** vertaalt die curve
met een regressor naar ``Aantal_studenten``. Dit module zoekt — via een kleine,
gerichte grid search — de instellingen die het beste presteren op de eigen
historie van een instelling, voor beide trappen afzonderlijk:

* :func:`tune_regressor` — hyperparameters van de regressor (stap 2).
* :func:`tune_sarima` — ARIMA-ordes ``(p,d,q)(P,D,Q,s)`` van SARIMA (stap 1).

Beide delen exact dezelfde tijd-bewuste cross-validatie en MAPE-metriek, zodat
hun scores onderling én met de ``benchmark``-stap vergelijkbaar zijn.

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
from studentprognose.models.sarima import SARIMAForecaster
from studentprognose.utils.constants import WEEKS_PER_YEAR


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


# Zoekruimte voor de SARIMA-ordes (stap 1). Bewust compact: een handvol
# niet-seizoens-ordes ``(p,d,q)`` × twee seizoens-ordes ``(P,D,Q,s)``. De vierde
# waarde van ``seasonal_order`` is de seizoenslengte; die wordt bij fit door
# ``shrink_season_length_to_period`` afgestemd op de werkelijke jaar-periode
# (net als productie en de benchmark), dus ``WEEKS_PER_YEAR`` is enkel de
# bovengrens. Een brede orde-zoektocht (zoals ``auto_arima``) per serie is bij
# deze reekslengtes duur en overfit-gevoelig; deze vaste, kleine set spiegelt de
# ordes die de pipeline historisch hardgecodeerd gebruikte.
DEFAULT_SARIMA_GRID: dict[str, list] = {
    "order": [(1, 0, 1), (1, 1, 1), (2, 1, 1)],
    "seasonal_order": [
        (1, 1, 0, WEEKS_PER_YEAR),
        (1, 1, 1, WEEKS_PER_YEAR),
    ],
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


def _rank_candidates(candidates: list[dict], evaluate) -> tuple[list[dict], dict | None]:
    """Scoor en rangschik kandidaten op gemiddelde MAPE (lager is beter).

    Args:
        candidates: Lijst van parameter-dicts (uit :func:`expand_grid`).
        evaluate: Callable ``params -> DataFrame`` met een ``mape``-kolom; de
            per-fold-evaluatie van één kandidaat. Wordt per kandidaat met de
            concrete params aangeroepen (geen closure-over-lus).

    Returns:
        Tuple ``(results, best)``: ``results`` is de lijst
        ``{"params", "mean_mape", "n_evals"}`` oplopend gesorteerd op MAPE
        (eindige scores eerst); ``best`` is de winnende rij, of ``None`` als geen
        enkele kandidaat een eindige score opleverde.
    """
    results = []
    for params in candidates:
        df = evaluate(params)
        if not df.empty and df["mape"].notna().any():
            mean_mape = float(df["mape"].mean())
        else:
            mean_mape = float("nan")
        results.append({"params": params, "mean_mape": mean_mape, "n_evals": int(len(df))})

    results.sort(key=lambda r: (not np.isfinite(r["mean_mape"]), r["mean_mape"]))
    finite = [r for r in results if np.isfinite(r["mean_mape"])]
    best = finite[0] if finite else None
    return results, best


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

        * ``target`` — ``"regressor"`` (welke pipeline-trap getuned is).
        * ``model`` / ``regressor_name`` — de getunede regressor.
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

    def evaluate(params):
        # Bind params via een default-arg zodat de closure de huidige waarde
        # vastlegt (niet de laatste lus-waarde) en evaluate_regressor_model per
        # fold een verse regressor krijgt.
        def factory(p=params):
            return REGRESSOR_REGISTRY[name](**p)

        return evaluate_regressor_model(
            full_data,
            data_studentcount,
            factory,
            config=config,
            min_training_year=min_training_year,
            min_train_years=min_train_years,
            numerus_fixus_list=numerus_fixus_list,
        )

    results, best = _rank_candidates(candidates, evaluate)

    return {
        "target": "regressor",
        "model": name,
        "regressor_name": name,
        "best_params": best["params"] if best else None,
        "best_mape": best["mean_mape"] if best else float("nan"),
        "results": results,
        "n_candidates": len(candidates),
    }


def tune_sarima(
    data_cumulative: pd.DataFrame,
    predict_week: int,
    config: dict,
    grid: dict[str, list] | None = None,
    min_training_year: int = 2016,
    min_train_years: int = 3,
) -> dict:
    """Zoek de beste SARIMA-ordes voor de tijdreeksextrapolatie via tijd-bewuste CV.

    Elke kandidaat — een ``{"order", "seasonal_order"}``-combinatie uit de grid —
    wordt geëvalueerd met
    :func:`~studentprognose.benchmark.evaluate_ts.evaluate_timeseries_model`, exact
    dezelfde per-fold-pipeline (pivot, seizoenslengte-afstemming, MAPE op de curve)
    als de ``benchmark``-stap. De gemiddelde MAPE over alle (serie × testjaar)-folds
    is de selectiemetriek; de laagste eindige gemiddelde MAPE wint.

    Args:
        data_cumulative: Cumulatieve vooraanmelddata in lang formaat (zelfde vorm
            als de benchmark-/productie-tijdreeksinput).
        predict_week: Week waarvandaan voorspeld wordt (bepaalt de horizon).
        config: Configuratie-dict (column_roles, model_config).
        grid: Zoekruimte met ``order``- en ``seasonal_order``-lijsten. Bij ``None``
            wordt :data:`DEFAULT_SARIMA_GRID` gebruikt.
        min_training_year: Vroegste trainingsjaar (zie ``time_series_split``).
        min_train_years: Minimaal aantal trainingsjaren per fold.

    Returns:
        Dict met dezelfde vorm als :func:`tune_regressor` (``target="sarima"``,
        ``model="sarima"``). ``best_params`` is een
        ``{"order", "seasonal_order"}``-dict, of ``None`` bij onvoldoende historie.
    """
    from studentprognose.benchmark.evaluate_ts import evaluate_timeseries_model

    search_grid = grid if grid is not None else DEFAULT_SARIMA_GRID
    candidates = expand_grid(search_grid)

    def evaluate(params):
        def factory(p=params):
            return SARIMAForecaster(**p)

        return evaluate_timeseries_model(
            data_cumulative,
            factory,
            predict_week,
            config=config,
            min_training_year=min_training_year,
            min_train_years=min_train_years,
        )

    results, best = _rank_candidates(candidates, evaluate)

    return {
        "target": "sarima",
        "model": "sarima",
        "best_params": best["params"] if best else None,
        "best_mape": best["mean_mape"] if best else float("nan"),
        "results": results,
        "n_candidates": len(candidates),
    }


def config_snippet(regressor_name: str, best_params: dict) -> str:
    """Bouw een ``regressor_params``-snippet om in configuration.json te plakken.

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


def sarima_config_snippet(best_params: dict) -> str:
    """Bouw een ``forecaster_params``-snippet voor de getunede SARIMA-ordes.

    Args:
        best_params: De gevonden ``{"order", "seasonal_order"}``-ordes.

    Returns:
        Een ingesprongen JSON-snippet als string, klaar om in
        ``configuration.json`` te plakken. De orde-arrays blijven op één regel
        (``json.dumps(indent=4)`` zou ze verticaal uitklappen).
    """
    params_lines = ",\n".join(
        f'                "{key}": {json.dumps(list(value))}'
        for key, value in best_params.items()
    )
    return (
        "\nPlak dit in je configuration.json om de ordes vast te leggen:\n"
        "{\n"
        '    "model_config": {\n'
        '        "forecaster_params": {\n'
        '            "sarima": {\n'
        f"{params_lines}\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "}"
    )


def format_tuning_results(result: dict) -> str:
    """Render het volledige kandidatenoverzicht + config-snippet als string.

    Bouwt de tabel (één rij per geteste parameterset, oplopend op MAPE) plus —
    bij een geldige winnaar — een config-snippet. Wordt gebruikt door zowel het
    ``tune``-CLI-commando als het API-pad (``run_pipeline_from_dataframes(...,
    tune=True)``), zodat beide identieke output tonen.

    Werkt voor beide trappen: het ``target``-veld van het resultaat bepaalt welk
    config-snippet onder de tabel komt (``"regressor"`` → ``regressor_params``,
    ``"sarima"`` → ``forecaster_params``). Bij ``tune="both"`` print de aanroeper
    dit twee keer — één tabel per trap, elk met een eigen ✓ voor de winnaar.

    Args:
        result: Het resultaat-dict van :func:`tune_regressor` of
            :func:`tune_sarima`.

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

    if best_params is None:
        lines.append(
            "\nGeen geldige resultaten: waarschijnlijk te weinig trainingsjaren voor "
            "een tijdreeks-split. Voeg meer historische collegejaren toe."
        )
        return "\n".join(lines)

    # Label + bijbehorend snippet hangen af van welke trap getuned is. Default
    # "regressor" houdt bestaande aanroepers (zonder target-veld) werkend.
    target = result.get("target", "regressor")
    label = result.get("model") or result.get("regressor_name")
    lines.append(
        f"\nBeste parameters voor '{label}' "
        f"(MAPE={result['best_mape']:.4f}, {result['n_candidates']} kandidaten):"
    )
    if target == "sarima":
        lines.append(sarima_config_snippet(best_params))
    else:
        lines.append(config_snippet(label, best_params))
    return "\n".join(lines)
