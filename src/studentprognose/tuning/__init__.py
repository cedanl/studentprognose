import os
import sys
import time

import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from studentprognose.tuning.cache import save_cached_params, save_sarima_orders
from studentprognose.tuning.xgboost_tuner import (
    tune_xgboost_classifier,
    tune_xgboost_regressor,
    _sample_param_combinations,
    PARAM_GRID,
)
from studentprognose.tuning.sarima_tuner import find_best_sarima_order
from studentprognose.utils.weeks import get_weeks_list, get_all_weeks_valid
from studentprognose.utils.constants import WEEKS_PER_YEAR
from studentprognose.data.transforms import transform_data


class _ProgressBar:
    """Inline progress bar using \\r overwrites."""

    def __init__(self, total: int, label: str):
        self.total = total
        self.current = 0
        self.label = label
        self.start_time = time.monotonic()
        self._print()

    def advance(self):
        self.current += 1
        self._print()

    def _print(self):
        elapsed = time.monotonic() - self.start_time
        if self.current > 0:
            remaining = elapsed / self.current * (self.total - self.current)
            eta = f"{remaining:.0f}s remaining"
        else:
            eta = "..."
        pct = self.current / self.total * 100 if self.total else 0
        filled = int(30 * self.current / self.total) if self.total else 0
        bar = "█" * filled + "░" * (30 - filled)
        line = f"\r    {self.label} {bar} {self.current}/{self.total} ({pct:.0f}%) {elapsed:.0f}s elapsed, {eta}"
        sys.stdout.write(line)
        sys.stdout.flush()

    def finish(self):
        elapsed = time.monotonic() - self.start_time
        sys.stdout.write(f"\r    {self.label} {'█' * 30} {self.total}/{self.total} (100%) {elapsed:.0f}s elapsed       \n")
        sys.stdout.flush()


def run_tuning(strategy, cfg, configuration):
    """Run hyperparameter tuning for all models and cache results."""
    print("\n" + "=" * 54)
    print("  Hyperparameter Tuning")
    print("=" * 54)

    hp_config = configuration.get("model_config", {}).get("hyperparameters", {})
    sarima_config = hp_config.get("sarima", {})
    n_iter = hp_config.get("tuning_n_iter", 50)

    _tune_xgboost(strategy, cfg, configuration, n_iter)

    if sarima_config.get("auto_order_selection", True):
        _tune_sarima(strategy, cfg, sarima_config)

    print("=" * 54 + "\n")


def _tune_xgboost(strategy, cfg, configuration, n_iter):
    """Tune XGBoost classifier and regressor with expanding-window CV."""
    from studentprognose.strategies.individual import IndividualStrategy
    from studentprognose.strategies.cumulative import CumulativeStrategy
    from studentprognose.strategies.combined import CombinedStrategy
    from studentprognose.models.xgboost_classifier import DEFAULT_STATUS_MAP

    model_config = configuration.get("model_config", {})
    status_map = model_config.get("status_mapping", DEFAULT_STATUS_MAP)
    min_year = model_config.get("min_training_year", 2016)

    # --- Classifier tuning ---
    individual = None
    if isinstance(strategy, (IndividualStrategy, CombinedStrategy)):
        individual = (
            strategy
            if isinstance(strategy, IndividualStrategy)
            else strategy.individual
        )

    if individual is not None:
        print("\n  Tuning XGBoost Classifier...")
        data = individual.data_individual.copy()
        data = data[data["Weeknummer"].isin(get_weeks_list(cfg.weeks[-1]))]
        data = data[data["Collegejaar"] >= min_year]
        data["Inschrijfstatus"] = data["Inschrijfstatus"].map(status_map)
        data = data.dropna(subset=["Inschrijfstatus"])

        numeric_cols = ["Collegejaar", "Sleutel_count", "is_numerus_fixus"]
        categorical_cols = [
            "Examentype",
            "Faculteit",
            "Croho groepeernaam",
            "Deadlineweek",
            "Herkomst",
            "Weeknummer",
            "Opleiding",
            "Type vooropleiding",
            "Nationaliteit",
            "EER",
            "Geslacht",
            "Plaats code eerste vooropleiding",
            "Studieadres postcode",
            "Studieadres land",
            "Geverifieerd adres plaats",
            "Geverifieerd adres land",
            "Geverifieerd adres postcode",
            "School code eerste vooropleiding",
            "School eerste vooropleiding",
            "Land code eerste vooropleiding",
        ]

        years_array = data["Collegejaar"].values
        y = data["Inschrijfstatus"].values.astype(float)
        X_raw = data.drop(["Inschrijfstatus"], axis=1)

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    "passthrough",
                    [c for c in numeric_cols if c in X_raw.columns],
                ),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    [c for c in categorical_cols if c in X_raw.columns],
                ),
            ]
        )
        X = preprocessor.fit_transform(X_raw)

        neg = int((y == 0).sum())
        pos = int((y == 1).sum())
        spw = neg / pos if pos > 0 else 1.0

        n_total = len(_sample_param_combinations(PARAM_GRID, n_iter))
        pb = _ProgressBar(n_total, "Classifier")
        best_params, best_score, n_folds, n_combos = tune_xgboost_classifier(
            X,
            y,
            years_array,
            scale_pos_weight=spw,
            n_iter=n_iter,
            on_progress=pb.advance,
        )
        pb.finish()

        if best_params:
            save_cached_params(
                "xgboost_classifier", best_params, best_score, n_folds, n_combos
            )
            print(
                f"    Best CV MAPE: {best_score:.1%} ({n_folds} folds, {n_combos} combinaties)"
            )
            _print_params(best_params)
        else:
            print("    Niet genoeg data voor tuning")

    # --- Regressor tuning ---
    cumulative = None
    if isinstance(strategy, (CumulativeStrategy, CombinedStrategy)):
        cumulative = (
            strategy
            if isinstance(strategy, CumulativeStrategy)
            else strategy.cumulative
        )

    if cumulative is not None and cumulative.data_studentcount is not None:
        print("\n  Tuning XGBoost Regressor...")
        data_cum = cumulative.data_cumulative.copy()
        data_cum = data_cum.merge(
            cumulative.data_studentcount[
                [
                    "Croho groepeernaam",
                    "Collegejaar",
                    "Herkomst",
                    "Examentype",
                    "Aantal_studenten",
                ]
            ],
            on=["Croho groepeernaam", "Collegejaar", "Herkomst", "Examentype"],
            how="inner",
        )
        data_cum = data_cum.drop_duplicates()
        data_cum = data_cum.dropna(subset=["Aantal_studenten"])
        data_cum["ts"] = (
            data_cum["Gewogen vooraanmelders"] + data_cum["Inschrijvingen"]
        )

        if not data_cum.empty:
            data_wide = transform_data(data_cum, "ts")
            data_wide = data_wide.dropna(subset=["Aantal_studenten"])

            numeric_cols = ["Collegejaar"] + [str(x) for x in get_weeks_list(38)]
            categorical_cols = [
                "Examentype",
                "Faculteit",
                "Croho groepeernaam",
                "Herkomst",
            ]

            years_array = data_wide["Collegejaar"].values
            y = data_wide["Aantal_studenten"].values.astype(float)
            X_raw = data_wide.drop(["Aantal_studenten"], axis=1)

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "numeric",
                        "passthrough",
                        [c for c in numeric_cols if c in X_raw.columns],
                    ),
                    (
                        "categorical",
                        OneHotEncoder(handle_unknown="ignore"),
                        [c for c in categorical_cols if c in X_raw.columns],
                    ),
                ]
            )

            try:
                X = preprocessor.fit_transform(X_raw)
                n_total = len(_sample_param_combinations(PARAM_GRID, n_iter))
                pb = _ProgressBar(n_total, "Regressor ")
                best_params, best_score, n_folds, n_combos = tune_xgboost_regressor(
                    X,
                    y,
                    years_array,
                    n_iter=n_iter,
                    on_progress=pb.advance,
                )
                pb.finish()

                if best_params:
                    save_cached_params(
                        "xgboost_regressor", best_params, best_score, n_folds, n_combos
                    )
                    print(
                        f"    Best CV MAE: {best_score:.1f} studenten ({n_folds} folds, {n_combos} combinaties)"
                    )
                    _print_params(best_params)
                else:
                    print("    Niet genoeg data voor tuning")
            except (ValueError, KeyError) as e:
                print(f"    Regressor tuning overgeslagen: {e}")


def _tune_sarima(strategy, cfg, sarima_config):
    """Tune SARIMA orders per programme/herkomst/examentype via AIC grid search."""
    from studentprognose.strategies.cumulative import CumulativeStrategy
    from studentprognose.strategies.combined import CombinedStrategy

    cumulative = None
    if isinstance(strategy, (CumulativeStrategy, CombinedStrategy)):
        cumulative = (
            strategy
            if isinstance(strategy, CumulativeStrategy)
            else strategy.cumulative
        )

    if cumulative is None:
        print("\n  SARIMA tuning overgeslagen (geen cumulatieve data)")
        return

    print("\n  Tuning SARIMA orders (AIC grid search)...")

    data = cumulative.data_cumulative.copy()
    data = data.astype({"Weeknummer": "int32", "Collegejaar": "int32"})
    data["ts"] = data["Gewogen vooraanmelders"] + data["Inschrijvingen"]
    min_year = cumulative.min_training_year

    data_wide = transform_data(data.drop_duplicates(), "ts")
    data_wide = data_wide[data_wide["Collegejaar"] >= min_year]
    data_wide["39"] = 0

    groups = data_wide.groupby(["Croho groepeernaam", "Herkomst", "Examentype"])

    tasks = []
    for (programme, herkomst, examentype), group_df in groups:
        week_cols = get_all_weeks_valid(group_df.columns)
        if not week_cols:
            continue
        ts_data = group_df[week_cols].values.flatten()
        ts_data = ts_data[np.isfinite(ts_data)]
        if len(ts_data) < 2 * WEEKS_PER_YEAR:
            continue
        tasks.append((programme, herkomst, examentype, ts_data))

    nr_CPU_cores = os.cpu_count()
    pb = _ProgressBar(len(tasks), "SARIMA    ")
    results = []
    for result in joblib.Parallel(n_jobs=nr_CPU_cores, return_as="generator")(
        joblib.delayed(_tune_single_sarima)(ts_data, sarima_config)
        for _, _, _, ts_data in tasks
    ):
        results.append(result)
        pb.advance()
    pb.finish()

    orders = {}
    n_successful = 0
    for (programme, herkomst, examentype, _), (order, seasonal_order, aic) in zip(
        tasks, results
    ):
        if order is not None:
            key = f"{programme}|{herkomst}|{examentype}|cumulative"
            orders[key] = {
                "order": list(order),
                "seasonal_order": list(seasonal_order),
                "aic": round(aic, 2),
            }
            n_successful += 1

    if orders:
        save_sarima_orders(orders)
        print(f"    {n_successful}/{len(tasks)} succesvol getuned")
        example_key = next(iter(orders))
        example = orders[example_key]
        print(
            f"    Voorbeeld: {example_key.split('|')[0]}|{example_key.split('|')[1]}"
            f" -> {tuple(example['order'])}{tuple(example['seasonal_order'])} AIC={example['aic']}"
        )
    else:
        print("    Geen SARIMA orders getuned")


def _tune_single_sarima(ts_data, sarima_config):
    return find_best_sarima_order(
        ts_data,
        max_p=sarima_config.get("max_p", 2),
        max_d=sarima_config.get("max_d", 1),
        max_q=sarima_config.get("max_q", 2),
        max_P=sarima_config.get("max_P", 1),
        max_D=sarima_config.get("max_D", 1),
        max_Q=sarima_config.get("max_Q", 1),
        seasonal_period=sarima_config.get("seasonal_period", WEEKS_PER_YEAR),
        criterion=sarima_config.get("selection_criterion", "aic"),
    )


def _print_params(params):
    parts = [f"{k}={v}" for k, v in sorted(params.items())]
    line = "    " + "  ".join(parts)
    print(line)
