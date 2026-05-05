import os

import pandas as pd

from studentprognose.benchmark.evaluate_ts import evaluate_timeseries_model
from studentprognose.benchmark.evaluate_regressor import evaluate_regressor_model
from studentprognose.benchmark.evaluate_classifier import evaluate_classifier_model
from studentprognose.benchmark.report import generate_cumulative_report, generate_individual_report
from studentprognose.config import load_configuration, get_columns
from studentprognose.data.loader import load_data
from studentprognose.models.sarima import SARIMAForecaster, _get_transformed_data
from studentprognose.models.forecasters import ETSForecaster, ThetaForecaster
from studentprognose.models.regressors import (
    XGBoostRegressor,
    RidgeRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from studentprognose.models.classifiers import (
    XGBoostClassifier,
    RandomForestClassifier,
    LogisticRegressionClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from studentprognose.strategies.cumulative import _add_engineered_features
from studentprognose.strategies.individual import preprocess_individual_data
from studentprognose.utils.weeks import DataOption


_FORECASTER_FACTORIES = {
    "sarima": lambda: SARIMAForecaster(),
    "ets": lambda: ETSForecaster(),
    "theta": lambda: ThetaForecaster(),
}

_REGRESSOR_FACTORIES = {
    "xgboost": lambda: XGBoostRegressor(),
    "ridge": lambda: RidgeRegressor(),
    "random_forest": lambda: RandomForestRegressor(),
    "gradient_boosting": lambda: GradientBoostingRegressor(),
    "extra_trees": lambda: ExtraTreesRegressor(),
}

_CLASSIFIER_FACTORIES = {
    "xgboost": lambda: XGBoostClassifier(),
    "random_forest": lambda: RandomForestClassifier(),
    "logistic_regression": lambda: LogisticRegressionClassifier(),
    "gradient_boosting": lambda: GradientBoostingClassifier(),
    "extra_trees": lambda: ExtraTreesClassifier(),
}


def main(
    configuration_path: str = "configuration/configuration.json",
    predict_week: int = 12,
    data_option: DataOption = DataOption.CUMULATIVE,
):
    """Draai benchmarks voor alle model-combinaties."""
    config = load_configuration(configuration_path)
    min_year = config.get("model_config", {}).get("min_training_year", 2016)
    nf_list = list(config.get("numerus_fixus", {}).keys())

    output_dir = os.path.join(os.getcwd(), "data", "output")
    os.makedirs(output_dir, exist_ok=True)

    if data_option == DataOption.CUMULATIVE:
        _run_cumulative_benchmark(config, predict_week, min_year, nf_list, output_dir)
    elif data_option == DataOption.INDIVIDUAL:
        _run_individual_benchmark(config, predict_week, min_year, output_dir)


def _run_cumulative_benchmark(config, predict_week, min_year, nf_list, output_dir):
    """Draai benchmark voor het cumulatieve spoor (tijdreeks + regressie)."""
    print("Benchmark: alternatieve modellen cumulatief spoor\n")

    data_cumulative, data_studentcount = _load_cumulative_benchmark_data(config)
    if data_cumulative is None:
        print("Geen cumulatieve data gevonden. Benchmark afgebroken.")
        return

    print(f"Predict week: {predict_week}, min trainingsjaar: {min_year}\n")

    # --- Stap 1: tijdreeksmodellen ---
    print("Stap 1: Tijdreeksmodellen evalueren")
    ts_results = []
    for name, factory in _FORECASTER_FACTORIES.items():
        print(f"  Evalueren: {name}...")
        result = evaluate_timeseries_model(
            data_cumulative, factory, predict_week,
            config=config,
            min_training_year=min_year,
        )
        result["model"] = name
        ts_results.append(result)

    ts_df = pd.concat(ts_results, ignore_index=True) if ts_results else pd.DataFrame()

    if not ts_df.empty:
        _print_ts_summary(ts_df)

    # --- Stap 2: regressiemodellen ---
    print("\nStap 2: Regressiemodellen evalueren")

    full_data = _get_transformed_data(data_cumulative.copy(), min_year)
    full_data["39"] = 0
    full_data = _add_engineered_features(full_data, data_cumulative, predict_week)

    reg_results = []
    for name, factory in _REGRESSOR_FACTORIES.items():
        print(f"  Evalueren: {name}...")
        result = evaluate_regressor_model(
            full_data, data_studentcount, factory,
            config=config,
            min_training_year=min_year,
            numerus_fixus_list=nf_list,
        )
        result["model"] = name
        reg_results.append(result)

    reg_df = pd.concat(reg_results, ignore_index=True) if reg_results else pd.DataFrame()

    if not reg_df.empty:
        _print_reg_summary(reg_df)

    # --- Opslaan ---
    if not ts_df.empty:
        ts_path = os.path.join(output_dir, "benchmark_timeseries.csv")
        ts_df.to_csv(ts_path, index=False)
        print(f"\nTijdreeksresultaten: {ts_path}")

    if not reg_df.empty:
        reg_path = os.path.join(output_dir, "benchmark_regressor.csv")
        reg_df.to_csv(reg_path, index=False)
        print(f"Regressieresultaten: {reg_path}")

    if not ts_df.empty and not reg_df.empty:
        generate_cumulative_report(ts_df, reg_df, output_dir)

    print("\nBenchmark voltooid.")


def _run_individual_benchmark(config, predict_week, min_year, output_dir):
    """Draai benchmark voor het individuele spoor (classificatie)."""
    print("Benchmark: alternatieve modellen individueel spoor\n")

    data_individual = _load_individual_benchmark_data(config)
    if data_individual is None:
        print("Geen individuele data gevonden. Benchmark afgebroken.")
        return

    print(f"Predict week: {predict_week}, min trainingsjaar: {min_year}\n")

    print("Stap 1: Classificatiemodellen evalueren")
    clf_results = []
    all_roc_curves = {}
    for name, factory in _CLASSIFIER_FACTORIES.items():
        print(f"  Evalueren: {name}...")
        result, roc_curves = evaluate_classifier_model(
            data_individual, factory, predict_week,
            config=config,
            min_training_year=min_year,
        )
        result["model"] = name
        clf_results.append(result)
        all_roc_curves[name] = roc_curves

    clf_df = pd.concat(clf_results, ignore_index=True) if clf_results else pd.DataFrame()

    if not clf_df.empty:
        _print_clf_summary(clf_df)

    # --- Opslaan ---
    if not clf_df.empty:
        clf_path = os.path.join(output_dir, "benchmark_classifier.csv")
        clf_df.to_csv(clf_path, index=False)
        print(f"\nClassificatieresultaten: {clf_path}")

        generate_individual_report(clf_df, output_dir, all_roc_curves)

    print("\nBenchmark voltooid.")


def _load_cumulative_benchmark_data(config):
    """Laad en preprocess cumulatieve data voor benchmark."""
    try:
        _, data_cumulative, data_studentcount, _, _ = load_data(
            config, DataOption.CUMULATIVE
        )
    except Exception:
        return None, None

    if data_cumulative is None:
        return None, None

    data_cumulative = _preprocess_cumulative(data_cumulative, config)

    return data_cumulative, data_studentcount


def _load_individual_benchmark_data(config):
    """Laad en preprocess individuele data voor benchmark."""
    try:
        data_individual, _, _, _, _ = load_data(
            config, DataOption.INDIVIDUAL
        )
    except Exception:
        return None

    if data_individual is None:
        return None

    nf_list = list(config.get("numerus_fixus", {}).keys())
    return preprocess_individual_data(data_individual, nf_list)


def _preprocess_cumulative(data, config):
    """Minimale preprocessing voor benchmark (zelfde als CumulativeStrategy.preprocess)."""
    c = get_columns(config)

    float_cols = [c.unweighted_applicants, c.weighted_applicants,
                  c.single_applicants, c.enrollments]
    for col in float_cols:
        if col not in data.columns:
            continue
        if pd.api.types.is_string_dtype(data[col].dtype):
            data[col] = data[col].str.replace(".", "").str.replace(",", ".")
        data[col] = pd.to_numeric(data[col], errors="coerce").astype("float64")

    data = data.rename(columns={
        c.higher_education_type: c.exam_type,
        c.croho_source: c.programme,
    })

    if c.higher_years in data.columns:
        data.loc[data[c.exam_type] == "Pre-master", c.higher_years] = "Nee"
        data = data[data[c.higher_years] == "Nee"]

    data = (
        data.groupby([
            c.academic_year, c.programme, c.faculty,
            c.exam_type, c.origin, c.week,
        ])
        .sum(numeric_only=False)
        .reset_index()
    )

    data["ts"] = data[c.weighted_applicants] + data[c.enrollments]

    return data


def _print_ts_summary(df):
    """Print samenvattingstabel voor tijdreeksmodellen."""
    summary = (
        df.groupby("model")
        .agg(
            mean_mape=("mape", "mean"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_time=("train_time_s", "mean"),
            convergence_rate=("converged", "mean"),
            n_evals=("converged", "count"),
        )
        .sort_values("mean_mape")
    )

    header = f"{'Model':<15} {'MAPE':>10} {'MAE':>10} {'RMSE':>10} {'Tijd (s)':>10} {'Conv.':>8} {'N':>5}"
    print(f"\n  Tijdreeksmodellen — Samenvatting\n  {header}")
    print(f"  {'─' * len(header)}")

    for model, row in summary.iterrows():
        mape_s = f"{row['mean_mape']:.4f}" if pd.notna(row["mean_mape"]) else "—"
        mae_s = f"{row['mean_mae']:.1f}" if pd.notna(row["mean_mae"]) else "—"
        rmse_s = f"{row['mean_rmse']:.1f}" if pd.notna(row["mean_rmse"]) else "—"
        print(
            f"  {str(model):<15} {mape_s:>10} {mae_s:>10} {rmse_s:>10} "
            f"{row['mean_time']:>10.3f} {row['convergence_rate']:>7.0%} {int(row['n_evals']):>5}"
        )


def _print_reg_summary(df):
    """Print samenvattingstabel voor regressiemodellen."""
    summary = (
        df.groupby("model")
        .agg(
            mean_mape=("mape", "mean"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_time=("train_time_s", "mean"),
            mean_train_size=("n_train", "mean"),
            n_evals=("n_train", "count"),
        )
        .sort_values("mean_mape")
    )

    header = f"{'Model':<15} {'MAPE':>10} {'MAE':>10} {'RMSE':>10} {'Tijd (s)':>10} {'Train N':>10} {'N':>5}"
    print(f"\n  Regressiemodellen — Samenvatting\n  {header}")
    print(f"  {'─' * len(header)}")

    for model, row in summary.iterrows():
        mape_s = f"{row['mean_mape']:.4f}" if pd.notna(row["mean_mape"]) else "—"
        mae_s = f"{row['mean_mae']:.1f}" if pd.notna(row["mean_mae"]) else "—"
        rmse_s = f"{row['mean_rmse']:.1f}" if pd.notna(row["mean_rmse"]) else "—"
        print(
            f"  {str(model):<15} {mape_s:>10} {mae_s:>10} {rmse_s:>10} "
            f"{row['mean_time']:>10.3f} {row['mean_train_size']:>10.0f} {int(row['n_evals']):>5}"
        )


def _print_clf_summary(df):
    """Print samenvattingstabel voor classificatiemodellen."""
    summary = (
        df.groupby("model")
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_auc_roc=("auc_roc", "mean"),
            mean_f1=("f1", "mean"),
            mean_agg_mae=("aggregate_mae", "mean"),
            mean_time=("train_time_s", "mean"),
            mean_train_size=("n_train", "mean"),
            n_evals=("n_train", "count"),
        )
        .sort_values("mean_auc_roc", ascending=False)
    )

    header = f"{'Model':<20} {'Accuracy':>10} {'AUC-ROC':>10} {'F1':>10} {'Agg MAE':>10} {'Tijd (s)':>10} {'Train N':>10} {'N':>5}"
    print(f"\n  Classificatiemodellen — Samenvatting\n  {header}")
    print(f"  {'─' * len(header)}")

    for model, row in summary.iterrows():
        acc_s = f"{row['mean_accuracy']:.4f}" if pd.notna(row["mean_accuracy"]) else "—"
        auc_s = f"{row['mean_auc_roc']:.4f}" if pd.notna(row["mean_auc_roc"]) else "—"
        f1_s = f"{row['mean_f1']:.4f}" if pd.notna(row["mean_f1"]) else "—"
        mae_s = f"{row['mean_agg_mae']:.1f}" if pd.notna(row["mean_agg_mae"]) else "—"
        print(
            f"  {str(model):<20} {acc_s:>10} {auc_s:>10} {f1_s:>10} {mae_s:>10} "
            f"{row['mean_time']:>10.3f} {row['mean_train_size']:>10.0f} {int(row['n_evals']):>5}"
        )
