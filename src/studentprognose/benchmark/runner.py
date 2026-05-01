import os

import pandas as pd

from studentprognose.benchmark.evaluate_ts import evaluate_timeseries_model
from studentprognose.benchmark.evaluate_regressor import evaluate_regressor_model
from studentprognose.config import load_configuration
from studentprognose.data.loader import load_data
from studentprognose.models.sarima import SARIMAForecaster, _get_transformed_data
from studentprognose.models.forecasters import ETSForecaster, ThetaForecaster, AutoARIMAForecaster
from studentprognose.models.regressors import XGBoostRegressor, RidgeRegressor, RandomForestRegressor
from studentprognose.strategies.cumulative import _add_engineered_features


_FORECASTER_FACTORIES = {
    "sarima": lambda: SARIMAForecaster(),
    "ets": lambda: ETSForecaster(),
    "theta": lambda: ThetaForecaster(),
    "auto_arima": lambda: AutoARIMAForecaster(),
}

_REGRESSOR_FACTORIES = {
    "xgboost": lambda: XGBoostRegressor(),
    "ridge": lambda: RidgeRegressor(),
    "random_forest": lambda: RandomForestRegressor(),
}


def main(configuration_path: str = "configuration/configuration.json", predict_week: int = 12):
    """Draai benchmarks voor alle model-combinaties."""
    config = load_configuration(configuration_path)
    min_year = config.get("model_config", {}).get("min_training_year", 2016)
    nf_list = list(config.get("numerus_fixus", {}).keys())

    print("Benchmark: alternatieve modellen cumulatief spoor\n")

    data_cumulative, data_studentcount = _load_benchmark_data(config)
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
            min_training_year=min_year,
            numerus_fixus_list=nf_list,
        )
        result["model"] = name
        reg_results.append(result)

    reg_df = pd.concat(reg_results, ignore_index=True) if reg_results else pd.DataFrame()

    if not reg_df.empty:
        _print_reg_summary(reg_df)

    # --- Opslaan ---
    output_dir = os.path.join(os.getcwd(), "data", "output")
    os.makedirs(output_dir, exist_ok=True)

    if not ts_df.empty:
        ts_path = os.path.join(output_dir, "benchmark_timeseries.csv")
        ts_df.to_csv(ts_path, index=False)
        print(f"\nTijdreeksresultaten: {ts_path}")

    if not reg_df.empty:
        reg_path = os.path.join(output_dir, "benchmark_regressor.csv")
        reg_df.to_csv(reg_path, index=False)
        print(f"Regressieresultaten: {reg_path}")

    print("\nBenchmark voltooid.")


def _load_benchmark_data(config):
    """Laad en preprocess cumulatieve data voor benchmark."""
    paths = config.get("paths", {})
    cumulative_path = paths.get("path_cumulative", "")
    studentcount_path = paths.get("path_student_count_first-years", "")

    data_cumulative = None
    data_studentcount = None

    try:
        data_cumulative = load_data(cumulative_path)
    except Exception:
        return None, None

    try:
        data_studentcount = load_data(studentcount_path)
    except Exception:
        pass

    if data_cumulative is not None:
        data_cumulative = _preprocess_cumulative(data_cumulative)

    return data_cumulative, data_studentcount


def _preprocess_cumulative(data):
    """Minimale preprocessing voor benchmark (zelfde als CumulativeStrategy.preprocess)."""
    for col in ["Ongewogen vooraanmelders", "Gewogen vooraanmelders",
                "Aantal aanmelders met 1 aanmelding", "Inschrijvingen"]:
        if col not in data.columns:
            continue
        if pd.api.types.is_string_dtype(data[col].dtype):
            data[col] = data[col].str.replace(".", "").str.replace(",", ".")
        data[col] = pd.to_numeric(data[col], errors="coerce").astype("float64")

    data = data.rename(columns={
        "Type hoger onderwijs": "Examentype",
        "Groepeernaam Croho": "Croho groepeernaam",
    })

    if "Hogerejaars" in data.columns:
        data.loc[data["Examentype"] == "Pre-master", "Hogerejaars"] = "Nee"
        data = data[data["Hogerejaars"] == "Nee"]

    data = (
        data.groupby([
            "Collegejaar", "Croho groepeernaam", "Faculteit",
            "Examentype", "Herkomst", "Weeknummer",
        ])
        .sum(numeric_only=False)
        .reset_index()
    )

    data["ts"] = data["Gewogen vooraanmelders"] + data["Inschrijvingen"]

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
