import time

import numpy as np
import pandas as pd

from studentprognose.benchmark.metrics import mape, mae, rmse
from studentprognose.benchmark.splitter import time_series_split
from studentprognose.models.regressors import build_preprocessor
from studentprognose.strategies.cumulative import ENGINEERED_FEATURE_COLS


def evaluate_regressor_model(
    full_data: pd.DataFrame,
    data_studentcount: pd.DataFrame,
    regressor_factory: callable,
    min_training_year: int = 2016,
    min_train_years: int = 3,
    numerus_fixus_list: list[str] | None = None,
) -> pd.DataFrame:
    """Evalueer een regressiemodel op stap 2 (voorspelling Aantal_studenten) via cross-validatie.

    Args:
        full_data: Wide-format data met SARIMA-voorspellingen en engineered features.
        data_studentcount: DataFrame met Aantal_studenten per (programme, jaar, herkomst, examentype).
        regressor_factory: Callable die een vers BaseRegressor-object retourneert.
        min_training_year: Vroegste trainingsjaar.
        min_train_years: Minimaal aantal trainingsjaren.
        numerus_fixus_list: Lijst van NF-programma's (voor aparte rapportage).

    Returns:
        DataFrame met per (examentype, test_year) de metrics en trainingsset-grootte.
    """
    if numerus_fixus_list is None:
        numerus_fixus_list = []

    merged = full_data.merge(
        data_studentcount[
            ["Croho groepeernaam", "Collegejaar", "Herkomst", "Examentype", "Aantal_studenten"]
        ],
        on=["Croho groepeernaam", "Collegejaar", "Herkomst", "Examentype"],
        how="inner",
    ).drop_duplicates(ignore_index=True)

    if merged.empty:
        return pd.DataFrame()

    splits = time_series_split(merged, min_training_year, min_train_years)
    results = []

    for train_df, test_df in splits:
        test_year = int(test_df["Collegejaar"].iloc[0])

        assert train_df["Collegejaar"].max() < test_year, "Leakage: train bevat testjaar"

        for examentype in test_df["Examentype"].unique():
            train_et = train_df[
                (train_df["Examentype"] == examentype)
                & (~train_df["Croho groepeernaam"].isin(numerus_fixus_list))
            ]
            test_et = test_df[
                (test_df["Examentype"] == examentype)
                & (~test_df["Croho groepeernaam"].isin(numerus_fixus_list))
            ]

            if train_et.empty or test_et.empty:
                continue

            result = _evaluate_single(
                train_et, test_et, regressor_factory,
                examentype, test_year, is_numerus_fixus=False,
            )
            if result:
                results.append(result)

        for nf_prog in numerus_fixus_list:
            train_nf = train_df[train_df["Croho groepeernaam"] == nf_prog]
            test_nf = test_df[test_df["Croho groepeernaam"] == nf_prog]

            if train_nf.empty or test_nf.empty:
                continue

            examentype = test_nf["Examentype"].iloc[0]
            result = _evaluate_single(
                train_nf, test_nf, regressor_factory,
                examentype, test_year, is_numerus_fixus=True,
                programme=nf_prog,
            )
            if result:
                results.append(result)

    return pd.DataFrame(results)


def _evaluate_single(
    train_df, test_df, regressor_factory,
    examentype, test_year, is_numerus_fixus, programme=None,
):
    """Evalueer een regressor op een enkele train/test split."""
    y_train = train_df["Aantal_studenten"].values
    y_test = test_df["Aantal_studenten"].values
    X_train = train_df.drop(columns=["Aantal_studenten"])
    X_test = test_df.drop(columns=["Aantal_studenten"])

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(ENGINEERED_FEATURE_COLS)

    try:
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)
    except ValueError:
        return None

    t0 = time.perf_counter()
    try:
        regressor = regressor_factory()
        regressor.fit(X_train_t, y_train)
        predictions = np.rint(regressor.predict(X_test_t)).astype(int)
        elapsed = time.perf_counter() - t0
    except Exception:
        return None

    return {
        "examentype": examentype,
        "test_year": test_year,
        "is_numerus_fixus": is_numerus_fixus,
        "programme": programme or "all",
        "n_train": len(y_train),
        "n_test": len(y_test),
        "mape": mape(y_test, predictions),
        "mae": mae(y_test, predictions),
        "rmse": rmse(y_test, predictions),
        "train_time_s": elapsed,
    }
