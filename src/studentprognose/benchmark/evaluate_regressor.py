import time

import numpy as np
import pandas as pd

from studentprognose.benchmark.metrics import mape, mae, rmse
from studentprognose.benchmark.splitter import time_series_split
from studentprognose.config import get_columns, get_model_features, get_final_academic_week
from studentprognose.models.regressors import build_preprocessor
from studentprognose.strategies.cumulative import ENGINEERED_FEATURE_COLS
from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK


def evaluate_regressor_model(
    full_data: pd.DataFrame,
    data_studentcount: pd.DataFrame,
    regressor_factory: callable,
    config: dict,
    min_training_year: int = 2016,
    min_train_years: int = 3,
    numerus_fixus_list: list[str] | None = None,
) -> pd.DataFrame:
    """Evalueer een regressiemodel op stap 2 (voorspelling Aantal_studenten) via cross-validatie.

    Args:
        full_data: Wide-format data met SARIMA-voorspellingen en engineered features.
        data_studentcount: DataFrame met studentaantallen per (programme, jaar, herkomst, examentype).
        regressor_factory: Callable die een vers BaseRegressor-object retourneert.
        config: Configuratie-dict met column_roles en model_features.
        min_training_year: Vroegste trainingsjaar.
        min_train_years: Minimaal aantal trainingsjaren.
        numerus_fixus_list: Lijst van NF-programma's (voor aparte rapportage).

    Returns:
        DataFrame met per (examentype, test_year) de metrics en trainingsset-grootte.
    """
    c = get_columns(config)
    mf = get_model_features(config)
    reg_categorical = mf.get("regressor", {}).get("categorical")
    final_week = get_final_academic_week(config)

    if numerus_fixus_list is None:
        numerus_fixus_list = []

    merge_cols = [c.programme, c.academic_year, c.origin, c.exam_type]

    merged = full_data.merge(
        data_studentcount[merge_cols + [c.student_count]],
        on=merge_cols,
        how="inner",
    ).drop_duplicates(ignore_index=True)

    if merged.empty:
        return pd.DataFrame()

    splits = time_series_split(merged, min_training_year, min_train_years, year_col=c.academic_year)
    results = []

    for train_df, test_df in splits:
        test_year = int(test_df[c.academic_year].iloc[0])

        assert train_df[c.academic_year].max() < test_year, "Leakage: train bevat testjaar"

        for examentype in test_df[c.exam_type].unique():
            train_et = train_df[
                (train_df[c.exam_type] == examentype)
                & (~train_df[c.programme].isin(numerus_fixus_list))
            ]
            test_et = test_df[
                (test_df[c.exam_type] == examentype)
                & (~test_df[c.programme].isin(numerus_fixus_list))
            ]

            if train_et.empty or test_et.empty:
                continue

            result = _evaluate_single(
                train_et, test_et, regressor_factory,
                examentype, test_year, is_numerus_fixus=False,
                target_col=c.student_count,
                year_col=c.academic_year,
                reg_categorical=reg_categorical,
                final_week=final_week,
            )
            if result:
                results.append(result)

        for nf_prog in numerus_fixus_list:
            train_nf = train_df[train_df[c.programme] == nf_prog]
            test_nf = test_df[test_df[c.programme] == nf_prog]

            if train_nf.empty or test_nf.empty:
                continue

            examentype = test_nf[c.exam_type].iloc[0]
            result = _evaluate_single(
                train_nf, test_nf, regressor_factory,
                examentype, test_year, is_numerus_fixus=True,
                programme=nf_prog,
                target_col=c.student_count,
                year_col=c.academic_year,
                reg_categorical=reg_categorical,
                final_week=final_week,
            )
            if result:
                results.append(result)

    return pd.DataFrame(results)


def _evaluate_single(
    train_df, test_df, regressor_factory,
    examentype, test_year, is_numerus_fixus, programme=None,
    target_col="Aantal_studenten", year_col="Collegejaar",
    reg_categorical=None, final_week=FINAL_ACADEMIC_WEEK,
):
    """Evalueer een regressor op een enkele train/test split."""
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])

    # Geef available_cols mee, net als de productie-pijplijn (predict_with_xgboost):
    # de wide-pivot bevat niet altijd élke weekkolom tot final_week (bijv. UvA met
    # final_week 36 levert ~43 i.p.v. 52 weken). Zonder deze filter eist de
    # ColumnTransformer ontbrekende kolommen op en faalt elke fold stil.
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(
        ENGINEERED_FEATURE_COLS,
        categorical_cols=reg_categorical,
        year_col=year_col,
        final_week=final_week,
        available_cols=list(X_train.columns),
    )

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
