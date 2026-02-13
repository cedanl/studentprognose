# evaluation.py

import numpy as np
import sys
from pathlib import Path

# --- Project modules ---
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

GROUP_COLS = [
    "Croho groepeernaam", "Faculteit",
    "Examentype", "Herkomst", "Weeknummer"
]

YEAR_COL = ['Collegejaar']

def evaluate_predictions(df, actual_col, pred_cols, configuration, args,
                         baseline_col=None, print_programmes=True,
                         absolute_margin=5, relative_margin=0.1):
    """
    Evaluate multiple prediction columns with optional baseline and configuration-based filtering.

    Prints MAE, RMSE, MAPE, Within Tolerance Rate, and wins/losses per programme.
    """

    df = df.copy()
    # --- Filter by Collegejaar ---
    df = df[df["Collegejaar"] >= 2021]

    # --- Apply configuration filters ---
    filtering = configuration.get("filtering", {})
    mask = np.ones(len(df), dtype=bool)

    if filtering.get("programme"):
        mask &= df["Croho groepeernaam"].isin(filtering["programme"])
    if filtering.get("herkomst"):
        mask &= df["Herkomst"].isin(filtering["herkomst"])
    if filtering.get("examentype"):
        mask &= df["Examentype"].isin(filtering["examentype"])
    
    predict_week = args.weeks[0]
    if predict_week != 999:
        mask &= df["Weeknummer"] == predict_week

    # Filter rows without predictions (or 0 when actual <= 5)
    mask &= (df[pred_cols].sum(axis=1) > 0) | (df[actual_col] <= 5)
    cols_to_keep = YEAR_COL + GROUP_COLS + [actual_col] + pred_cols

    if baseline_col:
        cols_to_keep.append(baseline_col)
    df = df.loc[mask, cols_to_keep].copy()
    df = df.dropna(subset=[actual_col])
    # Compute tolerance
    df['tolerance'] = np.maximum(absolute_margin, df[actual_col] * relative_margin)

    # Optionally filter by year
    predict_year = args.years[0]
    df = df[df["Collegejaar"] == predict_year]

    # --- Evaluate each prediction column ---
    for col in pred_cols:
        # Store errors in DataFrame for later use
        df['abs_error'] = (df[col] - df[actual_col]).abs()
        df['squared_error'] = (df[col] - df[actual_col]) ** 2
        df['mape_component'] = df['abs_error'] / (df[actual_col] + 1e-6)
        df['within_tol'] = df['abs_error'] <= df['tolerance']

        if baseline_col:
            df['baseline_abs_error'] = (df[baseline_col] - df[actual_col]).abs()
            df['baseline_squared_error'] = (df[baseline_col] - df[actual_col]) ** 2
            df['baseline_mape_component'] = df['baseline_abs_error'] / (df[actual_col] + 1e-6)
            df['baseline_within_tol'] = df['baseline_abs_error'] <= df['tolerance']

        print(f"\n--- Evaluation for '{col}' ---")
        print(f"MAE:                  {df['abs_error'].mean():.4f}", end='')
        if baseline_col:
            print(f" ({df['baseline_abs_error'].mean():.4f})", end='')
        print()

        print(f"RMSE:                 {np.sqrt(df['squared_error'].mean()):.4f}", end='')
        if baseline_col:
            print(f" ({np.sqrt(df['baseline_squared_error'].mean()):.4f})", end='')
        print()

        print(f"MAPE:                 {df['mape_component'].mean():.4f}", end='')
        if baseline_col:
            print(f" ({df['baseline_mape_component'].mean():.4f})", end='')
        print()

        print(f"Within Tolerance Rate: {df['within_tol'].mean():.2%}", end='')
        if baseline_col:
            print(f" ({df['baseline_within_tol'].mean():.2%})", end='')
        print()

        # --- Print programme-level wins/losses vs baseline ---
        if baseline_col and print_programmes:
            comparison_df = df.copy()
            comparison_df['model_better'] = comparison_df['abs_error'] < comparison_df['baseline_abs_error']

            wins = comparison_df[comparison_df['model_better']].sort_values('abs_error')
            losses = comparison_df[~comparison_df['model_better']].sort_values('abs_error', ascending=False)

            print(f"\nModel won in {len(wins)} rows, lost in {len(losses)} rows\n")
            display_cols = GROUP_COLS + [actual_col, col, baseline_col, 'model_better', 'abs_error', 'baseline_abs_error']
            print("Top 10 wins:")
            print(wins[display_cols].head(10))
            print("Top 10 losses:")
            print(losses[display_cols].head(10))
        print("-------------------------------")