# find_optimal_sarima_parameters.py

# --- Standard library ---
import os
import sys
import math
import json
import logging
import warnings
from datetime import date
from pathlib import Path
import itertools

# --- Third-party libraries ---
import numpy as np
from numpy import linalg as LA
import pandas as pd
import yaml
import joblib
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from dotenv import load_dotenv
# --- Project modules ---
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from scripts.utils.load_data import (
    load_individual,
    load_cumulative,
    load_distances,
    load_latest,
)
from scripts.utils.helper import *
from scripts.models.individual import Individual



# --- Warnings and logging setup ---
warnings.simplefilter("ignore", ConvergenceWarning)
logger = logging.getLogger(__name__)

# --- Environment setup ---
load_dotenv()
ROOT_PATH = os.getenv("ROOT_PATH")


class SarimaParameterFinder:
    """
    A class to find the optimal SARIMA parameters for time series data
    grouped by different program categories.
    """

    def __init__(self, configuration, data_latest, cumulative_model = None, individual_model = None):
        """
        Initializes the SarimaParameterFinder.
        """
        self.configuration = configuration
        self.cumulative_model = cumulative_model
        self.data_latest = data_latest
        self.individual_model = individual_model
        self.results_df = None

    def _create_time_series_individual(self, programme, herkomst, examentype, max_year):
        
        # --- Check if preapplicant probabilities are predicted ---
        if not self.individual_model.predicted:
            self.individual_model.predict_preapplicant_probabilities(2025, 38, predict = False)

        # --- Filter data based on the parameters given ---
        data = self.individual_model._filter_data(self.individual_model.data_individual.copy(), herkomst, max_year, programme, examentype)

        # --- Create time series data ---
        _, ts_data = self.individual_model._create_time_series(data, pred_len = 0)

        return ts_data

    def find_optimal_parameters(self, ts_data):
        """
        Performs a grid search to find the best SARIMA parameters (p,d,q)(P,D,Q)s.

        Args:
            ts_data (np.array): The time series data.

        Returns:
            tuple: A tuple containing the optimal parameters (p, d, q, P, D, Q).
        """
        # Define parameter ranges for the grid search
        p = d = q = range(0, 3)
        P = D = Q = range(0, 2)
        s = 52  # Seasonal cycle length (52 weeks in a year)

        # Generate all possible parameter combinations
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(itertools.product(P, D, Q))]

        best_aic = np.inf
        best_params = None
        best_seasonal_params = None

        # Grid search for the best parameters based on AIC
        for param in pdq:
            for seasonal_param in seasonal_pdq:
                print(param, seasonal_param)
                model = sm.tsa.statespace.SARIMAX(
                    ts_data,
                    order=param,
                    seasonal_order=seasonal_param,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                results = model.fit(disp=False)
                print(results.aic)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = param
                    best_seasonal_params = seasonal_param

        if not best_params:
            return 0, 0, 0, 0, 0, 0

        p, d, q = best_params
        P, D, Q, s = best_seasonal_params

        print('Best parameters:', best_params, best_seasonal_params)

        return p, d, q, P, D, Q

    def _run_single_combination(self, row):
        """
        Processes a single row (combination) to find its optimal SARIMA parameters.
        This function is designed to be used with pandas .apply().
        """
        opleiding = row["Croho groepeernaam"]
        herkomst = row["Herkomst"]
        examentype = row["Examentype"]

        print(f"Running for: {opleiding} | {herkomst} | {examentype}")

        # Step 1: Create the time series for the combination
        ts = self._create_time_series_individual(opleiding, herkomst, examentype, max_year = 2025)

        # Step 2: Find the optimal parameters
        p, d, q, P, D, Q = self.find_optimal_parameters(ts_data = ts)

        return pd.Series({"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q})

    def run_optimization(self):
        """
        Runs the entire optimization process for all unique program combinations.
        """
        # --- Apply filtering from configuration ---
        filtering = self.configuration["filtering"]

        # --- Filter data ---
        mask = np.ones(len(self.data_latest), dtype=bool) 

        # --- Apply conditional filters from configuration ---
        if filtering["programme"]:
            mask &= self.data_latest["Croho groepeernaam"].isin(filtering["programme"])
        if filtering["herkomst"]:
            mask &= self.data_latest["Herkomst"].isin(filtering["herkomst"])
        if filtering["examentype"]:
            mask &= self.data_latest["Examentype"].isin(filtering["examentype"])
        
        # --- Apply mask ---
        optimalization_df = self.data_latest.loc[mask, ["Croho groepeernaam", "Herkomst", "Examentype"]].copy()

        # --- Make sure the rows are unique ---
        optimalization_df = optimalization_df.drop_duplicates()

        # Apply the processing function to each row (combination)
        param_results = optimalization_df.apply(self._run_single_combination, axis=1)

        # Combine the original combinations with their new parameters
        self.results_df = pd.concat([optimalization_df, param_results], axis=1)

        print("\nOptimization complete.")
        return self.results_df

    def save_results(self):
        """
        Saves the resulting DataFrame with SARIMA parameters to an Excel file.
        """
        if self.results_df is not None:
            output_path = self.config["paths"]["path_sarima_paramters"]
            print(f"Saving results to {output_path}...")
            self.results_df.to_excel(output_path, index=False)
            print("Results saved successfully.")
        else:
            print("No results to save. Please run the optimization first.")


if __name__ == "__main__":
    # Define the path to your configuration file
    CONFIG_FILE = Path("configuration.yaml")
    with open(CONFIG_FILE, "r") as f:
        configuration = yaml.safe_load(f) 

    # --- Load data ---
    individual_data = load_individual()
    distances = load_distances()
    latest_data = load_latest()
    data_cumulative = load_cumulative()

    # --- Initialize models ---
    individual_model = Individual(individual_data, distances, latest_data, configuration, data_cumulative)

    # Step 1: Create an instance of the finder
    finder = SarimaParameterFinder(configuration, latest_data, individual_model = individual_model)

    # Step 2: Run the optimization for all program combinations
    finder.run_optimization()

    # Step 3: Save the final results to an Excel file
    #finder.save_results()
