# clustering.py

# --- Standard library ---
import sys
from pathlib import Path

# --- Third-party libraries ---
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# --- Project modules ---
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from scripts.utils.helper import get_weeks_list


def _get_cyclic_gaussian_week_weights(predict_week: int, sigma: float = 1.5) -> pd.Series:
    """
    Compute cyclic Gaussian weights. Weeks near `predict_week` receive higher weights, while distant weeks
    are down-weighted using a Gaussian kernel. 
    """

    # Define academic-year order: weeks 39–52, then 1–38
    weeks = np.array(get_weeks_list(predict_week))

    # Compute cyclic distance (wrap-around between 52 and 1)
    diff = np.abs(weeks - predict_week)
    diff = np.minimum(diff, 52 - diff)

    # Gaussian weighting
    weights = np.exp(-0.5 * (diff / sigma) ** 2)

    weights /= weights.sum()  # normalize so total weight = 1

    return pd.Series(weights, index=weeks)



def cluster_programme(data, programme, herkomst, examentype, predict_week, predict_year, configuration, ratio_threshold = 1.5, cat_weight = 2, force_programmes = None):
        """Function to assign a new programme to a cluster using KNN with categorical encoding."""

        data = data.copy()  

        valid_weeks = get_weeks_list(predict_week)
        valid_weeks = [str(x) for x in valid_weeks]
        df_wide = data[["Collegejaar", "Croho groepeernaam", "Faculteit", "Examentype", "Herkomst", 'Aantal_studenten'] + valid_weeks]

        # --- Prepare training and target data ---
        if programme in list(configuration["numerus_fixus"].keys()):
            population_mask = (
                (df_wide['Croho groepeernaam'] == programme) &
                (df_wide['Collegejaar'] < (predict_year)) &
                (df_wide['Herkomst'] == herkomst) &
                (df_wide['Examentype'] == examentype) &
                (df_wide["Collegejaar"] != configuration['covid_year'])
            )
        else:
            used_to_be_nf = configuration.get("used_to_be_numerus_fixus", {})

            # Precompute (programme, year) pairs to exclude
            exclude_pairs = {
                (prog, year)
                for prog, years in used_to_be_nf.items()
                for year in years
            }

            # Still apply NF mask if predicting for used to be NF
            if (programme, predict_year) in exclude_pairs:
                 population_mask = (
                    (df_wide['Croho groepeernaam'] == programme) &
                    (df_wide['Collegejaar'] != (predict_year - 1)) &
                    (df_wide['Herkomst'] == herkomst) &
                    (df_wide['Examentype'] == examentype) &
                    (df_wide["Collegejaar"] != configuration['covid_year'])
                )

            population_mask = (
                (df_wide["Collegejaar"] <= (predict_year - 1))
                & (df_wide["Herkomst"] == herkomst)
                & (df_wide["Examentype"] == examentype)
                & (~df_wide["Croho groepeernaam"].isin(configuration.get("numerus_fixus", [])))
                & (df_wide["Collegejaar"] != configuration['covid_year']) 
                & (~df_wide[["Croho groepeernaam", "Collegejaar"]]
                    .apply(tuple, axis=1)
                    .isin(exclude_pairs))
            )        


        population_data = df_wide[population_mask].copy()

        target_mask = (
            (df_wide['Croho groepeernaam'] == programme) &
            (df_wide['Collegejaar'] == predict_year) &
            (df_wide['Herkomst'] == herkomst) &
            (df_wide['Examentype'] == examentype)
        )
        target_data = df_wide[target_mask].copy()

        if not programme in list(configuration["new_programmes"].keys()):

            target_n = df_wide[
                (df_wide['Croho groepeernaam'] == programme) &
                (df_wide['Collegejaar'] <= predict_year) &
                (df_wide['Herkomst'] == herkomst) &
                (df_wide['Examentype'] == examentype)
            ]['Aantal_studenten'].mean()

            min_allowed = target_n - max(0.5 * target_n, 30)
            max_allowed = target_n + max(0.5 * target_n, 30)


            population_data = population_data[
                (population_data["Aantal_studenten"] >= min_allowed)
                & (population_data["Aantal_studenten"] <= max_allowed)
            ]

        # Drop Aantal_studenten before encoding / distance calculations
        population_data = population_data.drop(columns=["Aantal_studenten"], errors="ignore")
        target_data = target_data.drop(columns=["Aantal_studenten"], errors="ignore")

        originaL_population_data = population_data.copy()

        # Compute cyclic Gaussian weights for current prediction week
        week_weights = _get_cyclic_gaussian_week_weights(predict_week)
        week_weights.index = week_weights.index.astype(str)

        #print(week_weights)

        # Apply weights only to those columns (scales weeks before distance calculation)
        population_data[valid_weeks] = population_data[valid_weeks] * week_weights[valid_weeks].values
        target_data[valid_weeks] = target_data[valid_weeks] * week_weights[valid_weeks].values

        # --- Identify categorical columns ---
        cat_cols = ['Croho groepeernaam']

        # --- Fit encoder on population, transform both ---
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        population_encoded = encoder.fit_transform(population_data[cat_cols])
        target_encoded = encoder.transform(target_data[cat_cols])

        # --- Create DataFrames with proper column names ---
        encoded_cols = encoder.get_feature_names_out(cat_cols)
        population_encoded_df = pd.DataFrame(population_encoded, columns=encoded_cols, index=population_data.index)
        target_encoded_df = pd.DataFrame(target_encoded, columns=encoded_cols, index=target_data.index)

        
        # --- Apply categorical weight ---
        population_encoded_df *= cat_weight
        target_encoded_df *= cat_weight

        # --- Combine numeric columns back ---
        numeric_cols = population_data.select_dtypes(include='number').columns
        population_final = pd.concat([population_data[numeric_cols], population_encoded_df], axis=1)
        target_final = pd.concat([target_data[numeric_cols], target_encoded_df], axis=1)

        # --- Impute missing values ---
        imputer = SimpleImputer(fill_value=0)  
        population_imputed = pd.DataFrame(imputer.fit_transform(population_final),
                                        columns=population_final.columns,
                                        index=population_final.index)

        target_imputed = pd.DataFrame(imputer.transform(target_final),
                                    columns=target_final.columns,
                                    index=target_final.index)
        
        # --- Normalize all columns ---
        #scaler = StandardScaler()
        #population_imputed = pd.DataFrame(scaler.fit_transform(population_imputed),
        #                                columns=population_imputed.columns, index=population_imputed.index)
        #target_imputed = pd.DataFrame(scaler.transform(target_imputed),
        #                            columns=target_imputed.columns, index=target_imputed.index)

        # --- Fit NearestNeighbors on population ---
        n_samples = len(population_imputed)
        n_neighbors = min(10, n_samples - 1) if n_samples > 1 else 1
        
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
            
        # Fit the nearest neighbors model
        nn.fit(population_imputed)

        # --- Find closest population row for each target row ---
        distances, indices = nn.kneighbors(target_imputed)

        # Flatten results for a single target row
        distances = distances.flatten()
        indices = indices.flatten()

        # --- Sort by distance ascending ---
        sorted_idx = np.argsort(distances)
        distances_sorted = distances[sorted_idx]
        indices_sorted = indices[sorted_idx]

        # --- Filter out dissimilar programmes ---
        if len(distances_sorted) > 0:
            min_d = np.nanmin(distances_sorted)
            mask = distances_sorted <= ratio_threshold * min_d
        else:
            mask = np.array([], dtype=bool)

        indices_filt = indices_sorted[mask]
        distances_filt = distances_sorted[mask]

        # --- Guarantee at least N results if force_programmes is set ---
        if force_programmes is not None:
            n_force = int(force_programmes)
            if len(indices_filt) < n_force:
                # Take top N closest from the sorted arrays (not the filtered ones)
                indices_filt = indices_sorted[:n_force]
                distances_filt = distances_sorted[:n_force]

        # --- Use the final filtered arrays ---
        indices = indices_filt
        distances = distances_filt

        # --- Create a DataFrame of closest rows with distances ---
        closest_df = originaL_population_data.iloc[indices].copy()
        closest_df["distance"] = distances

        # Sort by distance ascending
        closest_df = closest_df.sort_values('distance', ascending=True)

        return closest_df[['Collegejaar', 'Croho groepeernaam', 'Faculteit', 'Examentype', 'Herkomst', 'distance']]

