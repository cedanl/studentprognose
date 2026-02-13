# bayesian_models.py

# --- Standard library ---
import sys
from pathlib import Path
import numpy as np
import warnings
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, RegressorMixin
import statsmodels.api as sm
from statsmodels.tsa.forecasting.theta import ThetaModel
from scipy.stats import norm
from typing import Optional, Dict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- Project modules ---
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from scripts.utils.helper import get_weeks_list, get_current_len, get_pred_len

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
#  Base Class: Bayesian Kalman-Smoothed Regressors
# ============================================================

class BaseBayesianRegressor(BaseEstimator, RegressorMixin):
    """
    Base class for models using:
    - Kalman-filtered local trend Poisson prior
    - RTS smoothing
    - Precision-weighted prior/posterior combination

    Subclasses implement domain-specific posterior likelihood.
    """

    LOG_OFFSET = 0.5
    EPS = 1e-9
    MAD_OUTLIER_MULT_WEEKLY = 1.5
    MAD_OUTLIER_MULT_YEAR = 1
    ABS_THRESHOLD = 25       
    REL_THRESHOLD = 0.7      

    # General colnames
    PROGRAMME_COL = 'Croho groepeernaam'
    ORIGIN_COL = 'Herkomst'
    EXAMTYPE_COL = 'Examentype'
    TARGET_COL = 'Aantal_studenten'

    # ------------------------------------------------------------
    def __init__(self, predict_week: int,  nf_programmes: dict = None, verbose: bool = True):
        self.verbose = verbose
        self.predict_week = predict_week
        self.nf_programmes = nf_programmes

    # ============================================================
    #  1) PRIOR: Sarima/Theta Trend 
    # ============================================================

    def local_trend_prior(self, counts: np.ndarray):
        """
        Fit a trend using Theta + Arima
        """
        counts = np.asarray(counts, dtype=float)


        self.last_year_value = counts[-1]
        
        if self.verbose:
            print(f'Students in earlier years: {counts}')

        n = len(counts)

        if n < 3:
            # Too few data points OR small counts detected → degenerate prior
            
            # Calculate mean_lambda from counts, or use a default if counts is empty
            lam = float(np.mean(counts)) if counts.size else 0.0
            
            # Original case for n < 3
            mean_rate = lam
            var_rate = np.nanvar(counts)

            alpha = mean_rate**2 / max(var_rate, self.EPS) if var_rate > 0 else 1.0 # Default to 1.0 if variance is 0
            beta = mean_rate / max(var_rate, self.EPS) if var_rate > 0 else 1.0 / max(lam, self.EPS)

            return {
                "alpha": alpha,
                "beta": beta,
                "mean_lambda": mean_rate,
                "var_lambda": var_rate
            }

        # Only run time series forecasting if we didn't trigger the degenerate prior
        time_series_result = self.forecaste_sarima_theta_ensemble(counts)

        # The rest of the original logic for processing the time_series_result follows...
        if time_series_result is not None:
            mean_rate = time_series_result['forecast']
            var_rate = time_series_result['variance']

            # Make sure mean is not below 0
            mean_rate = max(mean_rate, 0)

            # Note: If the time series *itself* produces a very high variance (var_rate), 
            alpha = mean_rate**2 / max(var_rate, self.EPS)
            beta = mean_rate / max(var_rate, self.EPS)

            return {
                "alpha": alpha,
                "beta": beta,
                "mean_lambda": mean_rate,
                "var_lambda": var_rate
            }
        
        # If the ensemble forecast failed, return the high-variance prior manually
        mean_rate = float(np.mean(counts)) if counts.size else 0.0
        var_rate = 10000.0
        alpha = mean_rate**2 / max(var_rate, self.EPS)
        beta = mean_rate / max(var_rate, self.EPS)
        
        return {
            "alpha": alpha,
            "beta": beta,
            "mean_lambda": mean_rate,
            "var_lambda": var_rate
        }
    
    # ------------------------------------------------------------
    def fit_sarimax(self, y: np.ndarray, horizon: int = 1) -> Optional[Dict[str, np.ndarray]]:
        """
        Fit a SARIMAX(0,1,1) model with no trend and return forecasts and their variances.
        """
        y_clean = y[~np.isnan(y)].astype(float)
        if y_clean.size < 1:
            if self.verbose:
                print("Insufficient data points to fit SARIMAX.")
            return None

        try:
            model = sm.tsa.SARIMAX(
                y_clean,
                order=(0, 1, 1),
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=True
            )
            result = model.fit(disp=False)
            forecast_obj = result.get_forecast(steps=horizon)

            # Convert to numpy arrays (even if horizon=1)
            forecast_array = np.asarray(forecast_obj.predicted_mean)
            variance_array = np.asarray(forecast_obj.var_pred_mean)

            return {
                "forecast": forecast_array,
                "variance": variance_array
            }

        except Exception as e:
            if self.verbose:
                print(f"SARIMAX fitting failed: {e}")
            return None

    def fit_theta(self, y: np.ndarray, horizon: int = 1) -> Optional[Dict[str, np.ndarray]]:
        """
        Fit a Theta model (period=1) and return forecasts and variance derived from prediction intervals.
        """
        y_clean = y[~np.isnan(y)].astype(float)
        if y_clean.size < 1:
            if self.verbose:
                print("Insufficient data points to fit Theta model.")
            return None

        try:
            model = ThetaModel(y_clean, period=1)
            result = model.fit()

            forecast_values = result.forecast(horizon)
            alpha = 0.05
            prediction_intervals = result.prediction_intervals(steps=horizon, alpha=alpha)

            # Correct: use .iloc for positional indexing
            z_score = norm.ppf(1 - alpha / 2)
            std_dev = (prediction_intervals.iloc[:, 1] - prediction_intervals.iloc[:, 0]) / (2 * z_score)
            variance_values = np.asarray(std_dev)

            return {
                "forecast": forecast_values,
                "variance": variance_values
            }

        except Exception as e:
            if self.verbose:
                print(f"Theta fitting failed: {e}")
            return None
        
    
    def forecaste_sarima_theta_ensemble(self, y: np.ndarray, horizon: int = 1) -> Optional[Dict[str, np.ndarray]]:
        """
        Fit both SARIMAX and Theta models and return an ensemble forecast
        and ensemble variance (averaged from both models).
        """
        sarimax_result = self.fit_sarimax(y, horizon)
        theta_result = self.fit_theta(y, horizon)

        # If both models fail, return None
        if sarimax_result is None and theta_result is None:
            if self.verbose:
                print("Both SARIMAX and Theta model fitting failed.")
            return None

        # If only one model succeeds, return that model's output
        if sarimax_result is None:
            return theta_result
        if theta_result is None:
            return sarimax_result

        # Both models succeeded: compute ensemble as the mean of forecasts and variances
        ensemble_forecast = (sarimax_result["forecast"] + theta_result["forecast"]) / 2
        ensemble_variance = (sarimax_result["variance"] + theta_result["variance"]) / 2

        return {
            "forecast": ensemble_forecast.item(),
            "variance": ensemble_variance.item()
        }

    # ============================================================
    #  Week weights
    # ============================================================
    
    def _compute_predictive_weights(self, X, y, predict_week, threshold=0.8, sharpness=5.0):
            """
            Compute weekly predictive weights based on Pearson R² scores.
            """
            def _get_r2(model, col, y_true):
                model.fit(col, y_true)
                y_pred = model.predict(col)
                r2 = r2_score(y, y_pred)
                return max(r2, 0)
            
            # --- Prepare data ---
            valid_weeks = list(map(str, reversed(get_weeks_list(predict_week))))
            X_mat = X[valid_weeks].to_numpy(dtype=float)
            y = np.asarray(y, dtype=float)
            n_weeks = len(valid_weeks)

            # Initialize informative_r2_ to 0 in case loop is skipped
            self.informative_r2_ = 0.0 
            r2_scores = np.zeros(n_weeks)
            model = LinearRegression(positive=True, fit_intercept=False)

            # --- Compute per-week R² scores ---
            # Only run if we have valid variation in y
            if y.size > 1 and y.std() > 0:
                for i, w in enumerate(valid_weeks):
                    col = X_mat[:, i]

                    # Skip constant columns
                    if col.std() > 0:
                        r2 = _get_r2(model, col.reshape(-1, 1), y)
                        r2_scores[i] = max(r2, 0)

                    # Check cumulative R2 up to this week
                    cols = X_mat[:, :i+1]
                    self.informative_r2_ = _get_r2(model, cols, y)  

                    # Stop early if threshold reached
                    if self.informative_r2_ >= threshold:
                        if self.verbose:
                            print(
                                f"R² = {self.informative_r2_:.2f} up to week {w}. "
                                "All weeks up to this week are selected."
                            )
                        # Return weights immediately
                        active_scores = r2_scores[:i+1]
                        
                        # Apply softmax ONLY to the active subset
                        # If i=0 (only Week 24), softmax([x]) results in [1.0]
                        active_weights = softmax(active_scores * sharpness)
                        
                        # Create full array and fill only the active slots
                        final_weights = np.zeros(n_weeks)
                        final_weights[:i+1] = active_weights
                        
                        # Flip to return to chronological order (Oldest -> Newest)
                        return np.flip(final_weights)
                
                # If loop finishes without returning, we calculate the final R2 on ALL data
                self.informative_r2_ = _get_r2(model, X_mat, y)

            if self.verbose:
                print(f"No week found that was above the threshold. Using all weeks")
                print(f" R² = {self.informative_r2_:.2f}. ")

            # --- Compute softmax weights ---
            if np.any(r2_scores > 0):
                weighted_scores = softmax(r2_scores * sharpness)
            else:
                # Fallback to uniform weights if no correlation found or y was constant
                weighted_scores = np.ones(n_weeks) / n_weeks

            # Return weights in chronological order (earliest week first)
            return np.flip(weighted_scores)

    # ============================================================
    #  Posterior Combination
    # ============================================================

    def _combine_prior_posterior(self, posterior, lambda_decay, posterior_trust, r2_adjustment_weight = 0.75):
        """Precision-weighted combination of prior + posterior."""
        posterior = np.asarray(posterior)

        if posterior.size == 0 or np.allclose(posterior, 0):
            if self.verbose:
                print("Posterior empty → using prior only.")
            return np.array([round(self.prior_mean_)])
        
        n = len(posterior)

        # indices from oldest → newest
        idx = np.arange(n)            # [0, 1, 2, ..., n-1]


        # we want highest weight for the LAST entry
        exponent = (n-1) - idx       

        w_value = np.exp(-lambda_decay * exponent)
        w_value /= w_value.sum()        # normalize

        # weighted mean
        m_post = float(np.sum(w_value * posterior))

        # weighted variance 
        v_post = float(np.sum(w_value * (posterior - m_post)**2) + self.EPS)

        m_prior = getattr(self, "prior_mean_", m_post)
        v_prior = getattr(self, "prior_var_", v_post)

        if hasattr(self, "informative_r2_"):
            
            r2_scaled = self.informative_r2_ * r2_adjustment_weight

            v_prior = v_prior / (1 - (r2_scaled + self.EPS))

            if r2_scaled == 0:
                v_post = v_prior * 10 # arbitrary big number
            else:
                v_post = v_post / (r2_scaled + self.EPS)

        
        if self.verbose:
            print("=== Prior ===")
            print(f"Mean lambda:     {m_prior:.2f}")
            print(f"Variance lambda: {v_prior:.2f}")
            print("=========================")
            print("=== Posterior ===")
            print(f"Mean: {m_post:.2f}")
            print(f"Var:  {v_post:.2f}")
            print("=================")
        
        # Calculate sequential week index (1 to 52)
        predict_week = self.predict_week # This is the current week number (e.g., 45 or 12)
        total_weeks_in_cycle = 52.0
        
        if predict_week >= 39:
            # Weeks 39 through 52
            sequential_week = predict_week - 38
        else:
            # Weeks 1 through 38
            sequential_week = predict_week + 14
        
        # Calculate progress (0.0 to 1.0)
        progress = sequential_week / total_weeks_in_cycle

        # 1. Calculate precisions based on R2 adjustments
        prec_prior = 1.0 / v_prior
        prec_post  = 1.0 / v_post

        # 2. Apply time decay to the Prior's precision
        time_decay_factor = 1.0 - progress 
        prec_prior = prec_prior * time_decay_factor

        # 3. Combine... (rest of the combination logic remains the same)
        inv = prec_prior + (prec_post * posterior_trust)

        # Safety check for div by zero
        if inv == 0: inv = 1.0 / self.EPS 
            
        v_final = 1.0 / inv
        m_final = v_final * (m_prior * prec_prior + m_post * (prec_post * posterior_trust))

        return np.array([round(m_final)])
        

        # 1. Calculate raw precisions
        prec_prior = 1.0 / v_prior
        prec_post  = 1.0 / v_post

        # 2. Boost the posterior precision
        prec_post_weighted = prec_post * posterior_trust

        # 3. Combine
        inv = prec_prior + prec_post_weighted
        v_final = 1.0 / inv
        
        m_final = v_final * (m_prior * prec_prior + m_post * prec_post_weighted)

        return np.array([round(m_final)])


# ============================================================
#  Ratio-Based Bayesian Regression
# ============================================================

class BayesianRatioRegressor(BaseBayesianRegressor):
    """
    Bayesian regressor using weekly ratios (X_week / y).
    """

    MAD_OUTLIER_MULT_WEEKLY = 1.5
    MAD_OUTLIER_MULT_YEAR = 1.0

    # ------------------------------------------------------------
    def fit(self, X, y):
        if self.verbose:
            print("=====================================")
            print("Ratio Model Fit")
            print("=====================================")
        
        # Replace nan with 0
        y = np.nan_to_num(y, nan=0)

        prior = self.local_trend_prior(y)
        self.prior_mean_ = prior["mean_lambda"]
        self.prior_var_ = prior["var_lambda"] + self.EPS
        self.prior_info_ = prior

        # Weekly weight selection
        self.week_weights_ = self._compute_predictive_weights(
            X, y, self.predict_week
        )

        # Store years
        self.years_ = np.array(X.Collegejaar)

        # Ratio matrix (weekly cumulative predictors / y)
        self.ratios_ = self._compute_weekly_ratios(X, y)
        self.predict_week_ = self.predict_week
        self.is_fitted_ = True

        return self

    # ------------------------------------------------------------
    def predict(self, X):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit() before predict().")
        
        evidence = self._likelihood_ratio_model(
            X, self.ratios_, self.week_weights_, self.predict_week_
        )
        return self._combine_prior_posterior(evidence, lambda_decay = 0.5, posterior_trust=2.0) # Higher trust in recent (weekly) data
 
    # ============================================================
    #  Ratio Model Helpers
    # ============================================================

    # ------------------------------------------------------------
    @staticmethod
    def _compute_weekly_ratios(X, y):
        y = np.asarray(y).reshape(-1, 1)
        valid_weeks = list(map(str, get_weeks_list(38)))
        ratios = X[valid_weeks].astype(float).to_numpy() / np.maximum(y, 1e-9)
        return ratios

    # ------------------------------------------------------------
    def _likelihood_ratio_model(self, X_test, ratios, w_week, predict_week):
        pred_len = get_current_len(predict_week)
        weeks = list(map(str, get_weeks_list(predict_week)))

        X_sub = X_test[weeks].astype(float).to_numpy()

        # Pad or trim historical ratios
        r = ratios
        if r.shape[1] < pred_len:
            pad = r[:, [-1]].repeat(pred_len - r.shape[1], axis=1)
            r = np.hstack([r, pad])
        else:
            r = r[:, :pred_len]

        # Normalized weekly weights
        w_week = np.asarray(w_week)
        if w_week.size < pred_len:
            w_week = np.pad(w_week, (0, pred_len - w_week.size), constant_values=w_week[-1])
        w_week = w_week[:pred_len]
        w_week /= w_week.sum()

        try:
            preds = X_sub / np.maximum(r, self.EPS)

            # Compute threshold
            threshold = np.maximum(self.last_year_value * self.REL_THRESHOLD, self.ABS_THRESHOLD)

            # Compute min/max bounds
            minimum_value = max(self.last_year_value - threshold,0)
            maximum_value = self.last_year_value + threshold

            # Clip the array
            preds = np.clip(preds, minimum_value, maximum_value)

            #print(preds)

            evidence = preds @ w_week

            if self.verbose:
                print("=== Evidence (Ratio Model) before the threshold ===")
                print([f"{x:.1f}" for x in evidence])
                print("==============================")

            # Keep only evidence values within thresholds 
            mask = (evidence >= minimum_value) & (evidence <= maximum_value)
            evidence = evidence[mask]

            # Also remove the values that are exactly the min or max value to not mess with the variance
            mask = (evidence != minimum_value) & (evidence != maximum_value)
            evidence = evidence[mask]

            if self.verbose:
                print("=== Evidence (Ratio Model) after the threshold ===")
                print([f"{x:.1f}" for x in evidence])
                print("==============================")

        except ValueError:
            evidence = []

        return evidence
    


# ============================================================
#  Cluster-Level Bayesian Regression
# ============================================================

class BayesianClusterRegressor(BaseBayesianRegressor):
    """
    Simple Bayesian model using cluster-level yearly counts
    as posterior evidence.
    """

    def fit(self, train_prog, train_total, target):
        if self.verbose:
            print("=====================================")
            print("Cluster Model Fit")
            print("=====================================")

        try:
            y_programme = train_prog['Aantal_studenten']

            # Replace nan with 0
            y_programme = np.nan_to_num(y_programme, nan=0)

            prior = self.local_trend_prior(y_programme)
            self.prior_mean_ = prior["mean_lambda"]
            self.prior_var_ = prior["var_lambda"] + self.EPS
            self.prior_info_ = prior

            # --- Get the weights ---
            self.weights = self._compute_predictive_weights(train_prog, y_programme, self.predict_week)
        except IndexError:
            pass # Skip prior when there is no history

        # --- Get the cluster ---
        try:
            self.clusterd_values = self._get_cluster(train_total, target)
        except ValueError:
            pass

        self.predict_week_ = self.predict_week
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------
    def predict(self):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit() before predict().")

        try:
            evidence = self._cluster_likelihood(self.clusterd_values)
        except AttributeError:
            evidence = []

        return self._combine_prior_posterior(evidence, lambda_decay=0.2, posterior_trust = 1.0) # Lower decay here, lower posterior trust

    # ------------------------------------------------------------

    def _cluster_likelihood(self, y_cluster):
        evidence = np.asarray(y_cluster, dtype=float)

        if self.verbose:
            print("=== Evidence (Cluster Model)  ===")
            print([f"{x:.1f}" for x in evidence])
            print("==============================")

        return evidence
    
            
    # ------------------------------------------------------------
    def _get_cluster(self, train, target, ratio_threshold = 1.5, force_programmes = 5):
        """Function to assign a new programme to a cluster using KNN with categorical encoding."""

        # --- Apply weekly weights ---
        valid_weeks = [str(x) for x in get_weeks_list(self.predict_week)] 

        try:

            # --- Filter unreasonable rows (outside threshold) --- 
            threshold = np.maximum(self.last_year_value * self.REL_THRESHOLD, self.ABS_THRESHOLD)

            # Compute min/max bounds
            minimum_value = self.last_year_value - threshold
            maximum_value = self.last_year_value + threshold

            # Keep only evidence values within thresholds
            mask = (train[self.TARGET_COL] >= minimum_value) & (train[self.TARGET_COL] <= maximum_value)
            train = train[mask]
        except AttributeError:
            pass # Skip this part when there is no history

        train_original = train.copy()

        train = train[valid_weeks].to_numpy(dtype=float)
        target = target[valid_weeks].to_numpy(dtype=float)


        try:
            train = train * self.weights
            target = target * self.weights
        except AttributeError:
            pass # Skip this part when there is no history

        # --- Fit NearestNeighbors on population ---
        n_samples = len(train)
        n_neighbors = min(10, n_samples - 1) if n_samples > 1 else 1
        
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
            
        # Fit the nearest neighbors model
        nn.fit(train)

        # --- Find closest population row for each target row ---
        distances, indices = nn.kneighbors(target)

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
        closest_df = train_original.iloc[indices].copy()
        closest_df["distance"] = distances

        # Sort by distance ascending
        closest_df = closest_df.sort_values('distance', ascending=False)

        if self.verbose:
            print('Cluster df:')
            print(closest_df[['Collegejaar', 'Croho groepeernaam', 'Faculteit', 'Examentype', 'Herkomst', 'distance']])

        y_values = np.array(closest_df[self.TARGET_COL])

        return y_values





# ============================================================
#  Time-series Bayesian Regression
# ============================================================

class BayesianTSRegressor(BaseBayesianRegressor):
    """
    Simple Bayesian model using time-series analysis
    as posterior evidence.
    """

    def fit(self, train_prog, train_total, target):
        if self.verbose:
            print("=====================================")
            print("Time-series Model Fit")
            print("=====================================")

        try:
            y_programme = train_prog['Aantal_studenten']

            # Replace nan with 0
            y_programme = np.nan_to_num(y_programme, nan=0)

            prior = self.local_trend_prior(y_programme)
            self.prior_mean_ = prior["mean_lambda"]
            self.prior_var_ = prior["var_lambda"] + self.EPS
            self.prior_info_ = prior

        except IndexError:
            pass # Skip prior when there is no history

        # --- Fit the model ---
        self.time_series_prediction(train_prog, target)

        self.predict_week_ = self.predict_week
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------
    def predict(self):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit() before predict().")

        evidence = self._cluster_likelihood(self.clusterd_values)
        return self._combine_prior_posterior(evidence, lambda_decay=0.2, posterior_trust = 1.0) # Lower decay here, lower posterior trust

    # ------------------------------------------------------------

    def _cluster_likelihood(self, y_cluster):
        evidence = np.asarray(y_cluster, dtype=float)

        if self.verbose:
            print("=== Evidence (Cluster Model)  ===")
            print([f"{x:.1f}" for x in evidence])
            print("==============================")

        return evidence
    
            
    # ------------------------------------------------------------
    def _create_time_series(self, train, target):

        # --- First create one large time series from the train data ---
        train = train.loc[:, '39':'38'] # Only the weekly data
        train_ts_data = train.values.flatten() # Flatten into a numpy array

        # --- Similar process to the target data
        valid_weeks = [str(x) for x in get_weeks_list(self.predict_week)] 
        target = target[valid_weeks]
        target_ts_data = target.values.flatten() # Flatten into a numpy array

        # --- Combine them togehter --- 
        ts_data = np.concatenate([train_ts_data, target_ts_data])

        return ts_data
    
    def _fit_sarima(self, ts_data: np.ndarray):

        sarimax_args = dict(
            order=(1, 0, 1),
            seasonal_order= (1, 1, 1, 52),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        model = sm.tsa.SARIMAX(ts_data, **sarimax_args)

        init_params = {
            'ar.L1': 0.9420447264310663,
            'ma.L1': -0.14166284881358818,
            'ar.S.L52': -0.1504389257703507,
            'ma.S.L52': -1.0000560934942428,
            'sigma2': 22.219914878554132,
        }
        param_vector = [init_params[name] for name in model.param_names]

        fitted_model = model.fit(start_params=param_vector, disp=False)

        return fitted_model


    def time_series_prediction(self, train_prog, target):

        ts_data = self._create_time_series(train_prog, target)

        model = self._fit_sarima(ts_data)

        # --- Get prediction length --- 
        pred_len = get_pred_len(self.predict_week)

        # --- Full forecast object ---
        forecast_obj = model.get_forecast(steps=pred_len)

        # Mean forecast (NumPy array or pandas Series)
        mean_forecast = forecast_obj.predicted_mean

        # Variance of predictive distribution (NumPy array)
        var_forecast = forecast_obj.var_pred_mean

        # Final values
        prediction = round(mean_forecast[-1])      
        variance   = var_forecast[-1]           

        print("Prediction:", prediction)
        print("Variance:", variance)

        return prediction, variance


