import pandas as pd
import xgboost as xgb
from typing import Dict, Any, List, Tuple 
import numpy as np
import os
import sys
import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.load_data import load_configuration

class HigherYearsPredictor:
    """
    Predicts student retention (registration in the next academic year) 
    based on historical data using an XGBoost model.

    The prediction is performed separately for each combination of 
    'Groepeernaam Croho' (program group) and 'EER-NL-nietEER' (origin).
    """

    def __init__(self, data_october: pd.DataFrame):
        """
        Initializes the predictor with the main dataset and XGBoost parameters.

        Args:
            data_october (pd.DataFrame): The input DataFrame containing student registration data.
                                         Expected columns include 'ID', 'Collegejaar', 
                                         'Croho opleiding code', 'Groepeernaam Croho', 
                                         'EER-NL-nietEER', etc.
            xgb_params (Dict[str, Any]): Dictionary of parameters for the XGBoost model.
                                         Example: {'learning_rate': 0.01}
        """
        if not isinstance(data_october, pd.DataFrame):
            raise TypeError("data_october must be a pandas DataFrame.")
            
        self.raw_data = data_october.copy() # Keep a copy of the raw data if needed
        self.xgb_params = {'learning_rate': 0.01, 'objective': 'binary:logistic', 'eval_metric': 'logloss'} 
        self.processed_data = self._preprocess_base_data(data_october)
        self.feature_columns = [
             'Collegejaar', 'Naam faculteit Nederlands', 'Groepeernaam Croho', 
             'Examentype code', 'Studievorm code', 'Geslacht', 
             'Type vooropleiding eerste vooropleiding', 'Aantal eerstejaarsinstelling', 
             'Aantal eerstejaars croho', 'Aantal Hoofdinschrijvingen', 
             'Aantal neveninschrijvingen', 'EER-NL-nietEER', 
             'hoeveelste_jaar', 'andere_opleiding'
        ]
        self.target_column = 'next_year_registered'
        self.id_column = 'ID'
        self.prediction_df = pd.DataFrame()
        self.trained_year = None

    def _preprocess_base_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Performs initial preprocessing steps common to all predictions.

        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The preprocessed DataFrame before group-specific filtering.
        """
        df = data.sort_values(by=['ID', 'Collegejaar', 'Croho opleiding code']).copy()

        # Create a column to check if the student is also registered in the next year
        df['next_year_registered'] = df.groupby(['ID', 'Croho opleiding code'])['Collegejaar'].shift(-1) == df['Collegejaar'] + 1
        df['next_year_registered'] = df['next_year_registered'].fillna(False).astype(int) # Handle NaNs and convert

        # Calculate year in program and if student has other registrations
        df['hoeveelste_jaar'] = df.groupby(['ID', 'Groepeernaam Croho']).cumcount() + 1
        df['andere_opleiding'] = df.duplicated(subset=['ID', 'Collegejaar'], keep=False).astype(int)
        
        # Filter based on general criteria
        valid_exam_types = ['Bachelor eerstejaars', 'Bachelor hogerejaars', 'Pre-master', 'Master']
        df = df[df['Examentype code'].isin(valid_exam_types)]
        df = df[df['Aantal Hoofdinschrijvingen'] == 1] # Assuming only primary registrations matter
        
        # Select relevant columns (features + ID + target)
        columns_to_keep = [
            'ID', 'Collegejaar', 'Naam faculteit Nederlands', 'Groepeernaam Croho', 
            'Examentype code', 'Studievorm code', 'Geslacht', 
            'Type vooropleiding eerste vooropleiding', 'Aantal eerstejaarsinstelling', 
            'Aantal eerstejaars croho', 'Aantal Hoofdinschrijvingen', 
            'Aantal neveninschrijvingen', 'EER-NL-nietEER', 
            'next_year_registered', 'hoeveelste_jaar', 'andere_opleiding'
        ]
        # Ensure all needed columns exist before selecting
        missing_cols = [col for col in columns_to_keep if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in input data: {missing_cols}")
            
        df = df[columns_to_keep]        
        return df

    def _split_and_encode(self, group_data: pd.DataFrame, train_year_max: int, test_year: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
        """
        Splits data into train/test sets based on year and performs one-hot encoding.

        Args:
            group_data (pd.DataFrame): Data filtered for a specific OPL/HERKOMST group.
            train_year_max (int): The latest year included in the training set.
            test_year (int): The year used for the test set.

        Returns:
            Tuple containing:
                - X_train_encoded (pd.DataFrame): Encoded training features.
                - y_train (pd.Series): Training target variable.
                - X_test_encoded (pd.DataFrame): Encoded test features.
                - y_test (pd.Series): Test target variable.
                - train_columns (List[str]): List of columns after encoding the training set.
        """
        train = group_data[group_data['Collegejaar'] <= train_year_max].copy()
        test = group_data[group_data['Collegejaar'] == test_year].copy()

        if train.empty:
             print(f"Warning: No training data found for year <= {train_year_max}.")
             # Return empty structures or handle as appropriate
             return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame(), pd.Series(dtype=int), []
        if test.empty:
             print(f"Warning: No testing data found for year == {test_year}.")
             # Return empty structures or handle as appropriate
             return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame(), pd.Series(dtype=int), []

        # Select features and target
        X_train = train[self.feature_columns]
        y_train = train[self.target_column]
        X_test = test[self.feature_columns]
        y_test = test[self.target_column]
        
        # One-hot encode categorical features
        X_train_encoded = pd.get_dummies(X_train, drop_first=True)
        X_test_encoded = pd.get_dummies(X_test, drop_first=True)

        # Align columns - crucial step
        train_columns = X_train_encoded.columns.tolist()
        X_test_encoded = X_test_encoded.reindex(columns=train_columns, fill_value=0)
        # Ensure columns are in the same order (reindex might change order)
        X_test_encoded = X_test_encoded[train_columns] 

        return X_train_encoded, y_train, X_test_encoded, y_test, train_columns


    def train_and_predict_for_group(self, opleiding: str, herkomst: str, train_year_max: int, test_year: int, num_boost_round: int = 200) -> float:
        """
        Trains an XGBoost model and predicts retention for a specific group.

        Args:
            opleiding (str): The 'Groepeernaam Croho' to filter by.
            herkomst (str): The 'EER-NL-nietEER' value to filter by.
            train_year_max (int): The latest year included in the training set.
            test_year (int): The year used for the test set.
            num_boost_round (int): Number of boosting rounds for XGBoost.

        Returns:
            float: The sum of predicted retention probabilities for the test year group. 
                   Returns 0.0 if no test data exists for the group.
        """
        print(f"\nProcessing Group: Opleiding='{opleiding}', Herkomst='{herkomst}'")
        
        # Filter data for the specific group
        group_data = self.processed_data[
            (self.processed_data['Groepeernaam Croho'] == opleiding) & 
            (self.processed_data['EER-NL-nietEER'] == herkomst)
        ]

        if group_data.empty:
            print("Warning: No data found for this group after initial filtering.")
            return 0

        # Split and encode
        X_train_encoded, y_train, X_test_encoded, y_test, train_columns = self._split_and_encode(
            group_data, train_year_max, test_year
        )
        
        # Check if data splitting resulted in usable sets
        if X_train_encoded.empty or X_test_encoded.empty:
             print("Skipping training/prediction due to empty train or test set after split.")
             return 0 # No predictions possible if test set is empty

        # Create DMatrix for XGBoost (optional, but can improve performance)
        try:
            dtrain = xgb.DMatrix(X_train_encoded, label=y_train, feature_names=train_columns)
            dtest = xgb.DMatrix(X_test_encoded, label=y_test, feature_names=train_columns)
        except Exception as e:
            print(f"Error creating DMatrix: {e}")
            print("Train columns:", train_columns)
            print("Test columns:", X_test_encoded.columns.tolist())
            # print("X_train shape:", X_train_encoded.shape, "dtype:", X_train_encoded.info())
            # print("X_test shape:", X_test_encoded.shape, "dtype:", X_test_encoded.info())
            return 0 # Cannot proceed without DMatrix

        # Train the XGBoost model
        try:
            model = xgb.train(self.xgb_params, dtrain, num_boost_round=num_boost_round)
        except Exception as e:
             print(f"Error during XGBoost training: {e}")
             return 0

        # Make predictions (probabilities)
        y_pred_proba = model.predict(dtest)

        # Return the sum of predicted probabilities for this group
        prediction_sum = round(y_pred_proba.sum())
        print(f"Predicted sum for group: {prediction_sum}")
        return prediction_sum

    def create_prediction_input(self, prediction_year: int) -> pd.DataFrame:
        """
        Creates a DataFrame defining the groups for which predictions are needed 
        for a specific year.

        Args:
            prediction_year (int): The 'Collegejaar' to generate prediction inputs for.

        Returns:
            pd.DataFrame: A DataFrame with unique combinations of 
                          'Groepeernaam Croho' and 'Herkomst' for the specified year.
                          Columns are renamed to 'Groepeernaam Croho' and 'Herkomst'.
        """
        print(f"\nCreating prediction input file for year {prediction_year}...")

        # Subtract one year to read 1oktobercijfers
        prediction_year = prediction_year - 1
        
        if 'Collegejaar' not in self.raw_data.columns:
             raise ValueError("'Collegejaar' column not found in raw data.")
        if 'Groepeernaam Croho' not in self.raw_data.columns:
             raise ValueError("'Groepeernaam Croho' column not found in raw data.")
        if 'EER-NL-nietEER' not in self.raw_data.columns:
             raise ValueError("'EER-NL-nietEER' column not found in raw data.")
             
        prediction_df = self.raw_data[self.raw_data['Collegejaar'] == prediction_year][
            ['Collegejaar', 'Groepeernaam Croho', 'EER-NL-nietEER']
        ].drop_duplicates().reset_index(drop=True)

        # Add the year back
        prediction_df['Collegejaar'] = prediction_df['Collegejaar'] + 1
        
        prediction_df = prediction_df.rename(columns={'EER-NL-nietEER': 'Herkomst', 'Groepeernaam Croho' : 'Croho groepeernaam'})
        return prediction_df

    def predict_all(self, prediction_year: int, train_year_max: int = None, test_year: int = None, num_boost_round: int = 200) -> pd.DataFrame:
        """
        Generates predictions for all relevant groups for a specified future year.

        Args:
            prediction_year (int): The 'Collegejaar' to generate predictions for (used to find groups).
                                   The actual test data used for prediction comes from `test_year`.
            train_year_max (int, optional): The latest year included in the training set for each group model.
                                        If None, it will be inferred (e.g., max year < test_year).
            test_year (int, optional): The year used as the test set for evaluating and predicting for each group model.
                                       If None, it will default to one year before prediction_year.
            num_boost_round (int): Number of boosting rounds for XGBoost.


        Returns:
            pd.DataFrame: The input DataFrame created by `create_prediction_input` 
                          with an added 'Prediction' column containing the sum of 
                          predicted probabilities for each group based on the `test_year` data.
        """

        # 1. Handle default year values
        if prediction_year is None:
            raise ValueError("prediction_year must be provided.")
        
        if test_year is None:
            test_year = prediction_year - 1
    
        if train_year_max is None:
            train_year_max = prediction_year - 2

        # 2. Create the file listing groups to predict for
        self.prediction_df = self.create_prediction_input(prediction_year)

        if self.prediction_df.empty:
            print(f"No groups found for prediction year {prediction_year}. Returning empty DataFrame.")
            return pd.DataFrame(columns=['Collegejaar', 'Groepeernaam Croho', 'Herkomst', 'Prediction'])

        # 3. Apply the prediction logic to each row (each group)
        self.prediction_df['Prediction_higheryears'] = self.prediction_df.apply(
            lambda row: self.train_and_predict_for_group(
                opleiding=row['Croho groepeernaam'], 
                herkomst=row['Herkomst'],
                train_year_max=train_year_max,
                test_year=test_year,
                num_boost_round=num_boost_round
            ),
            axis=1
        )

        # 4. Set trained_year
        self.trained_year = prediction_year

        return self.prediction_df


    def predict_and_modify(
        self,
        total,
        prediction_year: int,
        train_year_max: int = None,
        test_year: int = None,
        num_boost_round: int = 200,
        MAE=True,
        MAPE=True
    ):
        """
        Predicts values for higher year students and updates the total DataFrame
        with prediction, MAE, and MAPE metrics.
    
        Parameters:
            total (DataFrame): Original data with actual student numbers.
            prediction_year (int): The year to generate predictions for.
            train_year_max (int, optional): Latest year used for training the model.
            test_year (int, optional): Year used for testing.
            num_boost_round (int, optional): Number of boosting rounds for the model.
            MAE (bool): Whether to compute Mean Absolute Error.
            MAPE (bool): Whether to compute Mean Absolute Percentage Error.
        
        Returns:
            DataFrame: Updated DataFrame with prediction and evaluation metrics.
        """
    
        # Generate predictions using the internal prediction method
        if not self.trained_year == prediction_year:
            self.predict_all(prediction_year, train_year_max, test_year, num_boost_round)
    
        # Set multi-index for alignment based on year, program name, and origin
        total.set_index(['Collegejaar', 'Croho groepeernaam', 'Herkomst'], inplace=True)
        self.prediction_df.set_index(['Collegejaar', 'Croho groepeernaam', 'Herkomst'], inplace=True)
    
        # Sort indices to ensure consistency and avoid performance warnings
        total.sort_index(inplace=True)
        self.prediction_df.sort_index(inplace=True)
    
        # Ensure the prediction column exists in total; initialize if missing
        if 'Prediction_higheryears' not in total.columns:
            total['Prediction_higheryears'] = np.nan
    
        # Update the predictions in total using prediction_df values
        total.update(self.prediction_df[['Prediction_higheryears']])
    
        # Reset indices for both DataFrames to restore original structure
        total.reset_index(inplace=True)
        self.prediction_df.reset_index(inplace=True)
    
        # Compute Mean Absolute Error if enabled
        if MAE:
            if 'MAE_higheryears' not in total.columns:
                total['MAE_higheryears'] = np.nan
    
            # Apply row-wise absolute error computation
            total["MAE_higheryears"] = total.apply(
                lambda row: (
                    abs(row["Aantal_studenten_higher_years"] - row["Prediction_higheryears"])
                    if pd.notna(row["Prediction_higheryears"]) and pd.notna(row["Aantal_studenten_higher_years"])
                    else np.nan
                ),
                axis=1,
            )
    
        # Compute Mean Absolute Percentage Error if enabled
        if MAPE:
            if 'MAPE_higheryears' not in total.columns:
                total['MAPE_higheryears'] = np.nan
    
            # Apply row-wise percentage error computation
            total["MAPE_higheryears"] = total.apply(
                lambda row: (
                    abs((row["Aantal_studenten_higher_years"] - row["Prediction_higheryears"]) / row["Aantal_studenten_higher_years"])
                    if pd.notna(row["Prediction_higheryears"])
                    and pd.notna(row["Aantal_studenten_higher_years"])
                    and row["Aantal_studenten_higher_years"] != 0
                    else np.nan
                ),
                axis=1,
            )
    
        return total
        


if __name__ == "__main__":
    arguments = sys.argv
    predict_years = []
    year_slice = False

    for i in range(1, len(arguments)):
        arg = arguments[i]
        if arg == "-y" or arg == "-year":
            continue
        elif arg == ":":
            year_slice = True
        elif arg.isnumeric():
            if year_slice:
                last_year = predict_years.pop(-1)
                predict_years = predict_years + list(range(last_year, int(arg) + 1))
                year_slice = False
            else:
                predict_years.append(int(arg))

    if predict_years == []:
        predict_years.append(datetime.date.today().year)

    configuration = load_configuration("configuration/configuration.json")

    data_october = pd.read_excel(configuration["paths"]["path_october"])
    data_latest = pd.read_excel(configuration["paths"]["path_latest_cumulative"])

    data_october = data_october[data_october["Examentype code"] != "Master post-initieel"]
    data_october.loc[
        data_october["Groepeernaam Croho"] == "M LVHO in de Taal- en Cultuurwetenschappen",
        "Groepeernaam Croho"
    ] = "M LVHO in de Taal en Cultuurwetenschappen"


    print("Predicting higher years...")
    predictor = HigherYearsPredictor(data_october)

    for predict_year in predict_years:
        print(f"Predicting year {predict_year}")
        data_latest = predictor.predict_and_modify(data_latest, prediction_year=predict_year, num_boost_round=1) 

    data_latest.sort_values(
        by=["Croho groepeernaam", "Examentype", "Collegejaar", "Weeknummer", "Herkomst"],
        inplace=True,
        ignore_index=True,
    )

    
    print("Saving output...")
    CWD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    outfile = os.path.join(CWD, "data/output/output_higher-years_test.xlsx")

    data_latest.to_excel(outfile)
