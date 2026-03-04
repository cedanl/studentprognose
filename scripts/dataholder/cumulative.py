from scripts.dataholder.superclass import *
from scripts.helper import *
from scripts.transform_data import *

import numpy as np
from numpy import linalg as LA
import joblib
import os
import math
import gc
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator
import itertools

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# from scripts.interpolate import interpolate

warnings.simplefilter("ignore", ConvergenceWarning)


class Cumulative(Superclass):
    def __init__(
        self, data_cumulative, data_studentcount, configuration, helpermethods_initialise_material
    ):
        super().__init__(configuration, helpermethods_initialise_material)

        self.data_cumulative = data_cumulative
        self.data_studentcount = data_studentcount
        self.skip_years = 0
        self.faculty_transformation = configuration["faculty"]

    def preprocess(self):
        """
        Joins the main dataframe with the data with cumulative pre applications.

        Args:
            data (pd.DataFrame): The raw preapplication data to be joined and preprocessed.

        Returns:
            pd.DataFrame: A consolidated and preprocessed DataFrame containing preapplication
            information.

        This function performs the following tasks:
        1. Converts columns with comma-separated numbers to float64 data types.
        2. Groups the data by specific columns and sums non-numeric columns.
        3. Resets the index and selects relevant columns.
        4. Renames the 'Groepeernaam Croho' column to 'Croho groepeernaam'.
        """

        # Set cumulative data to small variable name for reading with more ease
        data = self.data_cumulative

        # We cast the string numbers to floats to be able do perform calculations with these
        # numbers.
        data = self._cast_string_to_float(data, "Ongewogen vooraanmelders")
        data = self._cast_string_to_float(data, "Gewogen vooraanmelders")
        data = self._cast_string_to_float(data, "Aantal aanmelders met 1 aanmelding")
        data = self._cast_string_to_float(data, "Inschrijvingen")

        # Rename certain columns to match columns of data individual
        data = data.rename(
            columns={
                "Type hoger onderwijs": "Examentype",
                "Groepeernaam Croho": "Croho groepeernaam",
            }
        )

        data.loc[
            (data["Examentype"] == "Pre-master"),
            "Hogerejaars",
        ] = "Nee"

        # We filter out the higher year students because we only want to first years.
        data = data[data["Hogerejaars"] == "Nee"]

        data = (
            data.groupby(
                [
                    "Collegejaar",
                    "Croho groepeernaam",
                    "Faculteit",
                    "Examentype",
                    "Herkomst",
                    "Weeknummer",
                ]
            )
            .sum(numeric_only=False)
            .reset_index()
        )
        data = data[
            [
                "Weeknummer",
                "Collegejaar",
                "Faculteit",
                "Examentype",
                "Herkomst",
                "Croho groepeernaam",
                "Gewogen vooraanmelders",
                "Ongewogen vooraanmelders",
                "Aantal aanmelders met 1 aanmelding",
                "Inschrijvingen",
            ]
        ]

        self.data_cumulative = data.sort_values(
            by=["Croho groepeernaam", "Herkomst", "Collegejaar", "Weeknummer"]
        )
        self.data_cumulative = data
        self.data_cumulative_backup = self.data_cumulative

        return self.data_cumulative

    def _cast_string_to_float(self, data, key):
        if pd.api.types.is_string_dtype(data[key].dtype):
            data[key] = data[key].str.replace(".", "")
            data[key] = data[key].str.replace(",", ".")
        data[key] = pd.to_numeric(data[key], errors="coerce")
        data[key] = data[key].astype("float64")

        return data

    def predict_nr_of_students(self, predict_year, predict_week, skip_years=0):
        """
        Predicts the number of students by first predicting the pre-registrations with SARIMA
        and afterwards predict the actual influx with XGBoost.

        Args:
            data (pd.DataFrame): The preprocessed preapplication data to be joined and preprocessed.
            predict_year (int): The year to be predicted
            predict_week (int): The week to be predicted
            skip_year (int): The years to be skipped if we want to predict more time ahead.

        Returns:
            pd.DataFrame: A DataFrame including the SARIMA_cumulative and predicted
            pre-applicants.
        """

        # Prepare the data_cumulative and use it to initialise full_data (data from which the
        # test and training data will be filtered for xgboost) and data_to_predict.
        self.data_cumulative = self.data_cumulative_backup.copy(deep=True)
        self.set_year_week(predict_year, predict_week, self.data_cumulative)
        self.prepare_data()
        self.data_cumulative = self.data_cumulative.astype(
            {"Weeknummer": "int32", "Collegejaar": "int32"}
        )

        full_data = self.get_transformed_data(self.data_cumulative.copy(deep=True))
        full_data["39"] = 0

        self.skip_years = skip_years

        # Filter all the cumulative data on predict year, predict week, programme and herkomst to
        # obtain the data_to_predict.
        data_to_predict = self.data_cumulative[
            (self.data_cumulative["Collegejaar"] == self.predict_year)
            & (self.data_cumulative["Weeknummer"] == self.predict_week)
        ]
        if self.programme_filtering != []:
            data_to_predict = data_to_predict[
                (data_to_predict["Croho groepeernaam"].isin(self.programme_filtering))
            ]
        if self.herkomst_filtering != []:
            data_to_predict = data_to_predict[
                (data_to_predict["Herkomst"].isin(self.herkomst_filtering))
            ]
        if self.examentype_filtering != []:
            data_to_predict = data_to_predict[
                (data_to_predict["Examentype"].isin(self.examentype_filtering))
            ]

        if len(data_to_predict) == 0:
            return None

        # Split the DataFrame into smaller chunks for parallel processing
        nr_CPU_cores = os.cpu_count()
        chunk_size = math.ceil(
            len(data_to_predict) / nr_CPU_cores
        )  # Make as much chunks as you have CPU cores

        chunks = [
            data_to_predict[i : i + chunk_size] for i in range(0, len(data_to_predict), chunk_size)
        ]

        print("Start parallel predicting...")
        # Use joblib.Parallel to parallelize the operation
        self.predicted_data = joblib.Parallel(n_jobs=nr_CPU_cores)(
            joblib.delayed(self.predict_with_sarima)(row)
            for chunk in chunks
            for _, row in chunk.iterrows()
        )

        # Create two new columns with NaN values. 'Voorspelde vooraanmelders' is just predicted
        # and stored in predicted_data. This will be added with 'add_predicted_preregistrations()'.
        # These values will be used to predict the 'SARIMA_cumulative'.
        data_to_predict["SARIMA_individual"] = np.nan
        data_to_predict["Voorspelde vooraanmelders"] = np.nan

        # Add predicted preregistrations to the data_to_predict dataframe.
        data_to_predict = self.helpermethods.add_predicted_preregistrations(
            data_to_predict, [x[: self.pred_len] for x in self.predicted_data]
        )

        # Predict the SARIMA_cumulative and add it to the dataframe.
        data_to_predict = self.predict_students_with_preapplicants(
            full_data, self.predicted_data, data_to_predict
        )

        return data_to_predict
        # data_to_predict could look something like this:
        # Weeknummer  Collegejaar Faculteit Examentype Herkomst Croho groepeernaam  ...  Inschrijvingen  Aantal_studenten     ts  SARIMA_individual  Voorspelde vooraanmelders  SARIMA_cumulative
        # 0       12         2024       FdM   Bachelor       NL    B Bedrijfskunde  ...             0.0               NaN  109.0                NaN                        NaN              201.0
        # 1       13         2024       FdM   Bachelor       NL    B Bedrijfskunde  ...             NaN               NaN    NaN                NaN                 126.392052                NaN
        # 2       14         2024       FdM   Bachelor       NL    B Bedrijfskunde  ...             NaN               NaN    NaN                NaN                 132.691171                NaN
        # 3       15         2024       FdM   Bachelor       NL    B Bedrijfskunde  ...             NaN               NaN    NaN                NaN                 157.651602                NaN
        #   .....

    # This method adds some data to the dataframe e.g. the studentcount and changes the faculty
    # codes.
    def prepare_data(self):
        if self.data_studentcount is not None:
            self.data_cumulative = self.data_cumulative.merge(
                self.data_studentcount,
                on=["Croho groepeernaam", "Collegejaar", "Herkomst", "Examentype"],
                how="left",
            )
        self.data_cumulative["ts"] = (
            self.data_cumulative["Gewogen vooraanmelders"] + self.data_cumulative["Inschrijvingen"]
        )

        self.data_cumulative = self.data_cumulative.drop_duplicates()

        # Change faculty codes
        self.data_cumulative["Faculteit"] = self.data_cumulative["Faculteit"].replace(
            self.faculty_transformation,
        )

        if int(self.predict_week) > 38:
            self.pred_len = 38 + 52 - int(self.predict_week)
        else:
            self.pred_len = 38 - int(self.predict_week)

    def get_transformed_data(self, data):
        data = data.drop_duplicates()
        data = data[data["Collegejaar"] >= 2016]

        # Makes a certain pivot_wider where it transforms the data from long to wide
        data = transform_data(data, "ts")
        return data

    def predict_with_sarima(self, row, already_printed=False):
        """
        Predicts pre-registrations with sarima per programme/origin/week

        Args:
            data (pd.DataFrame): The preprocessed preapplication data to be joined and preprocessed.
            row (pd.DataFrame): Chunks of the dataframe data_to_predict
            already_printed (bool): Indicates if the programme already printed output that it was
            predicting this specific case.

        Returns:
            list: A list with a prediction (float) of the pre-applicants for every week that has
            to be predicted.
        """

        programme = row["Croho groepeernaam"]
        herkomst = row["Herkomst"]
        examentype = row["Examentype"]

        if not already_printed:
            print(
                f"Prediction for {programme}, {herkomst}, year: {self.predict_year}, week: {self.predict_week}"
            )

        # We do this to reclaim memory by destroying objects that are no longer in use.
        gc.collect()

        # Transform data i.a. from long to wide
        self.data_cumulative = self.data_cumulative.astype(
            {"Weeknummer": "int32", "Collegejaar": "int32"}
        )
        data = self.get_transformed_data(self.data_cumulative.copy(deep=True))

        data = data[
            (data["Herkomst"] == herkomst)
            & (data["Collegejaar"] <= self.predict_year - self.skip_years)
            & (data["Croho groepeernaam"] == programme)
            & (data["Examentype"] == examentype)
        ]

        # Week 39 to 0
        data["39"] = 0

        def create_time_series(data: pd.DataFrame, pred_len: int) -> np.array:
            """
            Create a time series data array from a DataFrame for a given prediction length.

            Args:
                data (pd.DataFrame): The input DataFrame containing time series data.
                pred_len (int): The length of the time series to be created.

            Returns:
                np.ndarray: A NumPy array containing the time series data.
            """

            ts_data = data.loc[:, get_all_weeks_valid(data.columns)].values.flatten()
            ts_data = ts_data[:-pred_len]

            return np.array(ts_data)

        ts_data = create_time_series(data, self.pred_len)

        try:
            # We went back to the way of hardcoding the parameters but I didn't want to delete
            # the next lines for in case we want to revise it.

            # best_params = self._random_search_sarima(ts_data)
            # model = sm.tsa.statespace.SARIMAX(ts_data, order=best_params['order'],
            #                                  seasonal_order=best_params['seasonal_order'])

            """
            We initialise the model by giving the following parameters:
                ts_data: the array with observed time-series process
                TODO: Explain the following two parameters
                order=(p,d,q):
                seasonal_order=(P,D,Q,s):
            """
            model = sm.tsa.statespace.SARIMAX(
                ts_data, order=(1, 0, 1), seasonal_order=(1, 1, 1, 52)
            )

            # Fit the model by maximum likelihood via Kalman filter. disp is set to False to not
            # print convergence messages.
            results = model.fit(disp=0)

            # Out-of-sample forecast. The steps parameter is set the number of weeks it has to
            # predict.
            pred = results.forecast(steps=self.pred_len)

            # Returns a list with a prediction (float) of the pre-applicants for every week that
            # has to be predicted.
            return pred

        except (LA.LinAlgError, IndexError, ValueError) as error:
            print(f"Cumulative sarima error on: {programme}, {herkomst}")
            print(error)
            return []

    def predict_students_with_preapplicants(self, data, predictions, data_to_predict):
        """
        Predicts SARIMA_cumulative by training the model per examtype (or per programme for
        programmes on the numerus fixus list).

        Args:
            data (pd.DataFrame): Transformed, preprocessed cumulative data.
            predictions (list): Pre-application predictions from the SARIMA model.
            data_to_predict (pd.DataFrame): Dataframe with in every row a different item that
            has to be predicted.

        Returns:
            (pd.DataFrame): data_to_predict dataframe with the final XGBoost (SARIMA_cumulative)
            prediction added.
        """

        # Fill the transformed, preprocessed data with the predictions
        index = str(increment_week(self.predict_week))
        i = 0
        for _, row in data_to_predict.iterrows():
            programme = row["Croho groepeernaam"]
            herkomst = row["Herkomst"]
            examentype = row["Examentype"]

            if i == len(predictions):
                break

            if self.predict_week != 38 and len(predictions[i]) > 0:
                data.loc[
                    (data["Collegejaar"] == self.predict_year - self.skip_years)
                    & (data["Croho groepeernaam"] == programme)
                    & (data["Herkomst"] == herkomst)
                    & (data["Examentype"] == examentype),
                    index:"38",
                ] = predictions[i]
            i += 1

            # If the programme is in the numerus fixus list, the model must only be trained on
            # results of that specific programme in the past.
            if programme in self.numerus_fixus_list:
                train = data[
                    (data["Croho groepeernaam"] == programme) & (data["Examentype"] == examentype)
                ]
                test = data[
                    (data["Croho groepeernaam"] == programme)
                    & (data["Herkomst"] == herkomst)
                    & (data["Examentype"] == examentype)
                ]
                data_to_predict = self._predict_with_xgboost_extra_year(
                    train,
                    test,
                    data_to_predict,
                    (data_to_predict["Croho groepeernaam"] == programme)
                    & (data_to_predict["Herkomst"] == herkomst)
                    & (data_to_predict["Examentype"] == examentype),
                )

        # Predict with XGBoost for bachelor, pre-master and master programmes apart
        all_examtypes = data_to_predict["Examentype"].unique()
        for examtype in all_examtypes:
            train = data[
                (data["Examentype"] == examtype)
                & (~data["Croho groepeernaam"].isin(self.numerus_fixus_list))
            ]
            test = data[
                (data["Examentype"] == examtype)
                & (~data["Croho groepeernaam"].isin(self.numerus_fixus_list))
            ]

            data_to_predict = self._predict_with_xgboost_extra_year(
                train,
                test,
                data_to_predict,
                (data_to_predict["Examentype"] == examtype)
                & (~data_to_predict["Croho groepeernaam"].isin(self.numerus_fixus_list)),
            )

        return data_to_predict

    def _predict_with_xgboost_extra_year(self, train, test, data_to_predict, replace_mask):
        """
        Determines the right train and testdata to pass on to the XGBoost model.

        Args:
            train (pd.DataFrame): training data that still has to be filtered.
            test (pd.DataFrame): test data that still has to be filtered.
            data_to_predict (pd.DataFrame): Dataframe with in every row a different item that
            has to be predicted.
            replace_mask (pd.DataFrame): Dataframe with booleans indicating which rows in the
            data_to_predict are the rows we are going to predict.

        Returns:
            (pd.DataFrame): data_to_predict dataframe with the final XGBoost (SARIMA_cumulative)
            prediction added.
        """

        columns_to_match = [
            "Collegejaar",
            "Faculteit",
            "Examentype",
            "Herkomst",
            "Croho groepeernaam",
        ]

        if self.skip_years > 0:
            # Predict for extra year(s)

            # Second predict using y_train as skip_years number of students
            train2 = train[
                (train["Collegejaar"] < self.predict_year - (self.skip_years * 2))
            ]  # Times two is necessary because twice as less training data can be used
            train2["Collegejaar"] = train2["Collegejaar"] + self.skip_years
            test2 = test[(test["Collegejaar"] == self.predict_year - self.skip_years)]
            test2["Collegejaar"] = test2["Collegejaar"] + self.skip_years

            test2_merged = data_to_predict[
                (data_to_predict["Weeknummer"] == self.predict_week) & replace_mask
            ][columns_to_match].merge(test2, on=columns_to_match)
            if not test2_merged.empty:
                if train2.empty:
                    print(
                        f"WARNING: Skipping XGBoost prediction: no training data "
                        f"available (try increasing --ci test N)."
                    )
                    return data_to_predict
                test2["Collegejaar"] = test2["Collegejaar"] - self.skip_years
                ahead_predictions = self._predict_with_xgboost(train2, test2_merged)
                test2["Collegejaar"] = test2["Collegejaar"] + self.skip_years

                mask = (
                    data_to_predict[columns_to_match]
                    .apply(tuple, axis=1)
                    .isin(test2_merged[columns_to_match].apply(tuple, axis=1))
                )

                full_mask = (
                    replace_mask & (data_to_predict["Weeknummer"] == self.predict_week) & mask
                )

                data_to_predict.loc[full_mask, "Skip_prediction"] = ahead_predictions[
                    : full_mask.sum()
                ]
        else:
            # Predict for only the next academic year
            train = train[(train["Collegejaar"] < self.predict_year)]
            test = test[(test["Collegejaar"] == self.predict_year)]

            # Actual test data (test_merged) is obtained by filtering data_to_predict on the
            # 'Weeknummer' and the replace_mask, merged with the testdata based on 5 columns.
            test_merged = data_to_predict[
                (data_to_predict["Weeknummer"] == self.predict_week) & replace_mask
            ][columns_to_match].merge(test, on=columns_to_match)

            if not test_merged.empty:
                if train.empty:
                    print(
                        f"WARNING: Skipping XGBoost prediction: no training data "
                        f"available (try increasing --ci test N)."
                    )
                    return data_to_predict
                predictions = self._predict_with_xgboost(train, test_merged)

                # This mask indicates which items in data_to_predict are just predicted.
                mask = (
                    data_to_predict[columns_to_match]
                    .apply(tuple, axis=1)
                    .isin(test_merged[columns_to_match].apply(tuple, axis=1))
                )

                # Apply the masks
                full_mask = (
                    replace_mask & (data_to_predict["Weeknummer"] == self.predict_week) & mask
                )

                # Fill in the predictions in the dataframe
                data_to_predict.loc[full_mask, "SARIMA_cumulative"] = predictions[
                    : full_mask.sum()
                ]

        return data_to_predict

    def _predict_with_xgboost(self, train, test):
        if self.data_studentcount is not None:
            train = train.merge(
                self.data_studentcount[
                    [
                        "Croho groepeernaam",
                        "Collegejaar",
                        "Herkomst",
                        "Examentype",
                        "Aantal_studenten",
                    ]
                ],
                on=["Croho groepeernaam", "Collegejaar", "Herkomst", "Examentype"],
            )
        else:
            # Student count is required
            return np.nan

        train.drop_duplicates(inplace=True, ignore_index=True)

        if train.empty:
            return np.full(len(test), np.nan)

        X_train = train.drop(["Aantal_studenten"], axis=1)
        y_train = train.pop("Aantal_studenten")

        # Encode
        # Specify the numeric and categorical column names
        numeric_cols = ["Collegejaar"] + [str(x) for x in get_weeks_list(38)]
        categorical_cols = ["Examentype", "Faculteit", "Croho groepeernaam", "Herkomst"]

        # Create transformers for numeric and categorical columns
        numeric_transformer = "passthrough"  # No transformation for numeric columns
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        # Create the column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_cols),
                ("categorical", categorical_transformer, categorical_cols),
            ]
        )

        # Apply the preprocessing to the training and test data
        X_train = preprocessor.fit_transform(X_train)
        test = preprocessor.transform(test)

        # Model
        model = XGBRegressor(learning_rate=0.25)

        model.fit(X_train, y_train)

        predictions = model.predict(test)

        for i in range(len(predictions)):
            predictions[i] = int(round(predictions[i], 0))

        return predictions


"""
The SARIMAXWrapper method and class is not used anymore because we hardcoded the parameters of
the SARIMA model. The search for the best parameter values didn't work as good as expected. We
won't delete the code yet for in case we wan't to revise it.
def _random_search_sarima(self, ts_data):
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 52) for x in list(itertools.product(p, d, q))]

    param_dist = {
        'order': pdq,
        'seasonal_order': seasonal_pdq
    }

    sarima_search = RandomizedSearchCV(SARIMAXWrapper(), param_distributions=param_dist,
                                        n_iter=100, cv=5, n_jobs=-1, verbose=0)
    sarima_search.fit(ts_data)

    best_params = sarima_search.best_params_

    return best_params

class SARIMAXWrapper(BaseEstimator):
    def __init__(self, order=None, seasonal_order=None):
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if self.order is None or self.seasonal_order is None:
            raise ValueError("SARIMAXWrapper: order and seasonal_order must be provided.")

        model = sm.tsa.statespace.SARIMAX(X, order=self.order, seasonal_order=self.seasonal_order)
        results = model.fit(disp=0)
        return results.forecast(steps=len(X))

    def score(self, X, y=None):
        # Dummy score method, as it is not used
        return 0
"""
