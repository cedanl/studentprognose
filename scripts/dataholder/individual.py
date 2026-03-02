from scripts.dataholder.superclass import *
from scripts.helper import *
from scripts.transform_data import *

import datetime
import numpy as np
from numpy import linalg as LA
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os
import math
import gc
import statsmodels.api as sm
import collections


class Individual(Superclass):
    def __init__(
        self, data_individual, data_distances, configuration, helpermethods_initialise_material
    ):
        super().__init__(configuration, helpermethods_initialise_material)

        self.data_individual = data_individual
        self.data_distances = data_distances
        self.faculty_transformation = configuration["faculty"]

    def preprocess(self):
        """
        Preprocess the input data for further analysis.

        Member variables used:
            data_individual (pd.DataFrame): The raw data to be preprocessed.
            data_distances (pd.DataFrame): Data of distances from the Radboud.
            numerus_fixus_list (list): List of numerus fixus programmes.

        Returns:
            pd.DataFrame: A preprocessed DataFrame ready for analysis.

        This function performs a series of data preprocessing steps on the input DataFrame:
        1. Removal of unnecessary columns.
        2. Filtering based on specific conditions.
        3. Grouping and counting data.
        4. Date transformation to week numbers.
        5. Determining the 'Herkomst' column based on nationality and EER status.
        6. Filtering data based on 'Ingangsdatum', 'Faculteit', and other criteria.
        7. Creating a 'is_numerus_fixus' column.
        8. Standardizing 'Examentype' values.
        9. Filtering data based on various conditions.
        10. Replacing less frequent 'Nationaliteit' values with 'Overig'.
        11. Adding city distances.
        12. Creating a 'Deadlineweek' column based on specific conditions.
        13. Additional data filtering based on year and exam type.
        14. Removal of the 'Sleutel' column.
        """

        # Set individual data to small variable name for reading with more ease
        data = self.data_individual

        # Remove redudant columns
        data = data.drop(labels=["Aantal studenten"], axis=1)

        # Filter english language and culture
        data = data[
            ~(
                (data["Croho groepeernaam"] == "B English Language and Culture")
                & (data["Collegejaar"] == 2021)
                & (data["Examentype"] != "Propedeuse Bachelor")
            )
        ]

        # Group the dataframe by collegejaar, weeknummer, and Sleutel
        grouped = data.groupby(["Collegejaar", "Sleutel"])

        # Create a new column in the original dataframe using the counts
        data["Sleutel_count"] = grouped["Sleutel"].transform("count")

        # Weeknummers

        def to_weeknummer(date):
            try:
                split_data = date.split("-")

                year = int(split_data[2])
                month = int(split_data[1])
                day = int(split_data[0])

                weeknummer = datetime.date(year, month, day).isocalendar()[1]

                return weeknummer
            except AttributeError:
                return np.nan

        data["Datum intrekking vooraanmelding"] = data["Datum intrekking vooraanmelding"].apply(
            to_weeknummer
        )

        data["Weeknummer"] = data["Datum Verzoek Inschr"].apply(to_weeknummer)

        def get_herkomst(nat, eer):
            if nat == "Nederlandse":
                return "NL"
            elif nat != "Nederlandse" and eer == "J":
                return "EER"
            else:
                return "Niet-EER"

        data["Herkomst"] = data.apply(lambda x: get_herkomst(x["Nationaliteit"], x["EER"]), axis=1)

        # Ingangsdatum
        data = data[
            data["Ingangsdatum"].str.contains("01-09-")
            | data["Ingangsdatum"].str.contains("01-10-")
        ]

        # Faculteit
        data.Faculteit = data["Faculteit"].replace("RU", self.faculty_transformation["RU"])

        # Create numerus fixus kolom
        data["is_numerus_fixus"] = (
            data["Croho groepeernaam"].isin(self.numerus_fixus_list)
        ).astype(int)

        # Aanpassen 'Bachelor Eerstejaars' naar 'Bachelor'
        data["Examentype"] = data["Examentype"].replace("Propedeuse Bachelor", "Bachelor")

        # Filter data
        data = data[data["Inschrijfstatus"].notna()]

        data = data[data["Examentype"].isin(["Bachelor", "Master", "Pre-master"])]

        # Nationaliteit
        # Count the occurrences of each value in the 'Nationaliteit' column
        nationaliteit_counts = data["Nationaliteit"].value_counts()

        # Get the values that occur less than 5 times
        values_to_change = nationaliteit_counts[nationaliteit_counts < 100].index

        # Replace the values with 'Overig'
        data["Nationaliteit"] = data["Nationaliteit"].replace(values_to_change, "Overig")

        if self.data_distances is not None:
            # Add cities
            afstanden = self.data_distances

            data["Afstand"] = np.nan
            data["Afstand"] = data["Geverifieerd adres plaats"].map(
                afstanden.set_index("Geverifieerd adres plaats")["Afstand"]
            )

        # Define a function to apply the conditions
        def get_new_column(row):
            if (
                row["Weeknummer"] == 17
                and not row["Croho groepeernaam"] in self.numerus_fixus_list
            ):
                return True
            else:
                return False

        # Apply the function to create a new column 'NewColumn' in the DataFrame
        data["Deadlineweek"] = data.apply(get_new_column, axis=1)

        data = data.drop(["Sleutel"], axis=1)

        data.loc[
            data["Examentype"] == "Pre-master",
            ["Is eerstejaars croho opleiding", "Is hogerejaars", "BBC ontvangen"],
        ] = [1, 0, 0]

        data = data[
            (data["Is eerstejaars croho opleiding"] == 1)
            & (data["Is hogerejaars"] == 0)
            & (data["BBC ontvangen"] == 0)
        ]

        self.data_individual = data.drop(
            [
                "Eerstejaars croho jaar",
                "Is eerstejaars croho opleiding",
                "Ingangsdatum",
                "BBC ontvangen",
                "Croho",
                "Is hogerejaars",
            ],
            axis=1,
        )
        self.data_individual_backup = self.data_individual

        return self.data_individual

    def predict_nr_of_students(self, predict_year, predict_week, skip_years=0):
        self.data_individual = self.data_individual_backup.copy(deep=True)
        self.set_year_week(predict_year, predict_week, self.data_individual)

        # Call predict_applicant first
        print("Predicting preapplicants...")
        predicties = self.predict_applicant(self.data_individual)

        self.data_individual.loc[
            (self.data_individual["Collegejaar"] == self.predict_year)
            & (self.data_individual["Weeknummer"].isin(get_weeks_list(self.predict_week))),
            "Inschrijvingen_predictie",
        ] = predicties

        # Transform data to be used for sarima
        self.transform_data_individual()
        self.data_individual = transform_data(self.data_individual, "Cumulative_sum_within_year")

        data_to_predict = self.get_data_to_predict(
            self.data_individual,
            self.programme_filtering,
            self.herkomst_filtering,
            self.examentype_filtering,
        )

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

        data_to_predict["SARIMA_individual"] = self.predicted_data
        data_to_predict["SARIMA_cumulative"] = np.nan
        data_to_predict["Voorspelde vooraanmelders"] = np.nan

        return data_to_predict

    def predict_applicant(self, data, data_cumulative=None):
        # Input
        data_train = data[data["Weeknummer"].isin(get_weeks_list(self.predict_week))]

        if data_cumulative is not None:
            data_train = self._create_ratio(data_train)
        data_train = data_train.replace([np.inf, -np.inf], np.nan)

        # Output
        model = XGBClassifier(objective="binary:logistic", eval_metric="auc")

        # Train/test split
        if self.predict_year == self.max_year:
            train = data_train[
                (data_train["Collegejaar"] < self.predict_year)
                & (data_train["Collegejaar"] >= 2016)
            ]
            test = data_train[
                (data_train["Collegejaar"] == self.predict_year)
                & (data_train["Weeknummer"].isin(get_weeks_list(self.predict_week)))
            ]
        else:
            train = data_train[
                (data_train["Collegejaar"] != self.predict_year)
                & (data_train["Collegejaar"] >= 2016)
                & (data_train["Collegejaar"] != self.max_year)
            ]
            test = data_train[
                (data_train["Collegejaar"] == self.predict_year)
                & (data_train["Weeknummer"].isin(get_weeks_list(self.predict_week)))
            ]

        # Remove cancelled registrations
        if int(self.predict_week) <= 38:
            train = train[
                (train["Datum intrekking vooraanmelding"].isna())
                | (
                    (train["Datum intrekking vooraanmelding"] >= int(self.predict_week))
                    & (train["Datum intrekking vooraanmelding"] < 39)
                )
            ]
        elif int(self.predict_week) > 38:
            train = train[
                (train["Datum intrekking vooraanmelding"].isna())
                | (
                    (train["Datum intrekking vooraanmelding"] > int(self.predict_week))
                    | (train["Datum intrekking vooraanmelding"] < 39)
                )
            ]

        # Mutate inschrijfstatus
        status_map = {
            "Ingeschreven": 1,
            "Geannuleerd": 0,
            "Uitgeschreven": 1,
            "Verzoek tot inschrijving": 0,
            "Studie gestaakt": 0,
            "Aanmelding vervolgen": 0,
        }

        # Use the map function to apply the mapping to the column
        train["Inschrijfstatus"] = train["Inschrijfstatus"].map(status_map)

        X_train = train.drop(["Inschrijfstatus"], axis=1)
        y_train = train.pop("Inschrijfstatus")

        X_test = test.drop(["Inschrijfstatus"], axis=1)
        y_test = test.pop("Inschrijfstatus")

        # No individual students from selected programmes/herkomst
        if len(X_test) == 0:
            return np.nan

        # Encode
        # Specify the numeric and categorical column names
        if self.data_distances is not None:
            numeric_cols = [
                "Collegejaar",
                "Sleutel_count",
                "is_numerus_fixus",
                "Afstand",
            ]
        else:
            numeric_cols = ["Collegejaar", "Sleutel_count", "is_numerus_fixus"]

        categorical_cols = [
            "Examentype",
            "Faculteit",
            "Croho groepeernaam",
            "Deadlineweek",
            "Herkomst",
            "Weeknummer",
            "Opleiding",
            "Type vooropleiding",
            "Nationaliteit",
            "EER",
            "Geslacht",
            "Plaats code eerste vooropleiding",
            "Studieadres postcode",
            "Studieadres land",
            "Geverifieerd adres plaats",
            "Geverifieerd adres land",
            "Geverifieerd adres postcode",
            "School code eerste vooropleiding",
            "School eerste vooropleiding",
            "Land code eerste vooropleiding",
        ]

        if data_cumulative is not None:
            numeric_cols = numeric_cols + [
                "Gewogen vooraanmelders",
                "Ongewogen vooraanmelders",
                "Aantal aanmelders met 1 aanmelding",
                "Inschrijvingen",
            ]

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
        X_test = preprocessor.transform(X_test)

        # Train model on the training data
        model.fit(X_train, y_train)

        # Predict probability of all possible enrollment status of individual student
        voorspellingen = model.predict_proba(X_test)[:, 1]

        predicties = np.zeros(len(voorspellingen))

        # Filter: Als de vooraanmelding al geannuleerd is dan wordt deze automatisch 0
        for i, (voorspelling, real) in enumerate(zip(voorspellingen, y_test)):
            if real == "Geannuleerd" and test["Datum intrekking vooraanmelding"].iloc[
                i
            ] in get_weeks_list(self.predict_week):
                pred = 0
            else:
                pred = voorspelling

            predicties[i] = pred

        return predicties

    def _create_ratio(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio features by dividing specific columns by 'Ongewogen vooraanmelders' and handling
        missing values.

        Args:
            data (pd.DataFrame): The input data for creating ratio features.

        Returns:
            pd.DataFrame: A DataFrame with ratio features added.
        """

        data = data.copy()

        def fix_zero(number):
            if number == 0:
                return np.nan
            else:
                return number

        data["Ongewogen vooraanmelders"] = data["Ongewogen vooraanmelders"].apply(fix_zero)

        data["Gewogen vooraanmelders"] = (
            data["Gewogen vooraanmelders"] / data["Ongewogen vooraanmelders"]
        )
        data["Gewogen vooraanmelders"] = data["Gewogen vooraanmelders"].fillna(1)

        data["Aantal aanmelders met 1 aanmelding"] = (
            data["Aantal aanmelders met 1 aanmelding"] / data["Ongewogen vooraanmelders"]
        )
        data["Aantal aanmelders met 1 aanmelding"] = data[
            "Aantal aanmelders met 1 aanmelding"
        ].fillna(1)

        data["Inschrijvingen"] = data["Inschrijvingen"] / data["Ongewogen vooraanmelders"]
        data["Inschrijvingen"] = data["Inschrijvingen"].fillna(1)

        return data

    def transform_data_individual(self):
        """
        Transforms the dataframe into a workable dataframe suitable for prediction. It groups the data
        and creates the cumulative sum of the pre-applications.

        Args:
            data (pd.DataFrame): The input data to be transformed.
            target_year (int): The target year for the transformation.
            last_week (int): The last week considered for the transformation.

        Returns:
            pd.DataFrame: The transformed data ready for analysis.
        """

        data = self.data_individual

        data = data[data["Collegejaar"] <= self.predict_year]

        group_cols = [
            "Collegejaar",
            "Faculteit",
            "Herkomst",
            "Examentype",
            "Croho groepeernaam",
        ]

        # Create all weeks
        all_weeks = []
        all_weeks = all_weeks + [str(i) for i in range(39, 53)]
        all_weeks = all_weeks + [str(i) for i in range(1, 39)]

        # Create target weeks
        target_year_weeknummers = []
        if int(self.predict_week) > 38:
            target_year_weeknummers = target_year_weeknummers + [
                str(i) for i in range(39, int(self.predict_week) + 1)
            ]
        elif int(self.predict_week) < 39:
            target_year_weeknummers = target_year_weeknummers + [str(i) for i in range(39, 53)]
            target_year_weeknummers = target_year_weeknummers + [
                str(i) for i in range(1, int(self.predict_week) + 1)
            ]

        data = data[group_cols + ["Inschrijvingen_predictie", "Inschrijfstatus", "Weeknummer"]]

        data["Weeknummer"] = data["Weeknummer"].astype(str)

        # Muteren inschrijfstatus
        status_map = {
            "Ingeschreven": 1,
            "Geannuleerd": 0,
            "Uitgeschreven": 1,
            "Verzoek tot inschrijving": 0,
            "Studie gestaakt": 0,
            "Aanmelding vervolgen": 0,
        }

        # use the map function to apply the mapping to the column
        data["Inschrijfstatus"] = data["Inschrijfstatus"].map(status_map)

        data = data.groupby(group_cols + ["Weeknummer"]).sum(numeric_only=False).reset_index()

        def transform_data(input_data, target_col, weeknummers):
            data2 = input_data.reset_index().drop(["index", target_col], axis=1)
            # Pivot
            input_data = input_data.pivot(
                index=group_cols, columns="Weeknummer", values=target_col
            ).reset_index()

            # Reorder columns
            input_data.columns = map(str, input_data.columns)
            available_weeks = list(
                (
                    collections.Counter(weeknummers) & collections.Counter(input_data.columns)
                ).elements()
            )
            colnames = group_cols + available_weeks

            missing_weeks = []
            for element in weeknummers:
                if element not in available_weeks:
                    missing_weeks.append(element)
            missing_weeks = get_all_weeks_valid(missing_weeks)

            input_data = input_data[colnames]

            if target_col == "Inschrijvingen_predictie":
                for week in missing_weeks:
                    if week == "39":
                        input_data[week] = 0
                    else:
                        input_data[week] = input_data[str(decrement_week(int(week)))]

            input_data = input_data.fillna(0)
            input_data = input_data.melt(
                ignore_index=False, id_vars=group_cols, value_vars=weeknummers
            )

            input_data = input_data.rename(columns={"variable": "Weeknummer", "value": target_col})

            input_data = input_data.merge(data2, on=group_cols + ["Weeknummer"], how="left")

            input_data = input_data.fillna(0)

            input_data["Cumulative_sum_within_year"] = input_data.groupby(group_cols)[
                target_col
            ].transform(pd.Series.cumsum)

            return input_data

        # Real data
        data_real = data[data["Collegejaar"] != self.predict_year]
        data_real = transform_data(data_real, "Inschrijfstatus", all_weeks)

        # Data predict
        data_predict = data[data["Collegejaar"] == self.predict_year]
        data_predict = transform_data(
            data_predict, "Inschrijvingen_predictie", target_year_weeknummers
        )

        self.data_individual = pd.concat([data_real, data_predict])

    # Predicts nr of students with sarima per programme/origin/week
    def predict_with_sarima(self, row, data_exog=None, already_printed=False):
        data = self.data_individual.copy()
        programme = row["Croho groepeernaam"]
        herkomst = row["Herkomst"]
        examentype = row["Examentype"]

        if not already_printed:
            print(
                f"Prediction for {programme}, {herkomst}, {examentype}, year: {self.predict_year}, week: {self.predict_week}"
            )

        """
        Predicts a value using SARIMA (Seasonal Autoregressive Integrated Moving Average) modeling.

        Args:
            data (pd.DataFrame): Main data containing time series.
            data_exog (pd.DataFrame): Exogenous data used for modeling.
            programme (str): Study program.
            herkomst (str): Origin information (NL/EER/niet-EER).
            weeknummer (int): Week number for which prediction is made.
            jaar (int): Year for which prediction is made.
            max_year (int): Maximum year in the dataframe.

        Returns:
            float: Predicted amount of students for a specific combination of a opleiding/herkomst, in a
            certain year.
        """
        gc.collect()

        def filter_data(
            data: pd.DataFrame,
            programme: str,
            herkomst: str,
            examentype: str,
            jaar: int,
            max_year: int,
        ) -> pd.DataFrame:
            """
            Filters a given dataframe based on opleiding, herkomst and jaar. Returns the filtered
            dataframe.

            Args:
                data (pd.DataFrame): Main data to be filtered.
                programme (str): Study program code.
                herkomst (str): Origin information (NL/EER/niet-EER).
                jaar (int): Year for which filtering is applied.
                max_year (int): Maximum year in the dataframe.

            Returns:
                pd.DataFrame: Filtered data.
            """
            data = data[data["Herkomst"] == herkomst]

            if jaar != max_year:
                data = data[data["Collegejaar"] <= jaar]

            data = data[data["Croho groepeernaam"] == programme]

            data = data[data["Examentype"] == examentype]

            return data

        # Filter both datasets
        if data_exog is not None:
            data_exog = filter_data(
                data_exog,
                programme,
                herkomst,
                examentype,
                self.predict_year,
                self.max_year,
            )
        data = filter_data(data, programme, herkomst, examentype, self.predict_year, self.max_year)

        def deadline_week(weeknummer, croho, examentype):
            """
            Determines if a week corresponds to a deadline week for a specific study program.
            This is added as exogenous variabele. Master does not seem to have a strong deadline week.

            Args:
                weeknummer (int): Week number.
                croho (str): Croho group name.
                examentype (str): Type of exam.

            Returns:
                int: 1 if it's a deadline week, 0 otherwise.
            """

            if (
                weeknummer in [16, 17]
                and examentype == "Bachelor"
                and croho not in self.numerus_fixus_list
            ):
                return 1
            elif (
                weeknummer in [1, 2]
                and examentype == "Bachelor"
                and croho in self.numerus_fixus_list
            ):
                return 1
            else:
                return 0

        # Apply the 'deadline_week' function on the dataset
        if data_exog is not None:
            data_exog["Deadline"] = data_exog.apply(
                lambda x: deadline_week(x["Weeknummer"], x["Croho groepeernaam"], x["Examentype"]),
                axis=1,
            )

        try:
            if data_exog is not None:
                data_exog = transform_data(data_exog, "Deadline")

            if self.predict_week == 38:
                ts_data = data.loc[:, get_all_weeks_valid(data.columns)].values.flatten()
                try:
                    return ts_data[-1]
                except IndexError:
                    return np.nan

            if int(self.predict_week) > 38:
                pred_len = 38 + 52 - int(self.predict_week)
            else:
                pred_len = 38 - int(self.predict_week)

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

            def create_exogenous(data: pd.DataFrame, pred_len: int) -> np.array:
                """
                Create an exogenous time series data array from a DataFrame for a given prediction length.

                Args:
                    data (pd.DataFrame): The input DataFrame containing time series data.
                    pred_len (int): The length of the time series to be created.

                Returns:
                    np.ndarray: A NumPy array containing the time series data.
                """

                exg_data = data.loc[:, get_all_weeks_valid(data.columns)].values.flatten()
                exg_data_train = exg_data[:-pred_len]
                exg_data_test = exg_data[-pred_len:]

                return np.array(exg_data_train), np.array(exg_data_test)

            ts_data = create_time_series(data, pred_len)

            if data_exog is not None:
                exogenous_train_1, exg_data_test_1 = create_exogenous(data_exog, pred_len)
            else:
                exogenous_train_1 = None

            if ts_data.size == 0:
                return np.nan

            try:
                # Create SARIMA
                weeknummers = [17, 18, 19, 20, 21]
                if programme.startswith("B") and self.predict_week in weeknummers:
                    # This model seems to work better for bachelor programmes close to the deadline
                    model = sm.tsa.statespace.SARIMAX(
                        ts_data,
                        order=(1, 0, 1),
                        seasonal_order=(1, 1, 1, 52),
                        exog=exogenous_train_1,
                    )
                else:
                    # For the other time series I used these settings for SARIMA
                    model = sm.tsa.statespace.SARIMAX(
                        ts_data,
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 0, 52),
                        exog=exogenous_train_1,
                    )
                results = model.fit(disp=0)
                if data_exog is not None:
                    pred = results.forecast(steps=pred_len, exog=exg_data_test_1)
                else:
                    pred = results.forecast(steps=pred_len)

                # Only the last prediction is relevant
                return pred[-1]
            except (LA.LinAlgError, IndexError, ValueError) as error:
                print(f"Individual error on: {programme}, {herkomst}")
                print(error)
                return np.nan
        except KeyError as error:
            print(f"Individual key error on: {programme}, {herkomst}")
            print(error)