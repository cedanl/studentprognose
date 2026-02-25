from scripts.transform_data import *
from scripts.helper import *
import numpy as np
import pandas as pd
import os
from statistics import mean


class HelperMethods:
    """This class consists of the methods used for operations on dataholders and is loaded in
    the initialisation of DataHolderSuperclass (dataholder_superclass.py) as a variable. The
    methods in this class are used for several varying tasks in main.py and for adding the
    predicted preregistrations when calculating the cumulative values.
    """

    def __init__(self, configuration, helpermethods_initialise_material):
        self.data_latest = helpermethods_initialise_material[0]
        self.ensemble_weights = helpermethods_initialise_material[1]
        self.data_studentcount = helpermethods_initialise_material[2]

        self.numerus_fixus_list = configuration["numerus_fixus"]

        self.CWD = helpermethods_initialise_material[3]

        self.data_option = helpermethods_initialise_material[4]
        self.data = None

    # This method used when calculating the cumulative value. The input is the data_to_predict
    # this method adds for every row the 'Voorspelde vooraanmelders' until week 38. It also adds
    # a column called 'SARIMA_cumulative' with only NaN values. This will be filled in later.
    # Output of this method could look like this:

    #     Weeknummer  Collegejaar Faculteit Examentype Herkomst Croho groepeernaam  ...  Inschrijvingen  Aantal_studenten      ts  SARIMA_individual  Voorspelde vooraanmelders  SARIMA_cumulative
    # 0           12         2024       FdM   Bachelor      EER    B Bedrijfskunde  ...             0.0               NaN   52.83                NaN                        NaN                NaN
    # 1           12         2024       FdM   Bachelor       NL    B Bedrijfskunde  ...             0.0               NaN  109.00                NaN                        NaN                NaN
    # 2           13         2024       FdM   Bachelor      EER    B Bedrijfskunde  ...             NaN               NaN     NaN                NaN                  53.740341                NaN
    # ...
    # 27          38         2024       FdM   Bachelor      EER    B Bedrijfskunde  ...             NaN               NaN     NaN                NaN                  36.319705                NaN
    # 28          13         2024       FdM   Bachelor       NL    B Bedrijfskunde  ...             NaN               NaN     NaN                NaN                 126.392052                NaN
    # ...
    # 53          38         2024       FdM   Bachelor       NL    B Bedrijfskunde  ...             NaN               NaN     NaN                NaN                 370.173031                NaN
    def add_predicted_preregistrations(self, data, predicted_preregistrations):
        dict = {
            "Collegejaar": [],
            "Faculteit": [],
            "Examentype": [],
            "Herkomst": [],
            "Croho groepeernaam": [],
            "Weeknummer": [],
            "SARIMA_cumulative": [],
            "SARIMA_individual": [],
            "Voorspelde vooraanmelders": [],
        }

        index = 0
        for _, row in data.iterrows():
            if index >= len(predicted_preregistrations):
                print(f"Index {index} out of range: {len(predicted_preregistrations)}")
                continue

            # The current_predicted_preregistrations is a list out of a list of lists that
            # belongs to a row of the data_to_predict and contains values that are filled in
            # in the dataframe in the column of 'Voorspelde vooraanmelders' that will be created
            # for the next weeks.
            current_predicted_preregistrations = predicted_preregistrations[index]

            current_week = increment_week(row["Weeknummer"])
            for current_prediction in current_predicted_preregistrations:
                dict["Collegejaar"].append(row["Collegejaar"])
                dict["Faculteit"].append(row["Faculteit"])
                dict["Examentype"].append(row["Examentype"])
                dict["Herkomst"].append(row["Herkomst"])
                dict["Croho groepeernaam"].append(row["Croho groepeernaam"])
                dict["Weeknummer"].append(current_week)
                dict["SARIMA_cumulative"].append(np.nan)
                dict["SARIMA_individual"].append(np.nan)
                dict["Voorspelde vooraanmelders"].append(current_prediction)

                current_week = increment_week(current_week)

            index += 1

        # This concatenation method concats the data_to_predict with newly created dataframe. The
        # ignore_index=True makes sure that the dataframe will be indexed from 0 to n-1.
        return pd.concat([data, pd.DataFrame(dict)], ignore_index=True)

    # Helper function for prepare_data() (see below). Makes sure there will not be more predicted
    # students for a numerus fixus programme than the number of students that are allowed.
    def _numerus_fixus_cap(self, data, year, week):
        for nf in self.numerus_fixus_list:
            nf_data = data[
                (data["Collegejaar"] == year)
                & (data["Weeknummer"] == week)
                & (data["Croho groepeernaam"] == nf)
            ]
            if np.sum(nf_data["SARIMA_individual"]) > self.numerus_fixus_list[nf]:
                data = self._nf_students_based_on_distribution_of_last_years(
                    data, self.data_latest, nf, year, week, "SARIMA_individual"
                )

            if np.sum(nf_data["SARIMA_cumulative"]) > self.numerus_fixus_list[nf]:
                data = self._nf_students_based_on_distribution_of_last_years(
                    data, self.data_latest, nf, year, week, "SARIMA_cumulative"
                )

        return data

    def _nf_students_based_on_distribution_of_last_years(
        self, data, data_latest, nf, year, week, method
    ):
        last_years_data = data_latest[
            (data_latest["Collegejaar"] < year)
            & (data_latest["Collegejaar"] >= year - 3)
            & (data_latest["Weeknummer"] == week)
            & (data_latest["Croho groepeernaam"] == nf)
        ].fillna(0)
        distribution_per_herkomst = {"EER": [], "NL": [], "Niet-EER": []}
        for last_year in range(year - 3, year):
            total_students = last_years_data[last_years_data["Collegejaar"] == last_year][
                "Aantal_studenten"
            ].sum()
            for herkomst in distribution_per_herkomst:
                distribution_per_herkomst[herkomst].append(
                    last_years_data[
                        (last_years_data["Collegejaar"] == last_year)
                        & (last_years_data["Herkomst"] == herkomst)
                    ]["Aantal_studenten"].values[0]
                    / total_students
                )
        for herkomst in distribution_per_herkomst:
            data.loc[
                (data["Collegejaar"] == year)
                & (data["Weeknummer"] == week)
                & (data["Croho groepeernaam"] == nf)
                & (data["Herkomst"] == herkomst),
                method,
            ] = self.numerus_fixus_list[nf] * mean(distribution_per_herkomst[herkomst])
        return data

    # This method is used in the main class after predicting the number of students and processes
    # the data to be ready for output. The only calculation done is the numerus fixus cap. For the
    # rest it is only merging some dataframes, removing columns and replacing faculty codes.
    def prepare_data_for_output_prelim(self, data, year, week, data_cumulative=None, skip_years=0):
        self.data = data
        self.data = self._numerus_fixus_cap(self.data, year, week)

        # We will remove redundant columns we don't want in our output_prelim.
        columns_to_select = [
            "Croho groepeernaam",
            "Faculteit",
            "Examentype",
            "Collegejaar",
            "Herkomst",
            "Weeknummer",
            "SARIMA_cumulative",
            "SARIMA_individual",
            "Voorspelde vooraanmelders",
        ]
        if skip_years > 0:
            columns_to_select = columns_to_select + ["Skip_prediction"]
        self.data = self.data[columns_to_select]

        # We will add a column with the studentcount ('Aantal_studenten') if this data exists.
        if self.data_studentcount is not None:
            self.data = self.data.merge(
                self.data_studentcount,
                on=["Croho groepeernaam", "Collegejaar", "Herkomst", "Examentype"],
                how="left",
            )

        if data_cumulative is not None:
            # In the data_cumulative are wrong faculty codes because they haven't been replaced.
            # This causes mistakes with merging. Therefore we drop these 'old' ones.
            data_cumulative = data_cumulative.drop(columns="Faculteit")

            # This merge adds the following columns: 'Gewogen vooraanmelders', 'Ongewogen
            # vooraanmelders', 'Aantal aanmelders met 1 aanmelding', 'Inschrijvingen'
            self.data = self.data.merge(
                data_cumulative,
                on=[
                    "Croho groepeernaam",
                    "Collegejaar",
                    "Herkomst",
                    "Weeknummer",
                    "Examentype",
                ],
                how="left",
            )

        # Exportation to .xlsx
        output_path = os.path.join(self.CWD, "data", "output", "output_prelim.xlsx")
        self.data.to_excel(output_path, index=False)

    # This method predicts the influx of students by looking at the ratio between pre-registrants
    # in every calender week and the actual influx of students and it takes the average of the
    # relevant ratio of every year since 2021.
    def predict_with_ratio(self, data_cumulative, predict_year):
        average_ratio_between = (predict_year - 3, predict_year - 1)
        if self.data_studentcount is not None:
            data_vooraanmeldingen = data_cumulative[
                [
                    "Collegejaar",
                    "Weeknummer",
                    "Examentype",
                    "Croho groepeernaam",
                    "Herkomst",
                    "Ongewogen vooraanmelders",
                    "Inschrijvingen",
                ]
            ]

            # Convert it's type to float to use the number for calculations.
            data_vooraanmeldingen["Ongewogen vooraanmelders"] = data_vooraanmeldingen[
                "Ongewogen vooraanmelders"
            ].astype("float")

            data_vooraanmeldingen["Aanmeldingen"] = (
                data_vooraanmeldingen["Ongewogen vooraanmelders"]
                + data_vooraanmeldingen["Inschrijvingen"]
            )

            # We can only use the data from previous years because we don't know yet how many
            # students will apply this year. Therefore we can't calculate the ratio of this year.
            data_vooraanmeldingen = data_vooraanmeldingen[
                (data_vooraanmeldingen["Collegejaar"] >= average_ratio_between[0])
                & (data_vooraanmeldingen["Collegejaar"] <= average_ratio_between[1])
            ]

            # Adding the actual studentcounts of last year to the dataframe.
            data_merged = pd.merge(
                left=data_vooraanmeldingen,
                right=self.data_studentcount,
                on=["Collegejaar", "Examentype", "Croho groepeernaam", "Herkomst"],
                how="left",
            )

            # Calculating the actual ratio for every programme, for every year and week.
            data_merged["Average_Ratio"] = data_merged["Aanmeldingen"].divide(
                data_merged["Aantal_studenten"]
            )

            # We replace the infinite values with NaN's because this would otherwise lead to a
            # prediction of an infinite influx of students.
            data_merged["Average_Ratio"] = data_merged["Average_Ratio"].replace(np.inf, np.nan)

            # We remove the redundant columns so that we can merge it with self.data in a bit.
            average_ratios = (
                data_merged[
                    [
                        "Croho groepeernaam",
                        "Examentype",
                        "Herkomst",
                        "Weeknummer",
                        "Average_Ratio",
                    ]
                ]
                .groupby(
                    ["Croho groepeernaam", "Examentype", "Herkomst", "Weeknummer"],
                    as_index=False,
                )
                .mean()
            )

            self.data = self.data.rename(columns={"Aantal eerstejaarsopleiding": "EOI_vorigjaar"})

            self.data["Ongewogen vooraanmelders"] = self.data["Ongewogen vooraanmelders"].astype(
                "float"
            )
            self.data["Aanmelding"] = (
                self.data["Ongewogen vooraanmelders"] + self.data["Inschrijvingen"]
            )

            self.data["Ratio"] = (self.data["Aanmelding"]).divide(self.data["Aantal_studenten"])
            self.data["Ratio"] = self.data["Ratio"].replace(np.inf, np.nan)

            self.data = pd.merge(
                left=self.data,
                right=average_ratios,
                how="left",
                on=["Croho groepeernaam", "Examentype", "Herkomst", "Weeknummer"],
            )

            # Calculating the actual prognose ratio.
            for i, row in self.data.iterrows():
                if row["Average_Ratio"] != 0:
                    self.data.at[i, "Prognose_ratio"] = row["Aanmelding"] / row["Average_Ratio"]

            # Check if numerus fixus values are below the numerus fixus cap
            NFs = self.numerus_fixus_list
            for year in self.data["Collegejaar"].unique():
                for week in self.data["Weeknummer"].unique():
                    for nf in NFs:
                        nf_data = self.data[
                            (self.data["Collegejaar"] == year)
                            & (self.data["Weeknummer"] == week)
                            & (self.data["Croho groepeernaam"] == nf)
                        ]

                        if np.sum(nf_data["Prognose_ratio"]) > NFs[nf]:
                            self.data.loc[
                                (self.data["Collegejaar"] == year)
                                & (self.data["Weeknummer"] == week)
                                & (self.data["Croho groepeernaam"] == nf)
                                & (self.data["Herkomst"] == "NL"),
                                "Prognose_ratio",
                            ] = nf_data[nf_data["Herkomst"] == "NL"]["Prognose_ratio"] - (
                                np.sum(nf_data["Prognose_ratio"]) - NFs[nf]
                            )

    # Postprocess the total data by adding the forecasted and latest data and adding the ensemble
    # and error values.
    def postprocess(self, predict_year, predict_week):
        # Postprocess the total data, i.e. the forecasted and latest data
        if self.data_latest is not None:
            self.data = replace_latest_data(
                self.data_latest, self.data, predict_year, predict_week
            )

        # Calculate and add ensemble values ('Ensemble_prediction, 'Weighted_ensemble_prediction'
        # and 'Average_ensemble_prediction')
        # Only do this when predicting both datasets because the ensemble values won't be valid
        # otherwise.
        if self.data_option == DataOption.BOTH_DATASETS:
            self._create_ensemble_columns(predict_year, predict_week)

        # Calculate and add error values ('MAE_...' and 'MAPE_...')
        self._create_error_columns()

        self.data = self.data.drop_duplicates()

        self.data_latest = self.data

    def _create_ensemble_columns(self, predict_year, predict_week):
        self.data = self.data.sort_values(
            by=["Croho groepeernaam", "Herkomst", "Collegejaar", "Weeknummer"]
        )
        self.data = self.data.reset_index(drop=True)

        # Initialize columns
        self.data.loc[
            (self.data["Collegejaar"] == predict_year) & (self.data["Weeknummer"] == predict_week),
            "Ensemble_prediction",
        ] = np.nan
        self.data.loc[
            (self.data["Collegejaar"] == predict_year) & (self.data["Weeknummer"] == predict_week),
            "Weighted_ensemble_prediction",
        ] = -1.0

        # Compute ensemble predictions using vectorized operations

        for examentype in self.data[
            (self.data["Collegejaar"] == predict_year) & (self.data["Weeknummer"] == predict_week)
        ]["Examentype"].unique():
            self.data.loc[
                (self.data["Collegejaar"] == predict_year)
                & (self.data["Weeknummer"] == predict_week)
                & (self.data["Examentype"] == examentype),
                "Ensemble_prediction",
            ] = self.data[
                (self.data["Collegejaar"] == predict_year)
                & (self.data["Weeknummer"] == predict_week)
                & (self.data["Examentype"] == examentype)
            ].apply(
                self._get_normal_ensemble, axis=1
            )

            if self.ensemble_weights is not None:
                # Merge weights into data for vectorized calculations
                weights = self.ensemble_weights.rename(columns={"Programme": "Croho groepeernaam"})
                if "Average_ensemble_prediction" not in self.data.columns:
                    weights = weights.rename(
                        columns={
                            "Average_ensemble_prediction": "Average_ensemble_prediction_weight"
                        }
                    )
                self.data = self.data.merge(
                    weights,
                    on=["Collegejaar", "Croho groepeernaam", "Examentype", "Herkomst"],
                    how="left",
                    suffixes=("", "_weight"),
                )

                weighted_ensemble = (
                    self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Weeknummer"] == predict_week)
                        & (self.data["Examentype"] == examentype)
                    ]["SARIMA_cumulative"].fillna(0)
                    * self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Weeknummer"] == predict_week)
                        & (self.data["Examentype"] == examentype)
                    ]["SARIMA_cumulative_weight"].fillna(0)
                    + self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Weeknummer"] == predict_week)
                        & (self.data["Examentype"] == examentype)
                    ]["SARIMA_individual"].fillna(0)
                    * self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Weeknummer"] == predict_week)
                        & (self.data["Examentype"] == examentype)
                    ]["SARIMA_individual_weight"].fillna(0)
                    + self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Weeknummer"] == predict_week)
                        & (self.data["Examentype"] == examentype)
                    ]["Prognose_ratio"].fillna(0)
                    * self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Weeknummer"] == predict_week)
                        & (self.data["Examentype"] == examentype)
                    ]["Prognose_ratio_weight"].fillna(0)
                )

                self.data.loc[
                    (self.data["Collegejaar"] == predict_year)
                    & (self.data["Weeknummer"] == predict_week)
                    & (self.data["Examentype"] == examentype),
                    "Weighted_ensemble_prediction",
                ] = np.where(
                    self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Weeknummer"] == predict_week)
                        & (self.data["Examentype"] == examentype)
                    ]["Average_ensemble_prediction_weight"]
                    != 1,
                    weighted_ensemble,
                    self.data[
                        (self.data["Collegejaar"] == predict_year)
                        & (self.data["Weeknummer"] == predict_week)
                        & (self.data["Examentype"] == examentype)
                    ]["Weighted_ensemble_prediction"],
                )

                self.data = self.data.drop(
                    [
                        "SARIMA_cumulative_weight",
                        "SARIMA_individual_weight",
                        "Prognose_ratio_weight",
                        "Average_ensemble_prediction_weight",
                    ],
                    axis=1,
                )

        # Compute average ensemble predictions
        self.data["Average_ensemble_prediction"] = np.nan

        # Use groupby and rolling to calculate averages
        self.data["Average_ensemble_prediction"] = self.data.groupby(
            ["Croho groepeernaam", "Examentype", "Herkomst", "Collegejaar"]
        )["Ensemble_prediction"].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean().shift().bfill()
        )

        # Update Weighted_ensemble_prediction where needed
        self.data["Weighted_ensemble_prediction"] = np.where(
            self.data["Weighted_ensemble_prediction"] == -1.0,
            self.data["Average_ensemble_prediction"],
            self.data["Weighted_ensemble_prediction"],
        )

    def _get_normal_ensemble(self, row):
        sarima_cumulative = convert_nan_to_zero(row["SARIMA_cumulative"])
        sarima_individual = convert_nan_to_zero(row["SARIMA_individual"])

        if row["Croho groepeernaam"] in [
            "B Geneeskunde",
            "B Biomedische Wetenschappen",
            "B Tandheelkunde",
        ]:
            return sarima_cumulative
        elif row["Weeknummer"] in range(17, 24) and row["Examentype"] == "Master":
            return sarima_individual * 0.2 + sarima_cumulative * 0.8
        elif row["Weeknummer"] in range(30, 35):
            return sarima_individual * 0.6 + sarima_cumulative * 0.4
        elif row["Weeknummer"] in range(35, 38):
            return sarima_individual * 0.7 + sarima_cumulative * 0.3
        elif row["Weeknummer"] == 38:
            return sarima_individual
        else:
            return sarima_individual * 0.5 + sarima_cumulative * 0.5

    def _create_error_columns(self):
        # Only calculate the error values for the predictions that are made.
        if self.data_option == DataOption.BOTH_DATASETS:
            predictions = [
                "Weighted_ensemble_prediction",
                "Average_ensemble_prediction",
                "Ensemble_prediction",
                "Prognose_ratio",
                "SARIMA_cumulative",
                "SARIMA_individual",
            ]

        elif self.data_option == DataOption.INDIVIDUAL:
            predictions = [
                "SARIMA_individual",
            ]

        elif self.data_option == DataOption.CUMULATIVE:
            predictions = [
                "Prognose_ratio",
                "SARIMA_cumulative",
            ]

        mae_columns = [f"MAE_{pred}" for pred in predictions]
        mape_columns = [f"MAPE_{pred}" for pred in predictions]

        for col in mae_columns + mape_columns:
            self.data[col] = np.nan

        valid_rows = ~self.data["Croho groepeernaam"].isin(self.numerus_fixus_list)

        for pred in predictions:
            predicted = self.data[pred].apply(convert_nan_to_zero)
            self.data[f"MAE_{pred}"] = abs(self.data["Aantal_studenten"] - predicted).where(
                valid_rows
            )
            self.data[f"MAPE_{pred}"] = (
                abs(self.data["Aantal_studenten"] - predicted) / self.data["Aantal_studenten"]
            ).where(valid_rows, np.nan)

    def _calculate_errors(self, row):
        actual = row["Aantal_studenten"]
        errors = {}
        for key in [
            "Weighted_ensemble_prediction",
            "Average_ensemble_prediction",
            "Ensemble_prediction",
            "Prognose_ratio",
            "SARIMA_cumulative",
            "SARIMA_individual",
        ]:
            predicted = convert_nan_to_zero(row[key])
            errors[f"MAE_{key}"] = self._mean_absolute_error(actual, predicted)
            errors[f"MAPE_{key}"] = self._mean_absolute_percentage_error(actual, predicted)
        return errors

    def _mean_absolute_error(self, actual, predicted):
        return abs(actual - predicted)

    def _mean_absolute_percentage_error(self, actual, predicted):
        return abs((actual - predicted) / actual) if actual != 0 else np.nan

    def ready_new_data(self):
        self.data_latest = self.data

    def save_output(self, student_year_prediction):
        output_filename = "output_"

        if student_year_prediction == StudentYearPrediction.FIRST_YEARS:
            output_filename += "first-years"
        elif student_year_prediction == StudentYearPrediction.HIGHER_YEARS:
            output_filename += "higher-years"
        elif student_year_prediction == StudentYearPrediction.VOLUME:
            output_filename += "volume"

        output_filename += ".xlsx"

        output_path = os.path.join(self.CWD, "data", "output", output_filename)

        self.data.sort_values(
            by=["Croho groepeernaam", "Examentype", "Collegejaar", "Weeknummer", "Herkomst"],
            inplace=True,
            ignore_index=True,
        )

        self.data.to_excel(output_path, index=False)
