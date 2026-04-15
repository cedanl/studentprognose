import datetime
import numpy as np
import pandas as pd
import joblib
import os
import math

from studentprognose.strategies.base import PredictionStrategy
from studentprognose.utils.weeks import get_weeks_list, get_all_weeks_valid, decrement_week
from studentprognose.data.transforms import transform_data
from studentprognose.models.xgboost_classifier import predict_applicant, DEFAULT_STATUS_MAP
from studentprognose.models.sarima import predict_with_sarima_individual


class IndividualStrategy(PredictionStrategy):
    def __init__(self, data_individual, configuration,
                 data_latest, ensemble_weights, data_studentcount,
                 cwd, data_option, ci_test_n):
        super().__init__(configuration, data_latest, ensemble_weights,
                         data_studentcount, cwd, data_option, ci_test_n)

        self.data_individual = data_individual

    def preprocess(self):
        data = self.data_individual

        data = data.drop(labels=["Aantal studenten"], axis=1)

        data = data[
            ~(
                (data["Croho groepeernaam"] == "B English Language and Culture")
                & (data["Collegejaar"] == 2021)
                & (data["Examentype"] != "Propedeuse Bachelor")
            )
        ]

        grouped = data.groupby(["Collegejaar", "Sleutel"])
        data["Sleutel_count"] = grouped["Sleutel"].transform("count")

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

        data = data[
            data["Ingangsdatum"].str.contains("01-09-")
            | data["Ingangsdatum"].str.contains("01-10-")
        ]

        data["is_numerus_fixus"] = (
            data["Croho groepeernaam"].isin(self.numerus_fixus_list)
        ).astype(int)

        data["Examentype"] = data["Examentype"].replace("Propedeuse Bachelor", "Bachelor")

        data = data[data["Inschrijfstatus"].notna()]
        data = data[data["Examentype"].isin(["Bachelor", "Master", "Pre-master"])]

        nationaliteit_counts = data["Nationaliteit"].value_counts()
        values_to_change = nationaliteit_counts[nationaliteit_counts < 100].index
        data["Nationaliteit"] = data["Nationaliteit"].replace(values_to_change, "Overig")

        def get_new_column(row):
            if (
                row["Weeknummer"] == 17
                and not row["Croho groepeernaam"] in self.numerus_fixus_list
            ):
                return True
            else:
                return False

        data["Deadlineweek"] = data.apply(get_new_column, axis=1)

        data = data.drop(["Sleutel"], axis=1)

        # Ensure numeric types for flag columns (raw data may contain strings like "Nee"/"Ja")
        for col in ["Is eerstejaars croho opleiding", "Is hogerejaars", "BBC ontvangen"]:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0).astype(int)

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

        print("Predicting preapplicants...")
        predicties = predict_applicant(
            self.data_individual, self.predict_year, self.predict_week,
            self.max_year, configuration=self.configuration
        )

        self.data_individual.loc[
            (self.data_individual["Collegejaar"] == self.predict_year)
            & (self.data_individual["Weeknummer"].isin(get_weeks_list(self.predict_week))),
            "Inschrijvingen_predictie",
        ] = predicties

        self._transform_data_individual()
        self.data_individual = transform_data(self.data_individual, "Cumulative_sum_within_year")

        data_to_predict = self.get_data_to_predict(
            self.data_individual,
            self.programme_filtering,
            self.herkomst_filtering,
            self.examentype_filtering,
        )

        if len(data_to_predict) == 0:
            return None

        nr_CPU_cores = os.cpu_count()
        chunk_size = math.ceil(len(data_to_predict) / nr_CPU_cores)

        chunks = [
            data_to_predict[i : i + chunk_size] for i in range(0, len(data_to_predict), chunk_size)
        ]

        print("Start parallel predicting...")
        predicted_data = joblib.Parallel(n_jobs=nr_CPU_cores)(
            joblib.delayed(self._predict_sarima)(row)
            for chunk in chunks
            for _, row in chunk.iterrows()
        )

        data_to_predict["SARIMA_individual"] = predicted_data
        data_to_predict["SARIMA_cumulative"] = np.nan
        data_to_predict["Voorspelde vooraanmelders"] = np.nan

        return data_to_predict

    def _predict_sarima(self, row, data_exog=None, already_printed=False):
        return predict_with_sarima_individual(
            self.data_individual, row, self.predict_year, self.predict_week,
            self.max_year, self.numerus_fixus_list, data_exog, already_printed
        )

    def _transform_data_individual(self):
        data = self.data_individual

        data = data[data["Collegejaar"] <= self.predict_year]

        group_cols = [
            "Collegejaar",
            "Faculteit",
            "Herkomst",
            "Examentype",
            "Croho groepeernaam",
        ]

        all_weeks = [str(i) for i in range(39, 53)] + [str(i) for i in range(1, 39)]

        target_year_weeknummers = []
        if int(self.predict_week) > 38:
            target_year_weeknummers = [str(i) for i in range(39, int(self.predict_week) + 1)]
        elif int(self.predict_week) < 39:
            target_year_weeknummers = [str(i) for i in range(39, 53)] + [
                str(i) for i in range(1, int(self.predict_week) + 1)
            ]

        data = data[group_cols + ["Inschrijvingen_predictie", "Inschrijfstatus", "Weeknummer"]]
        data["Weeknummer"] = data["Weeknummer"].astype(str)

        status_map = self.configuration.get("model_config", {}).get(
            "status_mapping", DEFAULT_STATUS_MAP
        )
        data["Inschrijfstatus"] = data["Inschrijfstatus"].map(status_map)

        data = data.groupby(group_cols + ["Weeknummer"]).sum(numeric_only=False).reset_index()

        def _transform_inner(input_data, target_col, weeknummers):
            data2 = input_data.reset_index().drop(["index", target_col], axis=1)
            input_data = input_data.pivot(
                index=group_cols, columns="Weeknummer", values=target_col
            ).reset_index()

            input_data.columns = map(str, input_data.columns)
            col_set = set(input_data.columns)
            available_weeks = [w for w in weeknummers if w in col_set]
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
            else:
                for week in missing_weeks:
                    input_data[week] = 0

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

        data_real = data[data["Collegejaar"] != self.predict_year]
        data_real = _transform_inner(data_real, "Inschrijfstatus", all_weeks)

        data_predict = data[data["Collegejaar"] == self.predict_year]
        data_predict = _transform_inner(
            data_predict, "Inschrijvingen_predictie", target_year_weeknummers
        )

        self.data_individual = pd.concat([data_real, data_predict])
