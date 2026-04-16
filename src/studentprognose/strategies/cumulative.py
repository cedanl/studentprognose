import numpy as np
import pandas as pd
import joblib
import os
import math

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

from studentprognose.strategies.base import PredictionStrategy
from studentprognose.utils.weeks import increment_week
from studentprognose.models.sarima import predict_with_sarima_cumulative, _get_transformed_data
from studentprognose.models.xgboost_regressor import predict_with_xgboost


class CumulativeStrategy(PredictionStrategy):
    def __init__(self, data_cumulative, data_studentcount, configuration,
                 data_latest, ensemble_weights, cwd, data_option, ci_test_n):
        super().__init__(configuration, data_latest, ensemble_weights,
                         data_studentcount, cwd, data_option, ci_test_n)

        self.data_cumulative = data_cumulative
        self.data_studentcount = data_studentcount
        self.skip_years = 0

    def preprocess(self):
        data = self.data_cumulative

        data = self._cast_string_to_float(data, "Ongewogen vooraanmelders")
        data = self._cast_string_to_float(data, "Gewogen vooraanmelders")
        data = self._cast_string_to_float(data, "Aantal aanmelders met 1 aanmelding")
        data = self._cast_string_to_float(data, "Inschrijvingen")

        data = data.rename(
            columns={
                "Type hoger onderwijs": "Examentype",
                "Groepeernaam Croho": "Croho groepeernaam",
            }
        )

        data.loc[(data["Examentype"] == "Pre-master"), "Hogerejaars"] = "Nee"
        data = data[data["Hogerejaars"] == "Nee"]

        data = (
            data.groupby([
                "Collegejaar", "Croho groepeernaam", "Faculteit",
                "Examentype", "Herkomst", "Weeknummer",
            ])
            .sum(numeric_only=False)
            .reset_index()
        )
        data = data[[
            "Weeknummer", "Collegejaar", "Faculteit", "Examentype",
            "Herkomst", "Croho groepeernaam", "Gewogen vooraanmelders",
            "Ongewogen vooraanmelders", "Aantal aanmelders met 1 aanmelding",
            "Inschrijvingen",
        ]]

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
        self.data_cumulative = self.data_cumulative_backup.copy(deep=True)
        self.set_year_week(predict_year, predict_week, self.data_cumulative)
        self._prepare_data()
        self.data_cumulative = self.data_cumulative.astype(
            {"Weeknummer": "int32", "Collegejaar": "int32"}
        )

        min_training_year = self.configuration.get("model_config", {}).get("min_training_year", 2016)
        full_data = _get_transformed_data(self.data_cumulative.copy(deep=True), min_training_year)
        full_data["39"] = 0

        self.skip_years = skip_years

        data_to_predict = self.data_cumulative[
            (self.data_cumulative["Collegejaar"] == self.predict_year)
            & (self.data_cumulative["Weeknummer"] == self.predict_week)
        ]
        if self.programme_filtering != []:
            data_to_predict = data_to_predict[
                data_to_predict["Croho groepeernaam"].isin(self.programme_filtering)
            ]
        if self.herkomst_filtering != []:
            data_to_predict = data_to_predict[
                data_to_predict["Herkomst"].isin(self.herkomst_filtering)
            ]
        if self.examentype_filtering != []:
            data_to_predict = data_to_predict[
                data_to_predict["Examentype"].isin(self.examentype_filtering)
            ]

        if len(data_to_predict) == 0:
            return None

        nr_CPU_cores = os.cpu_count()
        chunk_size = math.ceil(len(data_to_predict) / nr_CPU_cores)
        chunks = [
            data_to_predict[i : i + chunk_size] for i in range(0, len(data_to_predict), chunk_size)
        ]

        print("Start parallel predicting...")
        self.predicted_data = joblib.Parallel(n_jobs=nr_CPU_cores)(
            joblib.delayed(self._predict_sarima)(row)
            for chunk in chunks
            for _, row in chunk.iterrows()
        )

        data_to_predict["SARIMA_individual"] = np.nan
        data_to_predict["Voorspelde vooraanmelders"] = np.nan

        data_to_predict = self.postprocessor.add_predicted_preregistrations(
            data_to_predict, [x[: self.pred_len] for x in self.predicted_data]
        )

        data_to_predict = self._predict_students_with_preapplicants(
            full_data, self.predicted_data, data_to_predict
        )

        return data_to_predict

    def _prepare_data(self):
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

        if int(self.predict_week) > 38:
            self.pred_len = 38 + 52 - int(self.predict_week)
        else:
            self.pred_len = 38 - int(self.predict_week)

    def _predict_sarima(self, row, already_printed=False):
        return predict_with_sarima_cumulative(
            self.data_cumulative, row, self.predict_year, self.predict_week,
            self.pred_len, self.skip_years, already_printed
        )

    def _predict_students_with_preapplicants(self, data, predictions, data_to_predict):
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
                    train, test, data_to_predict,
                    (data_to_predict["Croho groepeernaam"] == programme)
                    & (data_to_predict["Herkomst"] == herkomst)
                    & (data_to_predict["Examentype"] == examentype),
                )

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
                train, test, data_to_predict,
                (data_to_predict["Examentype"] == examtype)
                & (~data_to_predict["Croho groepeernaam"].isin(self.numerus_fixus_list)),
            )

        return data_to_predict

    def _predict_with_xgboost_extra_year(self, train, test, data_to_predict, replace_mask):
        columns_to_match = [
            "Collegejaar", "Faculteit", "Examentype", "Herkomst", "Croho groepeernaam",
        ]

        if self.skip_years > 0:
            train2 = train[(train["Collegejaar"] < self.predict_year - (self.skip_years * 2))]
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
                ahead_predictions = predict_with_xgboost(train2, test2_merged, self.data_studentcount)
                test2["Collegejaar"] = test2["Collegejaar"] + self.skip_years

                mask = (
                    data_to_predict[columns_to_match]
                    .apply(tuple, axis=1)
                    .isin(test2_merged[columns_to_match].apply(tuple, axis=1))
                )
                full_mask = (
                    replace_mask & (data_to_predict["Weeknummer"] == self.predict_week) & mask
                )
                data_to_predict.loc[full_mask, "Skip_prediction"] = ahead_predictions[: full_mask.sum()]
        else:
            train = train[(train["Collegejaar"] < self.predict_year)]
            test = test[(test["Collegejaar"] == self.predict_year)]

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
                predictions = predict_with_xgboost(train, test_merged, self.data_studentcount)

                mask = (
                    data_to_predict[columns_to_match]
                    .apply(tuple, axis=1)
                    .isin(test_merged[columns_to_match].apply(tuple, axis=1))
                )
                full_mask = (
                    replace_mask & (data_to_predict["Weeknummer"] == self.predict_week) & mask
                )
                data_to_predict.loc[full_mask, "SARIMA_cumulative"] = predictions[: full_mask.sum()]

        return data_to_predict
