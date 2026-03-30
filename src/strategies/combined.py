import numpy as np
import joblib
import os
import math

from src.strategies.base import PredictionStrategy
from src.strategies.individual import IndividualStrategy
from src.strategies.cumulative import CumulativeStrategy
from src.utils.weeks import get_weeks_list
from src.data.s06_transforms import transform_data
from src.models.s07_sarima import predict_with_sarima_individual, predict_with_sarima_cumulative, _get_transformed_data


class CombinedStrategy(PredictionStrategy):
    def __init__(self, data_individual, data_cumulative,
                 data_studentcount, configuration,
                 data_latest, ensemble_weights, cwd, data_option, ci_test_n,
                 years):
        super().__init__(configuration, data_latest, ensemble_weights,
                         data_studentcount, cwd, data_option, ci_test_n)

        self.individual = IndividualStrategy(
            data_individual, configuration,
            data_latest, ensemble_weights, data_studentcount,
            cwd, data_option, ci_test_n,
        )
        self.cumulative = CumulativeStrategy(
            data_cumulative, data_studentcount, configuration,
            data_latest, ensemble_weights, cwd, data_option, ci_test_n,
        )

        if not all(
            year in self.individual.data_individual["Collegejaar"].unique() for year in years
        ):
            raise ValueError(
                f"Selected years {years} not found in individual dataset. Proceeding with cumulative dataset."
            )
        self.years = years

    def preprocess(self):
        print("Preprocessing individual data...")
        self.individual.preprocess()
        print("Preprocessing cumulative data...")
        return self.cumulative.preprocess()

    def predict_nr_of_students(self, predict_year, predict_week, skip_years=0):
        self.individual.data_individual = self.individual.data_individual_backup.copy(deep=True)
        self.cumulative.data_cumulative = self.cumulative.data_cumulative_backup.copy(deep=True)

        self.set_year_week(predict_year, predict_week, self.cumulative.data_cumulative)
        self.individual.set_year_week(predict_year, predict_week, self.individual.data_individual)
        self.cumulative.set_year_week(predict_year, predict_week, self.cumulative.data_cumulative)

        self.individual.data_individual = self.individual.data_individual.merge(
            self.cumulative.data_cumulative,
            on=[
                "Croho groepeernaam", "Collegejaar", "Faculteit",
                "Examentype", "Weeknummer", "Herkomst",
            ],
            how="left",
        )

        from src.models.s05_xgboost_classifier import predict_applicant
        print("Predicting preapplicants...")
        predicties = predict_applicant(
            self.individual.data_individual, self.predict_year, self.predict_week,
            self.individual.max_year,
            self.cumulative.data_cumulative,
        )
        self.individual.data_individual.loc[
            (self.individual.data_individual["Collegejaar"] == self.predict_year)
            & (self.individual.data_individual["Weeknummer"].isin(get_weeks_list(self.predict_week))),
            "Inschrijvingen_predictie",
        ] = predicties

        self.individual._transform_data_individual()

        temp_data_individual = self.individual.data_individual.copy(deep=True)
        temp_data_individual["Weeknummer"] = self.individual.data_individual["Weeknummer"].astype(int)

        self.data_exog = temp_data_individual.merge(
            self.cumulative.data_cumulative,
            on=[
                "Croho groepeernaam", "Collegejaar", "Examentype",
                "Faculteit", "Weeknummer", "Herkomst",
            ],
            how="left",
        )

        self.individual.data_individual = transform_data(
            self.individual.data_individual, "Cumulative_sum_within_year"
        )

        self.cumulative._prepare_data()

        full_data = _get_transformed_data(self.cumulative.data_cumulative.copy(deep=True))
        full_data["39"] = 0

        self.skip_years = skip_years

        data_to_predict = self.cumulative.data_cumulative[
            (self.cumulative.data_cumulative["Collegejaar"] == self.predict_year)
            & (self.cumulative.data_cumulative["Weeknummer"] == self.predict_week)
            & (
                self.cumulative.data_cumulative["Croho groepeernaam"]
                != "M Educatie in de Mens- en Maatschappijwetenschappen"
            )
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
            joblib.delayed(self._predict_sarima_both)(row)
            for chunk in chunks
            for _, row in chunk.iterrows()
        )

        data_to_predict["SARIMA_individual"] = [x[0] for x in self.predicted_data]
        data_to_predict["Voorspelde vooraanmelders"] = np.nan

        if self.predict_week != 38:
            data_to_predict = self.postprocessor.add_predicted_preregistrations(
                data_to_predict, [x[1] for x in self.predicted_data]
            )

        data_to_predict = self.cumulative._predict_students_with_preapplicants(
            full_data, [x[1] for x in self.predicted_data], data_to_predict
        )

        return data_to_predict

    def _predict_sarima_both(self, row):
        print(
            f"Prediction for {row['Croho groepeernaam']}, {row['Examentype']}, {row['Herkomst']}, year: {self.predict_year}, week: {self.predict_week}"
        )

        sarima_individual = predict_with_sarima_individual(
            self.individual.data_individual, row, self.predict_year, self.predict_week,
            self.individual.max_year, self.numerus_fixus_list, self.data_exog, already_printed=True,
        )
        if self.predict_week == 38:
            return sarima_individual, []
        else:
            predicted_preregistration = predict_with_sarima_cumulative(
                self.cumulative.data_cumulative, row, self.predict_year, self.predict_week,
                self.cumulative.pred_len, self.cumulative.skip_years, already_printed=True,
            )

        return sarima_individual, predicted_preregistration
