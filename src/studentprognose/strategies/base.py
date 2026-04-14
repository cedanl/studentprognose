from abc import ABC, abstractmethod
import collections
import numpy as np
import pandas as pd

from studentprognose.utils.weeks import get_max_week
from studentprognose.output.postprocessor import PostProcessor


class PredictionStrategy(ABC):
    """Base class for prediction strategies (Individual, Cumulative, Combined)."""

    def __init__(self, configuration, data_latest, ensemble_weights,
                 data_studentcount, cwd, data_option, ci_test_n):
        self.configuration = configuration
        self.numerus_fixus_list = configuration["numerus_fixus"]

        self.postprocessor = PostProcessor(
            configuration, data_latest, ensemble_weights,
            data_studentcount, cwd, data_option, ci_test_n,
        )

        self.programme_filtering = []
        self.herkomst_filtering = []
        self.examentype_filtering = []

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def predict_nr_of_students(self, predict_year, predict_week, skip_years):
        pass

    def set_year_week(self, predict_year, predict_week, data):
        self.predict_year = predict_year
        self.predict_week = predict_week

        self.max_year = data["Collegejaar"].max()
        self.max_week = get_max_week(self.predict_year, self.max_year, data, "Collegejaar")

    def set_filtering(self, programme_filtering, herkomst_filtering, examentype_filtering):
        self.programme_filtering = programme_filtering
        self.herkomst_filtering = herkomst_filtering
        self.examentype_filtering = examentype_filtering

    def get_data_to_predict(
        self, data, programme_filtering=[], herkomst_filtering=[], examentype_filtering=[]
    ):
        predict_dict = {
            "Croho groepeernaam": [],
            "Herkomst": [],
            "Collegejaar": [],
            "Weeknummer": [],
            "Examentype": [],
            "Faculteit": [],
        }

        all_programmes = data["Croho groepeernaam"].unique()
        if programme_filtering != []:
            all_programmes = list(
                (
                    collections.Counter(all_programmes) & collections.Counter(programme_filtering)
                ).elements()
            )

        all_herkomsts = data["Herkomst"].unique()
        if herkomst_filtering != []:
            all_herkomsts = list(
                (
                    collections.Counter(all_herkomsts) & collections.Counter(herkomst_filtering)
                ).elements()
            )

        all_examentypes = data["Examentype"].unique()
        if examentype_filtering != []:
            all_examentypes = list(
                (
                    collections.Counter(all_examentypes)
                    & collections.Counter(examentype_filtering)
                ).elements()
            )

        for programme in np.sort(all_programmes):
            available_examentypes_for_programme = data[
                (data["Croho groepeernaam"] == programme)
                & (data["Collegejaar"] == self.predict_year)
            ]["Examentype"].unique()

            examentypes_to_consider = list(
                (
                    collections.Counter(available_examentypes_for_programme)
                    & collections.Counter(all_examentypes)
                ).elements()
            )
            for examentype in np.sort(examentypes_to_consider):
                for herkomst in np.sort(all_herkomsts):
                    predict_dict["Croho groepeernaam"].append(programme)
                    predict_dict["Herkomst"].append(herkomst)

                    predict_dict["Collegejaar"].append(self.predict_year)
                    predict_dict["Weeknummer"].append(self.predict_week)

                    predict_dict["Examentype"].append(examentype)

                    sample_row = data[data["Croho groepeernaam"] == programme].head(1)
                    predict_dict["Faculteit"].append(sample_row["Faculteit"].values[0])

        return pd.DataFrame(predict_dict)
