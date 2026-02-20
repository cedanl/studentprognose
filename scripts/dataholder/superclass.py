from scripts.helper import *
from scripts.dataholder.helpermethods import *
from abc import ABC, abstractmethod
import collections


class Superclass(ABC):
    """This class is the superclass of the BothDatasets, Cumulative and Individual class. These
    can be found respectively in botdatasets.py, cumulative.py and individual.py. This class is
    a subclass of the abc (Abstract Base Classes) module. This module provides the
    infrastructure for definiing abstract base classes in Python.
    """

    def __init__(self, configuration, helpermethods_initialise_material):
        # Load in capacity numbers of the numerus fixus studies.
        self.numerus_fixus_list = configuration["numerus_fixus"]

        # Load in all the helpermethods used on dataholders as a variable. These methods are
        # mostly used in main.py for several tasks. Also used to add predicted preregistrations
        # when calculating the cumulative values.
        self.helpermethods = HelperMethods(configuration, helpermethods_initialise_material)

        # programme_filtering and herkomst_filtering will be initialised as empty lists but will
        # be declared later in set_filtering().
        self.programme_filtering = []
        self.herkomst_filtering = []
        self.examentype_filtering = []

    # Abstract method that must be defined in the subclasses (Indivual, Cumulative, BothDatasets)
    # itself.
    @abstractmethod
    def preprocess(self, programme_filtering, herkomst_filtering):
        pass

    # Abstract method that must be defined in the subclasses (Indivual, Cumulative, BothDatasets)
    # itself.
    @abstractmethod
    def predict_nr_of_students(self, predict_year, predict_week, skip_years):
        pass

    # Sets the year and week to predict. These values are specified in the command line when
    # executing the code. Also looks at the max year and week by looking in the data.
    def set_year_week(self, predict_year, predict_week, data):
        self.predict_year = predict_year
        self.predict_week = predict_week

        self.max_year = data["Collegejaar"].max()
        self.max_week = get_max_week(self.predict_year, self.max_year, data, "Collegejaar")

    # Sets the programme and herkomst filtering. These values are specified in the configuration
    # file.
    def set_filtering(self, programme_filtering, herkomst_filtering, examentype_filtering):
        self.programme_filtering = programme_filtering
        self.herkomst_filtering = herkomst_filtering
        self.examentype_filtering = examentype_filtering

    # Method that returns a table with information about the data that will be predicted. This will
    # done repeatedly for every specified week and year to be predicted. This is an example of
    # what this method would return when we filter on programme 'B Bedrijfskunde' and
    # 'B Communicatiewetenschap' and herkomst 'NL' and 'EER':

    #          Croho groepeernaam Herkomst  Collegejaar  Weeknummer Examentype Faculteit
    # 0           B Bedrijfskunde      EER         2024          11   Bachelor       FdM
    # 1           B Bedrijfskunde       NL         2024          11   Bachelor       FdM
    # 2  B Communicatiewetenschap      EER         2024          11   Bachelor       FSW
    # 3  B Communicatiewetenschap       NL         2024          11   Bachelor       FSW

    # This method will only be called when predicting individual values only because this will
    # done slightly different when predicting cumulative or both datasets.
    def get_data_to_predict(
        self, data, programme_filtering=[], herkomst_filtering=[], examentype_filtering=[]
    ):
        # These are the columns that will be defined in the table of data to predict.
        predict_dict = {
            "Croho groepeernaam": [],
            "Herkomst": [],
            "Collegejaar": [],
            "Weeknummer": [],
            "Examentype": [],
            "Faculteit": [],
        }

        # Take the intersection of the programmes that are in the data and the programmes that
        # we filter on.
        all_programmes = data["Croho groepeernaam"].unique()
        if programme_filtering != []:
            all_programmes = list(
                (
                    collections.Counter(all_programmes) & collections.Counter(programme_filtering)
                ).elements()
            )

        # Take the intersection of the herkomst that are in the data and the herkomst that we
        # filter on.
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

        # Add one line in the table per unique programme, examentype and herkomst that has to
        # be predicted.
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
