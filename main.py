import datetime
import sys
from enum import Enum

from scripts.load_data import *
from scripts.dataholder.superclass import *
from scripts.dataholder.individual import *
from scripts.dataholder.cumulative import *
from scripts.dataholder.bothdatasets import *
from scripts.higher_years import *
from scripts.helper import DataOption, StudentYearPrediction


class Main:
    def __init__(self, arguments):
        self._parse_arguments(arguments)

    class Cmd(Enum):
        NO_ARG = 0
        WEEKS = 1
        YEARS = 2
        DATASETS = 3
        CONFIGURATION = 4
        FILTERING = 5
        STUDENT_YEAR_PREDICTION = 6
        SKIP_YEARS = 7

    def _parse_arguments(self, arguments):
        cmd_arg = Main.Cmd.NO_ARG

        self.weeks = []
        self.week_slice = False
        self.years = []
        self.year_slice = False
        self.data_option = DataOption.BOTH_DATASETS
        self.configuration_path = "configuration/configuration.json"
        self.filtering_path = "configuration/filtering/base.json"
        self.student_year_prediction = StudentYearPrediction.FIRST_YEARS
        self.skip_years = 0

        try:
            # First arguments is always the name of the python script
            for i in range(1, len(arguments)):
                arg = arguments[i]

                if cmd_arg == Main.Cmd.WEEKS and arg == ":":
                    self.week_slice = True
                elif cmd_arg == Main.Cmd.WEEKS and arg.isnumeric():
                    if self.week_slice:
                        last_week = self.weeks.pop(-1)
                        self.weeks = self.weeks + list(range(last_week, int(arg) + 1))
                        self.week_slice = False
                    else:
                        self.weeks.append(int(arg))
                elif cmd_arg == Main.Cmd.YEARS and arg == ":":
                    self.year_slice = True
                elif cmd_arg == Main.Cmd.YEARS and arg.isnumeric():
                    if self.year_slice:
                        last_year = self.years.pop(-1)
                        self.years = self.years + list(range(last_year, int(arg) + 1))
                        self.year_slice = False
                    else:
                        self.years.append(int(arg))
                elif cmd_arg == Main.Cmd.DATASETS:
                    if arg == "i" or arg == "individual":
                        self.data_option = DataOption.INDIVIDUAL
                        cmd_arg = Main.Cmd.NO_ARG
                    elif arg == "c" or arg == "cumulative":
                        self.data_option = DataOption.CUMULATIVE
                        cmd_arg = Main.Cmd.NO_ARG
                    elif arg == "b" or arg == "both":
                        self.data_option = DataOption.BOTH_DATASETS
                        cmd_arg = Main.Cmd.NO_ARG
                elif cmd_arg == Main.Cmd.CONFIGURATION:
                    if os.path.exists(arg):
                        self.configuration_path = arg
                    else:
                        raise Exception("Configuration path does not exist")
                    cmd_arg = Main.Cmd.NO_ARG
                elif cmd_arg == Main.Cmd.FILTERING:
                    if os.path.exists(arg):
                        self.filtering_path = arg
                    else:
                        raise Exception("Configuration path does not exist")
                elif cmd_arg == Main.Cmd.STUDENT_YEAR_PREDICTION:
                    if arg == "f" or arg == "first-years":
                        self.student_year_prediction = StudentYearPrediction.FIRST_YEARS
                        cmd_arg = Main.Cmd.NO_ARG
                    elif arg == "h" or arg == "higher-years":
                        self.student_year_prediction = StudentYearPrediction.HIGHER_YEARS
                        cmd_arg = Main.Cmd.NO_ARG
                    elif arg == "v" or arg == "volume":
                        self.student_year_prediction = StudentYearPrediction.VOLUME
                        cmd_arg = Main.Cmd.NO_ARG
                elif cmd_arg == Main.Cmd.SKIP_YEARS and arg.isnumeric():
                    self.skip_years = int(arg)
                    cmd_arg = Main.Cmd.NO_ARG

                if arg == "-w" or arg == "-W" or arg == "-week":
                    cmd_arg = Main.Cmd.WEEKS
                elif arg == "-y" or arg == "-Y" or arg == "-year":
                    cmd_arg = Main.Cmd.YEARS
                elif arg == "-d" or arg == "-D" or arg == "-dataset":
                    cmd_arg = Main.Cmd.DATASETS
                elif arg == "-c" or arg == "-C" or arg == "-configuration":
                    cmd_arg = Main.Cmd.CONFIGURATION
                elif arg == "-f" or arg == "-F" or arg == "-filtering":
                    cmd_arg = Main.Cmd.FILTERING
                elif arg == "-sy" or arg == "-SY" or arg == "-studentyear":
                    cmd_arg = Main.Cmd.STUDENT_YEAR_PREDICTION
                elif arg == "-sk" or arg == "-SK" or arg == "-skipyears":
                    cmd_arg = Main.Cmd.SKIP_YEARS
        except:
            print(
                "Something went wrong while parsing the arguments, read the README.md for usage."
            )

        weeks_specified = True
        if self.weeks == []:
            weeks_specified = False
            current_week = datetime.date.today().isocalendar()[1]
            # Max of 52 weeks, week 53 is an edge case where the user should manually input data
            if current_week > 52:
                print("Current week is week 53, check what weeknumber should be used")
                print("Now predicting for week 52")
                current_week = 52
            self.weeks = [current_week]

        if self.years == []:
            current_year = datetime.date.today().year

            if not weeks_specified and self.weeks[0] >= 40:
                current_year += 1

            self.years = [current_year]

    def run(self):

        # Make sure that the output files are closed. Otherwise the program will crash when
        # it tries to write to these files.
        try:
            mode_suffix = self.data_option.filename_suffix
            open(f"data/output/output_prelim_{mode_suffix}.xlsx", "w").close()

            if "test" not in self.filtering_path:
                match self.student_year_prediction:
                    case StudentYearPrediction.FIRST_YEARS:
                        open(f"data/output/output_first-years_{mode_suffix}.xlsx", "w").close()
                    case StudentYearPrediction.HIGHER_YEARS:
                        open(f"data/output/output_higher-years_{mode_suffix}.xlsx", "w").close()
                    case StudentYearPrediction.VOLUME:
                        open(f"data/output/output_volume_{mode_suffix}.xlsx", "w").close()

        except IOError:
            input(
                "Could not open output files because they are (probably) opened by another process. Please close Excel. Press Enter to continue."
            )

        print("Predicting for years: ", self.years, " and weeks: ", self.weeks)

        # Load configuration
        print("Loading configuration...")
        self.configuration = load_configuration(self.configuration_path)
        self.filtering = load_configuration(self.filtering_path)

        # Load data
        print("Loading data...")
        (
            self.data_individual,
            self.data_cumulative,
            self.data_student_numbers_first_years,
            self.data_latest,
            self.data_distances,
            self.ensemble_weights,
        ) = load_data(self.configuration, self.data_option)

        CWD = os.path.dirname(os.path.abspath(__file__))
        helpermethods_initialise_material = [
            self.data_latest,
            self.ensemble_weights,
            self.data_student_numbers_first_years,
            CWD,
            self.data_option,
        ]

        # Initialize dataholder
        self.dataholder = None
        if self.skip_years > 0 or self.data_option == DataOption.CUMULATIVE:
            if self.data_cumulative is None:
                raise Exception("Cumulative dataset not found")
            self.dataholder = Cumulative(
                self.data_cumulative,
                self.data_student_numbers_first_years,
                self.configuration,
                helpermethods_initialise_material,
            )
        elif self.data_option == DataOption.BOTH_DATASETS:
            if self.data_individual is None:
                raise Exception("Individual dataset not found")
            if self.data_cumulative is None:
                raise Exception("Cumulative dataset not found")
            try:
                self.dataholder = BothDatasets(
                    self.data_individual,
                    self.data_cumulative,
                    self.data_distances,
                    self.data_student_numbers_first_years,
                    self.configuration,
                    helpermethods_initialise_material,
                    self.years,
                )
            except ValueError as e:
                print(e)  # Log the error message
                # Initialize only the cumulative dataset:
                self.dataholder = Cumulative(
                    self.data_cumulative,
                    self.data_student_numbers_first_years,
                    self.configuration,
                    helpermethods_initialise_material,
                )
        elif self.data_option == DataOption.INDIVIDUAL:
            if self.data_individual is None:
                raise Exception("Individual dataset not found")
            self.dataholder = Individual(
                self.data_individual,
                self.data_distances,
                self.configuration,
                helpermethods_initialise_material,
            )

        # self.higher_years_dataholder = HigherYears(
        #     self.data_student_numbers_first_years,
        #     self.data_student_numbers_higher_years,
        #     self.data_student_numbers_volume,
        #     self.data_individual,
        #     self.configuration,
        #     self.data_october,
        #     self.data_ratios,
        #     CWD,
        # )

        if (
            self.student_year_prediction == StudentYearPrediction.FIRST_YEARS
            or self.student_year_prediction == StudentYearPrediction.VOLUME
        ):
            # Preprocess data
            print("Preprocessing...")
            self.data_cumulative = self.dataholder.preprocess()

        if self.student_year_prediction == StudentYearPrediction.HIGHER_YEARS:
            self.dataholder.helpermethods.data = self.dataholder.helpermethods.data_latest[
                [
                    "Croho groepeernaam",
                    "Collegejaar",
                    "Herkomst",
                    "Weeknummer",
                    "SARIMA_cumulative",
                    "SARIMA_individual",
                    "Voorspelde vooraanmelders",
                    "Aantal_studenten",
                    "Faculteit",
                    "Examentype",
                    "Gewogen vooraanmelders",
                    "Ongewogen vooraanmelders",
                    "Aantal aanmelders met 1 aanmelding",
                    "Inschrijvingen",
                    "Weighted_ensemble_prediction",
                ]
            ]

        # Set programme and origin filtering
        self.dataholder.set_filtering(
            self.filtering["filtering"]["programme"],
            self.filtering["filtering"]["herkomst"],
            self.filtering["filtering"]["examentype"],
        )

        for year in self.years:
            for week in self.weeks:
                # Predict first years student based on settings
                if (
                    self.student_year_prediction == StudentYearPrediction.FIRST_YEARS
                    or self.student_year_prediction == StudentYearPrediction.VOLUME
                ):
                    print(f"Predicting first-years: {year}-{week}...")
                    data_to_predict = self.dataholder.predict_nr_of_students(
                        year, week, self.skip_years
                    )
                    if data_to_predict is None:
                        continue
                    self.dataholder.helpermethods.prepare_data_for_output_prelim(
                        data_to_predict, year, week, self.data_cumulative, self.skip_years
                    )

                    if (
                        self.data_option == DataOption.CUMULATIVE
                        or self.data_option == DataOption.BOTH_DATASETS
                    ):
                        # Run ratio model
                        self.dataholder.helpermethods.predict_with_ratio(
                            self.data_cumulative, year
                        )

                    # Post process
                    print("Postprocessing...")
                    self.dataholder.helpermethods.postprocess(year, week)

                # Predicting higher years students and/or volume based on settings
                if (
                    self.student_year_prediction == StudentYearPrediction.HIGHER_YEARS
                    or self.student_year_prediction == StudentYearPrediction.VOLUME
                ):
                    print(f"Predicting higher-years: {year}-{week}...")
                    self.dataholder.helpermethods.data = (
                        self.higher_years_dataholder.predict_nr_of_students(
                            self.dataholder.helpermethods.data,
                            self.dataholder.helpermethods.data_latest,
                            year,
                            week,
                            self.skip_years,
                        )
                    )

                self.dataholder.helpermethods.ready_new_data()

        # Save final output
        if (
            "test" not in self.filtering_path
        ):  # We do this to ensure faster testing times for fast setups
            if self.dataholder.helpermethods.data is not None:
                print("Saving output...")
                self.dataholder.helpermethods.save_output(self.student_year_prediction)
            else:
                print("No data to save. Saving output skipped.")


if __name__ == "__main__":
    main = Main(sys.argv)
    main.run()
