import argparse
import datetime
import os
import sys

from scripts.load_data import load_data, load_configuration
from scripts.dataholder.individual import Individual
from scripts.dataholder.cumulative import Cumulative
from scripts.dataholder.bothdatasets import BothDatasets
from scripts.helper import DataOption, HelperMethodsMaterial, StudentYearPrediction

DEFAULT_CONFIGURATION_PATH = "configuration/configuration.json"
DEFAULT_FILTERING_PATH = "configuration/filtering/base.json"
MAX_WEEK_NUMBER = 52
ACADEMIC_YEAR_START_WEEK = 40

OUTPUT_DIR = "data/output"
OUTPUT_PRELIM = os.path.join(OUTPUT_DIR, "output_prelim.xlsx")
OUTPUT_FILES = {
    StudentYearPrediction.FIRST_YEARS: os.path.join(OUTPUT_DIR, "output_first-years.xlsx"),
    StudentYearPrediction.HIGHER_YEARS: os.path.join(OUTPUT_DIR, "output_higher-years.xlsx"),
    StudentYearPrediction.VOLUME: os.path.join(OUTPUT_DIR, "output_volume.xlsx"),
}


def parse_range(values: list[str]) -> list[int]:
    """Parse a list of string values that may contain ':' for ranges.

    For example: ['3', ':', '7'] -> [3, 4, 5, 6, 7]
    """
    result: list[int] = []
    i = 0
    while i < len(values):
        if i + 2 < len(values) and values[i + 1] == ":":
            start = int(values[i])
            end = int(values[i + 2])
            result.extend(range(start, end + 1))
            i += 3
        else:
            result.append(int(values[i]))
            i += 1
    return result


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Studentprognose: predict student enrollment numbers."
    )
    parser.add_argument(
        "-w", "-W", "-week",
        nargs="+",
        dest="weeks",
        default=[],
        help="Target week number(s). Supports ranges with ':' (e.g., 3 : 7).",
    )
    parser.add_argument(
        "-y", "-Y", "-year",
        nargs="+",
        dest="years",
        default=[],
        help="Target year(s). Supports ranges with ':' (e.g., 2020 : 2023).",
    )
    parser.add_argument(
        "-d", "-D", "-dataset",
        dest="dataset",
        choices=["i", "individual", "c", "cumulative", "b", "both"],
        default="b",
        help="Dataset to use: (i)ndividual, (c)umulative, or (b)oth (default: both).",
    )
    parser.add_argument(
        "-c", "-C", "-configuration",
        dest="configuration",
        default=DEFAULT_CONFIGURATION_PATH,
        help="Path to configuration JSON file.",
    )
    parser.add_argument(
        "-f", "-F", "-filtering",
        dest="filtering",
        default=DEFAULT_FILTERING_PATH,
        help="Path to filtering JSON file.",
    )
    parser.add_argument(
        "-sy", "-SY", "-studentyear",
        dest="studentyear",
        choices=["f", "first-years", "h", "higher-years", "v", "volume"],
        default="f",
        help="Prediction type: (f)irst-years, (h)igher-years, or (v)olume (default: first-years).",
    )
    parser.add_argument(
        "-sk", "-SK", "-skipyears",
        dest="skipyears",
        type=int,
        default=0,
        help="Number of years to skip.",
    )
    parser.add_argument(
        "-t", "--test",
        dest="test_mode",
        action="store_true",
        default=False,
        help="Run in test mode (skip writing final output files).",
    )

    args = parser.parse_args(arguments[1:])

    # Parse weeks (with range support)
    weeks_specified = len(args.weeks) > 0
    if weeks_specified:
        weeks = parse_range(args.weeks)
    else:
        current_week = datetime.date.today().isocalendar()[1]
        if current_week > MAX_WEEK_NUMBER:
            print("Current week is week 53, check what weeknumber should be used")
            print(f"Now predicting for week {MAX_WEEK_NUMBER}")
            current_week = MAX_WEEK_NUMBER
        weeks = [current_week]

    # Parse years (with range support)
    if len(args.years) > 0:
        years = parse_range(args.years)
    else:
        current_year = datetime.date.today().year
        if not weeks_specified and weeks[0] >= ACADEMIC_YEAR_START_WEEK:
            current_year += 1
        years = [current_year]

    # Map dataset flag to DataOption
    dataset_map = {
        "i": DataOption.INDIVIDUAL,
        "individual": DataOption.INDIVIDUAL,
        "c": DataOption.CUMULATIVE,
        "cumulative": DataOption.CUMULATIVE,
        "b": DataOption.BOTH_DATASETS,
        "both": DataOption.BOTH_DATASETS,
    }
    data_option = dataset_map[args.dataset]

    # Map studentyear flag to StudentYearPrediction
    studentyear_map = {
        "f": StudentYearPrediction.FIRST_YEARS,
        "first-years": StudentYearPrediction.FIRST_YEARS,
        "h": StudentYearPrediction.HIGHER_YEARS,
        "higher-years": StudentYearPrediction.HIGHER_YEARS,
        "v": StudentYearPrediction.VOLUME,
        "volume": StudentYearPrediction.VOLUME,
    }
    student_year_prediction = studentyear_map[args.studentyear]

    # Validate file paths
    if not os.path.exists(args.configuration):
        parser.error(f"Configuration path does not exist: {args.configuration}")
    if not os.path.exists(args.filtering):
        parser.error(f"Filtering path does not exist: {args.filtering}")

    return argparse.Namespace(
        weeks=weeks,
        years=years,
        data_option=data_option,
        configuration_path=args.configuration,
        filtering_path=args.filtering,
        student_year_prediction=student_year_prediction,
        skip_years=args.skipyears,
        test_mode=args.test_mode,
    )


class Main:
    def __init__(self, arguments: list[str]) -> None:
        args = parse_arguments(arguments)
        self.weeks: list[int] = args.weeks
        self.years: list[int] = args.years
        self.data_option: DataOption = args.data_option
        self.configuration_path: str = args.configuration_path
        self.filtering_path: str = args.filtering_path
        self.student_year_prediction: StudentYearPrediction = args.student_year_prediction
        self.skip_years: int = args.skip_years
        self.test_mode: bool = args.test_mode

    def _check_output_files_writable(self) -> None:
        """Check that output files are not locked by another process."""
        output_files = [OUTPUT_PRELIM]

        if not self.test_mode:
            output_file = OUTPUT_FILES.get(self.student_year_prediction)
            if output_file:
                output_files.append(output_file)

        for path in output_files:
            try:
                with open(path, "a"):
                    pass
            except IOError:
                input(
                    f"Could not open {path} because it is (probably) opened by another process. "
                    "Please close Excel. Press Enter to continue."
                )

    def _load_config_and_data(self) -> None:
        """Load configuration files and all datasets."""
        print("Loading configuration...")
        self.configuration: dict = load_configuration(self.configuration_path)
        self.filtering: dict = load_configuration(self.filtering_path)

        print("Loading data...")
        (
            self.data_individual,
            self.data_cumulative,
            self.data_student_numbers_first_years,
            self.data_latest,
            self.data_distances,
            self.ensemble_weights,
        ) = load_data(self.configuration, self.data_option)

    def _build_helper_material(self) -> HelperMethodsMaterial:
        """Construct the HelperMethodsMaterial passed to dataholder constructors."""
        cwd = os.path.dirname(os.path.abspath(__file__))
        return HelperMethodsMaterial(
            data_latest=self.data_latest,
            ensemble_weights=self.ensemble_weights,
            data_student_numbers_first_years=self.data_student_numbers_first_years,
            cwd=cwd,
            data_option=self.data_option,
        )

    def _init_dataholder(self, helper_material: HelperMethodsMaterial) -> None:
        """Initialize the appropriate dataholder based on data_option."""
        self.dataholder = None

        if self.skip_years > 0 or self.data_option == DataOption.CUMULATIVE:
            if self.data_cumulative is None:
                raise Exception("Cumulative dataset not found")
            self.dataholder = Cumulative(
                self.data_cumulative,
                self.data_student_numbers_first_years,
                self.configuration,
                helper_material,
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
                    helper_material,
                    self.years,
                )
            except ValueError as e:
                print(e)
                self.dataholder = Cumulative(
                    self.data_cumulative,
                    self.data_student_numbers_first_years,
                    self.configuration,
                    helper_material,
                )
        elif self.data_option == DataOption.INDIVIDUAL:
            if self.data_individual is None:
                raise Exception("Individual dataset not found")
            self.dataholder = Individual(
                self.data_individual,
                self.data_distances,
                self.configuration,
                helper_material,
            )

    def _preprocess(self) -> None:
        """Run preprocessing and set up data for the prediction loop."""
        self.preprocessed_data = None

        if self.student_year_prediction in (
            StudentYearPrediction.FIRST_YEARS,
            StudentYearPrediction.VOLUME,
        ):
            print("Preprocessing...")
            self.preprocessed_data = self.dataholder.preprocess()

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

        self.dataholder.set_filtering(
            self.filtering["filtering"]["programme"],
            self.filtering["filtering"]["herkomst"],
            self.filtering["filtering"]["examentype"],
        )

    def _predict_loop(self) -> None:
        """Run predictions for each year/week combination."""
        for year in self.years:
            for week in self.weeks:
                if self.student_year_prediction in (
                    StudentYearPrediction.FIRST_YEARS,
                    StudentYearPrediction.VOLUME,
                ):
                    print(f"Predicting first-years: {year}-{week}...")
                    data_to_predict = self.dataholder.predict_nr_of_students(
                        year, week, self.skip_years
                    )
                    if data_to_predict is None:
                        continue
                    self.dataholder.helpermethods.prepare_data_for_output_prelim(
                        data_to_predict, year, week, self.preprocessed_data, self.skip_years
                    )

                    if self.data_option in (DataOption.CUMULATIVE, DataOption.BOTH_DATASETS):
                        self.dataholder.helpermethods.predict_with_ratio(
                            self.preprocessed_data, year
                        )

                    print("Postprocessing...")
                    self.dataholder.helpermethods.postprocess(year, week)

                if self.student_year_prediction in (
                    StudentYearPrediction.HIGHER_YEARS,
                    StudentYearPrediction.VOLUME,
                ):
                    raise NotImplementedError(
                        "Higher-years prediction is not yet implemented. "
                        "The HigherYears dataholder needs to be configured before using -sy h or -sy v."
                    )

                self.dataholder.helpermethods.ready_new_data()

    def _save_output(self) -> None:
        """Save the final output file unless in test mode."""
        if self.test_mode:
            return

        if self.dataholder.helpermethods.data is not None:
            print("Saving output...")
            self.dataholder.helpermethods.save_output(self.student_year_prediction)
        else:
            print("No data to save. Saving output skipped.")

    def run(self) -> None:
        self._check_output_files_writable()
        print("Predicting for years: ", self.years, " and weeks: ", self.weeks)
        self._load_config_and_data()
        self._init_dataholder(self._build_helper_material())
        self._preprocess()
        self._predict_loop()
        self._save_output()


if __name__ == "__main__":
    main = Main(sys.argv)
    main.run()
