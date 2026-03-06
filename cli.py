import argparse
import datetime
import os

from scripts.helper import DataOption, StudentYearPrediction

DEFAULT_CONFIGURATION_PATH = "configuration/configuration.json"
DEFAULT_FILTERING_PATH = "configuration/filtering/base.json"
MAX_WEEK_NUMBER = 52
ACADEMIC_YEAR_START_WEEK = 40


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
