import argparse
import datetime
import os
from dataclasses import dataclass, field

from src.utils.weeks import DataOption, StudentYearPrediction


@dataclass
class PipelineConfig:
    weeks: list = field(default_factory=list)
    years: list = field(default_factory=list)
    weeks_specified: bool = False
    data_option: DataOption = DataOption.BOTH_DATASETS
    configuration_path: str = "configuration/configuration.json"
    filtering_path: str = "configuration/filtering/base.json"
    student_year_prediction: StudentYearPrediction = StudentYearPrediction.FIRST_YEARS
    skip_years: int = 0
    ci_test_n: int | None = None
    noetl: bool = False


def _expand_slices(tokens):
    """Expand a list of tokens like ['12', ':', '20'] into [12, 13, ..., 20]."""
    result = []
    i = 0
    while i < len(tokens):
        if i + 2 < len(tokens) and tokens[i + 1] == ":":
            start = int(tokens[i])
            end = int(tokens[i + 2])
            result.extend(range(start, end + 1))
            i += 3
        else:
            result.append(int(tokens[i]))
            i += 1
    return result


DATASET_MAP = {
    "i": DataOption.INDIVIDUAL,
    "individual": DataOption.INDIVIDUAL,
    "c": DataOption.CUMULATIVE,
    "cumulative": DataOption.CUMULATIVE,
    "b": DataOption.BOTH_DATASETS,
    "both": DataOption.BOTH_DATASETS,
}

STUDENT_YEAR_MAP = {
    "f": StudentYearPrediction.FIRST_YEARS,
    "first-years": StudentYearPrediction.FIRST_YEARS,
    "h": StudentYearPrediction.HIGHER_YEARS,
    "higher-years": StudentYearPrediction.HIGHER_YEARS,
    "v": StudentYearPrediction.VOLUME,
    "volume": StudentYearPrediction.VOLUME,
}


def parse_args(argv):
    """Parse CLI arguments and return a PipelineConfig."""
    parser = argparse.ArgumentParser(description="Student prognose pipeline")

    parser.add_argument("-w", "-W", "-week", nargs="*", default=None, dest="weeks")
    parser.add_argument("-y", "-Y", "-year", nargs="*", default=None, dest="years")
    parser.add_argument("-d", "-D", "-dataset", choices=list(DATASET_MAP.keys()), default=None, dest="dataset")
    parser.add_argument("-c", "-C", "-configuration", default="configuration/configuration.json", dest="configuration")
    parser.add_argument("-f", "-F", "-filtering", nargs="*", default=None, dest="filtering")
    parser.add_argument("-sy", "-SY", "-studentyear", choices=list(STUDENT_YEAR_MAP.keys()), default=None, dest="studentyear")
    parser.add_argument("-sk", "-SK", "-skipyears", type=int, default=0, dest="skipyears")
    parser.add_argument("--ci", nargs=2, metavar=("test", "N"), default=None)
    parser.add_argument("--noetl", action="store_true")

    args = parser.parse_args(argv[1:])

    cfg = PipelineConfig()
    cfg.noetl = args.noetl

    # Configuration path
    if args.configuration and os.path.exists(args.configuration):
        cfg.configuration_path = args.configuration

    # Filtering path (supports multiple files via -f path1 -f path2 or -f path1 path2)
    if args.filtering is not None:
        for path in args.filtering:
            if os.path.exists(path):
                cfg.filtering_path = path

    # Dataset
    if args.dataset is not None:
        cfg.data_option = DATASET_MAP[args.dataset]

    # Student year prediction
    if args.studentyear is not None:
        cfg.student_year_prediction = STUDENT_YEAR_MAP[args.studentyear]

    # Skip years
    cfg.skip_years = args.skipyears

    # CI test
    if args.ci is not None:
        if args.ci[0] != "test":
            raise SystemExit(f"Invalid --ci argument: '{args.ci[0]}'. Usage: --ci test <N>")
        try:
            cfg.ci_test_n = int(args.ci[1])
        except ValueError:
            raise SystemExit(f"Invalid --ci argument: '{args.ci[1]}'. Usage: --ci test <N>")

    # Weeks
    if args.weeks is not None and len(args.weeks) > 0:
        cfg.weeks = _expand_slices(args.weeks)
        cfg.weeks_specified = True
    else:
        current_week = datetime.date.today().isocalendar()[1]
        if current_week > 52:
            print("Current week is week 53, check what weeknumber should be used")
            print("Now predicting for week 52")
            current_week = 52
        cfg.weeks = [current_week]
        cfg.weeks_specified = False

    # Years
    if args.years is not None and len(args.years) > 0:
        cfg.years = _expand_slices(args.years)
    else:
        current_year = datetime.date.today().year
        if not cfg.weeks_specified and cfg.weeks[0] >= 40:
            current_year += 1
        cfg.years = [current_year]

    return cfg
