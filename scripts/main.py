# main.py

# --- Standard library ---
import sys
import logging
from pathlib import Path
import time

# --- Third-party libraries ---
import yaml

# --- Warnings and logging setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Project modules ---
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from scripts.utils.load_data import load_data
from scripts.models.individual import Individual
from scripts.models.cumulative import Cumulative
from scripts.models.ensemble import Ensemble
from scripts.models.baseline import Baseline
from scripts.utils.postprocess import postprocess

from cli import parse_args


# --- Main pipeline ---
def pipeline(configuration, args):
    """
    Main pipeline for running the prediction loop for all models.
    """
    start_time = time.time()

    # --- load data ---
    data = load_data()

    cumulative_data = data["cumulative"]
    individual_data = data["individual"]
    distances = data["distances"]
    latest_data = data["latest"]
    student_counts = data["student_numbers_first_years"]
    logger.info("Data loaded.")

    # --- Initialize models ---
    cumulative_model = Cumulative(cumulative_data, student_counts, latest_data, configuration)
    individual_model = Individual(individual_data, distances, latest_data, configuration)
    baseline_model = Baseline(cumulative_data, student_counts, latest_data, configuration)
    ensemble_model = Ensemble(latest_data, configuration)
    logger.info("Models initialized.")

    first_pred = True

    # --- Run prediction loop for all models---
    for year in args.years:
        for week in args.weeks:
            logger.info(f"Running prediction loop for year: {year}, week: {week}")

            if not first_pred:
                cumulative_model.data_latest = latest_data

            # --- Run cumulative prediction loop ---
            cumulative_model.run_full_prediction_loop(
                predict_year=year,
                predict_week=week,
                write_file=args.write_file,
                print_output=args.print,
                args=args
            )
            individual_model.data_latest = cumulative_model.data_latest.copy()

            # --- Run individual prediction loop ---
            individual_model.run_full_prediction_loop(
                predict_year=year,
                predict_week=week,
                write_file=args.write_file,
                print_output=args.print,
                args=args
            )

            ensemble_model.data_latest = individual_model.data_latest.copy()

            # --- Run ensemble prediction loop ---
            ensemble_model.run_full_prediction_loop(
                predict_year=year,
                predict_week=week,
                write_file=args.write_file,
                print_output=args.print,
                args=args
            )
            
            baseline_model.data_latest = ensemble_model.data_latest.copy()

            # --- Run higher-years prediction loop ---
            baseline_model.run_full_prediction_loop(
                predict_year=year,
                predict_week=week,
                write_file=args.write_file,
                print_output=args.print,
                args=args
            )
            
            latest_data = baseline_model.data_latest.copy()

            # --- Postprocess ---
            latest_data = postprocess(cumulative_model.data_cumulative, latest_data, year, week)

            first_pred = False

            
    logger.info("Prediction loop completed.")

    # --- Write the file ---
    output_path = configuration["paths"]["output"]["path_output"].replace("${time}", time.strftime("%Y%m%d_%H%M%S"))
    latest_data.to_excel(output_path, index=False, engine="xlsxwriter")

    logger.info(f"Output written to: {output_path}")
    
    end_time = time.time()
    logger.info(f"Total time: {(end_time - start_time) / 60:.2f} minutes")

def main():
    # --- Parse arguments ---
    args = parse_args()

    # --- Load configuration ---
    CONFIG_FILE = Path("configuration.yaml")
    with open(CONFIG_FILE, "r") as f:
        configuration = yaml.safe_load(f)  

    # --- Run pipeline ---
    pipeline(configuration, args)


if __name__ == "__main__":
    main()