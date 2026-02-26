import os as os
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.load_data import load_configuration


class AppendStudentCountAndComputeErrors:
    def __init__(self, data_latest, student_count_first_years, student_count_higher_years):
        self.data_latest = data_latest
        self.student_count_first_years = student_count_first_years
        self.student_count_higher_years = student_count_higher_years

    def append_studentcount(self):
        data_latest = self.data_latest
        student_count_first_years = self.student_count_first_years
        student_count_higher_years = self.student_count_higher_years

        data_latest.drop(
            columns=["Aantal_studenten", "Aantal_studenten_higher_years"], inplace=True
        )
        data_latest = pd.merge(
            data_latest,
            student_count_higher_years,
            how="left",
            on=["Collegejaar", "Croho groepeernaam", "Herkomst", "Examentype"],
        )
        data_latest.rename(
            columns={"Aantal_studenten": "Aantal_studenten_higher_years"}, inplace=True
        )
        data_latest = pd.merge(
            data_latest,
            student_count_first_years,
            how="left",
            on=["Collegejaar", "Croho groepeernaam", "Herkomst", "Examentype"],
        )

        self.data_latest = data_latest

    def compute_errors(self):
        data_latest = self.data_latest
        data_latest["Aantal_studenten"] = data_latest["Aantal_studenten"].fillna(0)

        all_prediction_columns = [
            "Weighted_ensemble_prediction",
            "Average_ensemble_prediction",
            "Ensemble_prediction",
            "Prognose_ratio",
            "SARIMA_cumulative",
            "SARIMA_individual",
        ]
        prediction_columns = [col for col in all_prediction_columns if col in data_latest.columns]

        for key in prediction_columns:
            data_latest[key] = data_latest[key].fillna(0)

            data_latest[f"MAE_{key}"] = data_latest.apply(
                lambda row, k=key: (abs(row["Aantal_studenten"] - row[k])),
                axis=1,
            )
            data_latest[f"MAPE_{key}"] = data_latest.apply(
                lambda row, k=key: (
                    (abs(row["Aantal_studenten"] - row[k]) / row["Aantal_studenten"])
                    if row["Aantal_studenten"] != 0
                    else np.nan
                ),
                axis=1,
            )

        self.data_latest = data_latest


if __name__ == "__main__":
    configuration = load_configuration("configuration/configuration.json")
    student_count_first_years = pd.read_excel(
        configuration["paths"]["path_student_count_first-years"]
    )
    student_count_higher_years = pd.read_excel(
        configuration["paths"]["path_student_count_higher-years"]
    )

    CWD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    variants = [
        ("individueel", configuration["paths"]["path_latest_individual"]),
        ("cumulatief", configuration["paths"]["path_latest_cumulative"]),
    ]

    for variant_name, path in variants:
        if not os.path.exists(path):
            print(f"Skipping {variant_name}: file {path} not found")
            continue

        data_latest = pd.read_excel(path)
        data_latest_cols = data_latest.columns.tolist()

        processor = AppendStudentCountAndComputeErrors(
            data_latest, student_count_first_years, student_count_higher_years
        )
        processor.append_studentcount()
        processor.compute_errors()

        new_data_latest = processor.data_latest
        new_data_latest = new_data_latest[data_latest_cols]

        outfile = os.path.join(CWD, f"data/output/totaal_{variant_name}.xlsx")
        new_data_latest.to_excel(outfile)
        print(f"Saved {outfile}")
