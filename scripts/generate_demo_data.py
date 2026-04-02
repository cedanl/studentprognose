"""Generate realistic demo data for the student prognosis pipeline.

Creates:
  - data/input_raw/telbestanden/telbestandY{year}W{week}.csv  (years 2020-2024, weeks 1-30)
  - data/input_raw/oktober_bestand.xlsx                        (years 2019-2023)
  - data/input_raw/individuele_aanmelddata.csv                 (year 2024)

Usage:
    uv run scripts/generate_demo_data.py
"""

import os
import random

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "input_raw")
TEL_DIR = os.path.join(RAW_DIR, "telbestanden")

# Programmes: (Isatcode, Type_HO, Groepeernaam, Faculteit, base_count_NL, base_count_EER, base_count_REER)
PROGRAMMES = [
    ("56604", "B", "B Psychologie",      "SOW", 10, 3, 2),
    ("56981", "B", "B Politicologie",    "MAN",  6, 1, 0),
    ("60139", "M", "M Rechtsgeleerdheid", "JUR", 4, 1, 0),
]

BRINCODE = "21PB"
YEARS = range(2020, 2025)  # 5 years of telbestanden
WEEKS_SPRING = list(range(1, 39))   # Weeks 1-38 (spring semester)
WEEKS_AUTUMN = list(range(39, 53))  # Weeks 39-52 (autumn semester)
ALL_WEEKS = WEEKS_AUTUMN + WEEKS_SPRING  # Full academic year: 39..52, 1..38
MEERCODE = {"B": 1.29, "M": 1.10}


def _cumulative_growth(base, week_index, total_weeks=52):
    """Simulate cumulative pre-registration growth over an academic year.

    week_index: 0-based position in the academic year (0 = week 39, 51 = week 38).
    """
    t = week_index / total_weeks
    growth = 1 / (1 + np.exp(-10 * (t - 0.4)))
    value = base * (0.1 + 0.9 * growth)
    value *= (1 + np.random.normal(0, 0.05))
    return max(1, round(value, 2))


def generate_telbestanden():
    """Generate weekly telbestanden CSVs for multiple years."""
    os.makedirs(TEL_DIR, exist_ok=True)

    # Remove old telbestanden
    for f in os.listdir(TEL_DIR):
        if f.startswith("telbestand") and f.endswith(".csv"):
            os.remove(os.path.join(TEL_DIR, f))

    for year in YEARS:
        # Slight year-over-year growth
        year_factor = 1 + 0.03 * (year - 2020)

        for week_index, week in enumerate(ALL_WEEKS):
            rows = []
            for isatcode, type_ho, groepeernaam, faculteit, base_nl, base_eer, base_reer in PROGRAMMES:
                for herkomst, base in [("N", base_nl), ("E", base_eer), ("R", base_reer)]:
                    if base == 0:
                        continue
                    aantal = _cumulative_growth(base * year_factor, week_index, len(ALL_WEEKS))
                    meercode = MEERCODE[type_ho]
                    rows.append(f"{BRINCODE};{year};{type_ho};{isatcode};{groepeernaam};{faculteit};{herkomst};N;N;{aantal};{meercode}")

            path = os.path.join(TEL_DIR, f"telbestandY{year}W{week:02d}.csv")
            header = "Brincode;Studiejaar;Type_HO;Isatcode;Groepeernaam;Faculteit;Herkomst;Hogerejaars;Herinschrijving;Aantal;meercode_V"
            with open(path, "w") as f:
                f.write(header + "\n")
                f.write("\n".join(rows) + "\n")

    print(f"Generated telbestanden for years {list(YEARS)}, weeks 39-52 + 1-38")


def generate_oktober_bestand():
    """Generate oktober_bestand.xlsx with actual student counts."""
    rows = []

    for year in range(2019, 2024):  # Ground truth for years before prediction
        year_factor = 1 + 0.03 * (year - 2019)
        for _, type_ho, naam, _, base_nl, base_eer, base_reer in PROGRAMMES:
            examentype = "Bachelor eerstejaars" if type_ho == "B" else "Master"
            for herkomst, base in [("NL", base_nl), ("EER", base_eer), ("Niet-EER", base_reer)]:
                if base == 0:
                    continue
                # Actual enrolled ~ 70-90% of final pre-registrations
                enrolled = max(1, int(base * year_factor * random.uniform(0.7, 0.9)))
                rows.append({
                    "Collegejaar": year,
                    "Groepeernaam Croho": naam,
                    "EER-NL-nietEER": herkomst,
                    "Examentype code": examentype,
                    "Aantal eerstejaars croho": 1,
                    "Aantal Hoofdinschrijvingen": enrolled,
                })

            # Add some higher-years data for volume
            if type_ho == "B":
                enrolled_hy = max(1, int(base_nl * year_factor * random.uniform(0.5, 0.7)))
                rows.append({
                    "Collegejaar": year,
                    "Groepeernaam Croho": naam,
                    "EER-NL-nietEER": "NL",
                    "Examentype code": "Bachelor hogerejaars",
                    "Aantal eerstejaars croho": 0,
                    "Aantal Hoofdinschrijvingen": enrolled_hy,
                })

    df = pd.DataFrame(rows)
    path = os.path.join(RAW_DIR, "oktober_bestand.xlsx")
    df.to_excel(path, index=False)
    print(f"Generated oktober_bestand.xlsx ({len(df)} rows, years 2019-2023)")


def generate_individuele_aanmelddata():
    """Generate individual application data for years 2020-2024."""
    rows = []
    sleutel = 1000

    faculteit_map = {
        "B Psychologie": "FSW",
        "B Politicologie": "FdM",
        "M Rechtsgeleerdheid": "FdR",
    }

    for year in YEARS:
        year_factor = 1 + 0.03 * (year - 2020)

        for _, type_ho, naam, _, base_nl, base_eer, _ in PROGRAMMES:
            examentype = "Propedeuse Bachelor" if type_ho == "B" else "Master"
            faculteit = faculteit_map.get(naam, "FNWI")

            for herkomst_label, base, eer in [("NL", base_nl, "N"), ("EER", base_eer, "J")]:
                # Generate enough records to cover all weeks in the academic year
                # Need at least 1 per week (52 weeks), scale up with base count
                count = max(52, int(base * year_factor * 10))
                for i in range(count):
                    # Spread across full academic year: ~30% in weeks 39-52, ~70% in weeks 1-30
                    if random.random() < 0.3:
                        week = random.randint(39, 52)
                        # Autumn weeks fall in the previous calendar year
                        cal_year = year - 1
                    else:
                        week = random.randint(1, 37)
                        cal_year = year
                    # Convert week number to a plausible date
                    import datetime as _dt
                    jan1 = _dt.date(cal_year, 1, 4)  # ISO week 1 always contains Jan 4
                    monday = jan1 - _dt.timedelta(days=jan1.weekday())
                    target_date = monday + _dt.timedelta(weeks=week - 1, days=random.randint(0, 6))
                    date_str = target_date.strftime("%d-%m-%Y")

                    # Most are enrolled in historical years, mix for predict year
                    if year < max(YEARS):
                        status = random.choice(["Ingeschreven", "Ingeschreven", "Ingeschreven", "Geannuleerd"])
                    else:
                        status = random.choice(["Ingeschreven", "Verzoek tot inschrijving", "Verzoek tot inschrijving"])

                    rows.append({
                        "Sleutel": sleutel,
                        "Datum Verzoek Inschr": date_str,
                        "Ingangsdatum": f"01-09-{year}",
                        "Collegejaar": year,
                        "Datum intrekking vooraanmelding": "",
                        "Inschrijfstatus": status,
                        "Faculteit": faculteit,
                        "Examentype": examentype,
                        "Croho": naam.split(" ", 1)[1] if " " in naam else naam,
                        "Croho groepeernaam": naam,
                        "Opleiding": naam,
                        "Hoofdopleiding": naam,
                        "Eerstejaars croho jaar": year,
                        "Is eerstejaars croho opleiding": 1,
                        "Is hogerejaars": 0,
                        "BBC ontvangen": 0,
                        "Type vooropleiding": "VWO" if type_ho == "B" else "WO Bachelor",
                        "Nationaliteit": "Nederlandse" if eer == "N" else "Duitse",
                        "EER": eer,
                        "Geslacht": random.choice(["M", "V"]),
                        "Geverifieerd adres postcode": f"{random.randint(1000, 9999)}AA",
                        "Geverifieerd adres plaats": "Nijmegen",
                        "Geverifieerd adres land": "NL" if eer == "N" else "DE",
                        "Studieadres postcode": f"{random.randint(1000, 9999)}BB",
                        "Studieadres land": "NL",
                        "School code eerste vooropleiding": f"{random.randint(10000, 99999)}",
                        "School eerste vooropleiding": "Demo School",
                        "Plaats code eerste vooropleiding": "0268",
                        "Land code eerste vooropleiding": "6030",
                        "Aantal studenten": 1,
                    })
                    sleutel += 1

    df = pd.DataFrame(rows)
    path = os.path.join(RAW_DIR, "individuele_aanmelddata.csv")
    df.to_csv(path, sep=";", index=False)
    print(f"Generated individuele_aanmelddata.csv ({len(df)} rows, years {list(YEARS)})")


if __name__ == "__main__":
    generate_telbestanden()
    generate_oktober_bestand()
    generate_individuele_aanmelddata()
    print("\nDone! Now run: uv run main.py -w 19 -y 2024 -D cumulative")
