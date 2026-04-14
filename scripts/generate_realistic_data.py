#!/usr/bin/env python3
"""Generate realistic synthetic demo data for the studentprognose pipeline.

Produces data in data/input_raw/ that matches the exact schema expected by
src/data/etl.py.  Run via:  uv run python scripts/generate_realistic_data.py
"""

import csv
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
RNG = np.random.default_rng(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "input_raw"
TEL_DIR = RAW_DIR / "telbestanden"

BRINCODE = "21PB"  # Radboud

YEARS_TEL = list(range(2016, 2025))  # 2016-2024: 9 academic years
YEARS_INDIVIDUAL = list(range(2016, 2025))  # match XGBoost filter >= 2016
YEARS_OKTOBER = list(range(2015, 2024))  # 2015-2023: ground truth

# ---------------------------------------------------------------------------
# Programme definitions
# ---------------------------------------------------------------------------
# (groepeernaam, croho, faculteit, type_ho, base_volume_NL, pct_eer, pct_niet_eer)
# base_volume_NL = typical NL first-year applications per year
# pct_eer / pct_niet_eer = fraction of NL volume that EER/Niet-EER adds
PROGRAMMES = [
    # --- FSW ---
    ("B Psychologie",                 56604, "FSW", "B", 750, 0.30, 0.06),
    ("B Pedagogische Wetenschappen",  56607, "FSW", "B", 280, 0.08, 0.02),
    ("B Communicatiewetenschap",      56615, "FSW", "B", 220, 0.12, 0.03),
    ("B Artificial Intelligence",     56981, "FSW", "B", 550, 0.35, 0.10),
    ("B Sociologie",                  56601, "FSW", "B", 100, 0.10, 0.04),
    # --- FdR ---
    ("B Rechtsgeleerdheid",           56827, "FdR", "B", 650, 0.15, 0.03),
    ("B Notarieel Recht",             56828, "FdR", "B",  75, 0.05, 0.01),
    ("B Fiscaal Recht",               56829, "FdR", "B",  90, 0.05, 0.01),
    # --- FdM ---
    ("B Bedrijfskunde",               50645, "FdM", "B", 480, 0.25, 0.08),
    ("B Economie",                    56401, "FdM", "B", 270, 0.20, 0.06),
    ("B Bestuurskunde",               56627, "FdM", "B", 180, 0.10, 0.03),
    ("B Politicologie",               56606, "FdM", "B", 140, 0.12, 0.04),
    ("B Geografie, Planologie en Milieu", 56838, "FdM", "B", 110, 0.08, 0.02),
    # --- FNWI ---
    ("B Biologie",                    56860, "FNWI", "B", 230, 0.15, 0.04),
    ("B Scheikunde",                  56857, "FNWI", "B",  95, 0.12, 0.05),
    ("B Natuurkunde en Sterrenkunde", 56984, "FNWI", "B", 110, 0.15, 0.06),
    ("B Wiskunde",                    56980, "FNWI", "B",  75, 0.18, 0.08),
    ("B Informatica",                 56964, "FNWI", "B", 280, 0.30, 0.10),
    ("B Molecular Life Sciences",     56860, "FNWI", "B",  90, 0.15, 0.05),
    # --- FMW ---
    ("B Geneeskunde",                 56551, "FMW", "B", 1800, 0.05, 0.02),
    ("B Biomedische Wetenschappen",   56990, "FMW", "B", 350, 0.12, 0.04),
    ("B Tandheelkunde",               56560, "FMW", "B", 280, 0.04, 0.01),
    # --- FdL ---
    ("B Engelse Taal en Cultuur",     56806, "FdL", "B",  95, 0.25, 0.08),
    ("B Geschiedenis",                56034, "FdL", "B", 180, 0.10, 0.03),
    ("B Filosofie",                   56081, "FdL", "B",  70, 0.15, 0.05),
    # --- Master programmes ---
    ("M Psychologie",                 66604, "FSW", "M", 380, 0.35, 0.08),
    ("M Pedagogische Wetenschappen",  66607, "FSW", "M", 180, 0.10, 0.03),
    ("M Artificial Intelligence",     66981, "FSW", "M", 220, 0.40, 0.15),
    ("M Rechtsgeleerdheid",           66827, "FdR", "M", 450, 0.12, 0.03),
    ("M Fiscaal Recht",               66829, "FdR", "M",  70, 0.05, 0.01),
    ("M Bedrijfskunde",               60645, "FdM", "M", 230, 0.30, 0.10),
    ("M Economie",                    66401, "FdM", "M", 130, 0.25, 0.08),
    ("M Bestuurskunde",               66627, "FdM", "M", 100, 0.12, 0.04),
    ("M Politicologie",               66606, "FdM", "M",  70, 0.15, 0.05),
    ("M Informatica",                 66964, "FNWI", "M", 140, 0.35, 0.15),
    ("M Biologie",                    66860, "FNWI", "M",  75, 0.20, 0.08),
    ("M Scheikunde",                  66857, "FNWI", "M",  55, 0.18, 0.10),
    ("M Geneeskunde",                 66551, "FMW", "M", 180, 0.08, 0.03),
    ("M Biomedische Wetenschappen",   66990, "FMW", "M",  90, 0.15, 0.06),
    ("M Geschiedenis",                66034, "FdL", "M",  55, 0.12, 0.04),
    ("M Filosofie",                   66081, "FFTR", "M",  45, 0.20, 0.08),
    ("M Taalwetenschap",              66810, "FdL", "M",  40, 0.25, 0.10),
]

# Numerus fixus programmes (deadline ~15 January)
NUMERUS_FIXUS = {"B Geneeskunde", "B Tandheelkunde"}

# ---------------------------------------------------------------------------
# Demographics for individual data
# ---------------------------------------------------------------------------
NATIONALITIES_NL = [
    ("Nederlandse", 1.0),
]

NATIONALITIES_EER = [
    ("Duitse", 0.45), ("Belgische", 0.15), ("Italiaanse", 0.08),
    ("Spaanse", 0.06), ("Franse", 0.05), ("Griekse", 0.04),
    ("Roemeense", 0.04), ("Bulgaarse", 0.03), ("Poolse", 0.03),
    ("Hongaarse", 0.03), ("Portugese", 0.02), ("Ierse", 0.02),
]

NATIONALITIES_NIET_EER = [
    ("Chinese", 0.25), ("Indonesische", 0.12), ("Turkse", 0.10),
    ("Amerikaanse", 0.08), ("Indiase", 0.08), ("Braziliaanse", 0.06),
    ("Mexicaanse", 0.04), ("Colombiaanse", 0.04), ("Nigeriaanse", 0.03),
    ("Iraanse", 0.03), ("Pakistaanse", 0.03), ("Japanse", 0.03),
    ("Zuid-Koreaanse", 0.03), ("Marokkaanse", 0.04), ("Russische", 0.04),
]

VOOROPLEIDING_BACHELOR = [
    ("VWO", 0.68), ("HAVO", 0.04), ("HBO Propedeuse", 0.12),
    ("MBO", 0.02), ("Buitenlands diploma", 0.14),
]

VOOROPLEIDING_MASTER = [
    ("WO Bachelor", 0.75), ("HBO Bachelor", 0.15),
    ("Buitenlands diploma", 0.10),
]

# Schools: (name, code, plaats_code, land_code)
SCHOOLS_NL = [
    ("Stedelijk Gymnasium Nijmegen", 17407, 268, 6030),
    ("Kandinsky College", 25788, 268, 6030),
    ("Dominicus College", 27651, 268, 6030),
    ("NSG Groenewoud", 23547, 268, 6030),
    ("Montessori College", 26188, 268, 6030),
    ("Het Stedelijk Lyceum Enschede", 14982, 153, 7500),
    ("Christelijk Lyceum Arnhem", 15821, 202, 6800),
    ("Het Rhedens Rozendaal", 16248, 202, 6800),
    ("Lorentz Casimir Lyceum", 14289, 14, 5600),
    ("Gymnasium Beekvliet", 22186, 820, 5461),
    ("Marnix College Ede", 20519, 228, 6710),
    ("Christelijk Gymnasium Utrecht", 14127, 344, 3500),
    ("Het Utrechts Stedelijk Gymnasium", 14115, 344, 3500),
    ("Vossius Gymnasium Amsterdam", 14067, 363, 1000),
    ("Barlaeus Gymnasium", 14066, 363, 1000),
    ("Stedelijk Gymnasium Leiden", 14093, 546, 2300),
    ("Erasmiaans Gymnasium Rotterdam", 14108, 599, 3000),
    ("Gymnasium Haganum Den Haag", 14087, 518, 2500),
    ("Sint-Joriscollege Eindhoven", 23003, 772, 5600),
    ("Philips van Horne SG", 20839, 988, 6001),
    ("Maastricht University College", 30115, 935, 6200),
    ("Gymnasium Celeanum Zwolle", 14118, 193, 8000),
    ("Het Nieuwe Lyceum Bilthoven", 24765, 310, 3720),
    ("Liemers College Zevenaar", 20412, 299, 6900),
    ("Titus Brandsma Lyceum Oss", 23784, 828, 5340),
    ("Merletcollege Cuijk", 20845, 1684, 5430),
    ("Elzendaalcollege Boxmeer", 20834, 756, 5830),
    ("Pax Christi College Druten", 21154, 225, 6650),
    ("Mondial College Nijmegen", 25789, 268, 6030),
    ("Citadel College Nijmegen", 30100, 268, 6030),
    ("Overbetuwe College Bemmel", 30101, 236, 6680),
    ("Lingecollege Tiel", 21046, 281, 4000),
    ("Lyceum Elst", 30102, 236, 6660),
    ("Raayland College Venray", 21174, 984, 5800),
    ("Comenius College Hilversum", 14181, 402, 1200),
    ("RSG Tromp Meesters Steenwijk", 14265, 164, 8330),
    ("Gymnasium Apeldoorn", 14199, 200, 7300),
    ("Bonhoeffer College Enschede", 21234, 153, 7500),
    ("Etty Hillesum Lyceum Deventer", 22321, 150, 7400),
    ("Carolus Borromeus College Helmond", 23011, 794, 5700),
]

SCHOOLS_HBO = [
    ("HAN", 25116, 268, 6030),
    ("Hogeschool van Arnhem en Nijmegen", 25116, 202, 6800),
    ("Hogeschool Utrecht", 25104, 344, 3500),
    ("Hogeschool van Amsterdam", 25093, 363, 1000),
    ("Avans Hogeschool", 25067, 772, 5600),
    ("Fontys Hogescholen", 25065, 772, 5600),
    ("Saxion", 25070, 153, 7500),
    ("Hogeschool Zuyd", 25049, 935, 6200),
    ("Hogeschool Rotterdam", 25091, 599, 3000),
    ("De Haagse Hogeschool", 25035, 518, 2500),
]

SCHOOLS_WO = [
    ("Radboud Universiteit", 21000, 268, 6030),
    ("Universiteit Utrecht", 21001, 344, 3500),
    ("Universiteit van Amsterdam", 21002, 363, 1000),
    ("Vrije Universiteit", 21003, 363, 1000),
    ("Universiteit Leiden", 21004, 546, 2300),
    ("Erasmus Universiteit Rotterdam", 21005, 599, 3000),
    ("TU Eindhoven", 21006, 772, 5600),
    ("Wageningen University", 21007, 289, 6700),
    ("Universiteit Twente", 21008, 153, 7500),
    ("Maastricht University", 21009, 935, 6200),
    ("Rijksuniversiteit Groningen", 21010, 14, 9700),
]

SCHOOLS_BUITENLAND = [
    ("Universitat zu Koln", 99001, 0, 9000),
    ("RWTH Aachen", 99002, 0, 9000),
    ("Westfalische Wilhelms-Universitat", 99003, 0, 9000),
    ("Universitat Duisburg-Essen", 99004, 0, 9000),
    ("Heinrich-Heine-Universitat", 99005, 0, 9000),
    ("KU Leuven", 99006, 0, 9001),
    ("Universitat de Barcelona", 99007, 0, 9002),
    ("Universita di Bologna", 99008, 0, 9003),
    ("Universite Paris-Saclay", 99009, 0, 9004),
    ("University of Athens", 99010, 0, 9005),
]

# Cities with postcode prefix, distance to Nijmegen, and relative probability
CITIES = [
    ("Nijmegen", "6500", 0.0, 0.18),
    ("Arnhem", "6800", 18.5, 0.08),
    ("Wijchen", "6600", 8.2, 0.04),
    ("Elst", "6660", 12.0, 0.03),
    ("Bemmel", "6680", 14.5, 0.02),
    ("Druten", "6650", 15.0, 0.02),
    ("Beuningen", "6640", 7.5, 0.02),
    ("Malden", "6580", 5.0, 0.02),
    ("Groesbeek", "6560", 10.0, 0.01),
    ("Cuijk", "5430", 25.0, 0.01),
    ("Boxmeer", "5830", 32.0, 0.01),
    ("Oss", "5340", 35.0, 0.02),
    ("Den Bosch", "5200", 42.3, 0.04),
    ("Tilburg", "5000", 75.0, 0.03),
    ("Eindhoven", "5600", 95.7, 0.04),
    ("Utrecht", "3500", 82.1, 0.07),
    ("Amsterdam", "1000", 120.4, 0.06),
    ("Rotterdam", "3000", 135.2, 0.03),
    ("Den Haag", "2500", 130.0, 0.03),
    ("Leiden", "2300", 125.0, 0.02),
    ("Ede", "6710", 30.0, 0.02),
    ("Wageningen", "6700", 35.0, 0.01),
    ("Apeldoorn", "7300", 55.0, 0.01),
    ("Deventer", "7400", 75.0, 0.01),
    ("Zwolle", "8000", 100.0, 0.01),
    ("Enschede", "7500", 115.0, 0.01),
    ("Groningen", "9700", 195.8, 0.02),
    ("Maastricht", "6200", 130.0, 0.01),
    ("Venlo", "5900", 75.0, 0.01),
    ("Venray", "5800", 50.0, 0.01),
    ("Heerlen", "6400", 140.0, 0.01),
    ("Tiel", "4000", 30.0, 0.01),
    ("Doetinchem", "7000", 40.0, 0.01),
    ("Zevenaar", "6900", 22.0, 0.01),
    ("Hilversum", "1200", 105.0, 0.01),
    ("Amersfoort", "3800", 80.0, 0.01),
    ("Breda", "4800", 90.0, 0.01),
    ("Helmond", "5700", 85.0, 0.01),
    ("Roermond", "6040", 95.0, 0.01),
    ("Leeuwarden", "8900", 220.0, 0.005),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _academic_week_to_date(year, week):
    """Convert academic year + week number to approximate date.

    Academic year 2020 runs from Sep 2019 to Aug 2020.
    Week 39-52 falls in calendar year (year-1), week 1-38 in calendar year.
    """
    if week >= 39:
        cal_year = year - 1
    else:
        cal_year = year
    # Use ISO week
    try:
        return datetime.strptime(f"{cal_year}-W{week:02d}-1", "%G-W%V-%u")
    except ValueError:
        return datetime(cal_year, 1, 1) + timedelta(weeks=week - 1)


def _pick_weighted(choices, rng):
    """Pick from list of (value, probability) tuples."""
    values, probs = zip(*choices)
    probs = np.array(probs, dtype=float)
    probs /= probs.sum()
    return values[rng.choice(len(values), p=probs)]


def _generate_cumulative_curve(total, type_ho, is_nf, rng):
    """Generate a realistic S-shaped cumulative application curve over 52 weeks.

    Returns array of length 52, representing weeks in academic order:
    [week39, week40, ..., week52, week1, week2, ..., week38]
    """
    t = np.linspace(0, 1, 52)

    if type_ho == "B":
        if is_nf:
            # Numerus fixus: most applications before mid-January (~week 2 = index 15)
            base = 1 / (1 + np.exp(-18 * (t - 0.28)))
        else:
            # Regular Bachelor: acceleration toward May 1 deadline (~week 17 = index 30)
            # Two-phase: slow early, then ramp before deadline
            early = 0.3 * (1 / (1 + np.exp(-10 * (t - 0.25))))
            late = 0.7 * (1 / (1 + np.exp(-14 * (t - 0.56))))
            base = early + late
    else:
        # Master: spread more evenly, slight peak in spring
        base = 1 / (1 + np.exp(-7 * (t - 0.45)))

    # Scale to total
    curve = base / base[-1] * total

    # Add small noise (proportional)
    noise_scale = np.maximum(curve * 0.02, 0.3)
    noise = rng.normal(0, noise_scale)
    curve = curve + noise

    # Ensure monotonically non-decreasing and positive
    curve = np.maximum(curve, 0.5)
    curve = np.maximum.accumulate(curve)

    # Ensure final value is close to total
    curve = curve / curve[-1] * total

    return np.round(curve, 2)


def _meercode_for_week(week_index, type_ho, rng):
    """Generate realistic meercode_V for a given week.

    Early in the year: higher (students applied to many programmes).
    Later: lower (students have committed).
    """
    # Week index 0 = week 39 (October), index 51 = week 38 (September)
    t = week_index / 51
    if type_ho == "B":
        base = 1.45 - 0.25 * t  # 1.45 early → 1.20 late
    else:
        base = 1.30 - 0.15 * t  # 1.30 early → 1.15 late
    return round(max(1.01, base + rng.normal(0, 0.02)), 2)


def _year_variation(year, base_volume, rng):
    """Apply year-over-year variation to base volume.

    Slight upward trend + random noise to mimic real enrollment dynamics.
    """
    # Slight growth trend: ~2% per year from baseline year 2016
    trend = 1.0 + 0.02 * (year - 2016)
    noise = rng.normal(1.0, 0.05)
    return max(5, int(base_volume * trend * noise))


# ---------------------------------------------------------------------------
# Telbestanden generator
# ---------------------------------------------------------------------------

def generate_telbestanden():
    """Generate realistic telbestand CSV files in data/input_raw/telbestanden/."""
    print("Generating telbestanden...")

    # Clear existing
    if TEL_DIR.exists():
        shutil.rmtree(TEL_DIR)
    TEL_DIR.mkdir(parents=True, exist_ok=True)

    # Academic week order: 39,40,...,52,1,2,...,38
    week_order = list(range(39, 53)) + list(range(1, 39))

    for year in YEARS_TEL:
        # Pre-compute cumulative curves for each programme × herkomst
        curves = {}
        for naam, croho, fac, type_ho, base_nl, pct_eer, pct_niet_eer in PROGRAMMES:
            is_nf = naam in NUMERUS_FIXUS
            vol_nl = _year_variation(year, base_nl, RNG)
            vol_eer = max(1, int(vol_nl * pct_eer)) if pct_eer > 0 else 0
            vol_niet_eer = max(1, int(vol_nl * pct_niet_eer)) if pct_niet_eer > 0 else 0

            for herkomst_code, vol in [("N", vol_nl), ("E", vol_eer), ("R", vol_niet_eer)]:
                if vol < 1:
                    continue
                curve = _generate_cumulative_curve(vol, type_ho, is_nf, RNG)
                curves[(naam, croho, fac, type_ho, herkomst_code)] = curve

        # Write one file per week
        for week_idx, week_num in enumerate(week_order):
            filename = f"telbestandY{year}W{week_num:02d}.csv"
            filepath = TEL_DIR / filename

            rows = []
            for (naam, croho, fac, type_ho, herkomst), curve in curves.items():
                aantal = curve[week_idx]
                meercode = _meercode_for_week(week_idx, type_ho, RNG)
                rows.append({
                    "Brincode": BRINCODE,
                    "Studiejaar": year,
                    "Type_HO": type_ho,
                    "Isatcode": croho,
                    "Groepeernaam": naam,
                    "Faculteit": fac,
                    "Herkomst": herkomst,
                    "Hogerejaars": "N",
                    "Herinschrijving": "N",
                    "Aantal": aantal,
                    "meercode_V": meercode,
                })

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "Brincode", "Studiejaar", "Type_HO", "Isatcode", "Groepeernaam",
                    "Faculteit", "Herkomst", "Hogerejaars", "Herinschrijving",
                    "Aantal", "meercode_V",
                ], delimiter=";")
                writer.writeheader()
                writer.writerows(rows)

    n_files = len(list(TEL_DIR.glob("*.csv")))
    print(f"  Created {n_files} telbestand files")


# ---------------------------------------------------------------------------
# Individual data generator
# ---------------------------------------------------------------------------

def generate_individual_data():
    """Generate realistic individual application records."""
    print("Generating individuele aanmelddata...")

    records = []
    student_key = 1000

    for year in YEARS_INDIVIDUAL:
        for naam, croho, fac, type_ho, base_nl, pct_eer, pct_niet_eer in PROGRAMMES:
            vol_nl = _year_variation(year, base_nl, RNG)
            vol_eer = max(1, int(vol_nl * pct_eer)) if pct_eer > 0 else 0
            vol_niet_eer = max(1, int(vol_nl * pct_niet_eer)) if pct_niet_eer > 0 else 0

            for herkomst_code, vol, nat_list in [
                ("N", vol_nl, NATIONALITIES_NL),
                ("E", vol_eer, NATIONALITIES_EER),
                ("R", vol_niet_eer, NATIONALITIES_NIET_EER),
            ]:
                if vol < 1:
                    continue

                # Determine EER code
                eer_code = "N" if herkomst_code == "N" else ("J" if herkomst_code == "E" else "N")
                # Correctly: NL and Niet-EER both get EER=N; EER gets EER=J
                if herkomst_code == "E":
                    eer_code = "J"
                else:
                    eer_code = "N"

                is_nf = naam in NUMERUS_FIXUS
                examentype = "Propedeuse Bachelor" if type_ho == "B" else "Master"

                # Vooropleiding distribution depends on Bachelor vs Master
                vooropl_choices = VOOROPLEIDING_BACHELOR if type_ho == "B" else VOOROPLEIDING_MASTER

                # Application dates follow similar S-curve pattern
                for _ in range(vol):
                    # Pick application date based on programme type
                    if type_ho == "B":
                        if is_nf:
                            # Most apply Oct-Jan
                            month_weights = [0.15, 0.20, 0.25, 0.25, 0.08, 0.03, 0.02, 0.01, 0.005, 0.005, 0.0, 0.0]
                        else:
                            # Spread Oct-May with peak around April
                            month_weights = [0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.14, 0.06, 0.03, 0.02, 0.01]
                    else:
                        # Master: spread Oct-Aug
                        month_weights = [0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.13, 0.10, 0.06, 0.03, 0.02]

                    # Months: Oct(year-1), Nov, Dec, Jan(year), Feb, Mar, Apr, May, Jun, Jul, Aug, Sep
                    month_weights = np.array(month_weights, dtype=float)
                    month_weights /= month_weights.sum()
                    month_idx = RNG.choice(12, p=month_weights)

                    # Map to calendar month
                    cal_months = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                    cal_month = cal_months[month_idx]
                    cal_year = year - 1 if cal_month >= 10 else year

                    # Pick a random day in that month
                    day = RNG.integers(1, 29)
                    try:
                        app_date = datetime(cal_year, cal_month, day)
                    except ValueError:
                        app_date = datetime(cal_year, cal_month, 28)

                    datum_verzoek = app_date.strftime("%d-%m-%Y")

                    # Start date is always Sep 1 or Oct 1 of the academic year
                    ingangsdatum = f"01-09-{year}" if type_ho == "B" else f"01-09-{year}"

                    # Enrollment status
                    # Realistic rates: ~60-68% enrolled, ~18-25% cancelled, ~8-15% pending
                    # Later years (2024) have more pending since year isn't over
                    if year == YEARS_INDIVIDUAL[-1]:
                        status_probs = [0.35, 0.25, 0.40]  # more pending in current year
                    else:
                        status_probs = [0.64, 0.22, 0.14]
                    status = RNG.choice(
                        ["Ingeschreven", "Geannuleerd", "Verzoek tot inschrijving"],
                        p=status_probs,
                    )

                    # Gender varies by programme
                    if "Psychologie" in naam or "Pedagogische" in naam or "Communicatie" in naam:
                        gender_probs = [0.30, 0.70]  # more female
                    elif "Informatica" in naam or "Natuurkunde" in naam or "Wiskunde" in naam:
                        gender_probs = [0.75, 0.25]  # more male
                    elif "Geneeskunde" in naam or "Biomedische" in naam or "Tandheelkunde" in naam:
                        gender_probs = [0.35, 0.65]  # slightly more female
                    elif "Rechtsgeleerdheid" in naam or "Fiscaal" in naam:
                        gender_probs = [0.45, 0.55]
                    else:
                        gender_probs = [0.50, 0.50]
                    geslacht = RNG.choice(["M", "V"], p=gender_probs)

                    # Nationality
                    nationaliteit = _pick_weighted(nat_list, RNG)

                    # Vooropleiding
                    vooropleiding = _pick_weighted(vooropl_choices, RNG)

                    # Address - NL students get Dutch cities, EER/Niet-EER get mix
                    if herkomst_code == "N":
                        city_probs = np.array([c[3] for c in CITIES], dtype=float)
                        city_probs /= city_probs.sum()
                        city_idx = RNG.choice(len(CITIES), p=city_probs)
                        city_name = CITIES[city_idx][0]
                        postcode_prefix = CITIES[city_idx][1]
                        postcode = f"{postcode_prefix}{RNG.choice(list('ABCDEFGHJKLMNPRSTVWXZ'))}{RNG.choice(list('ABCDEFGHJKLMNPRSTVWXZ'))}"
                        land = "NL"
                    elif herkomst_code == "E":
                        if RNG.random() < 0.3:  # 30% already live in NL
                            city_idx = RNG.choice(len(CITIES))
                            city_name = CITIES[city_idx][0]
                            postcode_prefix = CITIES[city_idx][1]
                            postcode = f"{postcode_prefix}{RNG.choice(list('ABCDEFGHJKLMNPRSTVWXZ'))}{RNG.choice(list('ABCDEFGHJKLMNPRSTVWXZ'))}"
                            land = "NL"
                        else:
                            city_name = RNG.choice(["Koln", "Dusseldorf", "Aachen", "Munster",
                                                     "Kleve", "Brussel", "Antwerpen", "Leuven",
                                                     "Madrid", "Roma", "Parijs", "Athene"])
                            postcode = f"{RNG.integers(10000, 99999)}"
                            land = RNG.choice(["DE", "BE", "ES", "IT", "FR", "GR"])
                    else:
                        city_name = RNG.choice(["Beijing", "Jakarta", "Istanbul", "New York",
                                                 "Mumbai", "Sao Paulo", "Mexico City",
                                                 "Lagos", "Tehran", "Seoul", "Tokyo"])
                        postcode = f"{RNG.integers(10000, 99999)}"
                        land = RNG.choice(["CN", "ID", "TR", "US", "IN", "BR", "MX",
                                            "NG", "IR", "KR", "JP"])

                    # School based on vooropleiding
                    if vooropleiding in ("VWO", "HAVO", "MBO"):
                        if herkomst_code == "N":
                            school = SCHOOLS_NL[RNG.integers(0, len(SCHOOLS_NL))]
                        else:
                            school = SCHOOLS_BUITENLAND[RNG.integers(0, len(SCHOOLS_BUITENLAND))]
                    elif vooropleiding == "HBO Propedeuse":
                        school = SCHOOLS_HBO[RNG.integers(0, len(SCHOOLS_HBO))]
                    elif vooropleiding in ("WO Bachelor",):
                        all_wo = SCHOOLS_WO + [("Radboud Universiteit", 21000, 268, 6030)]
                        school = all_wo[RNG.integers(0, len(all_wo))]
                    elif vooropleiding == "HBO Bachelor":
                        school = SCHOOLS_HBO[RNG.integers(0, len(SCHOOLS_HBO))]
                    else:  # Buitenlands diploma
                        school = SCHOOLS_BUITENLAND[RNG.integers(0, len(SCHOOLS_BUITENLAND))]

                    # Study address (usually Nijmegen or same as home)
                    if RNG.random() < 0.6:
                        studie_postcode = f"6500{RNG.choice(list('ABCDEFGHJKLMNPRSTVWXZ'))}{RNG.choice(list('ABCDEFGHJKLMNPRSTVWXZ'))}"
                        studie_land = "NL"
                    else:
                        studie_postcode = postcode
                        studie_land = land

                    records.append({
                        "Sleutel": student_key,
                        "Datum Verzoek Inschr": datum_verzoek,
                        "Ingangsdatum": ingangsdatum,
                        "Collegejaar": year,
                        "Datum intrekking vooraanmelding": "",
                        "Inschrijfstatus": status,
                        "Faculteit": fac,
                        "Examentype": examentype,
                        "Croho": naam.split(" ", 1)[1] if " " in naam else naam,
                        "Croho groepeernaam": naam,
                        "Opleiding": naam,
                        "Hoofdopleiding": naam,
                        "Eerstejaars croho jaar": year,
                        "Is eerstejaars croho opleiding": 1,
                        "Is hogerejaars": 0,
                        "BBC ontvangen": 0,
                        "Type vooropleiding": vooropleiding,
                        "Nationaliteit": nationaliteit,
                        "EER": eer_code,
                        "Geslacht": geslacht,
                        "Geverifieerd adres postcode": postcode,
                        "Geverifieerd adres plaats": city_name,
                        "Geverifieerd adres land": land,
                        "Studieadres postcode": studie_postcode,
                        "Studieadres land": studie_land,
                        "School code eerste vooropleiding": school[1],
                        "School eerste vooropleiding": school[0],
                        "Plaats code eerste vooropleiding": school[2],
                        "Land code eerste vooropleiding": school[3],
                        "Aantal studenten": 1,
                    })
                    student_key += 1

    # Write CSV
    output_path = RAW_DIR / "individuele_aanmelddata.csv"
    df = pd.DataFrame(records)
    df.to_csv(output_path, sep=";", index=False)
    print(f"  Created {len(records)} individual records ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


# ---------------------------------------------------------------------------
# Oktober-bestand generator
# ---------------------------------------------------------------------------

def generate_oktober_bestand():
    """Generate realistic oktober-bestand (ground truth enrollment data)."""
    print("Generating oktober_bestand...")

    rows = []
    for year in YEARS_OKTOBER:
        for naam, croho, fac, type_ho, base_nl, pct_eer, pct_niet_eer in PROGRAMMES:
            vol_nl = _year_variation(year, base_nl, RNG)
            vol_eer = max(1, int(vol_nl * pct_eer))
            vol_niet_eer = max(1, int(vol_nl * pct_niet_eer))

            for herkomst_label, vol, herkomst_code in [
                ("NL", vol_nl, "N"),
                ("EER", vol_eer, "E"),
                ("Niet-EER", vol_niet_eer, "R"),
            ]:
                if vol < 1:
                    continue

                # Conversion rate: applications → actual enrollments
                if naam in NUMERUS_FIXUS:
                    conversion = RNG.uniform(0.08, 0.15)  # NF programmes have low conversion
                elif type_ho == "B":
                    conversion = RNG.uniform(0.35, 0.55)
                else:
                    conversion = RNG.uniform(0.50, 0.70)

                enrolled = max(1, int(vol * conversion))

                # Eerstejaars row
                if type_ho == "B":
                    exam_code = "Bachelor eerstejaars"
                else:
                    exam_code = "Master"

                rows.append({
                    "Collegejaar": year,
                    "Groepeernaam Croho": naam,
                    "EER-NL-nietEER": herkomst_label,
                    "Examentype code": exam_code,
                    "Aantal eerstejaars croho": 1,
                    "Aantal Hoofdinschrijvingen": enrolled,
                })

                # Add Bachelor hogerejaars rows (~30-60% of eerstejaars for Bachelor)
                if type_ho == "B":
                    hogerejaars = max(1, int(enrolled * RNG.uniform(0.30, 0.60)))
                    rows.append({
                        "Collegejaar": year,
                        "Groepeernaam Croho": naam,
                        "EER-NL-nietEER": herkomst_label,
                        "Examentype code": "Bachelor hogerejaars",
                        "Aantal eerstejaars croho": 0,
                        "Aantal Hoofdinschrijvingen": hogerejaars,
                    })

    df = pd.DataFrame(rows)
    output_path = RAW_DIR / "oktober_bestand.xlsx"
    df.to_excel(output_path, index=False)
    print(f"  Created {len(rows)} rows ({output_path.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Afstanden generator
# ---------------------------------------------------------------------------

def generate_afstanden():
    """Generate realistic distance reference table."""
    print("Generating afstanden...")

    rows = [{"Geverifieerd adres plaats": city, "Afstand": dist}
            for city, _, dist, _ in CITIES]

    df = pd.DataFrame(rows)
    output_path = RAW_DIR / "afstanden.xlsx"
    df.to_excel(output_path, index=False)
    print(f"  Created {len(rows)} city distances")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Generating realistic demo data in {RAW_DIR}/")
    print(f"Seed: {SEED}")
    print()

    generate_telbestanden()
    generate_individual_data()
    generate_oktober_bestand()
    generate_afstanden()

    # Report total size
    total_bytes = sum(f.stat().st_size for f in RAW_DIR.rglob("*") if f.is_file())
    print(f"\nTotal input_raw size: {total_bytes / 1024 / 1024:.1f} MB")
    print("Run the pipeline (uv run main.py) to process via ETL.")


if __name__ == "__main__":
    main()
