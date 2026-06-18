import os
import shutil

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from studentprognose.utils.telbestand_filenames import (
    compile_patterns,
    match_telbestand,
    week_from_match,
)

# De 16 canonieke kolommen van vooraanmeldingen_cumulatief.csv (downstream-contract).
_OUTPUT_COLUMNS = [
    "Korte naam instelling",
    "Collegejaar",
    "Weeknummer rapportage",
    "Weeknummer",
    "Faculteit",
    "Type hoger onderwijs",
    "Groepeernaam Croho",
    "Naam Croho opleiding Nederlands",
    "Croho",
    "Herinschrijving",
    "Hogerejaars",
    "Herkomst",
    "Gewogen vooraanmelders",
    "Ongewogen vooraanmelders",
    "Aantal aanmelders met 1 aanmelding",
    "Inschrijvingen",
]

# Identiteitskolommen voor aggregatie: de 12 niet-numerieke van _OUTPUT_COLUMNS.
# De fijnmazige SQL-dimensies (Geslacht, Opl_vorm, Voertaal, Fixus, 1cHO_*,
# meercode_V) collapsen hierop door optelling van de twee maatkolommen.
_AGG_SUM_COLUMNS = ["Gewogen vooraanmelders", "Ongewogen vooraanmelders"]
_AGG_IDENTITY_COLUMNS = [c for c in _OUTPUT_COLUMNS if c not in _AGG_SUM_COLUMNS
                         and c not in ("Aantal aanmelders met 1 aanmelding", "Inschrijvingen")]

# De canonieke output is altijd ;-gescheiden (downstream leest met sep=";").
# De input-separator is configureerbaar: legacy instellingsformaat is ;, het
# UvA SQL-telbestand is ,-gescheiden.
_OUTPUT_SEPARATOR = ";"

# Defaults reproduceren het legacy instellingsformaat exact, zodat een lege
# config (en de packaged default) ongewijzigd blijft werken. Het UvA SQL-formaat
# overschrijft deze via configuration["cumulative_input"] (zie configuratie.md).
_DEFAULT_CUMULATIVE_INPUT = {
    "separator": ";",
    "rename": {
        "Brincode": "Korte naam instelling",
        "Studiejaar": "Collegejaar",
        "Type_HO": "Type hoger onderwijs",
        "Isatcode": "Croho",
        "Aantal": "Ongewogen vooraanmelders",
    },
    "value_maps": {
        "Type hoger onderwijs": {"P": "Bachelor", "B": "Bachelor", "M": "Master"},
        "Herinschrijving": {"J": "Ja", "N": "Nee"},
        "Hogerejaars": {"J": "Ja", "N": "Nee"},
        "Herkomst": {"N": "NL", "E": "EER", "R": "Niet-EER"},
    },
    # Bronkolom voor de leesbare opleidingsnaam; valt terug op Croho (Isatcode)
    # als de kolom ontbreekt (UvA SQL levert geen Groepeernaam — zie #232).
    "programme_name_source": "Groepeernaam",
    # Niet-lege sentinel voor een ontbrekende Faculteit. MOET niet-leeg zijn:
    # de cumulatieve groupby/pivot droppen rijen met een NaN-sleutel (#232).
    "faculteit_sentinel": None,
    # Aggregeer fijnmazige rijen naar de canonieke grain (UvA SQL is fijnmaziger).
    "aggregate": False,
    # Filter etl_is_deleted == 0 (UvA SQL soft-delete vlag; afwezig in legacy).
    "drop_deleted": False,
}


def run_etl(configuration):
    """Run the full ETL pipeline: raw external data → data/input/ files."""
    print("==== Processing raw input data ====")

    paths = configuration["paths"]
    cwd = os.getcwd()

    # ETL always writes to canonical (non-DEMO) paths
    os.makedirs(os.path.join(cwd, "data", "input"), exist_ok=True)
    output_cumulative = os.path.join(cwd, "data", "input", "vooraanmeldingen_cumulatief.csv")
    output_individual = os.path.join(cwd, "data", "input", "vooraanmeldingen_individueel.csv")
    # Step 1: Rowbind telbestanden → vooraanmeldingen_cumulatief.csv
    telbestanden_dir = os.path.join(cwd, paths.get("path_raw_telbestanden", "data/input_raw/telbestanden"))

    if os.path.isdir(telbestanden_dir) and os.listdir(telbestanden_dir):
        print("[1/4] Rowbinding telbestanden...        → data/input/vooraanmeldingen_cumulatief.csv")
        _rowbind_and_reformat(telbestanden_dir, output_cumulative, configuration)
    else:
        print(f"[1/4] Skipping rowbind (no telbestanden found in {telbestanden_dir})")

    # Step 2: Interpolate missing weeks
    if os.path.exists(output_cumulative):
        print("[2/4] Interpolating missing weeks...    → data/input/vooraanmeldingen_cumulatief.csv")
        _interpolate_missing_weeks(output_cumulative)
    else:
        print("[2/4] Skipping interpolation (no cumulative file)")

    # Step 3: Calculate student counts from telbestand studenten
    path_october_rel = paths.get("path_raw_october", "")
    if not path_october_rel:
        print("[3/4] Skipping student counts (path_raw_october not configured)")
    else:
        path_october = os.path.join(cwd, path_october_rel)
        oktober_columns = configuration.get("columns", {}).get("oktober", {})
        if os.path.exists(path_october):
            print("[3/4] Calculating student counts...     → data/input/student_count_*.xlsx")
            _calculate_student_counts(path_october, cwd, oktober_columns)
        else:
            print(f"[3/4] Skipping student counts (telbestand studenten not found: {path_october_rel})")

    # Step 4: Copy direct files (raw → canonical input paths)
    copied = _copy_direct_files(cwd, paths, output_individual)
    if copied:
        print(f"[4/4] Copying direct files...           → {', '.join(copied)}")
    else:
        print("[4/4] No direct files to copy")

    print("==== ETL complete ====")


def _rowbind_and_reformat(telbestanden_dir, output_path, configuration):
    """Voeg wekelijkse telbestand-snapshots samen tot één cumulatief bestand.

    Config-gedreven via ``configuration["cumulative_input"]`` (separator, kolom-
    rename, waardevertalingen, aggregatie). De defaults reproduceren het legacy
    Studielink instellingsformaat; het UvA SQL-formaat zet dezelfde bronkolommen
    om naar identieke 16-koloms output. Zie :data:`_DEFAULT_CUMULATIVE_INPUT` en
    docs/configuratie.md.
    """
    cfg = {**_DEFAULT_CUMULATIVE_INPUT, **(configuration or {}).get("cumulative_input", {})}
    value_maps = {**_DEFAULT_CUMULATIVE_INPUT["value_maps"], **cfg.get("value_maps", {})}
    patterns = compile_patterns(configuration)

    dataframes = []
    for filename in sorted(os.listdir(telbestanden_dir)):
        match = match_telbestand(filename, patterns)
        if not match:
            continue

        filepath = os.path.join(telbestanden_dir, filename)
        data = pd.read_csv(filepath, sep=cfg["separator"], low_memory=False)
        data["Weeknummer"] = week_from_match(match)
        dataframes.append(data)

    if not dataframes:
        print("  Warning: no telbestand files found")
        return

    data = pd.concat(dataframes, ignore_index=True)

    # ISO-week 53 komt voor in lange ISO-jaren (bijv. leverdatum eind dec 2020/2026).
    # Deze snapshots blijven in de output staan, maar het 52-weeks SARIMA-seizoens-
    # model (get_all_weeks_ordered) modelleert ze niet. Maak dat expliciet i.p.v. stil.
    if (data["Weeknummer"] == 53).any():
        print(
            "  Warning: ISO-week 53 aangetroffen (lange ISO-jaren). Deze weeksnapshots "
            "blijven in vooraanmeldingen_cumulatief.csv staan, maar vallen buiten het "
            "52-weeks seizoensmodel van het cumulatieve spoor."
        )

    # UvA SQL soft-delete vlag (afwezig in legacy): verwijder verwijderde rijen.
    if cfg["drop_deleted"] and "etl_is_deleted" in data.columns:
        data = data[data["etl_is_deleted"] == 0]

    # Status A = annulering (Studielink PvL §5.12). Voor die rijen is meercode_V
    # per definitie 0 en horen aantallen semantisch niet bij vooraanmelders.
    # Geverifieerd op het echte UvA-anker: na deze filter blijft geen meercode_V
    # == 0 over, dus de deling hieronder kan niet door nul gaan.
    data = data[data["Status"] != "A"].copy()

    data["Gewogen vooraanmelders"] = data["Aantal"] / data["meercode_V"]

    data.rename(columns=cfg["rename"], inplace=True)

    data["Weeknummer rapportage"] = data["Weeknummer"]

    # Leesbare opleidingsnaam uit de bronkolom indien aanwezig, anders Croho
    # (Isatcode) als placeholder — de UvA SQL levert geen Groepeernaam (#232).
    programme_name = data.get(cfg["programme_name_source"], data["Croho"])
    data["Groepeernaam Croho"] = programme_name
    data["Naam Croho opleiding Nederlands"] = programme_name

    data["Aantal aanmelders met 1 aanmelding"] = None
    data["Inschrijvingen"] = None

    # Faculteit ontbreekt in de UvA SQL → niet-lege sentinel zodat de cumulatieve
    # groupby/pivot de rijen niet als NaN-sleutel droppen (echte mapping: #232).
    if "Faculteit" not in data.columns:
        data["Faculteit"] = cfg["faculteit_sentinel"]

    for col, mapping in value_maps.items():
        if col in data.columns:
            _warn_unmapped_values(data[col], mapping, col)
            data[col] = data[col].replace(mapping)

    # De UvA SQL is fijnmaziger dan de canonieke grain → optellen, anders
    # crasht de pivot in transform_data op een dubbele index (no-op voor legacy).
    if cfg["aggregate"]:
        data = _aggregate_to_grain(data)

    data = data[_OUTPUT_COLUMNS]
    data.to_csv(output_path, sep=_OUTPUT_SEPARATOR, index=False)


def _aggregate_to_grain(data):
    """Tel fijnmazige rijen op naar de canonieke 16-koloms grain.

    De UvA SQL-levering splitst elke canonieke combinatie nog op Geslacht,
    Opl_vorm, Voertaal, Fixus en 1cHO. Die dimensies staan niet in de output;
    ``Gewogen``/``Ongewogen vooraanmelders`` worden per grain-combinatie
    gesommeerd (``Gewogen`` is al per rij berekend vóór de optelling).

    ``dropna=False`` houdt rijen met een niet-NaN-maar-sentinel sleutel
    (bijv. Faculteit) intact.
    """
    aggregated = (
        data.groupby(_AGG_IDENTITY_COLUMNS, dropna=False, as_index=False)[_AGG_SUM_COLUMNS]
        .sum()
    )
    aggregated["Aantal aanmelders met 1 aanmelding"] = None
    aggregated["Inschrijvingen"] = None
    return aggregated


def _warn_unmapped_values(series, mapping, column):
    """Print een waarschuwing voor waarden buiten de configureerbare mapping.

    ``Series.replace`` laat niet-gemapte waarden ongewijzigd door; dit maakt
    nieuwe/onverwachte codes (bijv. een toekomstige Herkomst- of Type_HO-waarde)
    zichtbaar i.p.v. dat ze stil door de pipeline glippen.
    """
    unmapped = set(series.dropna().unique()) - set(mapping)
    if unmapped:
        print(
            f"  Warning: kolom '{column}' bevat niet-gemapte waarde(n) "
            f"{sorted(map(str, unmapped))} — onveranderd doorgegeven."
        )


def _interpolate_missing_weeks(csv_path):
    """Interpolate missing weeks in the cumulative data file."""
    data = pd.read_csv(csv_path, sep=";", low_memory=True)

    # Convert Dutch numeric formats
    for col in ["Gewogen vooraanmelders", "Ongewogen vooraanmelders",
                "Aantal aanmelders met 1 aanmelding", "Inschrijvingen"]:
        data[col] = data[col].apply(_convert_to_float)

    # Auto-detect: find years and week gaps to interpolate
    years = data["Collegejaar"].unique()
    interpolated = False

    for year in years:
        year_data = data[(data["Collegejaar"] == year)
                         & (data["Herinschrijving"] == "Nee")
                         & (data["Hogerejaars"] == "Nee")]
        weeks = sorted(year_data["Weeknummer"].unique())

        if len(weeks) < 2:
            continue

        # Find gaps between consecutive weeks (accounting for wrap-around)
        for i in range(len(weeks) - 1):
            start_week = weeks[i]
            end_week = weeks[i + 1]
            if end_week - start_week > 1:
                data = _interpolate(data, year, start_week, end_week)
                interpolated = True

        # Check wrap-around gap (last week → first week)
        if weeks[-1] > weeks[0] + 40:  # Likely a wrap scenario (e.g., week 51 and week 1)
            data = _interpolate(data, year, weeks[-1], weeks[0])
            interpolated = True

    # Round results
    data["Gewogen vooraanmelders"] = data["Gewogen vooraanmelders"].round(2)
    for col in ["Ongewogen vooraanmelders", "Aantal aanmelders met 1 aanmelding", "Inschrijvingen"]:
        data[col] = data[col].where(data[col].notna(), np.nan)
        data[col] = np.round(data[col]).astype("Int64")

    data.to_csv(csv_path, sep=";", index=False)

    if not interpolated:
        print("  No gaps found to interpolate")


def _convert_to_float(value):
    if isinstance(value, str):
        value = value.replace(".", "").replace(",", ".")
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def _interpolate(data, year, start_week, end_week, max_week=52):
    """Linear interpolation for missing weeks between start_week and end_week."""
    is_wrap = start_week > end_week

    filtered = data[
        (data["Herinschrijving"] == "Nee")
        & (data["Hogerejaars"] == "Nee")
        & (data["Collegejaar"] == year)
        & (data["Weeknummer"].isin([start_week, end_week]))
    ]

    if not is_wrap:
        intermediate_weeks = list(range(start_week + 1, end_week))
    else:
        intermediate_weeks = list(range(start_week + 1, max_week + 1)) + list(range(1, end_week))

    if not intermediate_weeks:
        return data

    def _week_position(week):
        if not is_wrap:
            return week - start_week
        elif week >= start_week:
            return week - start_week
        else:
            return week + (max_week - start_week + 1)

    updates = {
        "Collegejaar": [], "Weeknummer": [], "Type hoger onderwijs": [],
        "Groepeernaam Croho": [], "Herkomst": [], "target_value": [], "value": [],
    }

    groups = filtered[["Type hoger onderwijs", "Groepeernaam Croho", "Herkomst"]].drop_duplicates()

    for _, group in groups.iterrows():
        croho = group["Groepeernaam Croho"]
        herkomst = group["Herkomst"]
        examentype = group["Type hoger onderwijs"]

        group_data = filtered[
            (filtered["Groepeernaam Croho"] == croho)
            & (filtered["Herkomst"] == herkomst)
            & (filtered["Type hoger onderwijs"] == examentype)
        ]

        for target_col in ["Gewogen vooraanmelders", "Ongewogen vooraanmelders",
                           "Aantal aanmelders met 1 aanmelding", "Inschrijvingen"]:
            val_start = group_data[group_data["Weeknummer"] == start_week][target_col]
            val_end = group_data[group_data["Weeknummer"] == end_week][target_col]

            if val_start.empty or val_end.empty:
                continue

            pos_end = _week_position(end_week)
            interpolator = interp1d(
                [0, pos_end],
                [float(val_start.iloc[0]), float(val_end.iloc[0])],
                kind="linear",
            )
            positions = [_week_position(w) for w in intermediate_weeks]
            values = interpolator(positions)

            for week, val in zip(intermediate_weeks, values):
                updates["Collegejaar"].append(year)
                updates["Weeknummer"].append(week)
                updates["Type hoger onderwijs"].append(examentype)
                updates["Groepeernaam Croho"].append(croho)
                updates["Herkomst"].append(herkomst)
                updates["target_value"].append(target_col)
                updates["value"].append(val)

    if not updates["Collegejaar"]:
        return data

    updates_df = pd.DataFrame(updates)
    updates_pivot = updates_df.pivot_table(
        index=["Collegejaar", "Weeknummer", "Type hoger onderwijs", "Groepeernaam Croho", "Herkomst"],
        columns="target_value",
        values="value",
    ).reset_index()

    data = data.merge(
        updates_pivot,
        on=["Collegejaar", "Weeknummer", "Type hoger onderwijs", "Groepeernaam Croho", "Herkomst"],
        how="left",
        suffixes=("", "_interpolated"),
    )

    for col in ["Gewogen vooraanmelders", "Ongewogen vooraanmelders",
                "Aantal aanmelders met 1 aanmelding", "Inschrijvingen"]:
        interp_col = f"{col}_interpolated"
        if interp_col in data.columns:
            data[col] = data[interp_col].combine_first(data[col])
            data.drop(columns=[interp_col], inplace=True)

    return data


def _calculate_student_count(data, volume):
    """Calculate student counts from the telbestand studenten.

    Returns a DataFrame with columns: Collegejaar, Croho groepeernaam,
    Herkomst, Aantal_studenten, Examentype.
    """
    result = {
        "Collegejaar": [],
        "Croho groepeernaam": [],
        "Herkomst": [],
        "Aantal_studenten": [],
        "Examentype": [],
    }

    # Joinsleutel = de landelijke Isatcode (CROHO-code), niet de
    # instellingsspecifieke opleidingsnaam. De code is stabiel en identiek aan
    # de Isatcode in de telbestanden, zodat het label (student_count) op exact
    # dezelfde sleutel als de features (vooraanmeldingen_cumulatief) joint.
    # De code wordt weggeschreven in de kolom "Croho groepeernaam" (#232:
    # voor de UvA-bron is dat sowieso al de code, geen leesbare naam).
    programme_key = "Isatcode"
    herkomst_key = "EER-NL-nietEER"
    year_key = "Collegejaar"
    examtype_key = "Examentype code"

    data.loc[(data[examtype_key] == "Pre-master"), "Aantal eerstejaars croho"] = 1

    for programme in data[programme_key].unique():
        for herkomst in data[herkomst_key].unique():
            years = data[year_key].unique()
            years = years[~np.isnan(years)]
            years = np.sort(years)
            for year in years:
                filtered = data[
                    (data[year_key] == year)
                    & (data[programme_key] == programme)
                    & (data[herkomst_key] == herkomst)
                ].copy()

                if not volume:
                    filtered = filtered[filtered["Aantal eerstejaars croho"] == 1]
                    filtered = filtered[
                        filtered[examtype_key].isin(
                            ["Bachelor eerstejaars", "Master", "Pre-master"]
                        )
                    ]
                else:
                    filtered = filtered[
                        filtered[examtype_key].isin(
                            ["Bachelor eerstejaars", "Master", "Pre-master", "Bachelor hogerejaars"]
                        )
                    ]

                filtered.loc[
                    filtered[examtype_key].str.contains("Bachelor"), examtype_key
                ] = "Bachelor"

                for examtype in filtered[examtype_key].unique():
                    student_count = np.sum(
                        filtered[filtered[examtype_key] == examtype]["Aantal Hoofdinschrijvingen"]
                    )
                    if student_count > 0:
                        result["Collegejaar"].append(year)
                        result["Croho groepeernaam"].append(programme)
                        result["Herkomst"].append(herkomst)
                        result["Aantal_studenten"].append(student_count)
                        result["Examentype"].append(examtype)

    return pd.DataFrame(result)


def _calculate_student_counts(path_october, cwd, oktober_columns=None):
    """Calculate student count files from the telbestand studenten.

    oktober_columns maps canonical column names to institution-specific names,
    matching the structure of configuration["columns"]["oktober"]. Columns are
    renamed to canonical names before processing so downstream logic stays clean.
    """
    data = pd.read_excel(path_october)
    if oktober_columns:
        rename_map = {inst: canonical for canonical, inst in oktober_columns.items() if inst != canonical}
        if rename_map:
            data = data.rename(columns=rename_map)

    # First-years
    dict_count = _calculate_student_count(data, volume=False)
    result = dict_count[dict_count["Aantal_studenten"] > 0]
    result.to_excel(os.path.join(cwd, "data", "input", "student_count_first-years.xlsx"), index=False)

    # Volume
    dict_volume = _calculate_student_count(data, volume=True)
    result_volume = dict_volume[dict_volume["Aantal_studenten"] > 0]
    result_volume.to_excel(os.path.join(cwd, "data", "input", "student_volume.xlsx"), index=False)

    # Higher-years (volume minus first-years)
    dict_higher = dict_volume.copy()
    for idx, row in result_volume.iterrows():
        match_row = result[
            (result["Collegejaar"] == row["Collegejaar"])
            & (result["Croho groepeernaam"] == row["Croho groepeernaam"])
            & (result["Herkomst"] == row["Herkomst"])
            & (result["Examentype"] == row["Examentype"])
        ]
        if not match_row.empty:
            dict_higher.at[idx, "Aantal_studenten"] = (
                row["Aantal_studenten"] - match_row["Aantal_studenten"].values[0]
            )

    result_higher = dict_higher[dict_higher["Aantal_studenten"] > 0]
    result_higher.to_excel(os.path.join(cwd, "data", "input", "student_count_higher-years.xlsx"), index=False)


def _copy_direct_files(cwd, paths, output_individual):
    """Copy files that need no transformation from input_raw to input."""
    copied = []

    copies = [
        (paths.get("path_raw_individueel", "data/input_raw/individuele_aanmelddata.csv"), output_individual),
    ]

    for src_rel, dst_abs in copies:
        src = os.path.join(cwd, src_rel)
        if os.path.exists(src):
            shutil.copy2(src, dst_abs)
            copied.append(os.path.relpath(dst_abs, cwd))

    return copied
