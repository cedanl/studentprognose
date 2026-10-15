import os
import shutil

import numpy as np
import pandas as pd

from studentprognose.config import get_final_academic_week
from studentprognose.utils.constants import FINAL_ACADEMIC_WEEK
from studentprognose.utils.telbestand_filenames import (
    compile_patterns,
    match_telbestand,
    week_from_match,
)
from studentprognose.utils.weeks import get_all_weeks_ordered

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
        _interpolate_missing_weeks(
            output_cumulative,
            get_final_academic_week(configuration),
        )
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
    # Pas instelling-specifieke kolomnamen toe → canonieke telbestand-namen.
    # Conventie spiegelt columns.oktober: canonical → institution in config.
    tel_col_map = (configuration or {}).get("columns", {}).get("telbestand", {})
    if tel_col_map:
        institution_to_canonical = {
            inst: canon for canon, inst in tel_col_map.items() if inst != canon
        }
        if institution_to_canonical:
            data.rename(columns=institution_to_canonical, inplace=True)

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


_INTERPOLATED_MEASURES = [
    "Gewogen vooraanmelders", "Ongewogen vooraanmelders",
    "Aantal aanmelders met 1 aanmelding", "Inschrijvingen",
]


def _interpolate_missing_weeks(csv_path, final_week: int = FINAL_ACADEMIC_WEEK):
    """Vul ontbrekende weken binnen een collegejaar aan in het cumulatieve bestand.

    Per reeks (alle identiteitskolommen + ``Collegejaar``) wordt gekeken welke
    weken ontbreken **tussen** de eerste en laatste aangeleverde week, in de
    volgorde van het academisch jaar (``final_week + 1`` → 52 → ``final_week``).
    Voor die weken wordt een nieuwe rij toegevoegd met lineair geïnterpoleerde
    aantallen. Weken vóór de eerste of ná de laatste aangeleverde week worden
    niet aangevuld: daar is geen waarneming om tussen te interpoleren (vóór de
    eerste snapshot waren er bijvoorbeeld nog geen aanmelders). ISO-week 53 valt
    buiten het 52-weeks rooster en blijft ongemoeid.

    Alleen eerstejaars-rijen (``Herinschrijving`` en ``Hogerejaars`` == "Nee")
    worden aangevuld, zoals voorheen.
    """
    data = pd.read_csv(csv_path, sep=";", low_memory=True)

    # Convert Dutch numeric formats
    for col in _INTERPOLATED_MEASURES:
        data[col] = data[col].apply(_convert_to_float)

    added = _interpolated_rows(data, final_week)
    if not added.empty:
        data = pd.concat([data, added[data.columns]], ignore_index=True)
        print(f"  {len(added)} ontbrekende weekrij(en) aangevuld door interpolatie")
    else:
        print("  No gaps found to interpolate")

    # Round results
    data["Gewogen vooraanmelders"] = data["Gewogen vooraanmelders"].round(2)
    for col in ["Ongewogen vooraanmelders", "Aantal aanmelders met 1 aanmelding", "Inschrijvingen"]:
        data[col] = data[col].where(data[col].notna(), np.nan)
        data[col] = np.round(data[col]).astype("Int64")

    data.to_csv(csv_path, sep=";", index=False)


def _interpolated_rows(data: pd.DataFrame, final_week: int) -> pd.DataFrame:
    """Nieuwe rijen voor ontbrekende weken tussen twee aangeleverde weken.

    Returns:
        DataFrame met dezelfde kolommen als ``data`` (leeg als er niets ontbreekt).
    """
    order = [int(w) for w in get_all_weeks_ordered(final_week)]
    position = {week: i for i, week in enumerate(order)}

    subset = data[
        (data["Herinschrijving"] == "Nee")
        & (data["Hogerejaars"] == "Nee")
        & data["Weeknummer"].isin(position)
    ]
    if subset.empty:
        return data.iloc[0:0]

    key_cols = [
        c for c in data.columns
        if c not in _INTERPOLATED_MEASURES and c not in ("Weeknummer", "Weeknummer rapportage")
    ]
    # NaN-sleutels (bijv. een lege Faculteit) tijdelijk vervangen: unstack kan
    # NaN niet als indexwaarde gebruiken.
    na_token = "__ontbreekt__"
    keys = subset[key_cols].astype(object).where(subset[key_cols].notna(), na_token)
    frame = pd.concat([keys, subset[_INTERPOLATED_MEASURES]], axis=1)
    frame["_pos"] = subset["Weeknummer"].map(position)

    grouped = frame.groupby(key_cols + ["_pos"], sort=False)
    present = grouped.size().unstack("_pos")
    measures = grouped[_INTERPOLATED_MEASURES].sum(min_count=1)

    # Alleen posities tussen de eerste en laatste aangeleverde week per reeks.
    present = present.reindex(columns=range(len(order)))
    has_row = present.notna()
    inside = has_row.cummax(axis=1) & has_row[has_row.columns[::-1]].cummax(axis=1)[has_row.columns]
    missing = inside & ~has_row
    if not missing.to_numpy().any():
        return data.iloc[0:0]

    new = missing.stack()
    new = new[new].index.to_frame(index=False)
    for col in _INTERPOLATED_MEASURES:
        wide = measures[col].unstack("_pos").reindex(columns=range(len(order)))
        wide = wide.interpolate(axis=1, limit_area="inside")
        values = wide.stack(future_stack=True).rename(col).reset_index()
        new = new.merge(values, on=key_cols + ["_pos"], how="left")

    new["Weeknummer"] = new["_pos"].map(dict(enumerate(order)))
    new["Weeknummer rapportage"] = new["Weeknummer"]
    new = new.drop(columns="_pos")
    for col in key_cols:
        new[col] = new[col].where(new[col] != na_token, np.nan)
        new[col] = new[col].astype(data[col].dtype, errors="ignore")
    return new[data.columns]


def _convert_to_float(value):
    if isinstance(value, str):
        value = value.replace(".", "").replace(",", ".")
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


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
