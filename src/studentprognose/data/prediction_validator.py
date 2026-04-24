"""Pre-prediction datakwaliteitschecks.

Drie checks worden uitgevoerd op de cumulatieve data vóór de modellen draaien:

  decimaalintegriteit – hard stop als 'Gewogen vooraanmelders' strings met
                        komma's of niet-numerieke waarden bevat.
  lege dataset        – hard stop als er geen data is voor de gevraagde week.
  historisch realisme – vergelijkt de huidige week met dezelfde week een jaar
                        eerder. Hard stop bij extreme afwijking, warning bij
                        matige afwijking. NF-opleidingen worden overgeslagen.

Drempelwaarden (niet configureerbaar, bewust conservatief):
  warning  : afwijking > max(15 absoluut, 30% relatief)
  hard stop: afwijking > max(25 absoluut, 70% relatief)
"""

import sys
import warnings

import pandas as pd


def run_pre_prediction_checks(
    data_cumulative: pd.DataFrame,
    predict_year: int,
    predict_week: int,
    numerus_fixus_list: dict,
    yes: bool = False,
) -> None:
    """Run all pre-prediction checks. Hard stops exit the process."""
    current = data_cumulative[
        (data_cumulative["Collegejaar"] == predict_year)
        & (data_cumulative["Weeknummer"] == predict_week)
    ]
    last_year = data_cumulative[
        (data_cumulative["Collegejaar"] == predict_year - 1)
        & (data_cumulative["Weeknummer"] == predict_week)
    ]

    # Decimal and empty checks are unconditional hard stops: corrupted or missing
    # data means the model cannot run at all — there is no safe value to fall back on.
    # Historical realism is a domain judgment (e.g. COVID years legitimately differ)
    # so --yes can demote it to a warning.
    _check_decimal_integrity(current, predict_year, predict_week)
    _check_empty_data(current, predict_year, predict_week)
    _check_historical_realism(current, last_year, numerus_fixus_list, predict_year, predict_week, yes)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_decimal_integrity(
    current: pd.DataFrame, predict_year: int, predict_week: int
) -> None:
    col = "Gewogen vooraanmelders"
    if col not in current.columns:
        return

    if pd.api.types.is_object_dtype(current[col]):
        if current[col].astype(str).str.contains(",", na=False).any():
            print(
                f"\n[HARD STOP] Decimaalfout jaar {predict_year} week {predict_week}: "
                f"'{col}' bevat komma's als decimaalscheidingsteken. "
                f"Controleer de brondata en verwerk opnieuw."
            )
            sys.exit(1)

    try:
        pd.to_numeric(current[col], errors="raise")
    except (ValueError, TypeError) as exc:
        print(
            f"\n[HARD STOP] Decimaalfout jaar {predict_year} week {predict_week}: "
            f"'{col}' bevat niet-numerieke waarden ({exc}). "
            f"Controleer de brondata en verwerk opnieuw."
        )
        sys.exit(1)


def _check_empty_data(
    current: pd.DataFrame, predict_year: int, predict_week: int
) -> None:
    if current.empty:
        print(
            f"\n[HARD STOP] Geen data voor jaar {predict_year} week {predict_week}. "
            f"Voeg de ontbrekende telbestanden toe via data/input_raw/ en verwerk opnieuw."
        )
        sys.exit(1)


def _check_historical_realism(
    current: pd.DataFrame,
    last_year: pd.DataFrame,
    numerus_fixus_list: dict,
    predict_year: int,
    predict_week: int,
    yes: bool = False,
) -> None:
    if last_year.empty:
        return

    group_cols = ["Herkomst", "Croho groepeernaam", "Examentype"]
    missing = [c for c in group_cols if c not in current.columns or c not in last_year.columns]
    if missing:
        return

    curr_agg = (
        current.groupby(group_cols)["Gewogen vooraanmelders"].sum().reset_index()
    )
    last_agg = (
        last_year.groupby(group_cols)["Gewogen vooraanmelders"].sum().reset_index()
    )
    merged = pd.merge(curr_agg, last_agg, on=group_cols, suffixes=("_curr", "_last"))

    for _, row in merged.iterrows():
        programme = row["Croho groepeernaam"]
        examentype = row["Examentype"]

        # NF-programma's hebben een bewust plafond; schommelingen zijn beleidsmatig,
        # niet een signaal van datakwaliteitsproblemen. Vals-positieve hard stops
        # vermijden.
        if programme in numerus_fixus_list and examentype == "Bachelor":
            continue

        val_curr = row["Gewogen vooraanmelders_curr"]
        val_last = row["Gewogen vooraanmelders_last"]
        abs_diff = abs(val_curr - val_last)
        label = f"{row['Herkomst']} | {programme} | {examentype}"

        # Nieuw programma of jaar zonder historische data: geen basis voor vergelijking,
        # niet per definitie een datakwaliteitsprobleem.
        if val_last <= 0:
            continue

        # max() floor voorkomt vals-positieven bij kleine opleidingen: een programma
        # met 10 studenten vorig jaar en 18 dit jaar (80% relatief, 8 absoluut) mag
        # geen hard stop triggeren. De absolute vloer beschermt kleine programma's
        # tegen relatieve drempels die op die schaal niet zinvol zijn.
        hard_threshold = max(25.0, 0.70 * val_last)
        if abs_diff > hard_threshold:
            msg = (
                f"Historisch realisme jaar {predict_year} week {predict_week}: "
                f"extreme afwijking voor {label}. "
                f"Huidig: {val_curr:.0f}, vorig jaar: {val_last:.0f}, "
                f"verschil: {abs_diff:.0f} (drempel: {hard_threshold:.0f})."
            )
            if yes:
                warnings.warn(f"[--yes] {msg}", UserWarning, stacklevel=2)
                continue
            print(f"\n[HARD STOP] {msg} Gebruik --yes om door te gaan.")
            sys.exit(1)

        warn_threshold = max(15.0, 0.30 * val_last)
        if abs_diff > warn_threshold:
            warnings.warn(
                f"Historisch realisme jaar {predict_year} week {predict_week}: "
                f"hoge afwijking voor {label}. "
                f"Huidig: {val_curr:.0f}, vorig jaar: {val_last:.0f}, "
                f"verschil: {abs_diff:.0f} (drempel: {warn_threshold:.0f}).",
                UserWarning,
                stacklevel=2,
            )
