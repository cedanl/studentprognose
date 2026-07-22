"""Lezen, valideren en schrijven van de filtering-JSON, plus filterstatistiek.

Pure logica (geen NiceGUI). De filterstatistiek (:func:`count_programmes`) telt
hoeveel opleidingen na de filters overblijven, zodat de editor een live preview
kan tonen zonder de pipeline te draaien.
"""

from __future__ import annotations

import json

import pandas as pd

#: Toegestane herkomst- en examentype-waarden (spiegelen de pipeline).
HERKOMST_CHOICES = ["NL", "EER", "Niet-EER"]
EXAMENTYPE_CHOICES = ["Bachelor", "Master", "Pre-master"]

#: Lege filtering = geen filters (alle data).
DEFAULT_FILTERING = {"filtering": {"programme": [], "herkomst": [], "examentype": []}}


def load_filtering(path: str) -> dict:
    """Laad een filtering-JSON; val terug op de default-structuur bij ontbreken."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return json.loads(json.dumps(DEFAULT_FILTERING))
    # Zorg dat de verwachte sleutels bestaan.
    filtering = data.setdefault("filtering", {})
    for key in ("programme", "herkomst", "examentype"):
        filtering.setdefault(key, [])
    return data


def save_filtering(path: str, data: dict) -> None:
    """Schrijf de filtering als nette JSON (UTF-8, 4 spaties)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write("\n")


def validate_filtering(data: dict) -> list[str]:
    """Controleer dat herkomst/examentype alleen geldige waarden bevatten.

    Returns:
        Lijst met foutmeldingen (leeg = geldig).
    """
    errors: list[str] = []
    filtering = data.get("filtering", {})

    bad_herkomst = set(filtering.get("herkomst", [])) - set(HERKOMST_CHOICES)
    if bad_herkomst:
        errors.append(
            f"Ongeldige herkomst-waarde(n): {sorted(bad_herkomst)}. "
            f"Toegestaan: {HERKOMST_CHOICES}."
        )

    bad_examentype = set(filtering.get("examentype", [])) - set(EXAMENTYPE_CHOICES)
    if bad_examentype:
        errors.append(
            f"Ongeldige examentype-waarde(n): {sorted(bad_examentype)}. "
            f"Toegestaan: {EXAMENTYPE_CHOICES}."
        )

    return errors


def count_programmes(
    df: pd.DataFrame,
    *,
    programme_col: str,
    origin_col: str,
    exam_col: str,
    programme: list[str],
    herkomst: list[str],
    examentype: list[str],
) -> tuple[int, int]:
    """Tel opleidingen vóór en na toepassing van de filters.

    Een lege filterlijst betekent "geen filter op die dimensie" (alle waarden).

    Args:
        df: Het student_count-DataFrame.
        programme_col: Kolomnaam met de opleiding (bijv. ``Croho groepeernaam``).
        origin_col: Kolomnaam met de herkomst.
        exam_col: Kolomnaam met het examentype.
        programme: Geselecteerde opleidingen (leeg = alle).
        herkomst: Geselecteerde herkomsten (leeg = alle).
        examentype: Geselecteerde examentypes (leeg = alle).

    Returns:
        ``(overgebleven, totaal)`` — het aantal unieke opleidingen na en vóór de
        filters.
    """
    total = int(df[programme_col].nunique())

    mask = pd.Series(True, index=df.index)
    if programme:
        mask &= df[programme_col].isin(programme)
    if herkomst and origin_col in df.columns:
        mask &= df[origin_col].isin(herkomst)
    if examentype and exam_col in df.columns:
        mask &= df[exam_col].isin(examentype)

    remaining = int(df.loc[mask, programme_col].nunique())
    return remaining, total
