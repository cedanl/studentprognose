"""Lezen, valideren en schrijven van de filtering-JSON, plus filterstatistiek.

Pure logica (geen NiceGUI). De filterstatistiek (:func:`count_programmes`) telt
hoeveel opleidingen na de filters overblijven, zodat de editor een live preview
kan tonen zonder de pipeline te draaien.
"""

from __future__ import annotations

import json

import pandas as pd

from studentprognose.utils.programme_key import (
    normalize_programme_series,
    normalize_programme_values,
)

#: Toegestane herkomst- en examentype-waarden (spiegelen de pipeline).
HERKOMST_CHOICES = ["NL", "EER", "Niet-EER"]
EXAMENTYPE_CHOICES = ["Bachelor", "Master", "Pre-master"]

#: Lege filtering = geen filters (alle data).
DEFAULT_FILTERING = {"filtering": {"programme": [], "herkomst": [], "examentype": []}}


def isatcode_str(value) -> str:
    """Canonieke string-weergave van een programmesleutel (isatcode of naam).

    De programmesleutel is sinds de isatcode-migratie een numerieke CROHO-code.
    Ingelezen uit Excel/CSV kan die als ``int`` (``30008``) of als ``float``
    (``30008.0``) binnenkomen; beide moeten dezelfde canonieke sleutel opleveren
    zodat dropdown-waarden, opgeslagen filters en de datakolom matchen. Legacy
    leesbare namen (``"B Psychologie"``) blijven ongewijzigd.

    Args:
        value: Een ruwe sleutelwaarde (int, float, str of ``None``).

    Returns:
        De sleutel als string; ``""`` voor ``None``/leeg.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    # "30008.0" -> "30008": isatcode die als float is ingelezen.
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def programme_name_map(df: pd.DataFrame, *, code_col: str, name_col: str) -> dict[str, str]:
    """Bouw een ``{isatcode_str: opleidingsnaam}``-map uit een bronframe.

    Args:
        df: Bronframe met zowel de isatcode- als de naamkolom.
        code_col: Kolomnaam met de isatcode (programmesleutel).
        name_col: Kolomnaam met de leesbare opleidingsnaam.

    Returns:
        Mapping van canonieke isatcode-string naar naam; leeg als een kolom
        ontbreekt.
    """
    if code_col not in df.columns or name_col not in df.columns:
        return {}
    pairs = df[[code_col, name_col]].dropna().drop_duplicates()
    mapping: dict[str, str] = {}
    for code, name in zip(pairs[code_col], pairs[name_col]):
        key = isatcode_str(code)
        if key:
            mapping.setdefault(key, str(name).strip())
    return mapping


def build_programme_options(codes, name_map: dict[str, str] | None = None) -> dict[str, str]:
    """Bouw een ``{isatcode_str: label}``-dropdownmap, numeriek gesorteerd.

    Het label is ``"<code> — <naam>"`` als de naam bekend is, anders alleen de
    code. De dict-vorm laat een NiceGUI-``select`` de leesbare labels tonen maar
    de isatcode als waarde opslaan.

    Args:
        codes: Iterable met ruwe isatcodes (bijv. de programmakolom van het
            student_count-frame, plus reeds geconfigureerde sleutels).
        name_map: Optionele ``{isatcode_str: naam}``-verrijking.

    Returns:
        Geordende ``{isatcode_str: label}``-map (dubbelen samengevoegd).
    """
    name_map = name_map or {}
    normalized = {
        key for key in (isatcode_str(c) for c in codes) if key
    }

    def _sort_key(code: str):
        try:
            return (0, int(code), "")
        except ValueError:
            return (1, 0, code)

    options: dict[str, str] = {}
    for code in sorted(normalized, key=_sort_key):
        name = name_map.get(code)
        options[code] = f"{code} — {name}" if name else code
    return options


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
        # De datakolom en de opgeslagen filters kunnen in dtype verschillen
        # (Int64-isatcode vs. string): normaliseer beide kanten via dezelfde
        # pipeline-regel zodat de match niet stil leegloopt op int-vs-str.
        col_norm = normalize_programme_series(df[programme_col])
        mask &= col_norm.isin(normalize_programme_values(programme))
    if herkomst and origin_col in df.columns:
        mask &= df[origin_col].isin(herkomst)
    if examentype and exam_col in df.columns:
        mask &= df[exam_col].isin(examentype)

    remaining = int(df.loc[mask, programme_col].nunique())
    return remaining, total
