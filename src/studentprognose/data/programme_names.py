"""Opleidingsnamen opzoeken bij CROHO-/Isatcodes via de DUO-decodeertabel.

Sinds de Isatcode-migratie (#231/#232) is de programmesleutel een numerieke
code. Levert de bron geen naamkolom, dan bevat de naamkolom alleen die code
(placeholder uit de ETL). Deze module zoekt de officiële naam op in
``Dec_opleidingscode.asc`` uit de 1CijferHO-levering van DUO.

De code blijft altijd de sleutel; de naam is alleen een etiket voor mensen.
Namen zijn niet uniek (bijv. "B Psychologie" = 34930 (hbo) én 56604 (wo)), dus
er wordt nooit op naam gekoppeld of gefilterd. De DUO-naam wordt ongewijzigd
overgenomen (inclusief voorvoegsel zoals ``B ``, ``M `` of ``Ad ``).
"""

from __future__ import annotations

import os
import warnings
from importlib.resources import files

import pandas as pd

BUNDLED_FILENAME = "Dec_opleidingscode.asc"

# Vaste breedte volgens Bestandsbeschrijving_Dec-bestanden.txt (DUO):
# Opleidingscode op positie 1-5, Naam opleiding op positie 6-185.
_CODE_SLICE = slice(0, 5)
_NAME_SLICE = slice(5, 185)

# Maximaal aantal codes dat in de waarschuwing wordt uitgeschreven.
_MAX_CODES_IN_WARNING = 10


def _decode(raw: bytes) -> str:
    """Decodeer het bestand: UTF-8 als dat kan, anders cp1252 (DUO-export)."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def parse_programme_names(text: str) -> dict[int, str]:
    """Lees de inhoud van een ``Dec_opleidingscode``-bestand.

    Alleen de opvulspaties en het regeleinde worden van de naam gehaald; verder
    blijft de naam precies zoals DUO hem levert.

    Args:
        text: Inhoud van het bestand (vaste breedte, één opleiding per regel).

    Returns:
        Een dict van opleidingscode (``int``) naar officiële naam. Regels zonder
        numerieke code of zonder naam worden overgeslagen.
    """
    names: dict[int, str] = {}
    for line in text.splitlines():
        code = line[_CODE_SLICE].strip()
        name = line[_NAME_SLICE].rstrip()
        if code.isdigit() and name.strip():
            names[int(code)] = name
    return names


def load_programme_names(path: str | None = None) -> dict[int, str]:
    """Laad de opzoektabel van opleidingscode naar naam.

    Args:
        path: Pad naar een eigen ``Dec_opleidingscode``-bestand. Leeg of
            ``None`` betekent: de met het package meegeleverde DUO-tabel. Bestaat
            het opgegeven bestand niet, dan volgt een waarschuwing en wordt de
            meegeleverde tabel gebruikt.

    Returns:
        Een dict van opleidingscode (``int``) naar officiële naam.
    """
    if path:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return parse_programme_names(_decode(f.read()))
        warnings.warn(
            f"Bestand met opleidingsnamen niet gevonden: {path!r}. "
            "De meegeleverde DUO-tabel wordt gebruikt.",
            stacklevel=2,
        )
    raw = files("studentprognose.configuration").joinpath(BUNDLED_FILENAME).read_bytes()
    return parse_programme_names(_decode(raw))


def _to_code(series: pd.Series) -> pd.Series:
    """Zet een kolom om naar opleidingscodes (``Int64``); niet-codes worden ``<NA>``.

    Accepteert getallen, tekst (``"34808"``) en floats zonder decimalen
    (``34808.0``). Lege waarden en tekst als ``"B Psychologie"`` worden ``<NA>``.
    """
    if pd.api.types.is_numeric_dtype(series):
        numbers = pd.to_numeric(series, errors="coerce").astype("Float64")
    else:
        numbers = pd.to_numeric(series.astype("string").str.strip(), errors="coerce")
    whole = numbers.notna() & (numbers % 1 == 0)
    return numbers.where(whole).astype("Int64")


def fill_programme_names(
    df: pd.DataFrame,
    code_column: str,
    name_column: str,
    names: dict[int, str],
) -> list[int]:
    """Vul de naamkolom aan met de officiële DUO-naam, in place.

    Per rij geldt de volgorde:

    1. de eigen naam uit de bron, als die er is en niet gelijk is aan de code;
    2. de DUO-naam bij de code;
    3. de code zelf (de naamkolom blijft dan zoals hij was).

    De codekolom wordt niet aangeraakt. Ontbreekt een van beide kolommen, dan
    gebeurt er niets.

    Args:
        df: De data, wordt in place aangepast.
        code_column: Kolom met de opleidingscode (bijv. ``Croho``).
        name_column: Kolom met de opleidingsnaam (bijv.
            ``Naam Croho opleiding Nederlands``).
        names: Opzoektabel uit :func:`load_programme_names`.

    Returns:
        De gesorteerde, unieke codes die geen eigen naam hadden en ook niet in
        de opzoektabel staan.
    """
    if df is None or code_column not in df.columns or name_column not in df.columns:
        return []

    code = _to_code(df[code_column])
    name = df[name_column]
    if pd.api.types.is_numeric_dtype(name):
        is_empty = name.isna()
    else:
        name_text = name.astype("string").str.strip()
        is_empty = name_text.isna() | (name_text == "")
    # Geen eigen naam = leeg, of gelijk aan de code (placeholder uit de ETL).
    no_own_name = is_empty | (_to_code(name) == code)
    needs_name = code.notna() & no_own_name.fillna(False)

    duo_name = code.map(names)
    fill = needs_name & duo_name.notna()
    if fill.any():
        # Naamkolom wordt tekst: een kolom met alleen codes (placeholder) mag geen
        # getaltype houden nu er namen in komen.
        df[name_column] = df[name_column].astype(object)
        df.loc[fill, name_column] = duo_name[fill]

    missing = code[needs_name & duo_name.isna()].unique()
    return sorted(int(c) for c in missing)


def warn_missing_programme_names(missing: list[int]) -> None:
    """Geef één waarschuwing voor codes zonder naam (geen fout)."""
    if not missing:
        return
    shown = ", ".join(str(c) for c in missing[:_MAX_CODES_IN_WARNING])
    extra = len(missing) - _MAX_CODES_IN_WARNING
    suffix = f" en {extra} meer" if extra > 0 else ""
    warnings.warn(
        f"Geen opleidingsnaam gevonden voor {len(missing)} code(s): {shown}{suffix}. "
        "Voor deze opleidingen wordt de code getoond. Werk zo nodig het bestand "
        "met opleidingsnamen bij (paths.path_programme_names).",
        stacklevel=2,
    )
