"""Canonieke normalisatie van de programmesleutel (CROHO/Isatcode).

De programmesleutel (kolom ``Croho groepeernaam``) is sinds de Isatcode-migratie
(#231/#232) een numerieke CROHO-code. Hij moet als één dtype door de hele
pipeline lopen, anders falen merges en ``.isin``-filters stil (of crashen) op
gemengde types. Numerieke sleutels worden ``Int64`` (nullable: NaN-veilig én
zonder float-``.0``-staart); legacy leesbare namen (bijv. ``"B Psychologie"``)
blijven string.

Dezelfde regel wordt toegepast op de config-zijde van elke vergelijking
(``numerus_fixus``-keys, ``programme_filtering``), zodat beide kanten van een
``.isin``/merge gegarandeerd hetzelfde type hebben.
"""

from __future__ import annotations

import pandas as pd


def normalize_programme_series(series: pd.Series | None) -> pd.Series | None:
    """Normaliseer een programmesleutel-kolom naar één canoniek dtype.

    Args:
        series: De te normaliseren kolom, of ``None``.

    Returns:
        ``Int64`` als de kolom volledig numeriek is (CROHO/Isatcode), anders de
        kolom als ``str`` (legacy leesbare namen). ``None`` blijft ``None``.
    """
    if series is None:
        return None
    coerced = pd.to_numeric(series, errors="coerce")
    # Volledig numeriek (coercion introduceert geen nieuwe NaN's t.o.v. de bron)
    # => CROHO/Isatcode. Int64 is NaN-veilig en heeft geen float-`.0`-staart.
    if coerced.notna().equals(series.notna()):
        return coerced.astype("Int64")
    return series.astype(str)


def normalize_programme_value(value):
    """Normaliseer één sleutelwaarde, passend bij :func:`normalize_programme_series`.

    Een puur-numerieke sleutel (Isatcode) wordt ``int``; een leesbare naam blijft
    string. Zo matcht een geconfigureerde sleutel (``numerus_fixus``-key,
    ``programme_filtering``-waarde) een ``Int64``-kolom respectievelijk een
    string-kolom.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def normalize_programme_values(values):
    """Normaliseer een lijst sleutelwaarden (bijv. ``programme_filtering``)."""
    if values is None:
        return values
    return [normalize_programme_value(v) for v in values]


def normalize_programme_keys(mapping):
    """Normaliseer de keys van een sleutel-gemapte dict (bijv. ``numerus_fixus``)."""
    if not mapping:
        return mapping
    return {normalize_programme_value(k): v for k, v in mapping.items()}


_PROGRAMME_KEY = "Croho groepeernaam"


def merge_on_programme_key(left, right, on, how="left", suffixes=("_x", "_y"), key=_PROGRAMME_KEY):
    """Merge twee frames op een sleutelset, met uitgelijnd dtype voor de
    programmesleutel.

    Het individuele spoor keyt (nog) op leesbare opleidingsnamen, het cumulatieve
    en het label op numerieke isatcodes (Int64). Die sleutelruimtes overlappen
    momenteel niet, dus een left-join die ze combineert voegt op de
    programmesleutel niets toe (0 matches) — een pre-existing eigenschap van de
    Isatcode-migratie, los van het sleutel-dtype (zie
    ``docs/methodologie/individueel.md`` en issue #238). Zonder dtype-uitlijning
    crasht pandas echter op str-vs-Int64. We lijnen de sleutel daarom uit op een
    gemeenschappelijke string wanneer de dtypes verschillen; bij gelijk dtype
    blijft het een echte (numerieke) join die vanzelf matcht.
    """
    if key in left.columns and key in right.columns and left[key].dtype != right[key].dtype:
        left = left.assign(**{key: left[key].astype(str)})
        right = right.assign(**{key: right[key].astype(str)})
    return left.merge(right, on=on, how=how, suffixes=suffixes)
