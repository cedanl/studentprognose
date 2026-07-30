"""Configureerbaar matchen van telbestand-bestandsnamen.

Instellingen leveren telbestanden aan met afwijkende naamconventies (Studielink
gebruikt ``telbestandY2024W10.csv``, andere instellingen mogelijk
``VU_telbestand_2024_W10.csv``). Deze module vertaalt simpele
placeholder-patronen uit ``configuration.json`` naar regex-objecten zonder dat
de gebruiker regex-syntax hoeft te kennen.

Gebruik
-------
.. code-block:: python

    patterns = compile_patterns(configuration)
    match = match_telbestand("telbestandY2024W10.csv", patterns)
    if match:
        week = week_from_match(match)

Placeholder-syntax
------------------
Ondersteunde placeholders: ``{year}``, ``{week}``, ``{date}`` (leverdatum
``YYYYMMDD``) en ``{volgnummer}`` (Studielink-volgnummer). Alle andere karakters
in het patroon worden letterlijk gematcht (punten, streepjes, underscores zijn
veilig). Elk patroon bevat ``{year}`` en moet een week kunnen opleveren: het
bevat dus ``{week}`` **of** ``{date}``. Bij ``{date}`` wordt het weeknummer als
ISO-kalenderweek uit de leverdatum afgeleid (zie :func:`week_from_match`).
Voorbeelden:

- ``telbestandY{year}W{week}`` (default, Studielink instellingsformaat)
- ``VU_telbestand_{year}_W{week}``
- ``telbestand_sl_{date}_v{volgnummer}_{year}`` (UvA SQL-telbestand)
"""

from __future__ import annotations

import datetime
import re
from typing import NamedTuple

#: Beide instellingsformaten worden standaard herkend wanneer er geen configuratie is.
DEFAULT_TELBESTAND_PATTERNS = [
    "telbestandY{year}W{week}",
    "telbestand_sl_{date}_v{volgnummer}_{year}",
]
# Alias voor backwards-compatibiliteit met bestaande imports.
DEFAULT_TELBESTAND_PATTERN = DEFAULT_TELBESTAND_PATTERNS[0]

# Placeholder -> named-group regex. ``year``/``date`` hebben een vaste lengte
# zodat ze ondubbelzinnig matchen naast ``volgnummer``/``week`` (variabel).
_GROUP_REGEX = {
    "year": r"(?P<year>\d{4})",
    "week": r"(?P<week>\d+)",
    "date": r"(?P<date>\d{8})",
    "volgnummer": r"(?P<volgnummer>\d+)",
}
_PLACEHOLDER_TOKEN = re.compile(r"\{(year|week|date|volgnummer)\}")


class TelbestandPattern(NamedTuple):
    """Een gecompileerd telbestand-patroon, met behoud van de leesbare bron.

    ``raw`` is wat de gebruiker in ``configuration.json`` heeft geschreven
    (bijv. ``"telbestandY{year}W{week}"``) en wordt gebruikt in foutmeldingen.
    ``regex`` is de gecompileerde versie met named groups ``year`` en ``week``.
    """

    raw: str
    regex: re.Pattern[str]


def _placeholder_to_regex(pattern: str) -> str:
    """Vertaal een placeholder-patroon naar een regex-string.

    Raises
    ------
    ValueError
        Als ``{year}`` ontbreekt, of als het patroon geen week kan opleveren
        (zowel ``{week}`` als ``{date}`` ontbreekt).
    """
    if "{year}" not in pattern:
        raise ValueError(
            f"Telbestand-patroon {pattern!r} mist een van de vereiste "
            f"placeholders: {{year}}."
        )
    if "{week}" not in pattern and "{date}" not in pattern:
        raise ValueError(
            f"Telbestand-patroon {pattern!r} mist een van de vereiste "
            f"placeholders: {{week}} of {{date}} (nodig om het weeknummer af "
            f"te leiden)."
        )

    parts: list[str] = []
    cursor = 0
    for match in _PLACEHOLDER_TOKEN.finditer(pattern):
        parts.append(re.escape(pattern[cursor : match.start()]))
        parts.append(_GROUP_REGEX[match.group(1)])
        cursor = match.end()
    parts.append(re.escape(pattern[cursor:]))
    return "".join(parts)


def week_from_match(match: re.Match[str]) -> int:
    """Leid het ISO-weeknummer af uit een telbestand-match.

    Een patroon met ``{week}`` levert de week direct; een patroon met ``{date}``
    (leverdatum ``YYYYMMDD``) levert de **ISO-kalenderweek** van die datum. Dit
    sluit aan op de bestaande ``W{week}``-conventie: de UvA-leverdatum
    ``20260525`` (volgnummer V34, studiejaar 2026) valt op ISO-week 22.

    Raises
    ------
    ValueError
        Als de match geen ``week``- noch een geldige ``date``-groep bevat.
    """
    groups = match.groupdict()
    week = groups.get("week")
    if week is not None:
        return int(week)

    date = groups.get("date")
    if date is not None:
        leverdatum = datetime.datetime.strptime(date, "%Y%m%d").date()
        return leverdatum.isocalendar()[1]

    raise ValueError(
        "Telbestand-match bevat geen 'week'- of 'date'-groep; kan geen "
        "weeknummer afleiden."
    )


def compile_patterns(configuration: dict | None) -> list[TelbestandPattern]:
    """Compileer telbestand-patronen uit configuratie.

    Accepteert ``None``, een lege of ontbrekende sleutel, een string, of een
    lijst van strings. Een lege lijst valt terug op het defaultpatroon.

    Raises
    ------
    ValueError
        Als een patroon de placeholders ``{year}`` of ``{week}`` mist.
    """
    raw = (configuration or {}).get("telbestand_filename_patterns")

    if raw is None or raw == [] or raw == "":
        raw_patterns: list[str] = list(DEFAULT_TELBESTAND_PATTERNS)
    elif isinstance(raw, str):
        raw_patterns = [raw]
    else:
        raw_patterns = list(raw)

    return [
        TelbestandPattern(raw=p, regex=re.compile(_placeholder_to_regex(p)))
        for p in raw_patterns
    ]


def match_telbestand(
    filename: str, compiled_patterns: list[TelbestandPattern]
) -> re.Match[str] | None:
    """Match een bestandsnaam tegen de gecompileerde patronen.

    Returns het eerste match-object (met named groups ``year`` en ``week``)
    of ``None`` als geen patroon matcht.
    """
    for pattern in compiled_patterns:
        match = pattern.regex.search(filename)
        if match is not None:
            return match
    return None
