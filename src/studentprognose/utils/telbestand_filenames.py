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
        year = int(match.group("year"))
        week = int(match.group("week"))

Placeholder-syntax
------------------
``{year}`` en ``{week}`` zijn de enige placeholders. Alle andere karakters in
het patroon worden letterlijk gematcht (punten, streepjes, underscores zijn
veilig). Voorbeelden:

- ``telbestandY{year}W{week}`` (default, Studielink)
- ``VU_telbestand_{year}_W{week}``
- ``tel.{year}.{week}``
"""

from __future__ import annotations

import re
from typing import NamedTuple

DEFAULT_TELBESTAND_PATTERN = "telbestandY{year}W{week}"

_YEAR_REGEX = r"(?P<year>\d{4})"
_WEEK_REGEX = r"(?P<week>\d+)"
_PLACEHOLDER_TOKEN = re.compile(r"\{(year|week)\}")


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
        Als ``{year}`` of ``{week}`` ontbreekt in het patroon.
    """
    if "{year}" not in pattern or "{week}" not in pattern:
        raise ValueError(
            f"Telbestand-patroon {pattern!r} mist de placeholders "
            f"{{year}} en/of {{week}}."
        )

    parts: list[str] = []
    cursor = 0
    for match in _PLACEHOLDER_TOKEN.finditer(pattern):
        parts.append(re.escape(pattern[cursor : match.start()]))
        parts.append(_YEAR_REGEX if match.group(1) == "year" else _WEEK_REGEX)
        cursor = match.end()
    parts.append(re.escape(pattern[cursor:]))
    return "".join(parts)


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
        raw_patterns: list[str] = [DEFAULT_TELBESTAND_PATTERN]
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
