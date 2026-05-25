"""Configurable filename matching for Studielink telbestanden.

Instellingen hanteren niet altijd hetzelfde bestandsnaampatroon
(bijv. ``telbestandY2024W10.csv`` versus eigen Timo- of VU-naamgeving).
Daarom worden patronen via ``configuration.json`` opgegeven onder de
sleutel ``telbestand_filename_patterns`` (lijst). Elk patroon is een
gewone string met twee placeholders:

- ``{year}`` — viercijferig jaartal
- ``{week}`` — week (1 of 2 cijfers)

Andere tekens in het patroon zijn letterlijke tekst die in de
bestandsnaam moet voorkomen.

Voorbeeldoverride in ``configuration.json``::

    {
        "telbestand_filename_patterns": [
            "telbestandY{year}W{week}",
            "VU_telbestand_{year}_W{week}"
        ]
    }
"""

import re
from collections.abc import Iterable

DEFAULT_TELBESTAND_PATTERN = "telbestandY{year}W{week}"

_YEAR_REGEX = r"(?P<year>\d{4})"
_WEEK_REGEX = r"(?P<week>\d{1,2})"
_PLACEHOLDER_TOKEN = re.compile(r"\{(year|week)\}")


def _get_pattern_strings(configuration):
    if configuration is None:
        return [DEFAULT_TELBESTAND_PATTERN]
    raw = configuration.get("telbestand_filename_patterns")
    if not raw:
        return [DEFAULT_TELBESTAND_PATTERN]
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, Iterable):
        raise TypeError(
            "telbestand_filename_patterns moet een string of een lijst van strings zijn"
        )
    return list(raw)


def _placeholder_to_regex(pattern):
    """Vertaal een placeholder-patroon (``telbestandY{year}W{week}``) naar regex.

    De overige tekens worden ge-escaped zodat ze letterlijk worden gematcht.
    """
    if "{year}" not in pattern or "{week}" not in pattern:
        raise ValueError(
            f"Telbestand-patroon {pattern!r} mist de placeholders "
            f"{{year}} en/of {{week}}."
        )

    parts = []
    last = 0
    for match in _PLACEHOLDER_TOKEN.finditer(pattern):
        parts.append(re.escape(pattern[last:match.start()]))
        parts.append(_YEAR_REGEX if match.group(1) == "year" else _WEEK_REGEX)
        last = match.end()
    parts.append(re.escape(pattern[last:]))
    return "".join(parts)


def compile_patterns(configuration):
    """Compile alle telbestand-naampatronen uit de configuratie.

    Onbruikbare patronen (ontbrekende ``{year}``/``{week}`` placeholders)
    leiden tot een ``ValueError`` met een duidelijke melding, zodat de
    gebruiker dit kan corrigeren in zijn configuratiebestand.
    """
    compiled = []
    for raw in _get_pattern_strings(configuration):
        regex = _placeholder_to_regex(raw)
        compiled.append(re.compile(regex))
    return compiled


def match_telbestand(filename, compiled_patterns):
    """Match ``filename`` tegen de gecompileerde patronen.

    Geeft een ``re.Match`` terug bij de eerste succesvolle match (met
    ``year`` en ``week`` named groups beschikbaar), anders ``None``.
    """
    for pat in compiled_patterns:
        match = pat.search(filename)
        if match:
            return match
    return None
