"""Configurable filename matching for Studielink telbestanden.

Instellingen hanteren niet altijd hetzelfde bestandsnaampatroon
(bijv. ``telbestandY2024W10.csv`` versus eigen Timo- of VU-naamgeving).
Daarom worden patronen via ``configuration.json`` opgegeven onder de
sleutel ``telbestand_filename_patterns`` (lijst). Elk patroon is een
regex met de named groups ``year`` en ``week``.

Voorbeeldoverride in ``configuration.json``::

    {
        "telbestand_filename_patterns": [
            "telbestandY(?P<year>\\d{4})W(?P<week>\\d{1,2})",
            "VU_telbestand_(?P<year>\\d{4})_W(?P<week>\\d{1,2})"
        ]
    }
"""

import re
from collections.abc import Iterable

DEFAULT_TELBESTAND_PATTERN = r"telbestandY(?P<year>\d{4})W(?P<week>\d{1,2})"


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


def compile_patterns(configuration):
    """Compile alle telbestand-naampatronen uit de configuratie.

    Onbruikbare patronen (verkeerde syntax of zonder ``year`` en ``week``
    named groups) leiden tot een ``ValueError`` met een duidelijke melding,
    zodat de gebruiker dit kan corrigeren in zijn configuratiebestand.
    """
    compiled = []
    for raw in _get_pattern_strings(configuration):
        try:
            pat = re.compile(raw)
        except re.error as exc:
            raise ValueError(
                f"Ongeldig regex-patroon voor telbestand: {raw!r} ({exc})"
            ) from exc
        if "year" not in pat.groupindex or "week" not in pat.groupindex:
            raise ValueError(
                f"Telbestand-patroon {raw!r} mist de named groups "
                f"(?P<year>...) en/of (?P<week>...)."
            )
        compiled.append(pat)
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
