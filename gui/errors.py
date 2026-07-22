"""Vertaal ruwe fouten naar actiegerichte meldingen voor de gebruiker.

Kernregel uit het UX-design (#272): een gebruiker ziet nooit een kale Python-
traceback, maar een zin die zegt wát er misging en wélke stap het oplost. Deze
module is pure logica (geen NiceGUI) zodat ze getest kan worden.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FriendlyError:
    """Een gebruikersvriendelijke foutmelding.

    Attributes:
        message: Wat er misging, in gewone taal.
        hint: Concrete herstelstap.
    """

    message: str
    hint: str


#: (trefwoord in ruwe tekst, message, hint). Eerste match wint.
_RULES: list[tuple[str, str, str]] = [
    (
        "no such file",
        "Een benodigd bestand ontbreekt.",
        "Controleer de paden in Configuratie of draai eerst de ETL zonder --noetl.",
    ),
    (
        "filenotfounderror",
        "Een benodigd bestand ontbreekt.",
        "Controleer de paden in Configuratie of draai eerst de ETL zonder --noetl.",
    ),
    (
        "permission denied",
        "Geen schrijfrechten op een outputbestand.",
        "Sluit het bestand als het in Excel open staat en probeer opnieuw.",
    ),
    (
        "laatste week van het academisch jaar",
        "De gekozen week is de laatste week van het academisch jaar.",
        "Kies bij Uitvoeren een eerdere week.",
    ),
    (
        "buiten de range",
        "De gekozen jaar/week-combinatie zit niet in de beschikbare data.",
        "Pas jaar of week aan bij Uitvoeren, of voeg trainingsdata toe.",
    ),
    (
        "numerus_fixus",
        "De numerus fixus-configuratie klopt niet met de data.",
        "Gebruik in Configuratie de exacte programmasleutel uit je data.",
    ),
]

#: Fallback wanneer geen regel matcht.
_GENERIC = FriendlyError(
    message="Er ging iets mis tijdens het uitvoeren.",
    hint="Bekijk de logregels hieronder voor de details.",
)


def humanize_error(raw: str) -> FriendlyError:
    """Zet ruwe fouttekst om naar een :class:`FriendlyError`.

    Args:
        raw: Ruwe fouttekst of tracebackregel(s).

    Returns:
        De eerste passende vriendelijke melding, of een generieke fallback.
    """
    text = (raw or "").lower()
    for keyword, message, hint in _RULES:
        if keyword in text:
            return FriendlyError(message=message, hint=hint)
    return _GENERIC
