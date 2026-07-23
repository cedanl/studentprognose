"""Uitleg over de drie voorspelsporen — één bron van waarheid.

Gebruikt door de startpagina (uitlegkaart), de pipeline-runner en de benchmark-
tab (tooltips), zodat de omschrijving overal identiek is. Pure data, geen NiceGUI.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Aanbevolen spoor voor wie twijfelt.
RECOMMENDED = "Beide"

#: Documentatie over de methodologie (achtergrond bij de sporen).
DOCS_URL = "https://cedanl.github.io/studentprognose/methodologie/"


@dataclass(frozen=True)
class Track:
    """Eén voorspelspoor.

    Attributes:
        label: Weergavenaam (zoals in de dropdowns).
        icon: Material-icoonnaam.
        short: Korte omschrijving (één zin) voor uitleg en tooltips.
    """

    label: str
    icon: str
    short: str


#: De drie sporen, in weergavevolgorde.
TRACKS: list[Track] = [
    Track(
        "Cumulatief",
        "show_chart",
        "voorspelt uit de wekelijkse aanmeldcurve van een opleiding "
        "(SARIMA-tijdreeks → XGBoost).",
    ),
    Track(
        "Individueel",
        "person_search",
        "voorspelt per aanmelding de kans op inschrijving en telt die op "
        "(XGBoost-classifier).",
    ),
    Track(
        "Beide",
        "merge_type",
        "combineert het cumulatieve en individuele spoor tot één "
        "ensemble-voorspelling — meestal het nauwkeurigst.",
    ),
]


def track(label: str) -> Track | None:
    """Zoek een spoor op label."""
    return next((t for t in TRACKS if t.label == label), None)


def dataset_tooltip(labels: list[str] | None = None) -> str:
    """Bouw een tooltip die de gevraagde sporen uitlegt (één regel per spoor).

    Args:
        labels: Welke sporen op te nemen (default: alle drie).
    """
    chosen = TRACKS if labels is None else [t for t in TRACKS if t.label in labels]
    lines = [f"{t.label}: {t.short}" for t in chosen]
    if any(t.label == RECOMMENDED for t in chosen):
        lines.append(f"Niet zeker? Kies '{RECOMMENDED}'.")
    return "\n".join(lines)
