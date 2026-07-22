"""Navigatiemodel: de enige bron van waarheid voor de app-flow.

Definieert de lineaire wizard-stappen (voor nieuwe gebruikers) en losse tools
(voor terugkerende gebruikers). Pagina's registreren zich als "beschikbaar" zodra
hun module is geladen; de shell toont nog-niet-gebouwde stappen grijs/uitgeschakeld
zodat de gebruiker de volledige flow altijd ziet.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NavItem:
    """Eén navigatiebestemming.

    Attributes:
        route: URL-pad (bijv. ``/config``).
        label: Weergavenaam in de zijbalk en stepper.
        icon: Material-icoonnaam.
        step: Volgnummer in de wizard-flow (1-based), of ``None`` voor losse
            tools die buiten de lineaire flow vallen (zoals benchmark).
    """

    route: str
    label: str
    icon: str
    step: int | None = None


#: De lineaire wizard-flow. Volgorde = de aanbevolen route voor nieuwe gebruikers.
WIZARD_FLOW: list[NavItem] = [
    NavItem("/wizard", "1. Project", "folder_open", step=1),
    NavItem("/config", "2. Configuratie", "tune", step=2),
    NavItem("/filtering", "3. Filteren", "filter_alt", step=3),
    NavItem("/run", "4. Uitvoeren", "play_arrow", step=4),
    NavItem("/output", "5. Resultaten", "insights", step=5),
]

#: Losse tools buiten de lineaire flow.
TOOLS: list[NavItem] = [
    NavItem("/benchmark", "Benchmark & tune", "science"),
]

#: Startpagina (geen stap, altijd bereikbaar).
HOME = NavItem("/", "Start", "home")


# --- Registry van beschikbare (gebouwde) routes ------------------------------
# Een pagina roept register_route() aan in zijn create(). De shell gebruikt dit
# om nog-niet-gebouwde stappen uitgeschakeld te tonen i.p.v. een 404 te geven.

_AVAILABLE: set[str] = {HOME.route}


def register_route(route: str) -> None:
    """Markeer een route als gebouwd/beschikbaar."""
    _AVAILABLE.add(route)


def is_available(route: str) -> bool:
    """True als de pagina voor deze route bestaat."""
    return route in _AVAILABLE


def all_items() -> list[NavItem]:
    """Alle navigatie-items in weergavevolgorde (home, flow, tools)."""
    return [HOME, *WIZARD_FLOW, *TOOLS]
