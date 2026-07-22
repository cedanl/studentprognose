"""Gedeelde applicatiestate voor de GUI.

Bewust minimaal: één dataclass die het gekozen project (werkmap) en de afgeleide
paden bijhoudt. Pagina's lezen en schrijven hier, zodat de wizard, configuratie-
editor en runner naar hetzelfde project verwijzen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AppState:
    """Sessiestate: welke projectmap is actief en waar liggen de bestanden.

    Attributes:
        project_dir: Absolute werkmap van het gekozen project (de map waarin
            ``studentprognose init`` de structuur heeft aangemaakt). ``None`` tot
            de gebruiker een project kiest of aanmaakt.
    """

    project_dir: str | None = None

    # --- Afgeleide paden (relatief aan project_dir) ----------------------------

    @property
    def config_path(self) -> str | None:
        """Pad naar ``configuration/configuration.json`` in het project."""
        if self.project_dir is None:
            return None
        return os.path.join(self.project_dir, "configuration", "configuration.json")

    @property
    def filtering_path(self) -> str | None:
        """Pad naar ``configuration/filtering/base.json`` in het project."""
        if self.project_dir is None:
            return None
        return os.path.join(self.project_dir, "configuration", "filtering", "base.json")

    @property
    def output_dir(self) -> str | None:
        """Pad naar ``data/output`` in het project."""
        if self.project_dir is None:
            return None
        return os.path.join(self.project_dir, "data", "output")

    @property
    def is_initialised(self) -> bool:
        """True als het gekozen project een geldige configuratie bevat."""
        return self.config_path is not None and os.path.isfile(self.config_path)


#: Enkelvoudige, module-brede state. De GUI is een lokale single-session app
#: (zie issue #273 "Buiten scope: multi-user"), dus module-state volstaat.
STATE = AppState()
