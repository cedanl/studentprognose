"""Gedeelde applicatiestate voor de GUI.

Bewust minimaal: één dataclass die het gekozen project (werkmap) en de afgeleide
paden bijhoudt. Pagina's lezen en schrijven hier, zodat de wizard, configuratie-
editor en runner naar hetzelfde project verwijzen.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass


@dataclass
class AppState:
    """Sessiestate: welke projectmap is actief en waar liggen de bestanden.

    Attributes:
        project_dir: Absolute werkmap van het gekozen project (de map waarin
            ``studentprognose init`` de structuur heeft aangemaakt). ``None`` tot
            de gebruiker een project kiest of aanmaakt.
        wizard_mode: Modus gekozen in wizard-stap 4 (``"cumulative"``,
            ``"individual"`` of ``"both"``). ``None`` als de wizard nog niet
            doorlopen is. Wordt door de run-pagina als standaardwaarde gebruikt.
        config_saved: True zodra de gebruiker de configuratie deze sessie heeft
            opgeslagen. Voedt de "afgerond"-markering van stap 2 in de zijbalk;
            een gedraaide voorspelling (:attr:`has_output`) impliceert dit ook.
    """

    project_dir: str | None = None
    wizard_mode: str | None = None
    config_saved: bool = False

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

    @property
    def has_output(self) -> bool:
        """True als er minstens één definitief outputbestand in het project staat.

        Bepaalt of de stappen "Uitvoeren" en "Resultaten" als afgerond gelden:
        een gedraaide voorspelling laat een ``output_*.xlsx`` in ``data/output``
        achter (prelim-/testbestanden tellen niet mee).
        """
        if self.output_dir is None or not os.path.isdir(self.output_dir):
            return False
        for path in glob.glob(os.path.join(self.output_dir, "output_*.xlsx")):
            name = os.path.basename(path)
            if name.startswith("output_prelim") or "_ci_test" in name:
                continue
            return True
        return False


#: Enkelvoudige, module-brede state. De GUI is een lokale single-session app
#: (zie issue #273 "Buiten scope: multi-user"), dus module-state volstaat.
STATE = AppState()
