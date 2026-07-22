"""Design-tokens voor de GUI.

Eén bron van waarheid voor kleuren, statussen en spacing. De volledige stijlgids
(inclusief motivatie en component-voorschriften) staat in ``gui/DESIGN.md`` (issue
#272). Wijzig kleuren hier, niet los in de pagina's.
"""

# Primaire huisstijl — indigo, gelijk aan de mkdocs-documentatie (mkdocs.yml).
PRIMARY = "#3f51b5"
SECONDARY = "#5c6bc0"
ACCENT = "#7986cb"

# Statuskleuren. Consistent over alle pagina's en states.
POSITIVE = "#2e7d32"  # succes / lage fout
WARNING = "#ed6c02"  # waarschuwing / matige fout
NEGATIVE = "#c62828"  # fout / hoge fout
INFO = "#0277bd"  # bezig / neutrale info

# Neutrale tinten.
DARK = "#1a1a2e"
DARK_PAGE = "#f4f5fb"

#: Quasar-kleurmap die op ``ui.colors(...)`` wordt toegepast in :mod:`gui.app`.
QUASAR_COLORS = {
    "primary": PRIMARY,
    "secondary": SECONDARY,
    "accent": ACCENT,
    "positive": POSITIVE,
    "negative": NEGATIVE,
    "warning": WARNING,
    "info": INFO,
    "dark": DARK,
    "dark-page": DARK_PAGE,
}
