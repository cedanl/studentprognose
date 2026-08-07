"""Design-tokens voor de GUI — CEDA/Npuls-huisstijl.

Eén bron van waarheid voor kleuren, statussen en grafiekpaletten. De waarden
volgen de gedeelde organisatie-stylesheet ``cedanl/.github`` →
``docs/stylesheets/extra.css`` (primair zwart, accent npuls-oranje, met de
Npuls-secundaire kleuren). Wijzig kleuren hier, niet los in de pagina's. Zie
``gui/DESIGN.md`` voor de volledige stijlgids.
"""

# --- Huisstijl-kern (CEDA/Npuls) ---------------------------------------------
PRIMARY = "#000000"  # zwart — app-bar, tekst, primaire knoppen
ACCENT = "#DD784B"  # npuls-oranje — accenten, actieve navigatie, highlights
SECONDARY = "#3D68EC"  # npuls-blauw

# Npuls-palet (secundaire kleuren).
NPULS_ORANGE = "#DD784B"
NPULS_BLUE = "#3D68EC"
NPULS_GREEN = "#00AF81"
NPULS_YELLOW = "#F4D74B"
NPULS_PINK = "#F4D9DC"

# --- Statuskleuren ------------------------------------------------------------
POSITIVE = "#00AF81"  # succes / lage fout (npuls-groen)
WARNING = "#E6A020"  # waarschuwing / matige fout (amber, leesbaar op wit)
NEGATIVE = "#C0392B"  # fout / hoge fout (warm rood — Npuls kent geen rood)
INFO = "#3D68EC"  # bezig / neutrale info (npuls-blauw)

# --- Neutrale tinten ----------------------------------------------------------
DARK = "#000000"
DARK_PAGE = "#ffffff"
INK = "#1a1a1a"  # tekst op wit
MUTED = "#6b6b6b"  # gedempte tekst

# --- Grafiekpalet -------------------------------------------------------------
#: Kwalitatief kleurenreeks voor grafieken (Npuls-volgorde).
CHART_SEQUENCE = [NPULS_ORANGE, NPULS_BLUE, NPULS_GREEN, NPULS_YELLOW, "#7B4BF4"]

#: MAPE-kleurbuckets (laag → hoog fout): groen, oranje, rood.
MAPE_GOOD = NPULS_GREEN
MAPE_MEDIUM = NPULS_ORANGE
MAPE_BAD = NEGATIVE

#: Quasar-kleurmap die op ``ui.colors(...)`` wordt toegepast in de paginaschil.
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
