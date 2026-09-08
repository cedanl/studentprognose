"""Brincode → instellingsnaam mapping voor de GUI.

Bronnen:
- Regeling financiën hoger onderwijs (wetten.overheid.nl) voor WO en HBO lijsten
- NSE deelnemerslijst 2025 en Kennisnet Entree federatie lijst
- DUO open onderwijsdata

Fallback: onbekende code → kale code tonen (geen crash).
"""

from __future__ import annotations

# Volledige mapping van bekende Brincodes naar leesbare namen.
# Samengesteld uit meerdere openbare bronnen (zie module docstring).
# Waar namen verschillen tussen bronnen is de meest recente/gebruikelijke naam gekozen.
BRINCODE_NAMES: dict[str, str] = {
    # --- Universiteiten (WO) ---
    "00DV": "Protestantse Theologische Universiteit",
    "21PB": "Universiteit Leiden",
    "21PC": "Rijksuniversiteit Groningen",
    "21PD": "Universiteit Utrecht",
    "21PE": "Erasmus Universiteit Rotterdam",
    "21PF": "Technische Universiteit Delft",
    "21PG": "Technische Universiteit Eindhoven",
    "21PH": "Universiteit Twente",
    "21PI": "Wageningen University & Research",
    "21PJ": "Maastricht University",
    "21PK": "Universiteit van Amsterdam",
    "21PL": "Vrije Universiteit Amsterdam",
    "21PM": "Radboud Universiteit",
    "21PN": "Tilburg University",
    "21QO": "Theologische Universiteit Apeldoorn",
    "22NC": "Open Universiteit",
    "23BF": "Universiteit voor Humanistiek",
    "25AV": "Theologische Universiteit Kampen",
    "01MC": "Nyenrode Business Universiteit",
    # --- Hogescholen (HBO) ---
    "00IC": "Hogeschool KPZ",
    "00MF": "HKU - Hogeschool voor de Kunsten Utrecht",
    "01VU": "Windesheim",
    "02BY": "Gerrit Rietveld Academie",
    "02NR": "Hotelschool The Hague",
    "02NT": "Design Academy Eindhoven",
    "07GR": "Avans Hogeschool",
    "08OK": "Hogeschool De Kempel",
    "09OT": "Iselinge Hogeschool",
    "10IZ": "Marnix Academie",
    "14NI": "Codarts Hogeschool voor de Kunsten",
    "15BK": "Driestar Hogeschool",
    "21CW": "HAS green academy",
    "21MI": "HZ University of Applied Sciences",
    "21QA": "Amsterdamse Hogeschool voor de Kunsten",
    "21RI": "Hogeschool Leiden",
    "21UG": "Hogeschool iPabo",
    "21UI": "Breda University of Applied Sciences",
    "21WN": "NHL Hogeschool",
    "22EX": "Stenden Hogeschool",
    "22HH": "Hogeschool Viaa",
    "22OJ": "Hogeschool Rotterdam",
    "23AH": "Saxion",
    "23KJ": "Hogeschool der Kunsten Den Haag",
    "25BA": "Christelijke Hogeschool Ede",
    "25BE": "Hanzehogeschool Groningen",
    "25DW": "Hogeschool Utrecht",
    "25JX": "Zuyd Hogeschool",
    "25KB": "Hogeschool van Arnhem en Nijmegen",
    "27NF": "ArtEZ University of the Arts",
    "27PZ": "Hogeschool Inholland",
    "27UM": "De Haagse Hogeschool",
    "28DN": "Hogeschool van Amsterdam",
    "30GB": "Fontys Hogeschool",
    "30HD": "Hogeschool Van Hall Larenstein",
    "30TX": "Aeres Hogeschool",
    "30VP": "Thomas More Hogeschool",
    # Extra codes uit data dumps
    "01DZ": "STOAS Hogeschool",
    "01MY": "Christelijke Agrarische Hogeschool Dronten",
    "04CS": "Hogeschool Helicon",
    "08YJ": "Hogeschool Edith Stein",
    "21US": "Hogeschool van Hall Larenstein (oud)",
    "22ND": "Internationale Agrarische Hogeschool Larenstein",
    "24LE": "Van Hall Instituut",
    "28DE": "Hogeschool van Amsterdam (oud)",
}


def get_name(code: str) -> str | None:
    """Geef de instellingsnaam voor een Brincode, of None als onbekend."""
    if not code:
        return None
    return BRINCODE_NAMES.get(code.strip().upper())


def label_for(code: str) -> str:
    """Geef een leesbaar label voor een Brincode: 'Naam (CODE)' of 'CODE' als fallback."""
    name = get_name(code)
    if name:
        return f"{name} ({code})"
    return code


def build_options(codes: list[str]) -> dict[str, str]:
    """Bouw {code: label} dict voor NiceGUI select, gesorteerd op naam.

    - Bekende codes krijgen label 'Naam (CODE)'
    - Onbekende codes krijgen label 'CODE' (fallback, geen crash)
    - Sortering: eerst op naam (alfabetisch), onbekende achteraan op code.
    """
    unique = sorted({c.strip().upper() for c in codes if c and c.strip()})
    # Sorteer: bekende namen alfabetisch, onbekende op code achteraan
    def sort_key(c: str):
        name = get_name(c)
        if name:
            return (0, name.lower(), c)
        return (1, "", c)

    sorted_codes = sorted(unique, key=sort_key)
    return {code: label_for(code) for code in sorted_codes}
