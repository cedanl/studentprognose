"""Instellingsnamen opzoeken bij Brincodes via de openbare DUO-lijst.

De Brincode (bijv. ``21PK``) blijft overal de sleutel: in de config
(``institution_filter``), op de CLI (``--institution``) en in de pipeline. Deze
module levert alleen een leesbaar etiket voor mensen, zoals
*Universiteit van Amsterdam*.

De meegeleverde tabel ``instellingen_ho.csv`` komt uit de DUO open
onderwijsdata *Instellingen hbo en wo* (kolommen ``INSTELLINGSCODE``,
``INSTELLINGSNAAM`` en ``SOORT HO``). De namen worden ongewijzigd overgenomen.
"""

from __future__ import annotations

import csv
import io
from functools import lru_cache
from importlib.resources import files

BUNDLED_FILENAME = "instellingen_ho.csv"


def parse_institution_names(text: str) -> dict[str, str]:
    """Lees de inhoud van ``instellingen_ho.csv``.

    Args:
        text: Puntkomma-gescheiden inhoud met kolommen ``Brincode`` en
            ``Instellingsnaam``.

    Returns:
        Een dict van Brincode (hoofdletters, zonder spaties) naar instellingsnaam.
        Regels zonder code of naam worden overgeslagen.
    """
    names: dict[str, str] = {}
    for row in csv.DictReader(io.StringIO(text), delimiter=";"):
        code = (row.get("Brincode") or "").strip().upper()
        name = (row.get("Instellingsnaam") or "").strip()
        if code and name:
            names[code] = name
    return names


@lru_cache(maxsize=1)
def load_institution_names() -> dict[str, str]:
    """Laad de meegeleverde opzoektabel van Brincode naar instellingsnaam.

    Returns:
        Een dict van Brincode naar instellingsnaam.
    """
    raw = files("studentprognose.configuration").joinpath(BUNDLED_FILENAME).read_text(
        encoding="utf-8"
    )
    return parse_institution_names(raw)


def institution_label(code: str, names: dict[str, str] | None = None) -> str:
    """Geef de leesbare naam bij een Brincode, of de code zelf als die onbekend is.

    Args:
        code: Brincode, bijv. ``"21PK"``.
        names: Eigen opzoektabel; ``None`` = de meegeleverde DUO-tabel.

    Returns:
        De instellingsnaam, of ``code`` als er geen naam bekend is.
    """
    lookup = load_institution_names() if names is None else names
    return lookup.get(str(code).strip().upper(), str(code))


def institution_options(
    codes: list[str], names: dict[str, str] | None = None
) -> dict[str, str]:
    """Bouw keuzelijst-opties van Brincode naar naam, alfabetisch op naam.

    Codes zonder bekende naam krijgen het label ``"<code> (naam onbekend)"`` en
    staan onderaan. Komen twee codes op dezelfde naam uit (bijv. een instelling
    die onder twee Brincodes levert), dan krijgen die opties de code erbij,
    zodat de keuze ondubbelzinnig blijft.

    Args:
        codes: De Brincodes die gekozen mogen worden.
        names: Eigen opzoektabel; ``None`` = de meegeleverde DUO-tabel.

    Returns:
        Een dict van Brincode naar weergavenaam: eerst de bekende instellingen
        op naam gesorteerd, daarna de onbekende codes.
    """
    lookup = load_institution_names() if names is None else names
    unique = list(dict.fromkeys(str(c).strip() for c in codes if str(c).strip()))
    known = {c: lookup[c.upper()] for c in unique if c.upper() in lookup}
    counts: dict[str, int] = {}
    for label in known.values():
        counts[label] = counts.get(label, 0) + 1
    shown = {c: (f"{label} ({c})" if counts[label] > 1 else label) for c, label in known.items()}
    unknown = {c: f"{c} (naam onbekend)" for c in unique if c not in known}
    ordered = sorted(shown.items(), key=lambda kv: kv[1].casefold())
    return dict(ordered + sorted(unknown.items()))
