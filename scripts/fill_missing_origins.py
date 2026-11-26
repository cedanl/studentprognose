#!/usr/bin/env python3
"""Vul de synthetische demodata aan zodat elke opleiding alle herkomsten heeft.

De meegeleverde demodata (``data/input_raw/``) is synthetisch, maar veel
opleidingen hadden maar één of twee herkomstgroepen (bijv. alleen NL), of een
herkomst die alleen uit annuleringen bestond. Het dashboard kon dan per
opleiding geen uitsplitsing NL / EER / Niet-EER tonen.

Dit script vult de data aan; bestaande rijen worden nooit gewijzigd en een
tweede run voegt niets meer toe (idempotent).

* **Telbestanden** — per opleiding (Brincode, Isatcode, Type_HO) en collegejaar
  krijgt elke herkomst *bruikbare* rijen in elk weekbestand waarin de opleiding
  bruikbare rijen heeft. Bruikbaar = wat de pipeline als eerstejaars-
  vooraanmelding meetelt: ``Status != "A"``, ``Hogerejaars == "N"`` en
  ``Herinschrijving == "N"``. Ontbreekt een herkomst in een week, dan worden de
  bruikbare rijen van de best gedekte herkomst in die week gekopieerd.
* **Oktober-bestand** — per (Isatcode, Collegejaar, Examentype code) krijgt
  elke herkomst een rij, op dezelfde manier.

Aantallen worden geschaald met de verhouding tussen beide herkomsten zoals die
in dezelfde opleiding al voorkomt (alle weken en jaren waarin beide er zijn). Is die er niet, dan met een vaste, reproduceerbare verhouding per
opleiding (NL = 1, EER en Niet-EER een fractie daarvan). Zo sluiten aangevulde
weken aan op de bestaande reeks.

Gebruik::

    uv run python scripts/fill_missing_origins.py
    uv run python scripts/fill_missing_origins.py --raw-dir pad/naar/input_raw
"""

from __future__ import annotations

import argparse
import glob
import os
from collections.abc import Callable, Hashable, Sequence

import numpy as np
import pandas as pd

SEED = 328

#: Herkomstgroepen; bij gelijke dekking is dit ook de voorkeursvolgorde als bron.
ORIGINS: tuple[str, ...] = ("NL", "EER", "Niet-EER")

#: Herkomstcodes in de Studielink-telbestanden → canonieke namen.
TEL_ORIGIN_CODES: dict[str, str] = {"N": "NL", "E": "EER", "R": "Niet-EER"}

#: Bandbreedte van de EER- en Niet-EER-fractie t.o.v. NL (per opleiding getrokken).
_EER_RANGE = (0.10, 0.35)
_NIET_EER_RANGE = (0.05, 0.20)

#: Interne hulpkolommen.
_ORIGIN = "_origin"
_PERIOD = "_period"
_USEFUL = "_useful"
_COUNT = "_n"


def origin_weights(programme: Hashable) -> dict[str, float]:
    """Vaste, reproduceerbare verhouding tussen herkomsten voor één opleiding.

    Args:
        programme: Opleidingscode (Isatcode), bijv. ``"50700"`` of ``50700``.

    Returns:
        Gewicht per canonieke herkomst; NL is altijd 1.
    """
    digits = "".join(ch for ch in str(programme) if ch.isdigit()) or "0"
    rng = np.random.default_rng(SEED + int(digits))
    return {
        "NL": 1.0,
        "EER": float(rng.uniform(*_EER_RANGE)),
        "Niet-EER": float(rng.uniform(*_NIET_EER_RANGE)),
    }


def fill_missing_origins(
    df: pd.DataFrame,
    key_cols: Sequence[str],
    origin_col: str,
    count_col: str,
    origins: Sequence[str],
    weights: Callable[[tuple], dict[str, float]],
    period_col: str | None = None,
    useful: pd.Series | None = None,
    ratio_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Geef de rijen terug die ontbrekende herkomsten aanvullen.

    Een sleutel (``key_cols``, bijv. opleiding × jaar) bestaat uit perioden
    (``period_col``, bijv. weekbestanden; zonder ``period_col`` is de hele
    sleutel één periode). In elke periode waarin de sleutel bruikbare rijen
    heeft, moet elke herkomst uit ``origins`` bruikbare rijen hebben. Ontbreekt
    een herkomst, dan worden de bruikbare rijen van de bron gekopieerd: de
    herkomst die in die periode aanwezig is en over de hele sleutel de meeste
    perioden dekt (bij gelijke dekking de eerste in ``origins``).

    De telling wordt geschaald met de verhouding doel/bron over alle perioden
    waarin beide bruikbaar aanwezig zijn, gepoold over ``ratio_cols`` (bijv.
    de opleiding over alle jaren, wat stabieler is dan één jaar); zonder zulke
    perioden met ``weights(key)[doel] / weights(key)[bron]``. Afgerond,
    minimaal 1.

    Args:
        df: Brondata; wordt niet aangepast.
        key_cols: Kolommen die samen één opleiding-jaar vormen.
        origin_col: Kolom met de herkomst (waarden uit ``origins``).
        count_col: Kolom met het aantal dat geschaald wordt.
        origins: Herkomsten die aanwezig moeten zijn.
        weights: Functie die bij een sleutel het gewicht per herkomst geeft.
        period_col: Kolom met de periode binnen een sleutel, of ``None``.
        useful: Booleaanse reeks (zelfde lengte als ``df``): welke rijen
            meetellen. ``None`` = alle rijen.
        ratio_cols: Deel van ``key_cols`` waarover de verhouding tussen
            herkomsten gepoold wordt. ``None`` = per sleutel.

    Returns:
        Alleen de nieuwe rijen, met dezelfde kolommen als ``df``. Leeg als er
        niets ontbreekt.
    """
    work = df.reset_index(drop=True).copy()
    work[_PERIOD] = work[period_col] if period_col else 0
    work[_USEFUL] = True if useful is None else np.asarray(useful, dtype=bool)
    work[_COUNT] = pd.to_numeric(work[count_col])
    use = work[work[_USEFUL]].reset_index(drop=True)

    group_cols = [*key_cols, _PERIOD, origin_col]
    totals = use.groupby(group_cols, sort=False)[_COUNT].sum()
    rows_at = use.groupby(group_cols, sort=False).indices

    per_key: dict[tuple, dict] = {}
    for (*key, period, origin), total in totals.items():
        per_key.setdefault(tuple(key), {}).setdefault(period, {})[origin] = total

    # Som per herkomstpaar over de perioden waarin beide aanwezig zijn,
    # gepoold per ratio-sleutel.
    pos = [list(key_cols).index(c) for c in (ratio_cols or key_cols)]
    num: dict[tuple, float] = {}
    den: dict[tuple, float] = {}
    for key, periods in per_key.items():
        rkey = tuple(key[i] for i in pos)
        for present in periods.values():
            for t in present:
                for s in present:
                    if t != s:
                        num[(rkey, t, s)] = num.get((rkey, t, s), 0.0) + present[t]
                        den[(rkey, t, s)] = den.get((rkey, t, s), 0.0) + present[s]

    positions: list[np.ndarray] = []
    targets: list[str] = []
    factors: list[float] = []
    for key, periods in per_key.items():
        if all(len(present) == len(origins) for present in periods.values()):
            continue
        coverage = {o: sum(o in p for p in periods.values()) for o in origins}
        ranking = sorted(origins, key=lambda o: (-coverage[o], origins.index(o)))
        w = weights(key)
        rkey = tuple(key[i] for i in pos)

        def ratio(t: str, s: str, rkey=rkey, w=w) -> float:
            n, d = num.get((rkey, t, s), 0.0), den.get((rkey, t, s), 0.0)
            return n / d if n > 0 and d > 0 else w[t] / w[s]

        for period, present in periods.items():
            source = next(o for o in ranking if o in present)
            for target in origins:
                if target not in present:
                    idx = rows_at[(*key, period, source)]
                    positions.append(idx)
                    targets.extend([target] * len(idx))
                    factors.extend([ratio(target, source)] * len(idx))

    if not positions:
        return df.iloc[0:0].copy()
    idx = np.concatenate(positions)
    rows = use.iloc[idx]
    new = rows[list(df.columns)].reset_index(drop=True)
    new[origin_col] = targets
    scaled = rows[_COUNT].to_numpy() * np.asarray(factors)
    new[count_col] = np.maximum(1, np.rint(scaled)).astype(int)
    return new


def _line_ending(path: str) -> str:
    """Regeleinde van een bestaand bestand (CRLF of LF), zodat dat behouden blijft."""
    with open(path, "rb") as fh:
        first_line = fh.readline()
    return "\r\n" if first_line.endswith(b"\r\n") else "\n"


def fill_telbestanden(tel_dir: str) -> int:
    """Vul ontbrekende herkomsten aan in alle telbestanden van ``tel_dir``.

    Args:
        tel_dir: Map met de ``telbestand_*.csv``-bestanden.

    Returns:
        Aantal toegevoegde rijen.
    """
    frames = []
    for path in sorted(glob.glob(os.path.join(tel_dir, "*.csv"))):
        data = pd.read_csv(path, dtype=str, keep_default_na=False)
        data["_file"] = path
        frames.append(data)
    if not frames:
        return 0
    tel = pd.concat(frames, ignore_index=True)
    tel = tel[tel["Herkomst"].isin(TEL_ORIGIN_CODES)].reset_index(drop=True)
    tel[_ORIGIN] = tel["Herkomst"].map(TEL_ORIGIN_CODES)
    useful = (
        (tel["Status"] != "A") & (tel["Hogerejaars"] == "N") & (tel["Herinschrijving"] == "N")
    )

    new = fill_missing_origins(
        tel,
        ["Brincode", "Isatcode", "Type_HO", "Studiejaar"],
        _ORIGIN,
        "Aantal",
        ORIGINS,
        weights=lambda key: origin_weights(key[1]),
        period_col="_file",
        useful=useful,
        ratio_cols=["Brincode", "Isatcode", "Type_HO"],
    )
    if new.empty:
        return 0
    to_code = {v: k for k, v in TEL_ORIGIN_CODES.items()}
    new["Herkomst"] = new[_ORIGIN].map(to_code)
    new["Aantal"] = new["Aantal"].astype(str)

    for path, extra in new.groupby("_file", sort=False):
        original = pd.read_csv(path, dtype=str, keep_default_na=False)
        combined = pd.concat([original, extra[original.columns]], ignore_index=True)
        combined.to_csv(path, index=False, lineterminator=_line_ending(path))
    return len(new)


def fill_oktober(path: str) -> int:
    """Vul ontbrekende herkomsten aan in het oktober-bestand.

    Args:
        path: Pad naar ``oktober_bestand.xlsx``.

    Returns:
        Aantal toegevoegde rijen.
    """
    okt = pd.read_excel(path)
    new = fill_missing_origins(
        okt,
        ["Isatcode", "Collegejaar", "Examentype code"],
        "EER-NL-nietEER",
        "Aantal Hoofdinschrijvingen",
        ORIGINS,
        weights=lambda key: origin_weights(key[0]),
        useful=okt["Aantal Hoofdinschrijvingen"] > 0,
        ratio_cols=["Isatcode", "Examentype code"],
    )
    if new.empty:
        return 0
    pd.concat([okt, new], ignore_index=True).to_excel(path, index=False)
    return len(new)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-dir",
        default=os.path.join(root, "data", "input_raw"),
        help="Map met telbestanden/ en oktober_bestand.xlsx (standaard data/input_raw).",
    )
    args = parser.parse_args(argv)

    n_tel = fill_telbestanden(os.path.join(args.raw_dir, "telbestanden"))
    n_okt = fill_oktober(os.path.join(args.raw_dir, "oktober_bestand.xlsx"))
    print(f"Telbestanden: {n_tel} rijen toegevoegd")
    print(f"Oktober-bestand: {n_okt} rijen toegevoegd")


if __name__ == "__main__":
    main()
