"""Rungeschiedenis: vind eerdere tmp-projectruns met outputbestanden.

Scant de ``tmp/``-map voor ``studentprognose{YYYYMMDDHHMMSS}``-directories die
outputbestanden bevatten. Gebruikt door de uitvoerpagina om een run-dropdown aan
te bieden zonder de globale sessiestate te wijzigen.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime

_STAMP_RE = re.compile(r"studentprognose(\d{14})$")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TMP_ROOT = os.path.join(_REPO_ROOT, "tmp")

_NL_MAANDEN = [
    "jan", "feb", "mrt", "apr", "mei", "jun",
    "jul", "aug", "sep", "okt", "nov", "dec",
]


@dataclass(frozen=True)
class RunEntry:
    """Metadata van één historische run."""

    output_dir: str
    timestamp: datetime
    n_files: int

    @property
    def label(self) -> str:
        dag = self.timestamp.day
        maand = _NL_MAANDEN[self.timestamp.month - 1]
        jaar = self.timestamp.year
        tijd = self.timestamp.strftime("%H:%M")
        n = self.n_files
        s = "en" if n != 1 else ""
        return f"{dag} {maand} {jaar}, {tijd} — {n} bestand{s}"


def find_historical_runs(exclude_dir: str | None = None) -> list[RunEntry]:
    """Zoek tmp-runs met outputbestanden, nieuwste eerst.

    Args:
        exclude_dir: Projectmap die overgeslagen wordt (de actieve run).

    Returns:
        Lijst van :class:`RunEntry`, gesorteerd op timestamp aflopend.
    """
    if not os.path.isdir(_TMP_ROOT):
        return []

    from gui import results_io  # lokale import vermijdt circulaire dependencies

    exclude_real = os.path.realpath(exclude_dir) if exclude_dir else None
    entries: list[RunEntry] = []

    for name in os.listdir(_TMP_ROOT):
        full = os.path.join(_TMP_ROOT, name)
        if not os.path.isdir(full):
            continue
        if exclude_real and os.path.realpath(full) == exclude_real:
            continue
        ts = _parse_stamp(name)
        if ts is None:
            continue
        output_dir = os.path.join(full, "data", "output")
        files = results_io.find_output_files(output_dir)
        if not files:
            continue
        entries.append(
            RunEntry(output_dir=output_dir, timestamp=ts, n_files=len(files))
        )

    entries.sort(key=lambda e: e.timestamp, reverse=True)
    return entries


def _parse_stamp(name: str) -> datetime | None:
    m = _STAMP_RE.match(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None
