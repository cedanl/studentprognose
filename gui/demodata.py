"""Demodata downloaden en uitpakken.

De init-wizard (#266) biedt aan om de demodataset op te halen zodat een nieuw
project meteen te proberen is. De data komt uit de laatste GitHub-release. Het
uitpakken (:func:`extract_zip`) is losgekoppeld van het downloaden zodat het
getest kan worden.
"""

from __future__ import annotations

import os
import time
import zipfile
from collections.abc import Callable

#: Download-URL van de demodataset (laatste release-asset).
DEMO_URL = (
    "https://github.com/cedanl/studentprognose/releases/latest/download/demo-data.zip"
)


def extract_zip(zip_path: str, dest_dir: str) -> list[str]:
    """Pak een zip veilig uit naar ``dest_dir``.

    Beschermt tegen "zip slip" (padtraversal): leden die buiten ``dest_dir``
    zouden landen worden overgeslagen.

    Args:
        zip_path: Pad naar het zip-bestand.
        dest_dir: Doelmap (wordt aangemaakt als die niet bestaat).

    Returns:
        De lijst van (relatieve) uitgepakte bestandsnamen.

    Raises:
        zipfile.BadZipFile: Als het bestand geen geldige zip is.
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest_root = os.path.abspath(dest_dir)
    extracted: list[str] = []

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            target = os.path.abspath(os.path.join(dest_root, member))
            if not target.startswith(dest_root + os.sep):
                # Padtraversal — overslaan.
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted.append(member)

    return extracted


#: Maximale wachttijd (seconden) per netwerkstap. Zonder timeout blijft de wizard
#: bij een haperende verbinding eindeloos "bezig" zonder melding (#322).
DOWNLOAD_TIMEOUT_S = 30

#: Maximale totale downloadtijd (seconden). Een verbinding die blijft sijpelen
#: haalt de timeout per stap nooit, maar mag de gebruiker niet eindeloos laten
#: wachten (#322).
DOWNLOAD_MAX_SECONDS = 300


def download_file(
    url: str,
    dest_path: str,
    progress_cb: Callable[[float], None] | None = None,
    timeout: float = DOWNLOAD_TIMEOUT_S,
    max_seconds: float = DOWNLOAD_MAX_SECONDS,
) -> None:
    """Download ``url`` naar ``dest_path`` met optionele voortgang.

    Args:
        url: Bron-URL.
        dest_path: Doelbestand.
        progress_cb: Callback met een fractie 0.0–1.0 (of ``-1`` als de grootte
            onbekend is). Optioneel.
        timeout: Maximale wachttijd in seconden voor verbinden en voor elk
            gelezen blok.
        max_seconds: Maximale totale duur van de download in seconden.

    Raises:
        urllib.error.URLError: Bij netwerkfouten.
        TimeoutError: Als de server te lang niets terugstuurt, of als de hele
            download langer duurt dan ``max_seconds``.
    """
    import urllib.request

    deadline = time.monotonic() + max_seconds
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 (vaste https-URL)
        total = int(response.headers.get("Content-Length", 0))
        read = 0
        chunk = 64 * 1024
        with open(dest_path, "wb") as out:
            while True:
                block = response.read(chunk)
                if not block:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"de download duurde langer dan {int(max_seconds)} seconden"
                    )
                out.write(block)
                read += len(block)
                if progress_cb is not None:
                    progress_cb(read / total if total else -1)
