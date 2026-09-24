"""Bouw demo-data.zip uit data/input_raw/ voor de release-workflow.

Verzamelt de telbestanden (cumulatief spoor), individuele_aanmelddata.csv
(individueel spoor) en oktober_bestand.xlsx in één zip, zodat
`studentprognose init` met één download beide sporen kan proberen. Draait als
stap in `.github/workflows/pypi-publish.yml` bij elke getagde release, maar
kan ook los gedraaid worden:

    uv run python scripts/build_demo_data.py
"""

from __future__ import annotations

import argparse
import os
import zipfile

DEFAULT_MEMBERS = [
    "telbestanden",
    "individuele_aanmelddata.csv",
    "oktober_bestand.xlsx",
]


def build_zip(
    source_dir: str, output_path: str, members: list[str] | None = None
) -> list[str]:
    """Bundel de opgegeven bestanden/mappen uit ``source_dir`` in een zip.

    Args:
        source_dir: Map met de demodatabestanden (doorgaans `data/input_raw`).
        output_path: Pad waar de zip geschreven wordt.
        members: Bestands-/mapnamen relatief aan `source_dir` om mee te nemen.
            Standaard: telbestanden, individuele_aanmelddata.csv en
            oktober_bestand.xlsx.

    Returns:
        De (relatieve) bestandsnamen die in de zip zijn beland, gesorteerd.

    Raises:
        FileNotFoundError: Als een opgegeven member niet bestaat in
            `source_dir`.
    """
    members = DEFAULT_MEMBERS if members is None else members
    written: list[str] = []

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for member in members:
            member_path = os.path.join(source_dir, member)
            if not os.path.exists(member_path):
                raise FileNotFoundError(
                    f"Verwacht demodata-bestand ontbreekt: {member_path}"
                )
            if os.path.isdir(member_path):
                for root, _dirs, files in os.walk(member_path):
                    for name in sorted(files):
                        full = os.path.join(root, name)
                        arcname = os.path.relpath(full, source_dir)
                        zf.write(full, arcname)
                        written.append(arcname)
            else:
                zf.write(member_path, member)
                written.append(member)

    return sorted(written)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default="data/input_raw",
        help="Map met de demodatabestanden (default: data/input_raw)",
    )
    parser.add_argument(
        "--output",
        default="demo-data.zip",
        help="Doelpad voor de zip (default: demo-data.zip)",
    )
    args = parser.parse_args()

    written = build_zip(args.source_dir, args.output)
    print(f"{args.output}: {len(written)} bestanden")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
