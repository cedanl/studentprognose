"""studentprognose init — scaffold a new project directory."""

import os
import sys
import urllib.request
import zipfile
from importlib.resources import files

from tqdm import tqdm


_DEMO_DATA_URL = (
    "https://github.com/cedanl/studentprognose"
    "/releases/latest/download/demo-data.zip"
)

_INPUT_RAW_README = """\
# data/input_raw

Plaats hier je ruwe Studielink-exportbestanden:

| Bestand/map | Beschrijving |
|---|---|
| `telbestanden/` | Wekelijkse Studielink-telbestanden (default: `telbestandY<jaar>W<week>.csv`, instelbaar via `telbestand_filename_patterns`) |
| `individuele_aanmelddata.csv` | Individuele vooraanmeldingen per student |
| `oktober_bestand.xlsx` | Telbestand studenten (eigen levering instelling — voor studentaantallen) |

Draai daarna `studentprognose` om de ETL te starten en voorspellingen te genereren.
"""

_NEXT_STEPS_DEMO = """\
Demodata staat in data/input_raw/. Draai nu:

  studentprognose -d c -y 2024      # cumulatief spoor, collegejaar 2024

Of met een dashboard:

  studentprognose -d c -y 2024 --dashboard

De demodata bevat alleen telbestanden (cumulatief spoor). Voor het individuele
spoor (-d i of -d b) heb je eigen individuele aanmelddata nodig.
Zie: https://cedanl.github.io/studentprognose/je-data-voorbereiden/
"""

_NEXT_STEPS_OWN = """\
Volgende stappen:

  1. Plaats je telbestanden in:     data/input_raw/telbestanden/
                                    (default-naam: telbestandY<jaar>W<week>.csv;
                                     overschrijf via telbestand_filename_patterns
                                     als je instelling een andere conventie gebruikt)
  2. Plaats je individuele data in: data/input_raw/individuele_aanmelddata.csv
  3. Optioneel — telbestand studenten:
                                    data/input_raw/oktober_bestand.xlsx
                                    (eigen levering instelling, voor
                                    berekening van studentaantallen)

  Draaien:
    studentprognose -w <week> -y <jaar>

  Voor geautomatiseerde runs (cron, taakplanner):
    studentprognose -w <week> -y <jaar> --yes

  Afwijkende kolomnamen in je Studielink-export?
  Voeg een "columns"-blok toe aan configuration/configuration.json.
  Zie: https://cedanl.github.io/studentprognose/configuratie/
"""


def _ask_demo() -> bool:
    """Vraag of de gebruiker demodata wil downloaden. Retourneert False als stdin geen TTY is."""
    if not sys.stdin.isatty():
        return False
    try:
        antwoord = input(
            "\nWil je demodata downloaden om direct te starten? (4 MB, ~10 sec) [j/n]: "
        ).strip().lower()
        return antwoord in ("j", "ja", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def _download_demo(cwd: str) -> bool:
    """Download en extraheer demo-data.zip naar data/input_raw/. Retourneert True bij succes."""
    zip_path = os.path.join(cwd, "_demo_data_tmp.zip")
    extract_to = os.path.join(cwd, "data", "input_raw")

    try:
        print("  Downloaden...", flush=True)
        with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc="  demo-data.zip",
            file=sys.stdout,
        ) as t:
            def _hook(block_num, block_size, total_size):
                if total_size > 0:
                    t.total = total_size
                t.update(block_size)

            urllib.request.urlretrieve(_DEMO_DATA_URL, zip_path, _hook)

        print("  Uitpakken...", flush=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)

        print("  Demodata geïnstalleerd in data/input_raw/ ✓")
        return True

    except Exception as exc:
        print(f"\n  Downloaden mislukt: {exc}")
        print(f"  Download handmatig via:\n  {_DEMO_DATA_URL}")
        return False

    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)


def run_init():
    cwd = os.getcwd()

    dirs = [
        os.path.join(cwd, "configuration", "filtering"),
        os.path.join(cwd, "data", "input"),
        os.path.join(cwd, "data", "input_raw", "telbestanden"),
        os.path.join(cwd, "data", "output"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    config_path = os.path.join(cwd, "configuration", "configuration.json")
    if not os.path.exists(config_path):
        content = files("studentprognose.configuration").joinpath("configuration.json").read_text(encoding="utf-8")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("  Aangemaakt: configuration/configuration.json")
    else:
        print("  Overgeslagen: configuration/configuration.json (bestaat al)")

    filtering_path = os.path.join(cwd, "configuration", "filtering", "base.json")
    if not os.path.exists(filtering_path):
        content = files("studentprognose.configuration.filtering").joinpath("base.json").read_text(encoding="utf-8")
        with open(filtering_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("  Aangemaakt: configuration/filtering/base.json")
    else:
        print("  Overgeslagen: configuration/filtering/base.json (bestaat al)")

    readme_path = os.path.join(cwd, "data", "input_raw", "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(_INPUT_RAW_README)
        print("  Aangemaakt: data/input_raw/README.md")

    demo_downloaded = False
    if _ask_demo():
        demo_downloaded = _download_demo(cwd)

    print()
    if demo_downloaded:
        print(_NEXT_STEPS_DEMO)
    else:
        print(_NEXT_STEPS_OWN)
