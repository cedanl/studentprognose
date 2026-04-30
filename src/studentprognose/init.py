"""studentprognose init — scaffold a new project directory."""

import os
from importlib.resources import files


_INPUT_RAW_README = """\
# data/input_raw

Plaats hier je ruwe Studielink-exportbestanden:

| Bestand/map | Beschrijving |
|---|---|
| `telbestanden/` | Wekelijkse Studielink-telbestanden (`telbestandY<jaar>W<week>.csv`) |
| `individuele_aanmelddata.csv` | Individuele vooraanmeldingen per student |
| `oktober_bestand.xlsx` | 1-cijfer HO oktober-telling (voor studentaantallen) |

Draai daarna `studentprognose` om de ETL te starten en voorspellingen te genereren.
"""


def run_init():
    cwd = os.getcwd()

    dirs = [
        os.path.join(cwd, "configuration", "filtering"),
        os.path.join(cwd, "data", "input"),
        os.path.join(cwd, "data", "input_raw", "telbestanden"),
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

    print("""
Volgende stappen:

  1. Plaats je telbestanden in:     data/input_raw/telbestanden/
                                    (bestandsnamen: telbestandY<jaar>W<week>.csv)
  2. Plaats je individuele data in: data/input_raw/individuele_aanmelddata.csv
  3. Optioneel — oktober-bestand:  data/input_raw/oktober_bestand.xlsx
                                    (voor berekening van studentaantallen)

  Draaien:
    studentprognose -w <week> -y <jaar>

  Voor geautomatiseerde runs (cron, taakplanner):
    studentprognose -w <week> -y <jaar> --yes

  Afwijkende kolomnamen in je Studielink-export?
  Voeg een "columns"-blok toe aan configuration/configuration.json.
  Zie: https://cedanl.github.io/studentprognose/configuratie/
""")
