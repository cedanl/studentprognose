#!/usr/bin/env bash
# Smoke-test voor het geïnstalleerde package als standalone.
#
# Wat dit test:
#   - Of de wheel installeerbaar is
#   - Of het CLI-commando beschikbaar is na installatie
#   - Of `studentprognose init` de juiste mapstructuur aanmaakt
#   - Of de pipeline doorloopt met demo-data (ETL + predict + output)
#   - Of foutscenario's een begrijpelijke melding geven
#
# Wat dit NIET test:
#   - Correctheid van de voorspellingen (alleen dat er output verschijnt)
#   - Alle combinaties van flags en datasets
#   - Werking met echte instellingsdata (andere kolomnamen, custom config)
#   - Performance of memory bij grote datasets
#   - Windows-compatibiliteit (script is bash-only)
#
# Aannames:
#   - uv is geïnstalleerd
#   - Demo-data staat in data/input_raw/ (telbestanden/, individuele_aanmelddata.csv,
#     oktober_bestand.xlsx) — deze data zit in de repo maar is geen garantie
#     dat het representatief is voor productiedata
#   - De wheel is al gebouwd (uv build), of dit script bouwt hem zelf
#
# Gebruik: bash scripts/test_package.sh [--skip-build]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKIP_BUILD="${1:-}"

echo "=== Build wheel ==="
if [ "$SKIP_BUILD" = "--skip-build" ]; then
    WHEEL="$(ls "$REPO_ROOT/dist"/*.whl 2>/dev/null | sort -V | tail -1)"
    [ -z "$WHEEL" ] && echo "Geen wheel gevonden in dist/ — verwijder --skip-build" && exit 1
    echo "Bestaande wheel: $WHEEL"
else
    cd "$REPO_ROOT"
    uv build --out-dir "$REPO_ROOT/dist" -q
    WHEEL="$(ls "$REPO_ROOT/dist"/*.whl | sort -V | tail -1)"
    echo "Gebouwde wheel: $WHEEL"
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
cd "$WORK_DIR"

echo ""
echo "=== Installeer in geïsoleerde omgeving ($WORK_DIR) ==="
uv init --no-readme -q
uv add "$WHEEL" -q

echo ""
echo "=== [1/5] --help ==="
uv run studentprognose --help > /dev/null && echo "OK"

echo ""
echo "=== [2/5] init ==="
uv run studentprognose init
[ -f configuration/configuration.json ] || { echo "FOUT: configuration.json ontbreekt"; exit 1; }
[ -f configuration/filtering/base.json ] || { echo "FOUT: base.json ontbreekt"; exit 1; }
[ -d data/input_raw/telbestanden ] || { echo "FOUT: telbestanden/ ontbreekt"; exit 1; }

echo ""
echo "=== Kopieer demo-data ==="
cp -r "$REPO_ROOT/data/input_raw/telbestanden" "$WORK_DIR/data/input_raw/"
cp "$REPO_ROOT/data/input_raw/individuele_aanmelddata.csv" "$WORK_DIR/data/input_raw/"
cp "$REPO_ROOT/data/input_raw/oktober_bestand.xlsx" "$WORK_DIR/data/input_raw/"

echo ""
echo "=== [3/5] Volledige run: both, week 6, jaar 2020 (inclusief ETL) ==="
uv run studentprognose -w 6 -y 2020 --yes 2>&1 | grep -E "(Dataset|Saving output|Fout|Error)"
[ -f data/output/output_first-years_beide.xlsx ] || { echo "FOUT: output ontbreekt"; exit 1; }

echo ""
echo "=== [4/5] Foutscenario: higher-years zonder totaal-bestand ==="
OUTPUT=$(uv run studentprognose -w 6 -y 2020 -d cumulative -sy higher-years --noetl --yes 2>&1 || true)
echo "$OUTPUT" | grep "Fout:" || { echo "FOUT: geen begrijpelijke foutmelding"; exit 1; }

echo ""
echo "=== [5/5] Niet-bestaand config pad geeft waarschuwing ==="
OUTPUT=$(uv run studentprognose -w 6 -y 2020 -d cumulative -c /bestaat/niet.json --noetl --yes 2>&1)
echo "$OUTPUT" | grep "Waarschuwing:" || { echo "FOUT: geen waarschuwing bij ontbrekend config"; exit 1; }

echo ""
echo "=== Klaar — output ==="
ls data/output/
