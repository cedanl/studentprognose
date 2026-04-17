#!/usr/bin/env bash
# Smoke-test: bouwt de wheel en verifieert de gebruikersjournery end-to-end.
#
# Wat dit test:
#   1. Wheel bouwen en installeren slaagt
#   2. CLI is beschikbaar na installatie (--help)
#   3. `init` maakt de juiste mapstructuur aan
#   4. Een volledige run met demo-data geeft output
#
# Gebruik: bash scripts/test_package.sh [--skip-build]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKIP_BUILD="${1:-}"

echo "=== [1] Build en installeer ==="
if [ "$SKIP_BUILD" = "--skip-build" ]; then
    WHEEL="$(ls "$REPO_ROOT/dist"/*.whl 2>/dev/null | sort -V | tail -1)"
    [ -z "$WHEEL" ] && echo "Geen wheel in dist/ — verwijder --skip-build" && exit 1
else
    cd "$REPO_ROOT" && uv build --out-dir "$REPO_ROOT/dist" -q
    WHEEL="$(ls "$REPO_ROOT/dist"/*.whl | sort -V | tail -1)"
fi
echo "Wheel: $(basename "$WHEEL")"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
cd "$WORK_DIR"
uv init --no-readme -q && uv add "$WHEEL" -q
echo "OK"

echo ""
echo "=== [2] --help werkt ==="
uv run studentprognose --help > /dev/null
echo "OK"

echo ""
echo "=== [3] init maakt structuur aan ==="
uv run studentprognose init > /dev/null
[ -f configuration/configuration.json ]   || { echo "FOUT: configuration.json ontbreekt"; exit 1; }
[ -f configuration/filtering/base.json ]  || { echo "FOUT: filtering/base.json ontbreekt"; exit 1; }
[ -d data/input_raw/telbestanden ]        || { echo "FOUT: data/input_raw/telbestanden ontbreekt"; exit 1; }
echo "OK"

echo ""
echo "=== [4] Volledige run met demo-data ==="
cp -r "$REPO_ROOT/data/input_raw/telbestanden"           data/input_raw/
cp    "$REPO_ROOT/data/input_raw/individuele_aanmelddata.csv" data/input_raw/
cp    "$REPO_ROOT/data/input_raw/oktober_bestand.xlsx"   data/input_raw/
uv run studentprognose -w 6 -y 2020 --yes > /dev/null
[ -f data/output/output_first-years_beide.xlsx ] || { echo "FOUT: output ontbreekt"; exit 1; }
echo "OK"

echo ""
echo "=== Klaar ==="
