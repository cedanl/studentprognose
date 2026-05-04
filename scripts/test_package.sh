#!/usr/bin/env bash
# Smoke-test: bouwt de wheel en verifieert de gebruikersjournery end-to-end.
#
# Wat dit test:
#   1. Wheel bouwen en installeren slaagt
#   2. CLI is beschikbaar na installatie (--help)
#   3. `init` maakt de juiste mapstructuur aan (puur standalone)
#   4. Integratiepijplijn met repo-demodata — controleert dat de volledige
#      pipeline tot outputbestand loopt; data wordt gekopieerd vanuit de repo,
#      NIET meegeleverd in het wheel.
#   5. --noetl --yes combinatie — pipeline slaagt met al verwerkte data,
#      ETL en validatie worden overgeslagen
#   6. -sk (skipyears) vlag — pipeline slaagt met skip_years > 0
#   7. Exitcode 1 bij harde validatiefout — kapot telbestand triggert sys.exit(1)
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
echo "=== [4] Integratiepijplijn met repo-demodata ==="
cp -r "$REPO_ROOT/data/input_raw/telbestanden"           data/input_raw/
cp    "$REPO_ROOT/data/input_raw/individuele_aanmelddata.csv" data/input_raw/
cp    "$REPO_ROOT/data/input_raw/oktober_bestand.xlsx"   data/input_raw/
uv run studentprognose -w 6 -y 2020 --yes > /dev/null
[ -f data/output/output_first-years_beide.xlsx ] || { echo "FOUT: output ontbreekt"; exit 1; }
echo "OK"

echo ""
echo "=== [5] --noetl --yes combinatie ==="
# Stap 4 heeft de verwerkte inputbestanden al aangemaakt in data/input/.
# --noetl slaat ETL en validatie over; de pipeline moet slagen op de bestaande data.
uv run studentprognose -w 6 -y 2020 --noetl --yes > /dev/null
[ -f data/output/output_first-years_beide.xlsx ] || { echo "FOUT: output ontbreekt na --noetl run"; exit 1; }
echo "OK"

echo ""
echo "=== [6] -sk (skipyears) vlag ==="
# Bewaakt regressie op issue #109: Skip_prediction KeyError bij skip_years > 0.
uv run studentprognose -w 6 -y 2022 -d cumulative -sk 1 --noetl --yes > /dev/null
[ -f data/output/output_prelim_cumulatief.xlsx ] || { echo "FOUT: output ontbreekt na -sk run"; exit 1; }
echo "OK"

echo ""
echo "=== [7] Exitcode 1 bij harde validatiefout ==="
# Maak een kapot telbestand aan zonder de vereiste kolommen.
# De validatie moet een hard error geven en afsluiten met exitcode 1.
BROKEN_DIR="$(mktemp -d)"
trap 'rm -rf "$BROKEN_DIR"' EXIT
mkdir -p "$BROKEN_DIR/data/input_raw/telbestanden"
echo "col1;col2" > "$BROKEN_DIR/data/input_raw/telbestanden/telbestandY2024W10.csv"
echo "a;b"       >> "$BROKEN_DIR/data/input_raw/telbestanden/telbestandY2024W10.csv"
if (cd "$BROKEN_DIR" && uv run --with "$WHEEL" studentprognose -w 10 -y 2024 --yes 2>/dev/null); then
    echo "FOUT: exitcode 0 verwacht bij harde validatiefout, maar pipeline slaagde"
    exit 1
fi
echo "OK"

echo ""
echo "=== Klaar ==="
