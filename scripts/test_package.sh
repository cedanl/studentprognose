#!/usr/bin/env bash
# Test het geïnstalleerde package als standalone (simuleert een pip install).
# Gebruik: bash scripts/test_package.sh
# Vereiste: uv, de demo-data in data/input_raw/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "=== Build wheel ==="
cd "$REPO_ROOT"
uv build --out-dir "$WORK_DIR/dist" -q
WHEEL="$(ls "$WORK_DIR/dist"/*.whl | head -1)"
echo "Wheel: $WHEEL"

echo ""
echo "=== Installeer in geïsoleerde omgeving ==="
cd "$WORK_DIR"
uv init --no-readme -q
uv add "$WHEEL" -q

echo ""
echo "=== Smoke test: --help ==="
uv run studentprognose --help > /dev/null && echo "OK"

echo ""
echo "=== studentprognose init ==="
uv run studentprognose init

echo ""
echo "=== Kopieer demo-data ==="
cp -r "$REPO_ROOT/data/input_raw/telbestanden" "$WORK_DIR/data/input_raw/"
cp "$REPO_ROOT/data/input_raw/individuele_aanmelddata.csv" "$WORK_DIR/data/input_raw/"
cp "$REPO_ROOT/data/input_raw/oktober_bestand.xlsx" "$WORK_DIR/data/input_raw/"

echo ""
echo "=== Run: both, week 6, jaar 2020 ==="
uv run studentprognose -w 6 -y 2020 --yes 2>&1 | grep -E "(Dataset|Saving|Fout|Error)"

echo ""
echo "=== Run: cumulative only ==="
uv run studentprognose -w 6 -y 2020 -d cumulative --noetl --yes 2>&1 | grep -E "(Dataset|Saving|Fout)"

echo ""
echo "=== Run: higher-years zonder totaal-bestand (verwacht: nette foutmelding) ==="
uv run studentprognose -w 6 -y 2020 -d cumulative -sy higher-years --noetl --yes 2>&1 | grep -E "(Fout|Error)" || true

echo ""
echo "=== Run: niet-bestaand config pad (verwacht: waarschuwing) ==="
uv run studentprognose -w 6 -y 2020 -d cumulative -c /bestaat/niet.json --noetl --yes 2>&1 | grep -E "(Waarschuwing|Dataset|Saving)"

echo ""
echo "=== Klaar ==="
ls "$WORK_DIR/data/output/"
