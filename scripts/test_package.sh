#!/usr/bin/env bash
# Smoke-test voor het geïnstalleerde package als standalone.
#
# Wat dit test:
#   - Of de wheel installeerbaar is
#   - CLI flags: -w -y -d -c -f -sy -sk --ci --noetl --yes en afkortingen
#   - Slice-notatie voor weken en jaren
#   - Verwachte output-bestanden per modus
#   - Verwachte waarschuwingen en foutmeldingen
#
# Wat dit NIET test:
#   - Correctheid van de voorspellingen (alleen dat er output verschijnt)
#   - Alle combinaties van flags
#   - Werking met echte instellingsdata (custom kolomnamen, afwijkende config)
#   - Performance bij grote datasets
#   - Windows-compatibiliteit (script is bash-only)
#
# Aannames:
#   - uv is geïnstalleerd
#   - Demo-data in data/input_raw/ (telbestanden/, individuele_aanmelddata.csv,
#     oktober_bestand.xlsx) — representatief voor structuur, niet voor omvang
#   - Demo-data bevat jaren 2020–2024, weken 1–52
#
# Gebruik: bash scripts/test_package.sh [--skip-build]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKIP_BUILD="${1:-}"
PASS=0
FAIL=0

# ── helpers ──────────────────────────────────────────────────────────────────

ok()   { echo "  ✓ $1"; PASS=$((PASS+1)); }
fail() { echo "  ✗ $1"; FAIL=$((FAIL+1)); }

assert_exit0() {
    local label="$1"; shift
    if "$@" &>/dev/null; then ok "$label"; else fail "$label (verwacht exit 0)"; fi
}

assert_exit1() {
    local label="$1"; shift
    if ! "$@" &>/dev/null; then ok "$label"; else fail "$label (verwacht exit 1)"; fi
}

assert_output_contains() {
    local label="$1"; local pattern="$2"; shift 2
    local out
    out=$("$@" 2>&1 || true)
    if echo "$out" | grep -qE "$pattern"; then ok "$label"; else fail "$label (patroon '$pattern' niet gevonden in output)"; echo "    output was: $(echo "$out" | head -5)"; fi
}

assert_file_exists() {
    local label="$1"; local file="$2"
    if [ -f "$file" ]; then ok "$label"; else fail "$label (bestand ontbreekt: $file)"; fi
}

assert_dir_exists() {
    local label="$1"; local dir="$2"
    if [ -d "$dir" ]; then ok "$label"; else fail "$label (directory ontbreekt: $dir)"; fi
}

assert_file_nonempty() {
    local label="$1"; local file="$2"
    if [ -s "$file" ]; then ok "$label"; else fail "$label (bestand leeg of ontbreekt: $file)"; fi
}

assert_file_missing() {
    local label="$1"; local file="$2"
    if [ ! -f "$file" ]; then ok "$label"; else fail "$label (bestand bestaat maar zou er niet moeten zijn: $file)"; fi
}

run() { uv run studentprognose "$@"; }

# ── setup ─────────────────────────────────────────────────────────────────────

echo "=== Build wheel ==="
if [ "$SKIP_BUILD" = "--skip-build" ]; then
    WHEEL="$(ls "$REPO_ROOT/dist"/*.whl 2>/dev/null | sort -V | tail -1)"
    [ -z "$WHEEL" ] && echo "Geen wheel in dist/ — verwijder --skip-build" && exit 1
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
echo "=== Installeer in geïsoleerde omgeving ==="
uv init --no-readme -q
uv add "$WHEEL" -q
echo "Geïnstalleerd: $(uv run studentprognose --version 2>/dev/null || echo 'versie onbekend')"

# ── [1] init ──────────────────────────────────────────────────────────────────

echo ""
echo "=== [1] init ==="
run init > /dev/null
assert_file_exists "configuration/configuration.json aangemaakt"  "configuration/configuration.json"
assert_file_exists "configuration/filtering/base.json aangemaakt" "configuration/filtering/base.json"
assert_file_exists "data/input_raw/README.md aangemaakt"          "data/input_raw/README.md"
assert_dir_exists  "data/input_raw/telbestanden/ aangemaakt"      "data/input_raw/telbestanden"

echo "  (init nogmaals draaien mag geen fout geven)"
assert_exit0 "init is idempotent" run init

cp -r "$REPO_ROOT/data/input_raw/telbestanden" "$WORK_DIR/data/input_raw/"
cp "$REPO_ROOT/data/input_raw/individuele_aanmelddata.csv" "$WORK_DIR/data/input_raw/"
cp "$REPO_ROOT/data/input_raw/oktober_bestand.xlsx"        "$WORK_DIR/data/input_raw/"

# ── [2] basis ETL + predict ───────────────────────────────────────────────────

echo ""
echo "=== [2] ETL + predict (both, week 6, jaar 2020) ==="
run -w 6 -y 2020 --yes > /dev/null
assert_file_nonempty "output_prelim_beide.xlsx aangemaakt"        "data/output/output_prelim_beide.xlsx"
assert_file_nonempty "output_first-years_beide.xlsx aangemaakt"   "data/output/output_first-years_beide.xlsx"

# ── [3] dataset afkortingen ───────────────────────────────────────────────────

echo ""
echo "=== [3] Dataset afkortingen ==="
run -w 6 -y 2020 -d c --noetl --yes > /dev/null
assert_file_nonempty "-d c → output_prelim_cumulatief.xlsx"       "data/output/output_prelim_cumulatief.xlsx"

run -w 6 -y 2020 -d i --noetl --yes > /dev/null
assert_file_nonempty "-d i → output_prelim_individueel.xlsx"      "data/output/output_prelim_individueel.xlsx"

run -w 6 -y 2020 -d b --noetl --yes > /dev/null
assert_file_nonempty "-d b → output_prelim_beide.xlsx"            "data/output/output_prelim_beide.xlsx"

run -w 6 -y 2020 -d cumulative --noetl --yes > /dev/null
assert_file_nonempty "-d cumulative werkt"                        "data/output/output_prelim_cumulatief.xlsx"

run -w 6 -y 2020 -d individual --noetl --yes > /dev/null
assert_file_nonempty "-d individual werkt"                        "data/output/output_prelim_individueel.xlsx"

run -w 6 -y 2020 -d both --noetl --yes > /dev/null
assert_file_nonempty "-d both werkt"                              "data/output/output_prelim_beide.xlsx"

# ── [4] studentyear afkortingen ───────────────────────────────────────────────

echo ""
echo "=== [4] Studentyear flags ==="
run -w 6 -y 2020 -d c -sy f --noetl --yes > /dev/null
assert_file_nonempty "-sy f → first-years output"                 "data/output/output_first-years_cumulatief.xlsx"

run -w 6 -y 2020 -d c -sy v --noetl --yes > /dev/null
assert_file_nonempty "-sy v → volume output"                      "data/output/output_volume_cumulatief.xlsx"

assert_output_contains "-sy h zonder totaal-bestand → nette fout" \
    "Fout:.*totaal" \
    run -w 6 -y 2020 -d c -sy h --noetl --yes

assert_output_contains "-sy higher-years zelfde fout" \
    "Fout:.*totaal" \
    run -w 6 -y 2020 -d c -sy higher-years --noetl --yes

# ── [5] meerdere weken en jaren ───────────────────────────────────────────────

echo ""
echo "=== [5] Meerdere weken en jaren ==="
OUTPUT=$(run -w 5 6 7 -y 2020 -d c --noetl --yes 2>&1)
assert_output_contains "3 weken: week 5 voorspeld" "Predicting first-years: 2020-5" echo "$OUTPUT"
assert_output_contains "3 weken: week 6 voorspeld" "Predicting first-years: 2020-6" echo "$OUTPUT"
assert_output_contains "3 weken: week 7 voorspeld" "Predicting first-years: 2020-7" echo "$OUTPUT"

OUTPUT=$(run -y 2020 2021 -w 6 -d c --noetl --yes 2>&1)
assert_output_contains "2 jaren: 2020 voorspeld" "Predicting first-years: 2020-6" echo "$OUTPUT"
assert_output_contains "2 jaren: 2021 voorspeld" "Predicting first-years: 2021-6" echo "$OUTPUT"

# ── [6] slice-notatie ─────────────────────────────────────────────────────────

echo ""
echo "=== [6] Slice-notatie ==="
OUTPUT=$(run -w 4 : 6 -y 2020 -d c --noetl --yes 2>&1)
assert_output_contains "slice 4:6 → week 4" "Predicting first-years: 2020-4" echo "$OUTPUT"
assert_output_contains "slice 4:6 → week 5" "Predicting first-years: 2020-5" echo "$OUTPUT"
assert_output_contains "slice 4:6 → week 6" "Predicting first-years: 2020-6" echo "$OUTPUT"

OUTPUT=$(run -w 4:6 -y 2020 -d c --noetl --yes 2>&1)
assert_output_contains "slice 4:6 (geen spaties) → week 4" "Predicting first-years: 2020-4" echo "$OUTPUT"

# ── [7] --ci test N ───────────────────────────────────────────────────────────

echo ""
echo "=== [7] --ci test N ==="
# CI subset filtert op year >= 2022 (seed=42), dus jaar moet in dat bereik vallen
assert_exit0 "--ci test 1 draait zonder fout" \
    run -w 6 -y 2022 -d c --ci test 1 --noetl --yes

assert_file_missing "--ci genereert geen normaal outputbestand" \
    "data/output/output_first-years_cumulatief_ci_test_N1.xlsx"

assert_output_contains "--ci ongeldige syntax geeft fout" \
    "Invalid --ci" \
    run --ci foo 1 --noetl --yes

# ── [8] -sk skipyears ─────────────────────────────────────────────────────────

echo ""
echo "=== [8] -sk skipyears ==="
# BEKENDE BUG: -sk veroorzaakt KeyError 'Skip_prediction' in postprocessor.
# Zie: cedanl/studentprognose (fix/pypi-standalone-packaging) — gerapporteerd als bevinding.
assert_output_contains "-sk 1 → bekende bug: KeyError Skip_prediction" \
    "Skip_prediction|KeyError" \
    run -w 6 -y 2020 -d c -sk 1 --noetl --yes

# ── [9] configuratie flags ────────────────────────────────────────────────────

echo ""
echo "=== [9] Configuratie flags ==="
assert_output_contains "-c niet-bestaand pad → waarschuwing" \
    "Waarschuwing:.*niet gevonden" \
    run -w 6 -y 2020 -d c -c /bestaat/niet.json --noetl --yes

cat > /tmp/custom.json << 'EOF'
{ "numerus_fixus": { "B Testprogramma": 100 } }
EOF
assert_exit0 "-c geldig custom config werkt" \
    run -w 6 -y 2020 -d c -c /tmp/custom.json --noetl --yes

FILTER_FILE="$WORK_DIR/configuration/filtering/base.json"
assert_exit0 "-f filtering file werkt" \
    run -w 6 -y 2020 -d c -f "$FILTER_FILE" --noetl --yes

assert_output_contains "-f niet-bestaand filteringbestand → waarschuwing" \
    "Waarschuwing:.*niet gevonden" \
    run -w 6 -y 2020 -d c -f /bestaat/niet.json --noetl --yes

# ── [10] randgevallen ─────────────────────────────────────────────────────────

echo ""
echo "=== [10] Randgevallen ==="
assert_output_contains "jaar niet in data → fout en exit" \
    "Waarschuwing:.*niet.*beschikbaar|niet.*volledig" \
    run -w 6 -y 2030 -d c --noetl --yes

assert_output_contains "week niet in data → fout en exit" \
    "Waarschuwing:.*niet.*beschikbaar|niet.*volledig" \
    run -w 99 -y 2020 -d c --noetl --yes

assert_output_contains "ongeldige -d waarde → argparse fout" \
    "invalid choice|error" \
    run -w 6 -y 2020 -d xyz --noetl --yes

assert_output_contains "--noetl zonder verwerkte data → melding of doorgaan" \
    "Loading data" \
    run -w 6 -y 2020 -d c --noetl --yes

# ── samenvatting ──────────────────────────────────────────────────────────────

echo ""
echo "=== Samenvatting ==="
echo "  Geslaagd:   $PASS"
echo "  Mislukt:    $FAIL"
[ "$FAIL" -eq 0 ] && echo "  Alle tests geslaagd." || { echo "  Er zijn mislukte tests."; exit 1; }
