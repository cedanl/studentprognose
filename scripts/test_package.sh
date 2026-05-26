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
#   8. Crash-injectie — STUDENTPROGNOSE_TEST_FORCE_WORKER_CRASH=1 forceert een
#      TerminatedWorkerError in de parallel-helper; de fallback uit PR #198 moet
#      kicken en de pipeline moet doorlopen tot outputbestand. Bewaakt issue #197
#      structureel, ook op Windows-runners.
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
trap 'rm -rf "$WORK_DIR" "$BROKEN_DIR"' EXIT
mkdir -p "$BROKEN_DIR/data/input_raw/telbestanden"
echo "col1;col2" > "$BROKEN_DIR/data/input_raw/telbestanden/telbestandY2024W10.csv"
echo "a;b"       >> "$BROKEN_DIR/data/input_raw/telbestanden/telbestandY2024W10.csv"
if (cd "$BROKEN_DIR" && uv run --with "$WHEEL" studentprognose -w 10 -y 2024 --yes 2>/dev/null); then
    echo "FOUT: exitcode 0 verwacht bij harde validatiefout, maar pipeline slaagde"
    exit 1
fi
echo "OK"

echo ""
echo "=== [8] Crash-injectie: fallback fires bij geforceerde worker-crash ==="
# Bewaakt issue #197/PR #198 structureel: STUDENTPROGNOSE_TEST_FORCE_WORKER_CRASH=1
# forceert een TerminatedWorkerError op de eerste joblib-poging in de SARIMA-stap.
# De fallback (n_jobs=2) moet kicken, de fallback-waarschuwing moet zichtbaar zijn
# in stdout, en de pipeline moet doorlopen tot een outputbestand.
#
# We zetten 'runtime.cpu_count' expliciet op 4 zodat n_jobs > FALLBACK_N_JOBS (=2),
# anders re-raised de helper de geforceerde crash (overeenkomstig echt OOM-gedrag).
uv run python - <<'PYEOF'
import json, pathlib
p = pathlib.Path("configuration/configuration.json")
c = json.loads(p.read_text(encoding="utf-8"))
c.setdefault("runtime", {})["cpu_count"] = 4
p.write_text(json.dumps(c, indent=2), encoding="utf-8")
PYEOF

CRASH_LOG="$WORK_DIR/crash-injection.log"
# Verwijder de output van eerdere stappen zodat de aanwezigheid-check
# bewijst dat deze run zelf schreef.
rm -f data/output/output_first-years_beide.xlsx
if ! STUDENTPROGNOSE_TEST_FORCE_WORKER_CRASH=1 \
        uv run studentprognose -w 6 -y 2020 --noetl --yes > "$CRASH_LOG" 2>&1; then
    echo "FOUT: pipeline crashte bij geforceerde worker-crash — fallback fired niet"
    tail -n 30 "$CRASH_LOG"
    exit 1
fi
[ -f data/output/output_first-years_beide.xlsx ] \
    || { echo "FOUT: output ontbreekt na crash-injectie run"; exit 1; }
grep -q "Opnieuw proberen met n_jobs=2" "$CRASH_LOG" \
    || { echo "FOUT: fallback-waarschuwing niet gevonden in log — fallback fired niet"; tail -n 30 "$CRASH_LOG"; exit 1; }
echo "OK"

echo ""
echo "=== Klaar ==="
