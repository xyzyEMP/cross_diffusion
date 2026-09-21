#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
set -a
source "$ROOT/local.env"
set +a
cd "$PROJECT_ROOT"
RUN_ID="stage02_transfer_primary_$(date -u +%Y%m%dT%H%M%SZ)_nogit"
RUN_ROOT="$OUTPUT_ROOT/00_baseline_freeze/$RUN_ID"
mkdir -p "$RUN_ROOT"
printf '%s\n' "$0" > "$RUN_ROOT/command.txt"
set +e
"$PYTHON_BIN" -m tartan.research_score.preflight --stage 02 --profile transfer_primary --config tartan/research_score/configs/stage02.yaml --report "$RUN_ROOT/preflight" --run-id "$RUN_ID" > "$RUN_ROOT/preflight.log" 2>&1
PREFLIGHT_RC=$?
set -e
cp "$RUN_ROOT/preflight/config.resolved.yaml" "$RUN_ROOT/config.resolved.yaml"
cp "$RUN_ROOT/preflight/config.sha256" "$RUN_ROOT/config.sha256"
"$PYTHON_BIN" -m pytest tartan/tests -q 2>&1 | tee "$RUN_ROOT/tartan_tests.log"
"$PYTHON_BIN" -m pytest tartan/research_score/preflight/tests -q 2>&1 | tee "$RUN_ROOT/preflight_tests.log"
"$PYTHON_BIN" -m pytest tartan/research_score/tests/test_source_regression.py -q 2>&1 | tee "$RUN_ROOT/source_regression_tests.log"
"$PYTHON_BIN" -m tartan.research_score.scripts.audit_source_model --args "$SOURCE_ARGS" --checkpoint "$SOURCE_CKPT" --device cpu --output "$RUN_ROOT" 2>&1 | tee "$RUN_ROOT/source_static_audit.log"
printf '{\n  "run_id": "%s",\n  "status": "AWAITING_GPU",\n  "preflight_exit_code": %s,\n  "preflight_expected_failure": "cuda.available",\n  "oracle_route": true,\n  "profile": "transfer_primary"\n}\n' "$RUN_ID" "$PREFLIGHT_RC" > "$RUN_ROOT/run_manifest.json"
echo "$RUN_ROOT"
