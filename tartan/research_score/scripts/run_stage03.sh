#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
set -a; source ./local.env; set +a
RUN_ID="stage03_data_$(date -u +%Y%m%dT%H%M%SZ)_nogit"
BASE="$OUTPUT_ROOT/01_data_audit"
for profile in transfer_primary proxy_pair_auxiliary; do
  "$PYTHON_BIN" -m tartan.research_score.preflight --stage 03 --profile "$profile" \
    --config tartan/research_score/configs/stage03_v1_2.yaml \
    --report "$OUTPUT_ROOT/preflight/${RUN_ID}_${profile}" --run-id "$RUN_ID"
done
"$PYTHON_BIN" -m tartan.research_score.scripts.build_manifests \
  --config tartan/research_score/configs/stage03_v1_2.yaml --profile all_available \
  --output "$BASE" --run-id "$RUN_ID"
"$PYTHON_BIN" -m tartan.research_score.scripts.audit_pairs --profile proxy_pair_auxiliary \
  --manifest-root "$BASE/$RUN_ID/proxy_pair_auxiliary" \
  --output "$BASE/$RUN_ID/proxy_pair_auxiliary/pairs"
"$PYTHON_BIN" -m tartan.research_score.scripts.audit_routes \
  --manifest-root "$BASE/$RUN_ID" --output "$BASE/$RUN_ID/route_set"
echo "$BASE/$RUN_ID"
