#!/usr/bin/env bash
# Unattended spill_wipe repair pass, S3 then S4. Resumable: it skips any pid
# already at ceiling, so re-invoking it after a kill costs nothing.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export OPENCAP_GPU="${OPENCAP_GPU:-6}"
source scripts/env_server.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES="$OPENCAP_GPU"   # re-derive; env_server fixes it at source time
export MUJOCO_EGL_DEVICE_ID="$OPENCAP_GPU"
for p in 8114 8115 8116 8122; do
  (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null || {
    echo "[batch] :$p is DOWN -- starting services"
    bash scripts/start_vision_servers.sh && bash scripts/start_molmo_server.sh || {
      echo "[batch] services would not start; refusing to score a dead service as a defect"
      exit 1; }
    break; }
done
for T in S3 S4; do
  echo "######## spill_wipe batch, tier $T ######## $(date -u)"
  python3 scripts/spill_wipe_batch.py --tier "$T"
done
echo "SPILL_BATCH_DONE $(date -u)"
