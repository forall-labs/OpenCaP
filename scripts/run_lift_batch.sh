#!/usr/bin/env bash
# cube_lifting batch, S3 then S4. Resumable.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export OPENCAP_GPU="${OPENCAP_GPU:-6}"
source scripts/env_server.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES="$OPENCAP_GPU"
export MUJOCO_EGL_DEVICE_ID="$OPENCAP_GPU"
for p in 8114 8115 8116 8122; do
  (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null || {
    bash scripts/start_vision_servers.sh && bash scripts/start_molmo_server.sh || exit 1; break; }
done
for T in S3 S4; do
  echo "######## cube_lifting batch, tier $T ######## $(date -u)"
  python3 scripts/spill_wipe_batch.py --task cube_lifting --tier "$T" \
      --out results/cube_lifting_batch.json
done
echo "LIFT_BATCH_DONE $(date -u)"
