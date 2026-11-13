#!/usr/bin/env bash
# Keep the repair queue full. Runs from cron; starts the next unfinished task
# batch whenever nothing is running, so the GPU never idles between tasks.
#
# Order is deliberate: spill_wipe first (deterministic, done), then the three
# grasping tasks. nut_assembly last -- its Gate 0 has not cleared, so its
# batch will tag but not accept, which is still useful output.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="${PATH:+$PATH:}/usr/bin:/bin:/usr/local/bin"
LOG=/data/capx-logs/pipeline.log
OFF=/data/capx-logs/pipeline.off
[ -e "$OFF" ] && exit 0
pgrep -f '[s]pill_wipe_batch.py' >/dev/null 2>&1 && exit 0

say(){ echo "[pipe $(date -u +'%m-%d %H:%M')] $*" >> "$LOG"; }

for T in cube_lifting cube_stack nut_assembly; do
  MARK="/data/capx-logs/${T}_batch.done"
  [ -e "$MARK" ] && continue
  # done when both tiers appear in its results file
  n=$(python3 - "$T" <<'PY' 2>/dev/null || echo 0
import json,sys,pathlib
f=pathlib.Path(f"results/{sys.argv[1]}_batch.json")
if not f.exists(): print(0); raise SystemExit
d=json.loads(f.read_text())
print(len({(e["tier"]) for e in d.values()}))
PY
)
  if [ "${n:-0}" -ge 2 ]; then say "$T: both tiers processed -- marking done"; : > "$MARK"; continue; fi
  say "starting $T batch"
  for TIER in S3 S4; do
    python3 scripts/spill_wipe_batch.py --task "$T" --tier "$TIER" \
        --out "results/${T}_batch.json" >> "$LOG" 2>&1
  done
  say "$T batch finished"
  exit 0
done
say "all task batches complete"
: > "$OFF"
