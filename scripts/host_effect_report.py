"""Is this host's qwen3-8b the same as Colab's? (HANDOFF §0.1 / §7 experiment 1)

    python3 scripts/host_effect_report.py [reference_store] [new_store]

The ladder is split across machines by VRAM, so host is perfectly confounded
with model size unless one rung is run on both. qwen3-8b is that rung
(models.OVERLAP_RUNG). If the paired difference here is significant, cells
collected on the two machines cannot be pooled without a host offset.

Reports per task and pooled, because a pooled null can hide two task effects
that cancel. Pairing is on (task, state_id): the seeding patch makes trial
index name a fixed initial scene, which is what verify_pairing_states.py
proves and what every number below depends on.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.analysis import host_effect, mcnemar, wilson  # noqa: E402
from opencap.cells import Cell  # noqa: E402
from opencap.models import OVERLAP_RUNG  # noqa: E402
from opencap.store import open_store  # noqa: E402

MODEL, TIER = OVERLAP_RUNG, "S1"


def rows(spec: str) -> list[dict]:
    store = open_store(spec)
    out = []
    for cell in sorted(store.completed()):
        if cell.model != MODEL or cell.tier != TIER:
            continue
        out.extend(store.read_cell(cell))
    return out


def main() -> int:
    ref_spec = sys.argv[1] if len(sys.argv) > 1 else "hf:savoji/capx-ab-on"
    new_spec = sys.argv[2] if len(sys.argv) > 2 else "hf:savoji/capx-host-check"
    ref, new = rows(ref_spec), rows(new_spec)
    if not ref or not new:
        print(f"reference: {len(ref)} episodes, new: {len(new)} -- need both")
        return 1

    def hosts(rs):
        return sorted({r.get("host_kind") or "?" for r in rs})
    print(f"reference {ref_spec}: {len(ref)} episodes, host {hosts(ref)}")
    print(f"new       {new_spec}: {len(new)} episodes, host {hosts(new)}")
    # A single host label on both sides is what makes this a host comparison.
    # Two labels in one arm would mean the arm is itself a mixture.
    if len(hosts(ref)) != 1 or len(hosts(new)) != 1:
        print("REFUSING: an arm spans more than one host_kind; "
              "it is a mixture, not a host.")
        return 1
    if hosts(ref) == hosts(new):
        print("REFUSING: both arms carry the same host_kind -- "
              "nothing to compare.")
        return 1

    by_task_ref = defaultdict(dict)
    by_task_new = defaultdict(dict)
    for r in ref:
        by_task_ref[r["task"]][r["state_id"]] = bool(r["success"])
    for r in new:
        by_task_new[r["task"]][r["state_id"]] = bool(r["success"])

    tasks = sorted(set(by_task_ref) & set(by_task_new))
    missing = sorted((set(by_task_ref) | set(by_task_new)) - set(tasks))
    print()
    print(f"{'task':<16}{'colab':>10}{'here':>10}{'delta':>9}"
          f"{'flips':>10}{'p':>9}")
    print("-" * 64)
    tot_a = tot_b = tot_n = 0
    for t in tasks:
        a, b = by_task_ref[t], by_task_new[t]
        shared = sorted(set(a) & set(b))
        sa = sum(a[s] for s in shared)
        sb = sum(b[s] for s in shared)
        mc = mcnemar({s: a[s] for s in shared}, {s: b[s] for s in shared})
        n = len(shared)
        tot_a += sa; tot_b += sb; tot_n += n
        print(f"{t:<16}{sa:>6}/{n:<3}{sb:>6}/{n:<3}{(sb-sa)/n:>+9.3f}"
              f"{mc.discordant:>10}{mc.p_value:>9.4f}")
    print("-" * 64)
    print(f"{'pooled':<16}{tot_a:>6}/{tot_n:<3}{tot_b:>6}/{tot_n:<3}"
          f"{(tot_b-tot_a)/tot_n:>+9.3f}")
    print(f"  colab {wilson(tot_a, tot_n)}")
    print(f"  here  {wilson(tot_b, tot_n)}")
    if missing:
        print(f"  tasks present in only one arm (excluded): {missing}")

    # The module's own test, pooled over tasks -- this is the number §0.1 asks
    # for. It hashes (task, state_id) so states stay distinct across tasks.
    he = host_effect(ref + new, MODEL, TIER)
    print()
    if he is None:
        print("host_effect(): not enough hosts to compare")
        return 1
    print(f"host_effect({MODEL}, {TIER}): {he}")
    # McNemar's `b` counts states the first arm won; sorted host labels put
    # colab-a100 first and h200 second, so b = colab-only, c = here-only.
    print(f"  discordant states: {he.discordant} of {tot_n}"
          f"  (colab-only wins {he.b}, here-only wins {he.c})")
    print()
    if he.p_value < 0.05:
        print("VERDICT: SIGNIFICANT. The two hosts do not agree. Cells "
              "collected here cannot be pooled with the Colab corpus without "
              "a host offset -- and every downstream comparison that crosses "
              "the two is suspect until that offset is characterised.")
        return 2
    print("VERDICT: no significant host effect on this rung. Pooling is "
          "defensible. Note what this does and does not say: it is one rung "
          "on three tasks, so it bounds the host effect where it was measured "
          "and assumes nothing about rungs that only ever ran on one machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
