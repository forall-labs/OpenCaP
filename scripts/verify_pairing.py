"""Prove that trial index -> deterministic initial state.

Everything in opencap.analysis assumes a given trial index means the same
initial state for every model. If CaP-X reseeds per run, that assumption is
false, `state_id` is meaningless, and every paired test in the paper is
invalid -- silently, because the numbers would still look plausible.

The oracle makes this cheap to check: its program is fixed and LLM-free, so
two runs can only differ if the *environment* differed.

    # on the pod, run the same oracle config twice into separate dirs
    for i in 1 2; do
      uv run capx/envs/launch.py \
        --config-path env_configs/human_oracle_code/robosuite/franka_robosuite_cube_stack_privileged_oracle.yaml \
        --total-trials 30 --num-workers 12 --record-video False \
        --output-dir /content/outputs/pairing_$i
    done
    python3 scripts/verify_pairing.py \
        /content/outputs/google_gemini-3.1-pro-preview/pairing_1 \
        /content/outputs/google_gemini-3.1-pro-preview/pairing_2
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from opencap.results import cross_check, parse_trials

# Long jobs are launched detached with stdout redirected to a file, where
# Python block-buffers by default -- the log stays empty until exit, so a
# multi-hour run shows no progress and a crash shows no context. Line
# buffering costs nothing and makes the log readable while it runs.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass



def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    a_dir, b_dir = sys.argv[1], sys.argv[2]

    for d in (a_dir, b_dir):
        ok, msg = cross_check(d)
        if not ok:
            print(f"[fail] {d} does not self-verify: {msg}")
            return 1

    a = {t.trial: t for t in parse_trials(a_dir)}
    b = {t.trial: t for t in parse_trials(b_dir)}
    shared = sorted(set(a) & set(b))
    if not shared:
        print("[fail] no shared trial indices")
        return 1

    mismatched = [i for i in shared
                  if a[i].success != b[i].success
                  or abs(a[i].reward - b[i].reward) > 1e-6]

    print(f"compared {len(shared)} trial indices")
    print(f"  success rate A: {sum(t.success for t in a.values())}/{len(a)}")
    print(f"  success rate B: {sum(t.success for t in b.values())}/{len(b)}")
    print(f"  mismatched:     {len(mismatched)}")
    for i in mismatched[:10]:
        print(f"    trial {i}: A(reward={a[i].reward}, ok={a[i].success}) "
              f"!= B(reward={b[i].reward}, ok={b[i].success})")

    if mismatched:
        print("\n[FAIL] trial index does NOT determine the initial state.")
        print("Paired analysis is invalid as designed. Options: seed the env")
        print("explicitly per trial, or drop pairing and pay for the wider")
        print("unpaired intervals (roughly +/-8pp at n=150 -- enough to hide")
        print("every effect this study is trying to measure).")
        return 1

    print("\n[OK] trial index determines the initial state. Pairing is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
