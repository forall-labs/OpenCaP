"""Which cells were cut off at the token cap, and therefore mislabelled?

    python3 scripts/truncation_audit.py                  # every phase, every model
    python3 scripts/truncation_audit.py --since 2026-09-21

CANDIDATE MEASUREMENT ARTIFACT #15
----------------------------------
`--max-tokens 2048` is a collection condition and is not in question. What is
missing is the RECORD. A program cut off mid-sentence enters the corpus with
`tokens_out = 0`, `truncation_fired` false, and a failure label of
`api_misuse` or `no_op` -- indistinguishable from a program the model finished
and got wrong. Measured in Phase 7: nine of eighty-five sampled cells hit the
cap on every request, seven of them "parsing" only because the cut fell inside
a comment.

It reaches back into the greedy corpus. Of 13,704 greedy-era requests from
Phases 5 and 6, 1,438 (10.5%) hit the cap, including 77% of qwen3.5-0.8B's and
73% of qwen3.5-2B's. Some of what the corpus reports as those models' defects
is the first 2,048 tokens of a program that had no ending.

At the extreme it stops being a labelling problem: the qwen3.5-27B sampled
draw that runs to the cap on 93% of its requests cannot be collected at all,
because two such generations queue past CaP-X's read timeout. See STATUS
2026-09-28.

THE ONLY RECORD IS THE SHIM'S
-----------------------------
scripts/vllm_shim.py writes one line per request to
$OPENCAP_LOGDIR/stats_<model>.jsonl with `completion_tokens` and a timestamp.
Nothing downstream reads it. This script does, and prints the per-model rate
so a cell can be checked before its number is believed.

The files are CUMULATIVE ACROSS PHASES -- a model's stats from the rebuild,
the leak A/B and the sampled arm all append to one file -- so `--since` is the
only way to ask about one run.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.analysis import wilson  # noqa: E402

#: The cap every cell in this corpus was collected under (run_cell.py's
#: --max-tokens). A request reporting this many completion tokens stopped
#: because it ran out of budget, not because the model finished.
TOKEN_CAP = 2048


def load(logdir: Path, since: float | None,
         until: float | None = None) -> dict:
    """{model: [completion_tokens, ...]} from the shim's stats files."""
    out: dict = defaultdict(list)
    for f in sorted(glob.glob(str(logdir / "stats_*.jsonl"))):
        model = Path(f).name[len("stats_"):-len(".jsonl")]
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = r.get("t", 0)
            if since is not None and t < since:
                continue
            if until is not None and t >= until:
                continue
            ct = r.get("completion_tokens")
            if isinstance(ct, int):
                out[model].append(ct)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir",
                    default=os.environ.get("OPENCAP_LOGDIR", "/data/capx-logs"))
    ap.add_argument("--since", default=None,
                    help="ISO date, e.g. 2026-09-21: ignore requests before it")
    ap.add_argument("--until", default=None,
                    help="ISO date: ignore requests at or after it. With "
                         "--until 2026-09-21 this gives the GREEDY-era rate, "
                         "which is the one the published tables were "
                         "collected under.")
    ap.add_argument("--cap", type=int, default=TOKEN_CAP)
    a = ap.parse_args()

    def _ts(v):
        return None if not v else datetime.fromisoformat(v).replace(
            tzinfo=timezone.utc).timestamp()

    since, until = _ts(a.since), _ts(a.until)
    by_model = load(Path(a.logdir), since, until)
    if not by_model:
        print(f"no stats files in {a.logdir}"
              + (f" since {a.since}" if a.since else "")
              + (f" before {a.until}" if a.until else ""))
        return 1

    print("=" * 72)
    print(f"TRUNCATION AUDIT -- requests that hit the {a.cap}-token cap".center(72))
    print("=" * 72)
    if a.since or a.until:
        print(f"\nwindow: {a.since or 'start'} .. {a.until or 'now'}")
    print(f"\n{'model':<20}{'requests':>10}{'at cap':>9}{'rate':>8}"
          f"{'95%':>16}{'mean tok':>10}")
    print("-" * 72)

    tot = cap_tot = 0
    rows = []
    for model, cts in by_model.items():
        n = len(cts)
        k = sum(1 for c in cts if c >= a.cap)
        tot += n
        cap_tot += k
        rows.append((k / n, model, k, n, sum(cts) / n))
    # Worst first: the models whose numbers are least trustworthy.
    for rate, model, k, n, mean_tok in sorted(rows, reverse=True):
        iv = wilson(k, n)
        flag = "  <-- audit" if rate >= 0.20 else ""
        print(f"{model:<20}{n:>10}{k:>9}{rate:>8.1%}"
              f"{f'[{iv.lo:.2f},{iv.hi:.2f}]':>16}{mean_tok:>10.0f}{flag}")

    iv = wilson(cap_tot, tot)
    print("-" * 72)
    print(f"{'ALL':<20}{tot:>10}{cap_tot:>9}{cap_tot/tot:>8.1%}"
          f"{f'[{iv.lo:.2f},{iv.hi:.2f}]':>16}")

    # THE CUT THAT MATTERS. "Everything falls with scale -- defects/program
    # 3.21 -> 1.15" is the project's headline, and it is anchored at the small
    # end. If truncation is concentrated there, some of those defects are
    # programs that were still being written.
    try:
        from opencap.models import BY_KEY  # noqa: E402

        def band_of(m: str) -> str:
            p = BY_KEY[m].params_b
            return "<=2B" if p <= 2 else "3-4B" if p <= 4 else \
                   "7-9B" if p <= 9 else ">=12B"

        band: dict = defaultdict(lambda: [0, 0])
        for _, model, k, n, _ in rows:
            if model in BY_KEY:
                b = band[band_of(model)]
                b[0] += k
                b[1] += n
        if band:
            print(f"\n{'band':<10}{'requests':>10}{'at cap':>9}{'rate':>8}{'95%':>16}")
            for b in ("<=2B", "3-4B", "7-9B", ">=12B"):
                if b not in band:
                    continue
                k, n = band[b]
                iv = wilson(k, n)
                print(f"{b:<10}{n:>10}{k:>9}{k/n:>8.1%}"
                      f"{f'[{iv.lo:.2f},{iv.hi:.2f}]':>16}")
            print("\nA defects-per-program trend measured across these bands is")
            print("measured across different truncation rates. That is a")
            print("confound, not a correction -- it says the small end needs")
            print("checking, not that the trend is wrong.")
    except Exception as exc:  # pragma: no cover - the per-model table still stands
        print(f"\n[band cut unavailable: {exc}]")

    bad = [r for r in rows if r[0] >= 0.20]
    print(f"\n{len(bad)} of {len(rows)} models truncate on 20% or more of their "
          f"requests.")
    print("For those, a `no_op` or `api_misuse` label is not safe to read as a")
    print("model defect without checking whether the program had an ending.")
    print("\nThe store cannot answer this: it records tokens_out = 0 for every")
    print("episode in the corpus. Until it carries a finish reason, this file")
    print("and the shim's stats are the only record that the question exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
