"""Per-model defect profile: each model's failed S1 programs vs its failed S3/S4 programs.

    python3 scripts/per_model_profile.py [--verified-only]

Answers whether the reduced-API defect families (FRAME, INTERFACE) come from a
few models or from the whole ladder. Hand tags on both sides
(results/tags_s1_hand.json, results/tags_reduced.json); truncation-only
records excluded, as in cross_tier_report.py.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from opencap.editclass import FAMILY  # noqa: E402
from opencap.models import BY_KEY  # noqa: E402

_spec = importlib.util.spec_from_file_location("_ctr", ROOT / "scripts/cross_tier_report.py")
ctr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ctr)


def profiles(verified_only: bool = False) -> dict[str, dict[str, dict]]:
    g = ctr.load(ROOT)
    g = {k: [r for r in v if not (r.get("truncated") and set(r["tags"]) <= {"syntax"})] for k, v in g.items()}
    if verified_only:
        g = {k: [r for r in v if not str(r.get("level") or "").startswith("L5")] for k, v in g.items()}
    out: dict[str, dict[str, dict]] = collections.defaultdict(lambda: {
        "S1": {"n": 0, "tags": 0, "FRAME": 0, "INTERFACE": 0},
        "S3S4": {"n": 0, "tags": 0, "FRAME": 0, "INTERFACE": 0}})
    for tier, key in (("S1", "S1"), ("S3", "S3S4"), ("S4", "S3S4")):
        for r in g[tier]:
            fams = {FAMILY[t] for t in r["tags"] if t in FAMILY}
            n = len([t for t in r["tags"] if t in FAMILY])
            for m in r["models"]:
                p = out[m][key]
                p["n"] += 1
                p["tags"] += n
                p["FRAME"] += "FRAME" in fams
                p["INTERFACE"] += "INTERFACE" in fams
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified-only", action="store_true")
    a = ap.parse_args()
    prof = profiles(a.verified_only)
    models = sorted(prof, key=lambda m: (BY_KEY[m].params_b, m))
    print(f"{'model':18s} {'B':>5s} | {'S1 n':>4s} {'tags/p':>6s} {'FRAME':>6s} {'INTF':>6s} | {'S3/4 n':>6s} {'tags/p':>6s} {'FRAME':>6s} {'INTF':>6s}")
    for m in models:
        a1, b1 = prof[m]["S1"], prof[m]["S3S4"]
        def f(p, k): return f"{p[k]}/{p['n']}" if p["n"] else "-"
        def t(p): return f"{p['tags'] / p['n']:.2f}" if p["n"] else "-"
        print(f"{m:18s} {BY_KEY[m].params_b:5.1f} | {a1['n']:4d} {t(a1):>6s} {f(a1,'FRAME'):>6s} {f(a1,'INTERFACE'):>6s} | "
              f"{b1['n']:6d} {t(b1):>6s} {f(b1,'FRAME'):>6s} {f(b1,'INTERFACE'):>6s}")
    with_frame = [m for m in models if prof[m]["S3S4"]["FRAME"] > 0]
    with_intf = [m for m in models if prof[m]["S3S4"]["INTERFACE"] > 0]
    print(f"\nmodels with >=1 FRAME defect at S3/S4: {len(with_frame)}/{len(models)}; at S1: "
          f"{sum(1 for m in models if prof[m]['S1']['FRAME'] > 0)}/{len(models)}")
    print(f"models with >=1 INTERFACE defect at S3/S4: {len(with_intf)}/{len(models)}; at S1: "
          f"{sum(1 for m in models if prof[m]['S1']['INTERFACE'] > 0)}/{len(models)}")
    up = sum(1 for m in models if prof[m]["S1"]["n"] and prof[m]["S3S4"]["n"]
             and prof[m]["S3S4"]["tags"] / prof[m]["S3S4"]["n"] > prof[m]["S1"]["tags"] / prof[m]["S1"]["n"])
    both = sum(1 for m in models if prof[m]["S1"]["n"] and prof[m]["S3S4"]["n"])
    print(f"models whose defects/program rise from S1 to S3/S4: {up}/{both}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
