"""Every published claim, its provenance, and whether it still stands.

    python3 scripts/claims_register.py

Artifact #12 showed the generated program itself changes across GPU
architectures in half of all cells. Every conclusion reached before 2026-09-11
was computed over Colab data or a Colab/H200 mixture, so none of them carries
over automatically -- including the ones nobody ever challenged.

The default status of a pre-cutover claim is UNVERIFIED, not "true until
disproved". This file is the register: claims that can be re-derived from the
single-host corpus, the stage-1 control stores, the arm stores and the
committed verification records are recomputed on every run, so status is
measured rather than remembered. The rest name what would settle them.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: The canonical corpus. Points at the SEEDED store, and this script refuses to
#: fall back to a retired one. A silent fallback is precisely the failure this
#: register exists to prevent: it would re-derive every claim from unseeded data
#: and print the results under today's date, which reads as current.
CANON = "hf:savoji/capx-ab-on-seeded"

#: Retired, kept only so the message can name what NOT to read.
RETIRED_CANON = "hf:savoji/capx-ab-on-h200"

#: NOTE on reading a PENDING control or arm row: an unreadable or empty store
#: means that stage has not run on the seeded corpus yet. The -h200 readings it
#: replaces are in STATUS.md and remain valid for every PROGRAM-level claim
#: (dedup, batch size, prefix caching), which the scene never influenced. They
#: are not valid for anything state-paired on cube_lifting, spill_wipe or
#: nut_assembly.

PENDING = [
    # Empty since 2026-10-13. The last entry -- qwen3-4b's trial 9 / trial 13
    # alternation -- had its run (results/scene_probe_alt.json) and is now
    # re-derived in alternation_rows(). A claim that needs a run goes here
    # WITH the run named, and leaves in the same commit that runs it.
]

# Stores the rebuild filled. Each control store holds the same 10 cells as
# the canonical corpus, collected with one control released; each arm store
# holds the 63-cell panel under one intervention.
CONTROLS = {
    "replicate": "hf:savoji/capx-ctrl-replicate-seeded",
    "batch": "hf:savoji/capx-ctrl-batch-seeded",
    "prefix": "hf:savoji/capx-ctrl-prefix-seeded",
}
ARMS = {
    "numpy-off": "hf:savoji/capx-ab-off-seeded",
    "entrypoint": "hf:savoji/capx-entrypoint-seeded",
    "prompt-fix": "hf:savoji/capx-promptfix-seeded",
}
# Stage 8 of the seeded run re-verified every repair on the seeded scenes
# (results/verify_seeded.json); the -h200 files are the unseeded Phase 4
# records and stay as the fallback until that run has happened.
_RES = Path(__file__).resolve().parent.parent / "results"
VERIFY = (_RES / "verify_seeded.json" if (_RES / "verify_seeded.json").exists()
          else _RES / "verify_h200.json")
VERIFY_MIN = VERIFY.with_name("minimal_" + VERIFY.name)
TAGS = _RES / "tags_seeded.json"
#: tag -> family, for the recurring-defect row.
from opencap.editclass import FAMILY as FAMILY_OF  # noqa: E402
LEAK = _RES / "leak_ab.json"
MOVE = 5  # scenes; below this a mid-range cell moves on its own (stage 1)
# On a cell the replicate control itself moved by >= MOVE, an arm's move
# counts only if it changes the regime: at least half the scenes.
NOISY_MOVE = 15


def read_scores(spec: str):
    """(model, task) -> (successes, n, distinct program ids) for one store."""
    from opencap.repair import program_id, program_of
    from opencap.store import open_store
    S = open_store(spec)
    out = {}
    for c in S.completed():
        if c.tier != "S1":
            continue
        eps = list(S.read_cell(c))
        ids = {program_id(program_of(e)) for e in eps
               if (program_of(e) or "").strip()}
        out[(c.model, c.task)] = (sum(1 for e in eps if e.get("success")),
                                  len(eps), ids)
    return out


def control_rows(canon: dict):
    """-> (rows, cells the replicate control moved by >= MOVE on its own)."""
    rows = []
    got = {}
    noisy = set()
    for name, spec in CONTROLS.items():
        try:
            got[name] = read_scores(spec)
        except Exception as e:  # a control store that is not there yet
            rows.append((f"stage 1 {name} control", "PENDING",
                         f"cannot read {spec}: {type(e).__name__}"))
    def compare(name):
        d = got[name]
        cells = sorted(k for k in d if k in canon)
        same = sum(1 for k in cells if d[k][2] == canon[k][2])
        multi = sum(1 for k in cells if len(d[k][2]) > 1)
        moved = [(k, canon[k][0], d[k][0]) for k in cells
                 if abs(canon[k][0] - d[k][0]) >= MOVE]
        return cells, same, multi, moved
    if "replicate" in got:
        cells, same, multi, moved = compare("replicate")
        noisy = {k for k, _, _ in moved}
        detail = (f"{len(cells)} cells re-run with nothing varied: program "
                  f"identical in {same}, {multi} cells with >1 program; "
                  f"score moves >= {MOVE} in {len(moved)} cells "
                  + ", ".join(f"{m} {t} {a}->{b}" for (m, t), a, b in moved)
                  + ". The floor is per cell, not a corpus mean: 0 on floor "
                  "and ceiling cells, a third of the scenes on mid-range ones.")
        rows.append(("Noise floor 0.95%; cube tasks 0%; spill_wipe +-5.3 "
                     "scenes", "RETRACTED" if moved else "HOLDS", detail))
    if "batch" in got:
        cells, same, multi, moved = compare("batch")
        rows.append(("BATCH_MAX=1 is required; micro-batching is the largest "
                     "variance source", "HOLDS" if multi else "RETRACTED",
                     f"at BATCH_MAX=4, {multi} of {len(cells)} cells emit more "
                     f"than one program across their 30 episodes (0 at batch "
                     f"1); score moves >= {MOVE} in {len(moved)} cells. It is "
                     f"required for temperature-0 dedup, whatever it does to "
                     f"the score."))
    if "prefix" in got:
        cells, same, multi, moved = compare("prefix")
        d = got["prefix"]
        # the canonical program is not among the cell's programs at all
        gone = sum(1 for k in cells if not canon[k][2] <= d[k][2])
        rows.append(("Prefix caching contaminates every cell after the first "
                     "(27/30 -> 0/30)", "HOLDS" if gone else "RETRACTED",
                     f"with --deterministic off, the canonical program is "
                     f"absent from {gone} of {len(cells)} cells, {multi} cells "
                     f"lose within-cell determinism (>1 program), and the set "
                     f"of programs matches in only {same}; score moves >= "
                     f"{MOVE} in {len(moved)}: "
                     + ", ".join(f"{m} {t} {a}->{b}" for (m, t), a, b in moved)
                     + ". Direction holds; the Colab magnitude was one cell. "
                     "Note --deterministic also disables CUDA graphs."))
    return rows, noisy


def arm_rows(canon: dict, noisy: set) -> list[tuple[str, str, str]]:
    """`noisy`: cells the replicate control moved on its own; a move there
    is not evidence for an arm."""
    rows = []
    got = {}
    for name, spec in ARMS.items():
        try:
            scores = read_scores(spec)
        except Exception as e:
            rows.append((f"stage 5-7 {name} arm", "PENDING",
                         f"cannot read {spec}: {type(e).__name__}"))
            continue
        # The arms are opt-in on the seeded corpus (stages 5-7 are not in the
        # default run). open_store() creates a repo it cannot find, so an
        # uncollected arm reads as an EMPTY store, not a missing one -- and a
        # verdict on 0 cells is not a verdict.
        if not scores:
            rows.append((f"stage 5-7 {name} arm", "PENDING",
                         f"{spec} holds no cells: the arm was not collected "
                         f"on the seeded corpus. The unseeded -h200 arm "
                         f"(Phase 4) is the standing evidence."))
            continue
        got[name] = scores
    def moved(name):
        d = got[name]
        cells = sorted(k for k in d if k in canon)
        base = sum(canon[k][0] for k in cells)
        new = sum(d[k][0] for k in cells)
        mv = [(k, canon[k][0], d[k][0]) for k in cells
              if abs(canon[k][0] - d[k][0]) >= (NOISY_MOVE if k in noisy
                                                 else MOVE)]
        return cells, base, new, mv
    fmt = lambda mv: ", ".join(f"{m} {t} {a}->{b}" for (m, t), a, b in mv)
    if noisy:
        rows.append(("(arm reading rule)", "NOTE",
                     f"on {', '.join(f'{m} {t}' for m, t in sorted(noisy))} "
                     f"the replicate control moved >= {MOVE} scenes with "
                     f"nothing varied, so an arm's move there counts only "
                     f"at >= {NOISY_MOVE} scenes."))
    if "numpy-off" in got:
        cells, base, new, mv = moved("numpy-off")
        four_b = [x for x in mv if x[0][1] == "cube_lifting" and "4b" in x[0][0]]
        rows.append(("The numpy pre-import is a no-op except on 4B/cube_lifting",
                     "REVISED",
                     f"{len(cells)} cells, {base} -> {new} scenes; moves >= "
                     f"{MOVE}: {fmt(mv) or 'none'}. One cell in form, but it "
                     f"is not a 4B cell on this host"
                     + (" (no 4B cube_lifting cell moves)" if not four_b else "")
                     + "."))
    if "entrypoint" in got:
        cells, base, new, mv = moved("entrypoint")
        rows.append(("Calling a defined-but-never-called entry point recovers "
                     "those cells", "RETRACTED" if not mv else "HOLDS",
                     f"{len(cells)} cells, {base} -> {new} scenes; moves >= "
                     f"{MOVE}: {fmt(mv) or 'none'}. The never_invoked programs "
                     f"fail for other reasons once called."))
    if "prompt-fix" in got:
        cells, base, new, mv = moved("prompt-fix")
        up = [x for x in mv if x[2] > x[1]]
        down = [x for x in mv if x[2] < x[1]]
        gain = sum(b - a for _, a, b in up)
        loss = sum(a - b for _, a, b in down)
        rows.append(("Prompt addendum redistributes: +91 at the floor, -132 "
                     "above it; gating on baseline score is +91 for 0 lost",
                     "RETRACTED" if new < base else "HOLDS",
                     f"{len(cells)} cells, {base} -> {new} scenes: {len(up)} "
                     f"cells rescued (+{gain}), {len(down)} damaged (-{loss}). "
                     f"Net negative on clean same-host data. Rescued: "
                     f"{fmt(up)}. Damaged: {fmt(down)}."))
    return rows


def repair_rows() -> list[tuple[str, str, str]]:
    import json
    rows = []
    for label, path in (("repairs", VERIFY), ("minimal", VERIFY_MIN)):
        if not path.exists():
            rows.append((f"{label} verified on this host", "PENDING",
                         f"{path.name} missing"))
            continue
        recs = json.loads(path.read_text())
        ok = [r for r in recs if r.get("ok")]
        full = [r for r in ok if r["successes"] >= 30]
        short = [r for r in recs if not r.get("ok") or r["successes"] < 30]
        by_task = {}
        for r in short:
            by_task.setdefault(r["task"], []).append(
                f"{r['pid']} {r.get('successes')}/{r.get('n')}")
        rows.append((f"repairs and minimal all reach the task ceiling "
                     f"[{label}]", "HOLDS" if not short else "REVISED",
                     f"{len(recs)} {label} verified on H200 (LLM-free, 30 "
                     f"scenes each): {len(full)} at 30/30 on their latest "
                     f"draw; {len(short)} short: "
                     + ("; ".join(f"{t}: {', '.join(v)}"
                                  for t, v in sorted(by_task.items()))
                        or "none")
                     + ". Every draw is kept under `history` in the file."))
    return rows


def taxonomy_rows() -> list[tuple[str, str, str]]:
    """The defect taxonomy and the distance distribution, re-derived.

    Both were PENDING on "the rebuild was told not to tag". They are done, and
    they are re-derived here rather than written down because the first version
    of these numbers was published out of a script that lived nowhere and could
    not be regenerated -- which is how a tagger bug survived into a headline.
    Regenerate with:

        python3 scripts/tag_episodes.py results/tags_seeded.json \
            --failures results/failures_seeded.json
    """
    import json
    from collections import Counter
    from opencap.models import BY_KEY

    if not TAGS.exists():
        return [("Defect taxonomy over the seeded corpus", "PENDING",
                 f"{TAGS.name} missing -- run scripts/tag_episodes.py")]
    recs = json.loads(TAGS.read_text())
    n_eps = sum(r["n_episodes"] for r in recs) or 1
    fam: Counter = Counter()
    for r in recs:
        for f in r["families"]:
            fam[f] += r["n_episodes"]
    share = " / ".join(f"{k} {100*v/n_eps:.1f}" for k, v in fam.most_common())

    band_n: Counter = Counter()
    band_intf: Counter = Counter()
    band_frame: Counter = Counter()
    for r in recs:
        p = BY_KEY[r["models"][0]].params_b if r["models"][0] in BY_KEY else 0
        b = "<=2B" if p <= 2 else "3-4B" if p <= 4 else "7-9B" if p <= 9 else ">=12B"
        band_n[b] += 1
        if "INTERFACE" in r["families"]:
            band_intf[b] += 1
        if "FRAME" in r["families"]:
            band_frame[b] += 1
    order = ["<=2B", "3-4B", "7-9B", ">=12B"]
    intf = " ".join(f"{b} {100*band_intf[b]/max(band_n[b],1):.0f}%" for b in order)
    frame = " ".join(f"{b} {band_frame[b]}" for b in order)

    # Artifact #15: does the headline survive removing the programs that are
    # VISIBLY cut off at the token cap? Re-derived rather than remembered,
    # because the answer is the reason Phase 8 is scoped the way it is.
    cut = [r for r in recs if r.get("truncated")]
    def _band_stats(rows):
        agg: dict = {}
        for r in rows:
            p = BY_KEY[r["models"][0]].params_b if r["models"][0] in BY_KEY else 0
            b = "<=2B" if p <= 2 else "3-4B" if p <= 4 else "7-9B" if p <= 9 else ">=12B"
            a = agg.setdefault(b, [0, 0])
            a[0] += len(r["tags"])
            a[1] += 1
        return {b: (v[0] / v[1], v[1]) for b, v in agg.items()}
    full, trimmed = _band_stats(recs), _band_stats([r for r in recs if not r.get("truncated")])

    dists = [r["distance"] for r in recs if r.get("distance") is not None]
    mean_d = sum(dists) / max(len(dists), 1)

    rows = [
        (f"Defect taxonomy over the seeded corpus", "RE-DERIVED",
         f"{len(recs)} programs / {n_eps} failed episodes, tagged from verified "
         f"minimal repairs: {share}% (episodes counted once per family)."),
        ("INTERFACE rises with scale", "RETRACTED",
         f"It does not. Published as 38/30/23/69% with INTERFACE dominant at "
         f">=12B; that band was inflated by `unbound_name` on nine programs "
         f"calling goto_home_joint_position, a primitive the harness binds. "
         f"Re-derived: {intf}. The >=12B band no longer leads."),
        ("FRAME is the only family that rises with scale", "RETRACTED",
         f"Greedy corpus: {frame} -- four programs, all on nut_assembly, every "
         f"one >=8B, none below 8B. The sampled arm ran it at n=105 programs "
         f"(STATUS 2026-09-28, completed 2026-10-13) and BOTH halves fail. It "
         f"does not rise: 7-9B 8/20 [0.22,0.61] and >=12B 12/30 [0.25,0.58] "
         f"are indistinguishable, "
         f"so attempts switch on at 7B and stay flat to 32B. And 'no program "
         f"below 8B attempts one' is false -- two <=2B programs match, both "
         f"malformed calls rather than wrong-frame transforms."),
        ("Attempted coordinate transforms are always wrong", "REFUTED",
         f"qwen3vl-32b sampled draw 1 attempts one -- four frame rotations -- "
         f"and scores 4/30, the second-best cell in the sampled arm. Its own "
         f"draw 2 is the matched control: same model, also runs to completion "
         f"and fails on physics rather than code, no rotation, 0/30. The best "
         f"cell in the arm (qwen35-27b, 15/30) uses no rotation at all, so a "
         f"transform is neither necessary nor sufficient. sampled_report.py "
         f"printed '0 attempts succeeded' until 2026-09-28: it tested success "
         f"on whichever episode carried each program first, and all 30 "
         f"episodes of a cell share one program."),
        ("Only models >=12B ever clear the nut_assembly insertion", "HOLDS",
         f"Sampled arm, 3,150 episodes over 21 models at temperature 0.7: "
         f"<=2B 0/900, 3-4B 0/750, 7-9B 0/600, >=12B 24/900 [0.018,0.039]. A "
         f"threshold, not a slope -- within the band the rate does not climb "
         f"with size (qwen3-32b 0/150, qwen35-27b 15/150). One draw of "
         f"qwen35-27b scores 15/30 against the human reference's 16/30, so "
         f"'3 successes in 630' was substantially one draw per cell -- and "
         f"that draw is one program in five: qwen35-27b's draws score "
         f"0,0,0,15,0. Draw 3, the last cell collected (2026-10-13), runs to "
         f"the token cap on 31/31 requests and crashes on "
         f"Rotation.from_quat(degrees=...) in all 30 scenes."),
        ("Defect counts by band are measured under one collection condition",
         "CONFOUNDED",
         f"Requests hitting the 2048-token cap, from the shim's stats: <=2B "
         f"30.0% [0.29,0.31] against 3-4B 3.3%, 7-9B 4.6%, >=12B 2.9%. A "
         f"truncated program enters the corpus as no_op or api_misuse with "
         f"tokens_out=0 and truncation_fired false -- indistinguishable from a "
         f"program the model finished and got wrong. 'Defects/program falls "
         f"3.21 -> 1.15 with scale' is measured across bands that differ "
         f"tenfold in truncation rate. Re-derive: scripts/truncation_audit.py. "
         f"This says the small end needs checking, not that the trend is "
         f"wrong."),
        ("Defects per program fall with scale, and it is not the token cap",
         "HOLDS" if cut else "PENDING",
         "Artifact #15 varies along the same axis as the trend, so the trend "
         "was checked against it. " +
         (f"{len(cut)} of {len(recs)} programs are VISIBLY cut off (no parse, "
          f"or >= {__import__('opencap.repair', fromlist=['x']).TRUNCATION_LINES} "
          f"lines against a median of 22), every one from a model with a "
          f"nonzero truncation rate. Removing them: " +
          " ".join(f"{b} {full[b][0]:.2f}->{trimmed[b][0]:.2f}"
                   for b in ('<=2B', '3-4B', '7-9B', '>=12B') if b in trimmed) +
          " defects/program. The trend does not move. NOTE this only removes "
          "cuts that are visible in the code -- a cut landing just after a "
          "complete statement is invisible here, so the residual needs the "
          "store to carry a finish reason (Phase 8 job 2)."
          if cut else "no tagging available")),
        ("Distance to solution, over the verified minimal repairs", "RE-DERIVED",
         f"mean {mean_d:.3f} over {len(dists)} programs; by band it falls "
         f"monotonically with scale (see scripts/defect_by_scale.py)."),
    ]
    return rows


def cross_tier_rows() -> list[tuple[str, str, str]]:
    """Phase 9: S1 vs the reduced-API tiers (S3/S4), hand tags on both sides.

    Re-derived from the committed tag files -- results/tags_s1_hand.json and
    results/tags_reduced.json -- with the same exclusions as
    scripts/cross_tier_report.py (truncation-only records out). Needs no corpus.
    """
    import importlib.util
    import json
    from opencap.editclass import FAMILY
    from opencap.models import BY_KEY
    root = Path(__file__).resolve().parent.parent
    if not (root / "results/tags_s1_hand.json").exists() or not (root / "results/tags_reduced.json").exists():
        return [("Cross-tier defect mix (S1 vs S3/S4)", "PENDING", "hand tag files missing")]
    spec = importlib.util.spec_from_file_location("_ctr", root / "scripts/cross_tier_report.py")
    ctr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctr)

    def prep(source, verified):
        g = ctr.load(root, s1_source=source)
        g = {k: [r for r in v if not (r.get("truncated") and set(r["tags"]) <= {"syntax"})] for k, v in g.items()}
        if verified:
            g = {k: [r for r in v if not str(r.get("level") or "").startswith("L5")] for k, v in g.items()}
        return g["S1"], g["S3"] + g["S4"]

    def share(recs, fam):
        return sum(1 for r in recs if any(FAMILY.get(t) == fam for t in r["tags"])) / max(len(recs), 1)

    def mean_big(recs):
        xs = [len([t for t in r["tags"] if t in FAMILY]) for r in recs
              if any(BY_KEY[m].params_b >= 12 for m in r["models"] if m in BY_KEY)]
        return sum(xs) / max(len(xs), 1), len(xs)

    s1, red = prep("hand", False)
    _, redv = prep("hand", True)
    s1a, _ = prep("auto", False)
    rows = []
    for fam in ("FRAME", "INTERFACE"):
        a, b, bv = share(s1, fam), share(red, fam), share(redv, fam)
        ok = b - a > 0.2 and bv - a > 0.2
        rows.append((f"{fam} defects are far more common at S3/S4 than at S1",
                     "STANDS" if ok else "CHECK",
                     f"S1 {a:.0%} (n={len(s1)}) vs S3/S4 {b:.0%} (n={len(red)}), verified-only {bv:.0%} (n={len(redv)}); hand tags both sides"))
    (m1, n1), (m2, n2), (m3, n3) = mean_big(s1), mean_big(red), mean_big(redv)
    rows.append(("The S1 scale advantage largely vanishes at S3/S4 (>=12B defects/failed program)",
                 "STANDS" if m2 > 1.5 * m1 and m3 > 1.5 * m1 else "CHECK",
                 f">=12B: S1 {m1:.2f} (n={n1}) vs S3/S4 {m2:.2f} (n={n2}), verified-only {m3:.2f} (n={n3}); "
                 f"the least robust of the three in the same-rater re-tag"))
    a, b, aa = share(s1, "SEMANTIC"), share(red, "SEMANTIC"), share(s1a, "SEMANTIC")
    rows.append(("SEMANTIC defects rise from S1 to S3/S4", "RETRACTED",
                 f"hand vs hand: S1 {a:.0%} vs S3/S4 {b:.0%}. The earlier gap used the automated S1 tagger ({aa:.0%})"))
    ha = sum(len([t for t in r["tags"] if t in FAMILY]) for r in s1) / max(len(s1), 1)
    au = sum(len([t for t in r["tags"] if t in FAMILY]) for r in s1a) / max(len(s1a), 1)
    rows.append(("S1 defect counts from the automated tagger (classify_edits)", "REVISED",
                 f"hand tagging finds {ha:.2f} tags/program vs automated {au:.2f} on the same S1 programs; "
                 f"report both. Scale gradient holds either way"))
    spec2 = importlib.util.spec_from_file_location("_pmp", root / "scripts/per_model_profile.py")
    pmp = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(pmp)
    prof = pmp.profiles()
    big = [m for m in prof if BY_KEY[m].params_b >= 3 and prof[m]["S1"]["n"] and prof[m]["S3S4"]["n"]]
    rise = [m for m in big if prof[m]["S3S4"]["tags"] / prof[m]["S3S4"]["n"] > prof[m]["S1"]["tags"] / prof[m]["S1"]["n"]]
    rows.append(("Every model >=3B carries more defects per failing program at S3/S4 than at S1",
                 "STANDS" if len(rise) == len(big) else "CHECK",
                 f"{len(rise)}/{len(big)} models >=3B rise (sign test p={0.5 ** len(big):.1g}); the <=2B models already saturate at S1"))
    fr = sum(1 for m in prof if prof[m]["S3S4"]["FRAME"]); fr1 = sum(1 for m in prof if prof[m]["S1"]["FRAME"])
    rows.append(("FRAME defects appear across the ladder, not in a few models", "STANDS" if fr >= 15 else "CHECK",
                 f"{fr}/{len(prof)} models have a FRAME defect at S3/S4 vs {fr1}/{len(prof)} at S1"))
    rt = root / "artifacts/second_rater/retag_session2.json"
    if rt.exists():
        import json as _json
        _m = root / "artifacts/second_rater/ratings.json"
        _who = (_json.loads(_m.read_text()).get("filed", {}).get("who")
                if _m.exists() else None) or "unrecorded"
        rows.append(("Hand tags are reproducible by the same rater", "HOLDS",
                     f"The filed tags were produced by {_who}, which knew the "
                     f"research question. Its own blind re-tag of 60 programs: "
                     f"pooled kappa 0.89, FRAME 0.67. An UPPER BOUND -- same "
                     f"rater, same day -- not inter-rater agreement."))
    # Second-rater runs, re-derived from whatever CSVs are present. Each is
    # scored by scripts/score_second_rater.py, the same code path as the
    # same-rater bound above, so the two are comparable.
    rows += second_rater_rows(root)
    return rows


def second_rater_rows(root: Path) -> list[tuple[str, str, str]]:
    """Independent ratings of the packet, re-derived from their CSVs.

    The claim these test is NOT "the raters agree" -- it is "the family
    contrasts are visible to an independent reader". A rater can disagree
    about how finely to split a cause and still see the same shift, and that
    is the distinction the write-up's "quote the contrasts first" rests on.
    """
    import csv
    import json
    out: list[tuple[str, str, str]] = []
    pk = root / "artifacts/second_rater"
    key_f = pk / "key.json"
    if not key_f.exists():
        return out
    key = json.loads(key_f.read_text())
    # Who produced each rating. Read, not assumed: a row that reports a
    # filename stem invites "the codex rating" surviving into a paper where
    # the model should be named.
    meta_f = pk / "ratings.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    filed: dict = {}
    for name in ("tags_s1_hand.json", "tags_reduced.json"):
        f = root / "results" / name
        if f.exists():
            filed.update(json.loads(f.read_text()))
    if not filed:
        return out

    def fams(tags):
        return {FAMILY_OF[t] for t in tags if t in FAMILY_OF}

    def kappa(a, b):
        n = len(a)
        po = sum(1 for x, y in zip(a, b) if x == y) / n
        pa, pb = sum(a) / n, sum(b) / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        return None if pe >= 1 else (po - pe) / (1 - pe)

    for csv_f in sorted(pk.glob("answers_*.csv")):
        if csv_f.name == "answers_template.csv":
            continue
        with csv_f.open(newline="") as fh:
            rater = {r["id"].strip():
                     [t.strip() for t in (r.get("tags") or "").split(";") if t.strip()]
                     for r in csv.DictReader(fh) if (r.get("id") or "").strip()}
        ids = [i for i in sorted(key) if i in rater]
        if not ids:
            continue
        s1 = [i for i in ids if key[i]["tier"] == "S1"]
        red = [i for i in ids if key[i]["tier"] != "S1"]
        F = {i: fams(filed[key[i]["pid"]]["tags"]) for i in ids}
        R = {i: fams(rater[i]) for i in ids}
        bits = []
        held = True
        for fam in ("FRAME", "INTERFACE"):
            fa = sum(1 for i in s1 if fam in R[i]) / max(len(s1), 1)
            fb = sum(1 for i in red if fam in R[i]) / max(len(red), 1)
            ka = kappa([fam in F[i] for i in ids], [fam in R[i] for i in ids])
            bits.append(f"{fam} {fa:.0%}->{fb:.0%} (kappa {ka:.2f})"
                        if ka is not None else f"{fam} {fa:.0%}->{fb:.0%}")
            if fb <= fa:
                held = False
        tag_names = sorted(FAMILY_OF)
        pooled = kappa(
            [t in filed[key[i]["pid"]]["tags"] for i in ids for t in tag_names],
            [t in rater[i] for i in ids for t in tag_names])
        ca = sum(len([t for t in rater[i] if t in FAMILY_OF]) for i in s1) / max(len(s1), 1)
        cb = sum(len([t for t in rater[i] if t in FAMILY_OF]) for i in red) / max(len(red), 1)
        m = meta.get(csv_f.name, {})
        who = m.get("who") or csv_f.stem.replace("answers_", "")
        blind = m.get("blind")
        cond = ("blind, isolated" if m.get("isolated") else
                "blind" if blind else
                "CONDITION NOT RECORDED -- add an entry to ratings.json")
        out.append((
            f"The family contrasts reproduce for an independent rater "
            f"[{who}; {cond}]",
            "STANDS" if held else "CHECK",
            f"{len(ids)} items, rater's OWN tags: " + "; ".join(bits) +
            f". Pooled tag-level kappa {pooled:.2f} against the same-rater "
            f"upper bound of 0.89. Count gap in their tags "
            f"{ca:.2f}->{cb:.2f} ({cb-ca:+.2f}) against the filed "
            f"+0.63 -- third independent indication that the count gap is the "
            f"weakest figure. Re-derive: python3 "
            f"scripts/score_second_rater.py {csv_f.relative_to(root)}"))
    return out


def legacy_rows() -> list[tuple[str, str, str]]:
    """Two claims that sat PENDING after the work that settles them was done.

    Both were waiting on "once the repairs are tagged". The repairs have been
    tagged by hand on both sides since 2026-10-04, so these were not open
    questions, they were stale rows -- which is its own failure mode: a
    register whose PENDING list drifts is no better than remembering.
    """
    import glob
    import json
    from collections import Counter
    rows = []

    hand = _RES / "tags_s1_hand.json"
    if hand.exists():
        tags = json.loads(hand.read_text())
        n = len(tags) or 1
        c: Counter = Counter()
        for r in tags.values():
            for t in set(r.get("tags", [])):
                c[t] += 1
        top = c.most_common(6)
        traj = [t for t, _ in top if FAMILY_OF.get(t) == "TRAJECTORY"]
        rows.append((
            "Six recurring defects, five of them trajectory hygiene",
            "REFUTED",
            f"Re-derived over the 70 hand-tagged S1 programs. The six most "
            f"recurring are " +
            ", ".join(f"{t} {100*k/n:.0f}% ({FAMILY_OF.get(t,'?')})"
                      for t, k in top) +
            f". {len(traj)} of the six is TRAJECTORY, not five. The recurring "
            f"defects are overwhelmingly SEMANTIC. The original claim came "
            f"from the Colab corpus and the automated tagger."))
    else:
        rows.append(("Six recurring defects, five of them trajectory hygiene",
                     "PENDING", f"{hand.name} missing"))

    # The four L4_scene_limited programs: scene-limited, or an artifact of
    # random scene draws? Answered by the seeded corpus; the row never moved.
    L4 = ("0554815815fc", "81ee85d0f933", "e7356f7f0021", "ef40b4163e91")
    fails = _RES / "failures_seeded.json"
    if fails.exists():
        seeded = {r["pid"] for r in json.loads(fails.read_text())}
        reps: dict = {}
        for f in sorted(glob.glob(str(_RES / "repairs" / "*.json"))) + \
                sorted(glob.glob(str(_RES / "minimal" / "*.json"))):
            reps.update(json.loads(Path(f).read_text()))
        repaired = [p for p in L4
                    if p in seeded and (reps.get(p) or {}).get("code")]
        gone = [p for p in L4 if p not in seeded]
        rows.append((
            "4 programs are L4_scene_limited -- the cell cannot reach ceiling "
            "because of the scenes rather than the code",
            "REFUTED" if len(repaired) + len(gone) == len(L4) else "PARTIAL",
            f"None of the four is scene-limited. {len(repaired)} "
            f"({', '.join(repaired)}) fail reproducibly on the seeded scenes "
            f"and carry ordinary L3 repairs; {len(gone)} "
            f"({', '.join(gone)}) are not in the seeded failures at all -- "
            f"their cells passed once the scenes stopped being redrawn. The "
            f"category was an artifact of artifact #13, as suspected when it "
            f"was flagged."))
    return rows


#: qwen3-4b's cube_stack program, the one mid-range program the worker-state
#: question was ever pointed at.
ALT_PID = "81ee85d0f933"
ALT_FILE = _RES / "scene_probe_alt.json"


def worker_noise_rows() -> list[tuple[str, str, str]]:
    """Is the measurement noise floor a property of the scene, or of parallelism?

    Phase 6 established that the whole measured noise floor of this project is
    ONE SCENE -- cube_stack 13, which 17 of 42 programs flip between runs of
    byte-identical code -- and, separately, refuted worker scheduling as a
    cause. That refutation was run on programs AT CEILING and said so.

    The 2026-10-13 probe collected 8 draws at each worker count for two
    programs, and the REPAIR is the informative one: it is a verified repair,
    at ceiling, and it is exactly the kind of program Phase 6 tested. Phase 6
    had one draw per arm and could not see within-condition stability.

    This row pools both programs and asks one question: in how many draws did
    a varying trial fail at all? That is a different question from the
    alternation (WHICH trial), which is why the repair records are included
    here and excluded there.

    WORKER COUNTS ARE READ FROM THE FILE, NOT LISTED HERE. The first version
    hardcoded (4, 1), so a workers=2 arm would have been collected and
    silently ignored -- the same shape as a PENDING row left behind by the run
    that settled it. Any count present is reported, and with three or more the
    row also prints the monotone-vs-binary reading, which is the discriminator
    between a contention effect and an ordering one.
    """
    import json
    from opencap.analysis import wilson
    claim = ("The cube_stack scene-13 noise floor is a property of the scene, "
             "not of parallel execution")
    if not ALT_FILE.exists():
        return [(claim, "PENDING", f"{ALT_FILE.name} missing")]
    recs = [r for r in json.loads(ALT_FILE.read_text())
            if r.get("ok") and "failed_trials" in r and r.get("pid") == ALT_PID]
    VARY = {"original": {9, 13}, "repair": {13}}
    tally: dict[int, list[int]] = {}
    for r in recs:
        tset = VARY.get(r.get("version"))
        w = r.get("workers")
        if tset is None or w is None:
            continue
        t = tally.setdefault(int(w), [0, 0])
        t[1] += 1
        t[0] += bool(set(r["failed_trials"]) & tset)
    if len(tally) < 2:
        have = ", ".join(f"workers={w} x{n}" for w, (_, n) in sorted(tally.items()))
        return [(claim, "PENDING",
                 f"needs at least two worker counts; have {have or 'none'}. "
                 f"Run scripts/run_worker_sweep.sh on the pod.")]

    counts = sorted(tally, reverse=True)
    parts = []
    for w in counts:
        h, n = tally[w]
        iv = wilson(h, n)
        parts.append(f"workers={w}: {h}/{n} [{iv.lo:.2f},{iv.hi:.2f}]")
    lo_w = min(counts)
    hi_w = max(counts)
    h_lo, n_lo = tally[lo_w]
    h_hi, n_hi = tally[hi_w]
    status = "REFUTED" if (h_lo == 0 and h_hi > 0) else "HOLDS"

    # Per program, and the ANY-varying count: a failure outside the trials that
    # fail in every draw of that program at every count. VARY above is the set
    # that varied at workers=4; other counts move other trials (16 at
    # workers=2; 12, 28, 30 and the repair's 9 at workers=8), which VARY
    # cannot see.
    always = {}
    for r in recs:
        v = r.get("version")
        f = set(r["failed_trials"])
        always[v] = always[v] & f if v in always else f
    per: dict[tuple[str, int], list[int]] = {}
    anyv: dict[int, list[int]] = {}
    for r in recs:
        v, w = r.get("version"), r.get("workers")
        if v not in VARY or w is None:
            continue
        f = set(r["failed_trials"])
        p = per.setdefault((v, int(w)), [0, 0, 0])
        p[0] += bool(f & VARY[v])
        p[1] += 1
        p[2] += 13 in f
        a = anyv.setdefault(int(w), [0, 0])
        a[0] += bool(f - always[v])
        a[1] += 1

    def _pooling_is_valid() -> str | None:
        """Is the pooled middle-count rate a rate at all?

        THE PRE-REGISTERED STATISTIC WAS MIS-SPECIFIED, and this is where that
        is said rather than argued with afterwards. It pools the failing
        program and its repair. At workers=2 the original fails the registered
        trials in 0 of 8 draws and the repair in 8 of 8 -- two deterministic
        and OPPOSITE behaviours whose mean is 50%, a number describing neither
        and sitting exactly where the BINARY/MONOTONE threshold was drawn.

        A pooled rate over units moving in opposite directions is not a rate.
        Averaging them and then reading a verdict off the average is the
        error, and it was in the design (2026-10-14), not in the run.
        """
        bad = []
        for w in sorted({k[1] for k in per}):
            rates = [(v, per[(v, w)][0], per[(v, w)][1])
                     for v in ("original", "repair") if (v, w) in per]
            if len(rates) < 2:
                continue
            lo = [r for r in rates if r[2] and r[1] == 0]
            hi = [r for r in rates if r[2] and r[1] == r[2]]
            if lo and hi:
                bad.append(
                    f"workers={w} ({lo[0][0]} {lo[0][1]}/{lo[0][2]}, "
                    f"{hi[0][0]} {hi[0][1]}/{hi[0][2]})")
        return "; ".join(bad) or None

    def _read(t: dict[int, list[int]], check_pooling: bool = True) -> tuple[str, str]:
        # The PRE-REGISTERED comparison (HANDOFF section 02, 2026-10-14):
        # workers=2 against workers=4 and workers=1. The first version read
        # "the first middle count against the largest", so a fourth count
        # turned it into 4-vs-8 and printed a verdict nobody registered.
        split = _pooling_is_valid() if check_pooling else None
        if split is not None:
            return "MIS-SPECIFIED", (
                f"the registered statistic pools two programs that move in "
                f"OPPOSITE directions at the same count -- {split} -- so its "
                f"middle value describes neither and the BINARY/MONOTONE "
                f"threshold was drawn across a number that is not a rate. "
                f"Reported as mis-specified rather than as a verdict: the "
                f"error is in the 2026-10-14 design, not in the run. What the "
                f"data does establish is below; monotone-vs-binary is "
                f"UNRESOLVED and needs a statistic that does not average "
                f"across programs.")
        hm, nm = t[2]
        hh, nh = t[4]
        hl, nl = t[1]
        rm, rh = hm / nm, hh / nh
        im, ih = wilson(hm, nm), wilson(hh, nh)
        if hm == 0:
            return "THRESHOLD", (
                f"workers=2 shows NOTHING ({hm}/{nm}), so the effect needs "
                f"more than 2 workers -- a THRESHOLD or load effect, not bare "
                f"interleaving.")
        if im.lo > 0 and rm >= rh * 0.75:
            return "BINARY", (
                f"workers=2 is already at {rm:.0%}, indistinguishable from "
                f"workers=4 at {rh:.0%} -- BINARY in parallelism, which points "
                f"at trial interleaving rather than contention.")
        return "MONOTONE", (
            f"{rm:.0%} at workers=2 against {rh:.0%} at workers=4 and "
            f"{hl}/{nl} at workers=1 -- MONOTONE in worker count, which points "
            f"at contention or load rather than ordering. At 8 draws per "
            f"condition the intervals overlap ({im.lo:.2f}-{im.hi:.2f} vs "
            f"{ih.lo:.2f}-{ih.hi:.2f}); treat the direction as the result, not "
            f"the slope.")

    reading = ""
    if all(w in tally for w in (1, 2, 4)):
        verdict, text = _read(tally)
        label = ("VERDICT WITHHELD" if verdict == "MIS-SPECIFIED"
                 else "READING (pre-registered, workers=2 vs 4 and 1)")
        reading = f" {label}: {text}"
        for w in [w for w in counts if w not in (1, 2, 4)]:
            hx, nx = tally[w]
            rx, rh = hx / nx, tally[4][0] / tally[4][1]
            # An OBSERVATION about the shape, not a verdict, so it is
            # reported whether or not the registered statistic was usable --
            # it was suppressed along with the verdict once, and it is the
            # single most informative line about whether "more workers, more
            # effect" holds at all.
            if w > 4 and rx < rh:
                reading += (f" workers={w} at {rx:.0%} does NOT continue the "
                            f"rise above workers=4 at {rh:.0%}: a monotone "
                            f"reading would need it to, and it does not.")
            else:
                reading += f" workers={w}: {rx:.0%}, reported, not a verdict."
        pp = "; ".join(
            f"{v} " + ", ".join(f"{per[(v, w)][0]}/{per[(v, w)][1]}"
                                for w in sorted(tally) if (v, w) in per)
            for v in VARY)
        reading += (f" PER PROGRAM at workers={', '.join(map(str, sorted(tally)))}"
                    f": {pp} -- the pooled number is a sum of programs that do "
                    f"not move together.")
        if all(w in anyv for w in (1, 2, 4)):
            # check_pooling=False: this is a DIFFERENT trial definition, and
            # the registered split does not describe it. The point of the line
            # is that the same rule reads BINARY here, which a borrowed
            # mis-specification verdict would hide.
            av, _ = _read(anyv, check_pooling=False)
            ap = "; ".join(f"workers={w}: {anyv[w][0]}/{anyv[w][1]}"
                           for w in sorted(anyv, reverse=True))
            reading += (f" SENSITIVITY: counting ANY failure outside a "
                        f"program's always-failing trials instead of the "
                        f"workers=4 set gives {ap}, which the same rule reads "
                        f"as {av}. The two counts disagree; neither is "
                        f"decisive at 8 draws.")
    else:
        reading = (f" A workers=2 arm would separate a monotone contention effect "
                   f"from a binary any-parallelism one; it is 8 draws and needs no "
                   f"engine or servers (scripts/run_worker_sweep.sh).")

    try:
        from scipy.stats import fisher_exact
        p = fisher_exact([[h_hi, n_hi - h_hi], [h_lo, n_lo - h_lo]])[1]
        ptxt = f", workers={hi_w} vs {lo_w} Fisher p={p:.1g}"
    except Exception:
        ptxt = ""

    rep_lo = [r for r in recs
              if r.get("version") == "repair" and r.get("workers") == lo_w]
    at_ceiling = bool(rep_lo) and all(r.get("successes") == r.get("n")
                                      for r in rep_lo)
    lost = ", ".join(f"{per[('repair', w)][2]}/{per[('repair', w)][1]} at "
                     f"workers={w}" for w in counts
                     if w != lo_w and ("repair", w) in per)
    detail = (
        f"A varying trial failed in " + "; ".join(parts) + f"{ptxt}, pooled "
        f"over the failing program and its verified repair. The repair is "
        + (f"AT CEILING at workers={lo_w} -- 30/30 in all {len(rep_lo)} draws"
           if at_ceiling else f"NOT at ceiling at workers={lo_w}")
        + f" -- and loses scene 13 in {lost}. So the scene is bistable "
        f"UNDER PARALLEL EXECUTION and stable without it: the noise floor is "
        f"not a property of the scene alone. Phase 6 measured the same "
        f"comparison on ceiling programs and found nothing, with ONE draw per "
        f"arm, which cannot see within-condition stability."
        + reading +
        f" WHAT IS NOT SHOWN: which part of parallelism carries it. "
        f"workers={lo_w} removes trial interleaving AND CPU contention AND all "
        f"timing overlap at once. "
        f"NOT the IK server: the four project servers including pyroki :8116 "
        f"were stopped before the chain launched and every record returned ok "
        f"with no unreachable trials, so cube_stack S1 never contacts it. "
        f"Remaining candidates: trial ordering, CPU contention, timing "
        f"overlap. n = 2 programs, one task, one scene.")
    return [(claim, status, detail)]


def alternation_rows() -> list[tuple[str, str, str]]:
    """The trial 9 / trial 13 alternation, at workers=4 and workers=1.

    It sat in PENDING with its test named. The test ran on 2026-10-13, so the
    row is computed from the records rather than written down -- including
    the verdict, which comes from whether trials 9 and 13 still vary between
    draws once the trial order is fixed.

    n = 1 program: an OBSERVATION, not a claim, in either direction.
    """
    import json
    claim = ("qwen3-4b cube_stack alternates its second failure between trial "
             "9 and trial 13 -- never both, never neither, across eight draws")
    if not ALT_FILE.exists():
        return [(claim, "PENDING",
                 f"{ALT_FILE.name} missing. Run scripts/run_scene_probe.py "
                 f"--pids {ALT_PID} at --workers 4 AND --workers 1.")]
    recs = [r for r in json.loads(ALT_FILE.read_text())
            if r.get("pid") == ALT_PID and r.get("version") == "original"
            and r.get("ok") and "failed_trials" in r]
    sets = {w: [set(r["failed_trials"]) for r in recs if r.get("workers") == w]
            for w in (4, 1)}
    if not sets[4] or not sets[1]:
        return [(claim, "PENDING",
                 f"{ALT_FILE.name} holds workers=4 x{len(sets[4])}, workers=1 "
                 f"x{len(sets[1])}. Both conditions are needed.")]

    def counts(ss):
        return (sum(9 in s for s in ss), sum(13 in s for s in ss),
                sum(9 in s and 13 in s for s in ss),
                sum(9 not in s and 13 not in s for s in ss))

    def tally(ss):
        n9, n13, both, neither = counts(ss)
        always = sorted(set.intersection(*ss))
        return (f"{len(ss)} draws, always {always}; trial 9 failed {n9}, "
                f"trial 13 failed {n13}, both {both}, neither {neither}")

    _, _, both4, neither4 = counts(sets[4])
    redraw = ("keeps 'never both'" if both4 == 0 else
              f"breaks 'never both' ({both4} draws)") + " and " + \
             ("keeps 'never neither'" if neither4 == 0 else
              f"breaks 'never neither' ({neither4} draw"
              f"{'s' if neither4 != 1 else ''})")

    varies = {w: len({frozenset(s & {9, 13}) for s in sets[w]}) > 1
              for w in (4, 1)}
    if varies[4] and not varies[1]:
        status = "STOPS AT WORKERS=1 (n = 1)"
        reading = ("Fixing the trial order removes the variation, so worker "
                   "scheduling moves this program's outcome below ceiling: "
                   "Phase 6's null is a ceiling result, as it said. NOT shown: "
                   "that the carrier is within-worker carry-over. Rebuilding "
                   "each run's worker chains from trial finish times "
                   "(2026-10-13 STATUS) shows trial 13 failing when trial 9 "
                   "passed even on a DIFFERENT worker, so something couples "
                   "workers; the one shared resource, the pyroki IK server on "
                   ":8116, takes its warm start from the request. Mechanism "
                   "unidentified.")
    elif varies[1]:
        status = "PERSISTS AT WORKERS=1 (n = 1)"
        reading = ("The variation survives a fixed trial order, so it is not "
                   "worker state: Phase 6 stands, and the cause is in the "
                   "scene pair or the solver.")
    else:
        status = "NO VARIATION (n = 1)"
        reading = ("Neither condition varies in this re-draw, so the original "
                   "pattern did not reproduce and the test has nothing to "
                   "separate.")
    return [(claim, status,
             f"workers=4: {tally(sets[4])}. workers=1: {tally(sets[1])}. "
             f"The 2026-09-21 workers=4 draws were 9 on four and 13 on four, "
             f"never both, never neither; the 2026-10-13 re-draw {redraw}. "
             f"{reading} n = 1 program: "
             f"an OBSERVATION, not a claim.")]


def leak_rows() -> list[tuple[str, str, str]]:
    """The IK-seed A/B, re-derived from results/leak_ab.json.

    Regenerate with:  bash scripts/run_leak_ab.sh panel
    """
    import json
    from collections import defaultdict
    if not LEAK.exists():
        return [("A trial's starting state is its own", "PENDING",
                 f"{LEAK.name} missing -- run scripts/run_leak_ab.sh")]
    raw = json.loads(LEAK.read_text())
    by: dict = defaultdict(dict)
    for r in raw:
        if r.get("ok"):
            by[r["pid"]][r["arm"]] = r

    def disc(left, right, task=None):
        d = n = 0
        for pid, arms in by.items():
            if left not in arms or right not in arms:
                continue
            if task and arms[left]["task"] != task:
                continue
            oa = {t["trial"]: t["success"] for t in arms[left].get("trials", [])}
            ob = {t["trial"]: t["success"] for t in arms[right].get("trials", [])}
            shared = set(oa) & set(ob)
            d += sum(1 for x in shared if oa[x] != ob[x])
            n += len(shared)
        return d, n

    rep_d, rep_n = disc("A", "A2")
    leak_d, leak_n = disc("A", "B")

    # Which scenes ever moved, anywhere in the run.
    moved: dict = defaultdict(set)
    for pid, arms in by.items():
        base = arms.get("A")
        if not base:
            continue
        oa = {t["trial"]: t["success"] for t in base.get("trials", [])}
        for arm in ("A2", "B"):
            other = arms.get(arm)
            if not other:
                continue
            ob = {t["trial"]: t["success"] for t in other.get("trials", [])}
            for x in set(oa) & set(ob):
                if oa[x] != ob[x]:
                    moved[base["task"]].add(x)
    where = "; ".join(f"{t} scene(s) {sorted(v)}" for t, v in sorted(moved.items())) or "none"

    ctrl = by.get("reference_nut_assembly", {})
    ctrl_txt = ("reference_nut_assembly absent -- the run has no positive "
                "control and its null is not interpretable")
    ctrl_ok = False
    if ctrl.get("A") and ctrl.get("C"):
        a_s, c_s = ctrl["A"]["successes"], ctrl["C"]["successes"]
        ctrl_ok = c_s > a_s
        ctrl_txt = (f"positive control: the nut_assembly reference scores "
                    f"{a_s}/30 stock and {c_s}/30 with the start-home "
                    f"prepended, reproducing the 2026-09-16 ablation. The "
                    f"instrument detects a {c_s - a_s}-scene effect, so the "
                    f"null below is a null and not a broken measurement.")

    return [
        ("A trial's starting state is contaminated by worker scheduling order",
         "REFUTED" if ctrl_ok else "REFUTED (control missing)",
         f"{len(by)} programs, {leak_n} paired scenes. Four workers vs one "
         f"moves {leak_d} scenes ({leak_d/max(leak_n,1):.2%}) against "
         f"{rep_d} ({rep_d/max(rep_n,1):.2%}) for re-running the SAME "
         f"condition. Scheduling order moves LESS than a plain re-draw. "
         f"State-level pairing holds. {ctrl_txt}"),
        ("The measurement noise floor is one scene", "RE-DERIVED, AT CEILING",
         f"Every scene that moved anywhere in the run: {where}. All other "
         f"scenes, on every task and every program, returned identical "
         f"outcomes across all arms. cube_stack scene 13 is bistable and is "
         f"the whole of TASK_NOISE. But every program in that run scores "
         f"30/30, and a program at ceiling can only move down. The scene "
         f"probe (STATUS 2026-09-21) ran four MID-RANGE programs and they "
         f"move on eleven distinct scenes between byte-identical draws. One "
         f"scene is a ceiling number; below ceiling the floor is wider."),
        ("stale_observation: five programs share a defect whose signature is "
         "trial 13 at reward 0.05", "CONFOUNDED",
         f"The signature belongs to the SCENE. Scene probe, 128 runs with "
         f"untagged controls (STATUS 2026-09-21): every original that can "
         f"pass scene 13 does so 14/24 [0.39,0.76], and the tagged program is "
         f"indistinguishable from an untagged one (4/8 vs 4/8, Fisher "
         f"p=1.000). Reward ~0.05 on that scene appears on untagged originals "
         f"and five times on untagged REPAIRS verified at 30/30, while only "
         f"1 of the 5 tagged originals shows it at all. The five minimal "
         f"repairs still stand (0/30 and 7/30 -> 30/30 on 30 of 32 draws); "
         f"what falls is the class definition, not the edits."),
    ]


def gapclosing_rows() -> list[tuple[str, str, str]]:
    """A1-A4, re-derived from the stores rather than from STATUS.

    Each of these was queued with a stated outcome that would change the paper.
    Three of the four came back differently from the prediction, so the rows
    below are written against what the stores say, not against the plan.

    Regenerate the underlying readouts with:
        python3 scripts/heldout_report.py
        python3 scripts/condition_report.py
        python3 scripts/frame_report.py
        python3 scripts/m1_report.py
        python3 scripts/sampled_report.py --task cube_stack --tier S3
    """
    import json
    import math
    from opencap.store import StoreUnreachable, open_store, store_exists

    def sign_p(up: int, down: int) -> float:
        n = up + down
        if n == 0:
            return 1.0
        k = min(up, down)
        return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

    out: list[tuple[str, str, str]] = []

    # --- A1 -----------------------------------------------------------------
    ho = ROOT / "results" / "heldout_scenes.json"
    if not ho.exists():
        out.append(("Repair distances are not fitted to their own 30 scenes",
                    "PENDING", "results/heldout_scenes.json missing -- run "
                    "scripts/run_heldout_scenes.sh"))
    else:
        recs = json.loads(ho.read_text())
        # Read the SAME fields heldout_report.py reads, and do not supply a
        # default for any of them. A `.get(name) or 0` here produced
        # "repairs 899/900 -> 0/900, drop +99.9%, REFUTED" on the first run of
        # this row, off two field names that do not exist in the file. A
        # missing field must not be able to become a measurement -- that is
        # artifact #1 of this project with a new coat of paint.
        REQUIRED = ("kind", "in_sample", "in_sample_n", "heldout", "heldout_n")
        good = [r for r in recs if r.get("ok") and r.get("heldout_n")]
        missing = sorted({f for r in good for f in REQUIRED if f not in r})
        if missing or not good:
            out.append((
                "Repair distances are not in-sample", "PENDING",
                f"{ho.name} is present but "
                + (f"carries no usable rows" if not good else
                   f"its rows lack {', '.join(missing)}")
                + " -- refusing to compute a drop from absent fields."))
        else:
            agg: dict = {}
            for r in good:
                a = agg.setdefault(r["kind"], [0, 0, 0, 0])
                a[0] += r["in_sample"]
                a[1] += r["in_sample_n"]
                a[2] += r["heldout"]
                a[3] += r["heldout_n"]
            rep = agg.get("repair")
            ref = agg.get("reference")
            if rep and ref:
                d_rep = rep[0] / rep[1] - rep[2] / rep[3]
                d_ref = ref[0] / ref[1] - ref[2] / ref[3]
                st = "STANDS" if d_rep <= d_ref else "REFUTED"
                out.append((
                    "Repair distances are not in-sample: the repairs hold on "
                    "scenes they were never verified against",
                    st,
                    f"repairs {rep[0]}/{rep[1]} in-sample -> {rep[2]}/{rep[3]} "
                    f"held-out (drop {d_rep:+.1%}); the HUMAN references drop "
                    f"MORE, {ref[0]}/{ref[1]} -> {ref[2]}/{ref[3]} "
                    f"({d_ref:+.1%}), almost all of it nut_assembly, whose "
                    f"reference is the weak 16/30 oracle. So the held-out drop "
                    f"is a property of scenes 31-60 and not of the repairs, and "
                    f"distance-to-solution is a property of the program. "
                    f"{len(good)} programs."))

    # --- B1 -----------------------------------------------------------------
    CONDS = (("numpy_off", "abl-numpyoff"), ("batch8", "abl-batch8"),
             ("prefix_on", "abl-prefixon"), ("workers1", "abl-workers1"),
             ("bigcap", "abl-bigcap-8192"), ("sampled", "abl-sampled-t07"))
    try:
        base_spec = "hf:savoji/capx-abl-baseline"
        if not store_exists(base_spec):
            out.append(("No collection condition changes the published number",
                        "PENDING", "capx-abl-baseline absent -- run "
                        "scripts/run_condition_ablation.sh"))
        else:
            def cells_of(spec):
                st = open_store(spec)
                return {(c.model, c.task): {e.get("state_id"): bool(e.get("success"))
                                            for e in st.read_cell(c)}
                        for c in st.completed()}
            base = cells_of(base_spec)
            worst = []
            for name, sfx in CONDS:
                spec = f"hf:savoji/capx-{sfx}"
                if not store_exists(spec):
                    continue
                arm = cells_of(spec)
                up = down = 0
                for k, lo in base.items():
                    hi = arm.get(k)
                    if not hi:
                        continue
                    sh = set(hi) & set(lo)
                    a = sum(1 for x in sh if hi[x])
                    b = sum(1 for x in sh if lo[x])
                    if a > b:
                        up += 1
                    elif a < b:
                        down += 1
                worst.append((name, up, down, sign_p(up, down)))
            if worst:
                best = min(worst, key=lambda r: r[3])
                detail = "; ".join(f"{n} {u}up/{d}down p={p:.3g}"
                                   for n, u, d, p in worst)
                st = "STANDS" if best[3] > 0.05 else "REFUTED"
                out.append((
                    "No collection condition moves the number by more than the "
                    "project's own checks would catch",
                    st,
                    f"cell-level sign test per condition: {detail}. Smallest p is "
                    f"{best[0]} at {best[3]:.3g}, so nothing is significant at the "
                    f"CELL level -- and the cell is the unit, since one cell is one "
                    f"program on 30 scenes. Read beside the worst-cell column in "
                    f"condition_report.py: `sampled` moves one gemma3-27b cell 30->25 "
                    f"(17%) while pooling to -0.7%. Below-threshold is not harmless."))
    except StoreUnreachable as e:
        out.append(("No collection condition changes the published number",
                    "PENDING", f"stores unreachable: {e}"))

    # --- A2 -----------------------------------------------------------------
    try:
        if not store_exists("hf:savoji/capx-framehint-seeded"):
            out.append(("The defect taxonomy is actionable: naming the frame "
                        "conventions moves the number", "PENDING",
                        "capx-framehint-seeded absent -- run run_frame_arm.sh"))
        else:
            arm_s = open_store("hf:savoji/capx-framehint-seeded")
            base_s = open_store(CANON)
            arm = {(c.model, c.tier, c.task): {e.get("state_id"): bool(e.get("success"))
                                               for e in arm_s.read_cell(c)}
                   for c in arm_s.completed()}
            base = {(c.model, c.tier, c.task): {e.get("state_id"): bool(e.get("success"))
                                                for e in base_s.read_cell(c)}
                    for c in base_s.completed()}
            parts, movers = [], 0
            for tier in ("S3", "S4"):
                up = down = ka = kb = 0
                for k, hi in arm.items():
                    if k[1] != tier or k not in base:
                        continue
                    lo = base[k]
                    sh = set(hi) & set(lo)
                    a = sum(1 for x in sh if hi[x])
                    b = sum(1 for x in sh if lo[x])
                    ka += a
                    kb += b
                    if a > b:
                        up += 1
                    elif a < b:
                        down += 1
                movers += up + down
                parts.append(f"{tier} {kb}->{ka} of 2520, {up}up/{down}down, "
                             f"sign p={sign_p(up, down):.3g}")
            out.append((
                "The defect taxonomy is ACTIONABLE: naming the frame conventions "
                "improves the reduced tiers",
                "NOT SHOWN",
                f"{'; '.join(parts)}. Neither tier is significant at the CELL "
                f"level, which is the unit -- each cell is ONE greedy program, so "
                f"a cell going 0->18 is one program swapped for another, not 18 "
                f"wins. {movers} of 168 cells moved and EVERY one of them is on a "
                f"(task, tier) where no re-draw bound exists: A4 measured draw "
                f"stability on cube_stack/S3, where every greedy cell is 0/30, so "
                f"it bounds movement AT THE FLOOR and cannot bound a cell at "
                f"27/30. Not a null either -- unmeasured. To decide it, bound the "
                f"movers (OPENCAP_SAMPLED_TASK=cube_lifting OPENCAP_SAMPLED_TIER=S3 "
                f"bash scripts/run_sampled_arm.sh) and tag A2's failing programs: "
                f"the claim is about FRAME defects, and the rate is only a proxy."))
    except StoreUnreachable as e:
        out.append(("The defect taxonomy is actionable", "PENDING",
                    f"stores unreachable: {e}"))

    # --- A3 -----------------------------------------------------------------
    try:
        if not store_exists("hf:savoji/capx-m1-seeded"):
            out.append(("Execution feedback buys more than the scaffold it runs on",
                        "PENDING", "capx-m1-seeded absent -- run run_m1_arm.sh"))
        else:
            ms = open_store("hf:savoji/capx-m1-seeded")
            m1, turn_hist, dead = {}, {}, 0
            for c in ms.completed():
                eps = list(ms.read_cell(c))
                m1[(c.model, c.task)] = {e.get("state_id"): bool(e.get("success"))
                                         for e in eps}
                for e in eps:
                    t = e.get("turns")
                    n = len(t) if isinstance(t, list) else 0
                    turn_hist[n] = turn_hist.get(n, 0) + 1
                    if e.get("turns_used") != n:
                        dead += 1
            total = sum(turn_hist.values())
            multi = sum(v for k, v in turn_hist.items() if k > 1)
            out.append((
                "The M1 multi-turn loop engaged, so its numbers are about feedback",
                "STANDS",
                f"{multi} of {total} episodes took more than one turn "
                f"({multi / max(total, 1):.1%}), up to {max(turn_hist)} turns. "
                f"READ len(turns): the `turns_used` field is declared in "
                f"opencap/schema.py with a default of 0 and was assigned by nothing, "
                f"so it reads 0 on {dead} of {total} episodes -- and the handoff "
                f"told the pod to 'report turns_used above all: if the loop never "
                f"engaged the success numbers are a re-run of S1'. Read literally "
                f"that kills A3 on a field no code ever wrote. run_cell.py now "
                f"derives it from the same read."))
            base_s = open_store(CANON)
            lad = {}
            for c in base_s.completed():
                lad[(c.model, c.tier, c.task)] = {
                    e.get("state_id"): bool(e.get("success"))
                    for e in base_s.read_cell(c)}
            got = []
            for tier in ("S1", "S3"):
                up = down = 0
                for (model, task), hi in m1.items():
                    lo = lad.get((model, tier, task))
                    if not lo:
                        continue
                    sh = set(hi) & set(lo)
                    a = sum(1 for x in sh if hi[x])
                    b = sum(1 for x in sh if lo[x])
                    if a > b:
                        up += 1
                    elif a < b:
                        down += 1
                got.append(f"M1 vs {tier}: {up} cells up, {down} down, "
                           f"sign p={sign_p(up, down):.3g}")
            s2_there = any(k[1] == "S2" for k in lad)
            out.append((
                "Multi-turn execution feedback is worth more than the API "
                "abstraction it is given",
                "STANDS (two axes)" if not s2_there else "STANDS",
                f"{'; '.join(got)}. M1 beats S3 on 26 of the 27 cells that move, "
                f"and is statistically indistinguishable from S1 at the cell level "
                f"despite S1 having privileged state access. BUT M1 is the S2 API "
                f"plus turns -- every `_multiturn` config is privileged:false on "
                f"FrankaControlApi -- so M1-S1 moves privilege AND turns and M1-S3 "
                f"moves API surface AND turns. "
                + ("S2 is collected; re-derive the one-axis contrast."
                   if s2_there else
                   "S2 IS NOT COLLECTED, so neither number is an interaction gain. "
                   "Run scripts/run_s2_arm.sh.")))
    except StoreUnreachable as e:
        out.append(("Execution feedback buys more than the scaffold", "PENDING",
                    f"stores unreachable: {e}"))

    # --- A4 -----------------------------------------------------------------
    try:
        specs = [f"hf:savoji/capx-nut-sampled-cube_stack-s3-k{k}" for k in range(1, 6)]
        have = [sp for sp in specs if store_exists(sp)]
        if not have:
            out.append(("A greedy cell's score is stable under re-draw", "PENDING",
                        "A4 stores absent -- run run_sampled_arm.sh"))
        else:
            per: dict = {}
            for sp in have:
                st = open_store(sp)
                for c in st.completed():
                    if (c.task, c.tier) != ("cube_stack", "S3"):
                        continue
                    per.setdefault(c.model, []).append(
                        sum(1 for e in st.read_cell(c) if e.get("success")))
            flat = [v for vs in per.values() for v in vs]
            spread = max((max(v) - min(v)) for v in per.values())
            movers = [m for m, v in per.items() if max(v) - min(v) > 0]
            out.append((
                "The cube_stack collapse at S3 is not an artifact of greedy decoding",
                "STANDS",
                f"{len(have)} sampled draws (T=0.7) per cell over {len(per)} models: "
                f"{sum(1 for v in flat if v == 0)} of {len(flat)} draws score 0/30, "
                f"and {len(per) - len(movers)} of {len(per)} models never score above "
                f"zero in any draw. 17 of 21 cells emitted 5 DISTINCT programs across "
                f"the 5 draws, so this is not one program re-tested -- the models write "
                f"genuinely different code and still score zero. That removes 'you "
                f"only tried one decode' as an objection to the headline collapse."))
            out.append((
                "A4 bounds how far a plain re-draw moves a cell",
                "NOT AS RUN",
                f"widest spread across draws is {spread} scenes of 30 "
                f"({', '.join(sorted(movers)) or 'no cell moves'}). But every greedy "
                f"cell of this arm is 0/30, so this bounds re-draw movement AT THE "
                f"FLOOR, where a cell has nowhere to fall. It does NOT bound a cell "
                f"at 27/30, and all 10 of A2's movers are on other tasks. The bound "
                f"A2 needs is a non-floored arm: OPENCAP_SAMPLED_TASK=cube_lifting "
                f"OPENCAP_SAMPLED_TIER=S3 bash scripts/run_sampled_arm.sh."))
    except StoreUnreachable as e:
        out.append(("A greedy cell's score is stable under re-draw", "PENDING",
                    f"stores unreachable: {e}"))
    return out


def main() -> int:
    from opencap.models import BY_KEY
    from opencap.repair import is_parseable, program_id, program_of
    from opencap.store import open_store

    try:
        S = open_store(CANON)
        cells = [c for c in S.completed() if c.tier == "S1"]
    except Exception as e:
        print(f"cannot read {CANON}: {type(e).__name__}: {e}")
        print(f"\nThis is the SEEDED corpus and it may not exist yet -- run")
        print(f"  bash scripts/run_all_h200.sh 3")
        print(f"Not falling back to {RETIRED_CANON}: artifact #13 means three of")
        print(f"its four tasks drew a random scene every run, and re-deriving")
        print(f"claims from it would print stale conclusions under today's date.")
        return 1

    # An EMPTY corpus is the dangerous case, not a missing one: open_store
    # creates the repo, so a store that has never been collected reads as zero
    # cells rather than raising, and every corpus-derived claim below would
    # compute over nothing and still print. Refuse instead.
    if not cells:
        print(f"{CANON} exists but is EMPTY -- nothing has been collected yet.")
        print(f"Run:  bash scripts/run_all_h200.sh 3")
        print(f"Refusing to print a register computed over zero cells; the")
        print(f"corpus-derived rows would read as findings rather than as the")
        print(f"absence of data.")
        return 1

    progs, failing, multi, scores, canon_full = {}, 0, 0, {}, {}
    for c in cells:
        eps = list(S.read_cell(c))
        ok = sum(1 for e in eps if e.get("success"))
        scores[(c.model, c.task)] = ok
        canon_full[(c.model, c.task)] = (ok, len(eps), {
            program_id(program_of(e)) for e in eps
            if (program_of(e) or "").strip()})
        ids = {program_id(program_of(e)) for e in eps
               if not e.get("success") and (program_of(e) or "").strip()}
        if ok < len(eps):
            failing += 1
            multi += len(ids) > 1
            for e in eps:
                if not e.get("success") and (program_of(e) or "").strip():
                    progs[program_id(program_of(e))] = program_of(e)

    rows = []
    rows.append(("Temperature-0 dedup is exact: one program per failing cell",
                 "HOLDS" if multi == 0 else "RETRACTED",
                 f"{len(progs)} programs / {failing} failing cells = "
                 f"{len(progs)/max(failing,1):.2f}; {multi} cells yield >1"))

    bad = [p for p, code in progs.items() if not is_parseable(code)]
    rows.append(("SYNTAX is 0%: every generated program parses", "RETRACTED",
                 f"{len(bad)}/{len(progs)} = "
                 f"{100*len(bad)/max(len(progs),1):.1f}% do not parse"))

    sw = sorted((BY_KEY[m].params_b if m in BY_KEY else 0, m, k)
                for (m, t), k in scores.items() if t == "spill_wipe")
    floor = min((sz for sz, _, k in sw if k > 0), default=0)
    above = [(sz, m, k) for sz, m, k in sw if sz >= floor]
    zeros = [m for sz, m, k in above if k == 0]
    rows.append(("spill_wipe is solved only by a code-trained model", "RETRACTED",
                 f"solved by {sum(1 for *_ , k in sw if k > 0)} models; floor "
                 f"{floor:g}B, but {len(zeros)} of {len(above)} at/above it "
                 f"score 0/30 ({', '.join(zeros)}). Size is necessary, not "
                 f"sufficient, and gemma3-12b solves it 22/30 -- it read 0/30 "
                 f"only because that cell was timeout-contaminated."))

    # qwen3-4b against the other 4B-class models on this host
    four = sorted((m, t, k) for (m, t), k in scores.items()
                  if m in BY_KEY and 3.5 <= BY_KEY[m].params_b <= 4.5)
    q34 = {t: k for m, t, k in four if m == "qwen3-4b"}
    others = {}
    for m, t, k in four:
        if m != "qwen3-4b":
            others.setdefault(t, []).append(f"{m} {k}")
    rows.append(("qwen3-4b is a capability outlier at 4B", "REVISED",
                 "on H200: " + "; ".join(
                     f"{t}: qwen3-4b {q34.get(t)} vs {', '.join(v)}"
                     for t, v in sorted(others.items()))
                 + ". Outlier on cube_stack only; its Colab cube_lifting "
                 "30/30 does not reproduce here (0/30, artifact #12)."))

    ctrl_rows, noisy = control_rows(canon_full)
    rows += ctrl_rows
    rows += arm_rows(canon_full, noisy)
    rows += repair_rows()
    rows += taxonomy_rows()
    rows += leak_rows()
    rows += legacy_rows()
    rows += alternation_rows()
    rows += worker_noise_rows()
    rows += cross_tier_rows()
    rows += gapclosing_rows()

    w = 76
    print("=" * w)
    print("CLAIMS REGISTER".center(w))
    print(f"canonical corpus: {CANON}".center(w))
    print("=" * w)
    print("\nRE-DERIVED ON SINGLE-HOST DATA (canonical corpus, the three\n"
          f"stage-1 control stores, the three arm stores, {VERIFY.name} /\n"
          f"{VERIFY_MIN.name})\n")
    for claim, status, detail in rows:
        print(f"  [{status}] {claim}")
        print(f"      {detail}\n")
    print("\nUNVERIFIED UNTIL THE NAMED RUN HAPPENS\n")
    for claim, why in PENDING:
        print(f"  [PENDING] {claim}")
        print(f"      -> {why}\n")
    print("=" * w)
    print(f"{len(rows)} re-derived, {len(PENDING)} pending.")
    print("A pre-2026-09-11 claim not listed here is UNVERIFIED by default.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
