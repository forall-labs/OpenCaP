> **Start here instead:** `OPERATOR.md` if you are running a rater,
> `RATER_TASK.md` if you *are* the rater.
>
> **This file names the study design and is on the rater's do-not-open list.**
> It is kept as the original packet note. Do not send it to a rater.

# Second-rater packet

60 failed robot programs (30 S1, 30 S3/S4; random, seed 41), shown as the diff from the
model's original code to its VERIFIED minimal repair. Tier labels are hidden (the API in
the code may still reveal it). Shared kit code (environment workarounds) is collapsed.

1. Read `taxonomy.md`.
2. For each program in `packet.md`, list ONE tag per distinct cause the repair had to fix.
3. Fill `answers_template.csv` (`tags` = tag names separated by `;`). Do not open `key.json`
   or anything under `results/` until done.

`retag_session2.json` is the original rater's own blind re-tag (same day; an upper bound on
reliability, not inter-rater). Against the filed tags: pooled tag-level kappa 0.89; family
kappa FRAME 0.67, INTERFACE 0.90, SEMANTIC 1.00, STRUCTURE 0.96, TRAJECTORY 0.96.

To score a second rater: compare their CSV with `results/tags_s1_hand.json` /
`results/tags_reduced.json` via `key.json` (same computation as STATUS 2026-10-04 ~23:00).
