# Running a second rater (operator notes)

**Do not give this file to the rater.** It names what the exercise measures.
The rater gets `RATER_TASK.md`, which is deliberately blind.

## Status, 2026-10-13 — closed

An independent rating is **done**: GPT-6 Astra (high reasoning), via codex,
blind and isolated. The family contrasts reproduced in its own tags and FRAME
agreement came back at κ 0.82, above the same-rater 0.67. Numbers in STATUS
2026-10-13 and §3.3 of `docs/WRITEUP.md`; provenance in `ratings.json`.

**A human rating is NOT planned** (owner decision, 2026-10-13). The packet
stays in the repository and the scoring path is unchanged, so one can be run
later with no code change — drop the CSV in as `answers_<name>.csv` and the
claims register picks it up. Everything below describes how.

## What this is for

The Phase 9 tags were produced by one rater on both sides of the comparison
(S1 and S3/S4). The write-up's limitation section says so. Without a second
rater there is no inter-rater agreement figure, and the family contrasts —
FRAME 4% → 47%, INTERFACE 26% → 60% — rest on one rater's judgement.

`retag_session2.json` is the first rater's own blind re-tag of the same 60
programs: pooled κ 0.89, FRAME 0.67. That is an **upper bound** on reliability,
not inter-rater agreement, because it is the same rater on the same day.

## What the first rater actually was

The filed tags were produced by **an LLM agent session**, not a human. The
`tagged_by` field says `"hand (CLI session, 2026-10-04)"` — "hand" there means
a human-like reading pass rather than the automated `classify_edits` tagger; it
does not mean a person.

This matters for how you describe the result:

| second rater | what it measures | how to describe it |
|---|---|---|
| A different LLM, different family | whether the tags are inferable from the diff by an independent reader of the same kind | "a second independent LLM rater (model X), blind to the hypothesis and to the filed tags" |
| A human familiar with robot code | whether the taxonomy means anything to a domain expert at all | "inter-rater agreement with a human rater" |

They are different claims. **Never report either as plain "inter-rater
agreement" without saying which.** Both are worth having; the LLM one is free
and immediate, the human one is what a reviewer is asking for.

Also note the asymmetry: the first rater knew the research question; the second
does not. Blind is the better condition, but the two passes were not run under
the same one, and the write-up should say so.

## Procedure — an LLM rater (e.g. codex / an OpenAI model)

The risk is contamination: the answers, the hypothesis and the conclusions are
all in this repository. Isolate the material rather than trusting an
instruction not to look.

**Preferred — a clean directory.** Nothing to resist:

```bash
mkdir -p /tmp/rater && cd /tmp/rater
cp <repo>/artifacts/second_rater/RATER_TASK.md .
cp <repo>/artifacts/second_rater/taxonomy_full.md .
cp <repo>/artifacts/second_rater/packet.md .
# point the model at /tmp/rater and nothing else
```

Then: *"Read RATER_TASK.md and do what it says."* Give it nothing else — no
framing, no mention of tiers, APIs, families or what is being compared.

**If it runs in the repo instead**, `RATER_TASK.md` lists the files not to open,
and asks the rater to stop and say so if it believes it has seen the key. Treat
any such report as a disqualified run, not a nuisance.

Either way, record: which model, which date, whether it ran isolated or in the
repo, and whether it reported seeing anything it should not have. Put that in
the STATUS entry alongside the numbers.

## Procedure — a human rater

Send them `RATER_TASK.md`, `taxonomy_full.md`, `packet.md`. About 2–3 hours at
2–4 minutes per item. Do not tell them the hypothesis, and do not answer
"should this be X or Y" questions while they work — those judgements are the
measurement. Collect the CSV.

## Scoring

```bash
python3 scripts/score_second_rater.py artifacts/second_rater/answers_<name>.csv \
    --label "gpt-5 codex, isolated, 2026-10-14"
```

It prints pooled tag-level κ, per-family κ, per-tag κ, mean per-program
Jaccard, and — the part that matters — whether the **FRAME and INTERFACE
contrasts reproduce in the second rater's own tags**. Same computation as the
same-rater re-tag, so the two are comparable.

## How to read the outcome

- **Family contrasts reproduce, κ ≳ 0.6 on FRAME and INTERFACE** → the headline
  is rater-independent. This is the result the write-up needs.
- **Contrasts reproduce, κ low** → the two raters draw the family boundaries
  differently but see the same shift. Report both; the contrast is what the
  claim rests on.
- **Contrasts do not reproduce** → the family result is one rater's reading.
  That is a finding about the method and it has to be published as such. Do not
  re-brief the rater and re-run until it agrees; that is fitting the rater to
  the hypothesis.
- **The count gap** (defects per program) was already the least robust figure in
  the same-rater check. Expect it to be the first thing to fail here, and keep
  reporting the family contrasts ahead of it.

A disagreeing second rater is a usable result. A second rater coached into
agreement is not.
