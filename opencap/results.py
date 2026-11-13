"""Parse CaP-X output directories into our Episode rows.

Schema learned from a real run on 2026-09-04 (cube_stack S1 oracle, n=30).

Two traps, both of which would silently corrupt success rates:

  * **CaP-X rewrites the output path.** Passing `--output-dir /x/run` produces
    `/x/<model_slug>/run`. The model slug is inserted one level up.
  * **A trial index can produce several directories.** `trial_03_..._reward_0.000`
    and `trial_03_..._reward_1.000` can both exist. Observed 4s apart, so this
    is a worker collision on the trial index, not a TRIAL_TIMEOUT_SECONDS retry
    (those are ~1000s apart). Whatever the cause, CaP-X's own summary counts
    the **latest-written** directory, so that is the outcome of record.
    Picking the *best* attempt instead biases every success rate upward -- the
    cross_check() below exists precisely to catch that class of mistake, and
    did catch it on first run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: trial_05_sandboxrc_0_reward_1.000_taskcompleted_1
# Reward can be negative (penalty terms): Qwen2.5-Coder-7B produced dir names
# like trial_07_sandboxrc_0_reward_-10.143_taskcompleted_0. A [\d.]+ reward
# group matched none of those, so parse_trials silently returned a SUBSET --
# 14 of 30 -- and any success rate computed from it would be over a biased
# sample. cross_check's trial-count test is what caught it.
TRIAL_DIR = re.compile(
    r"^trial_(?P<trial>\d+)_sandboxrc_(?P<rc>-?\d+)"
    r"_reward_(?P<reward>-?[\d.]+)_taskcompleted_(?P<done>-?\d+)$"
)

DONE_FLAG = "aaa_done_flag"


def resolved_output_dir(requested: str | Path, model_repo: str) -> Path:
    """Where CaP-X actually writes, given what we asked for."""
    requested = Path(requested)
    slug = model_repo.replace("/", "_")
    return requested.parent / slug / requested.name


def is_complete(run_dir: str | Path) -> bool:
    """CaP-X drops `aaa_done_flag` when a run finishes -- cheaper and more
    reliable than re-reading summaries.txt."""
    return (Path(run_dir) / DONE_FLAG).exists()


@dataclass(frozen=True)
class TrialResult:
    trial: int
    reward: float
    success: bool
    sandbox_rc: int
    attempts: int
    dir_name: str


def parse_trials(run_dir: str | Path) -> list[TrialResult]:
    """One result per trial index, taking the final attempt."""
    run_dir = Path(run_dir)
    by_trial: dict[int, list[re.Match]] = {}
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        m = TRIAL_DIR.match(child.name)
        if m:
            by_trial.setdefault(int(m["trial"]), []).append(m)

    out = []
    for trial, matches in sorted(by_trial.items()):
        # Latest mtime wins -- this is what CaP-X's summaries.txt agrees with.
        final = max(matches, key=lambda m: (run_dir / m.string).stat().st_mtime)
        out.append(TrialResult(
            trial=trial,
            reward=float(final["reward"]),
            success=final["done"] == "1",
            sandbox_rc=int(final["rc"]),
            attempts=len(matches),
            dir_name=final.string,
        ))
    return out


#: A trial whose LLM request timed out produces an empty program, reward 0.0
#: and `Num Code Blocks: 0` -- and enters the corpus as a model failure. It is
#: not one: the model was never asked a question it could answer.
#:
#: Measured 2026-09-10: 102 of 820 episodes across four cells (qwen35-0.8b and
#: qwen35-2b on cube_stack and spill_wipe) hit CaP-X's 200s HTTP read timeout
#: while five sim workers queued behind a batch-size-1 shim. Those cells read
#: 0/30 with ~25 of each 30 scenes never attempted. Same shape as the 39-cell
#: corpus with no stderr: a clean-looking number nobody can attribute.
MAX_TIMEOUT_FRACTION = 0.20


def timeout_fraction(run_dir: str | Path) -> tuple[float, int, int]:
    """-> (fraction, n_timed_out, n_trials) over the run's trial directories."""
    run_dir = Path(run_dir)
    dirs = [d for d in run_dir.glob("trial_*") if d.is_dir()]
    if not dirs:
        return 0.0, 0, 0
    n_to = 0
    for d in dirs:
        f = d / "summary.txt"
        if f.exists():
            try:
                if "timed out after" in f.read_text():
                    n_to += 1
            except OSError:
                continue
    return n_to / len(dirs), n_to, len(dirs)


def classify_failure(t: "TrialResult") -> str:
    """Map a trial onto the Sec. 04 failure taxonomy.

    `sandbox_rc` is the load-bearing signal: non-zero means the generated
    program raised before finishing, so the model failed at *writing code*.
    Zero with no task completion means the code ran fine and the robot did the
    wrong thing -- a reasoning failure, not a coding one. Collapsing the two
    (as the first version did) makes the taxonomy useless exactly where it
    matters most, at S3/S4.
    """
    if t.success:
        return "success"
    if t.sandbox_rc != 0:
        return "api_misuse"       # refined by log inspection in P5
    if t.reward <= 1e-6:
        return "no_op"            # ran clean, nothing moved
    return "physical_failure"     # executed, wrong outcome


@dataclass(frozen=True)
class RunSummary:
    trials: int
    codegen_rate: float
    avg_reward: float
    completed: int
    avg_code_blocks: float
    avg_regenerations: float

    @property
    def success_rate(self) -> float:
        return self.completed / self.trials if self.trials else 0.0


def parse_summary(run_dir: str | Path) -> RunSummary | None:
    """Read CaP-X's own aggregate. Used to cross-check parse_trials()."""
    path = Path(run_dir) / "summaries.txt"
    if not path.exists():
        return None
    text = path.read_text()

    def num(pattern: str, cast=float):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    # Average reward can be NEGATIVE (collision/penalty terms). The original
    # [\d.]+ silently failed to match "1.000/-10.143/0", and the caller then
    # reported "no summaries.txt" for a file that exists and is well-formed.
    triple = re.search(r"^(-?[\d.]+)/(-?[\d.]+)/(-?\d+)\s*$", text, re.M)
    if not triple:
        return None
    return RunSummary(
        trials=num(r"Total number of trials:\s*(\d+)", int) or 0,
        codegen_rate=float(triple.group(1)),
        avg_reward=float(triple.group(2)),
        completed=int(triple.group(3)),
        avg_code_blocks=num(r"Average code blocks:\s*([\d.]+)") or 0.0,
        avg_regenerations=num(r"Average regenerations:\s*([\d.]+)") or 0.0,
    )


def cross_check(run_dir: str | Path) -> tuple[bool, str]:
    """Directory parse must agree with CaP-X's own summary.

    Disagreement means the de-duplication is wrong, which is exactly the failure
    that would quietly bias every success rate in the corpus. Callers must treat
    a False here as fatal for the cell and refuse to commit it.
    """
    summary = parse_summary(run_dir)
    trials = parse_trials(run_dir)
    if summary is None:
        # Distinguish absent from unparseable: conflating them sent a real
        # debugging session hunting a path bug that did not exist.
        exists = (Path(run_dir) / "summaries.txt").exists()
        return False, ("summaries.txt unparseable" if exists
                       else "no summaries.txt")
    n, done = len(trials), sum(t.success for t in trials)
    if n != summary.trials:
        return False, f"trial count {n} != summary {summary.trials}"
    if done != summary.completed:
        return False, f"successes {done} != summary {summary.completed}"
    return True, f"{done}/{n} agree"
