"""Distance-to-solution: how far is a failed program from one that works?

WHY THIS EXISTS
---------------
A benchmark score of 0/30 is the same number whether the model wrote a correct
program that hit a missing import, or wrote nothing that could ever move a
robot. The root-cause taxonomy separates those categorically; this separates
them CONTINUOUSLY, by asking what it costs to repair each one.

    L0  passes already
    L1  recovered by a scaffold fix, no model change and no code change
    L2  recovered by a prompt fix
    L3  recovered by editing the program -- distance is the size of that edit
    L4  not recovered within budget

The measurement is the level, plus the normalised edit size at L3. A repair is
only counted once it has been RE-EXECUTED against the same 30 scene seeds: an
unverified repair is an opinion.

DEDUPLICATION
-------------
Generation is greedy (temperature 0) against a fixed prompt, so all 30 episodes
of a cell share one program and differ only in scene. 926 failed episodes in the
current corpus collapse to 32 distinct programs. Repair operates on programs,
and results fan back out to the episodes that share them.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping

#: Levels, cheapest first. The level is the smallest one that recovers.
LEVELS = ("L0_passes", "L1_scaffold", "L2_prompt", "L3_edit", "L4_unrecovered")


def program_of(episode: Mapping) -> str:
    """The code an episode ran, concatenated across turns."""
    return "\n".join((t.get("code") or "") for t in (episode.get("turns") or [])).strip()


def program_id(code: str) -> str:
    return hashlib.sha1(code.encode()).hexdigest()[:12]


def strip_markers(code: str) -> str:
    """Drop CaP-X's `# Code block N` separators, which are not model output."""
    return re.sub(r"^# Code block \d+\s*$", "", code, flags=re.M).strip()


@dataclass(frozen=True)
class Program:
    """One distinct generated program and every episode that produced it."""
    pid: str
    code: str
    task: str
    tier: str
    models: tuple[str, ...]
    n_episodes: int
    root_causes: tuple[str, ...]
    median_reward: float
    error_excerpt: str = ""


#: A program the model finished ends on a statement. One cut at the token cap
#: ends wherever the 2,048th token fell -- mid-assignment, mid-call, or inside
#: a comment, which is the case that still PARSES and is therefore the one that
#: reaches the corpus looking like a model defect.
#:
#: MEASUREMENT ARTIFACT #15. A truncated program enters the store with
#: tokens_out = 0, truncation_fired false, and a no_op or api_misuse label --
#: indistinguishable from a program the model finished and got wrong. The shim's
#: own stats are the only record that it happened (scripts/truncation_audit.py).
#: This is the corpus-side half: given only the code, does it look cut off?
#:
#: The line threshold is not arbitrary. Across the 70-program seeded corpus the
#: programs that parse and end on a statement run 5-83 lines, median 22; every
#: program at or above 100 lines is cut off mid-token. Both halves of the test
#: are needed: 5 of the 8 cut-off programs PARSE, because the cut landed inside
#: a trailing comment.
TRUNCATION_LINES = 100


def looks_truncated(code: str) -> bool:
    """Does this program look cut off at the token cap?

    Conservative on purpose. It catches a cut that is VISIBLE in the code and
    says nothing about one that landed just after a complete statement -- so a
    False here is not evidence the program was finished, and a corpus-level
    claim needs the shim's per-request record as well.
    """
    import ast as _ast
    lines = [l for l in (code or "").rstrip().splitlines() if l.strip()]
    if not lines:
        return False
    if len(lines) >= TRUNCATION_LINES:
        return True
    try:
        _ast.parse(code)
    except SyntaxError:
        return True
    return False


def normalised_edit_distance(before: str, after: str) -> float:
    """0.0 = identical, 1.0 = shares nothing. Line-level, whitespace-normalised.

    Line-level rather than character-level because the unit of repair is a
    statement: adding one missing `goto_pose` call should cost the same whether
    the variable names are long or short.
    """
    def lines(s: str) -> list[str]:
        return [ln.strip() for ln in strip_markers(s).splitlines() if ln.strip()]
    a, b = lines(before), lines(after)
    if not a and not b:
        return 0.0
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return 1.0 - sm.ratio()


def edit_ops(before: str, after: str) -> dict[str, int]:
    """Counts of added / removed / changed lines -- the shape of the repair."""
    def lines(s: str) -> list[str]:
        return [ln.strip() for ln in strip_markers(s).splitlines() if ln.strip()]
    a, b = lines(before), lines(after)
    out = {"added": 0, "removed": 0, "changed": 0, "before_lines": len(a),
           "after_lines": len(b)}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
        if tag == "insert":
            out["added"] += j2 - j1
        elif tag == "delete":
            out["removed"] += i2 - i1
        elif tag == "replace":
            out["changed"] += max(i2 - i1, j2 - j1)
    return out


def is_parseable(code: str) -> bool:
    try:
        ast.parse(strip_markers(code))
        return True
    except SyntaxError:
        return False


@dataclass
class RepairResult:
    """One repair attempt, with the verification that makes it a measurement."""
    pid: str
    level: str
    repaired_code: str = ""
    distance: float | None = None
    ops: dict[str, int] = field(default_factory=dict)
    #: Filled in only after re-execution against the original seeds.
    verified_successes: int | None = None
    verified_n: int | None = None
    note: str = ""

    @property
    def verified(self) -> bool:
        return self.verified_successes is not None

    @property
    def recovered(self) -> bool:
        """A repair counts only if the env agrees. Never infer this."""
        return bool(self.verified and self.verified_successes)


def group_failures(cells: Iterable[tuple], read) -> dict[str, Program]:
    """Collapse failed episodes into distinct programs.

    `cells` yields Cell objects; `read(cell)` yields its episodes.
    """
    from statistics import median
    from collections import defaultdict
    buckets: dict[str, dict] = defaultdict(
        lambda: {"models": set(), "causes": [], "rewards": [], "n": 0,
                 "code": "", "task": "", "tier": "", "err": ""})
    from .failures import classify_root_cause
    for cell in cells:
        for e in read(cell):
            if e.get("success"):
                continue
            code = program_of(e)
            if not code:
                continue
            b = buckets[program_id(code)]
            b["code"] = code
            b["task"], b["tier"] = cell.task, cell.tier
            b["models"].add(cell.model)
            b["causes"].append(classify_root_cause(e))
            b["rewards"].append(float(e.get("reward") or 0.0))
            b["n"] += 1
            if not b["err"]:
                tail = "".join((t.get("summary_tail") or "")
                               for t in (e.get("turns") or []))
                m = re.search(r"\w*Error:[^\n]{0,160}", tail)
                if m:
                    b["err"] = m.group(0)
    out = {}
    for pid, b in buckets.items():
        out[pid] = Program(
            pid=pid, code=b["code"], task=b["task"], tier=b["tier"],
            models=tuple(sorted(b["models"])), n_episodes=b["n"],
            root_causes=tuple(sorted(set(b["causes"]))),
            median_reward=median(b["rewards"]) if b["rewards"] else 0.0,
            error_excerpt=b["err"])
    return out
