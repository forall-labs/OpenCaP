"""Root-cause failure taxonomy.

`classify_failure` in results.py is coarse: it splits on sandbox_rc and reward,
giving api_misuse / no_op / physical_failure. That was enough to show the split
exists, but it lumps together failures with completely different remedies --
a missing import and a wrong API signature both land in `api_misuse`, yet one
is fixed by a prompt line and the other needs the model to understand the API.

This module classifies the ROOT CAUSE, which is what tells you how to improve a
model rather than merely how it scored. Two families:

EXECUTION failures -- the program raised, so the robot never got a fair trial:
  missing_import    NameError on a module the prompt said to import (numpy).
                    Pure instruction-following; a prompt fix erases it.
  undefined_name    NameError on something else -- hallucinated variable/API.
  syntax            SyntaxError: malformed output, e.g. reasoning tokens
                    written into the code buffer.
  wrong_signature   TypeError on an API call -- wrong arity or argument types.
  bad_index         IndexError/KeyError -- misunderstood a return shape, e.g.
                    indexing a scalar returned by a dot product.
  bad_value         ValueError -- right shape, invalid content.
  attribute         AttributeError -- called a method that does not exist.
  timeout           exceeded TRIAL_TIMEOUT_SECONDS.
  other_exception   anything else.

SEMANTIC failures -- the program ran cleanly and did the wrong thing:
  no_op             nothing moved (reward ~ 0). Often a plan that never acts.
  incomplete        real partial progress then stopped -- e.g. grasped the cube
                    and never lifted it. Distinguishing this from no_op is what
                    separates "cannot act" from "cannot finish".
  mock_shadowing    redefined the provided APIs as local stubs, so the program
                    runs against its own print statements and never reaches the
                    robot. gemma3-12b wrote a self-contained "testable script"
                    with mock goto_pose/get_object_pose in a __main__ block.
                    Remedy is one prompt line, so this is an L2 recovery -- the
                    model's task reasoning may be entirely sound underneath.
  never_invoked     defined a function holding the whole program and never
                    called it, with no top-level code. The module binds a name
                    and exits: reward 0, no exception, nothing to see in the
                    score. Induced by the low-level tiers, where the prompt
                    reads like a coding task -- so the remedy is the harness or
                    the prompt, not the model's reasoning.
  no_revision       emitted the same program on every turn -- the multi-turn
                    loop bought nothing, so more turns cannot help. Distinct
                    from `incomplete`: the remedy is making the model react to
                    feedback, not longer-horizon planning.
  marginal          moved the scene slightly, nowhere near the goal.
  incomplete        substantial progress toward the goal, did not finish.

The point of the split is prescriptive: missing_import wants a prompt change,
wrong_signature wants better API grounding, incomplete wants longer-horizon
planning. Aggregated by model scale it says WHICH failure mode shrinks as
models grow -- and which does not.
"""
from __future__ import annotations

import re
from typing import Iterable, Mapping

#: Ordered: first match wins, most specific first.
_EXC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("missing_import", re.compile(r"NameError:.*name '(np|numpy|math|scipy)' is not defined")),
    ("undefined_name", re.compile(r"NameError:")),
    ("syntax", re.compile(r"(SyntaxError|IndentationError|TabError):")),
    ("wrong_signature", re.compile(r"TypeError:")),
    ("bad_index", re.compile(r"(IndexError|KeyError):")),
    ("bad_value", re.compile(r"ValueError:")),
    ("attribute", re.compile(r"AttributeError:")),
    ("timeout", re.compile(r"(TimeoutError|timed out)")),
)

#: "Execution" means the program never ran as written -- it crashed, or it was
#: never invoked. Both are fixed outside the model's task reasoning.
EXECUTION = {"missing_import", "undefined_name", "syntax", "wrong_signature",
             "bad_index", "bad_value", "attribute", "timeout", "other_exception",
             "never_invoked", "mock_shadowing"}
SEMANTIC = {"no_op", "marginal", "incomplete", "no_revision"}

#: Reward below this counts as "nothing happened".
NO_OP_REWARD = 0.05
#: At or above this, the episode made substantial progress toward the goal;
#: between the two thresholds it moved the scene but got nowhere near it.
INCOMPLETE_REWARD = 0.25


#: Names the harness injects into the execution namespace. A program that
#: rebinds any of these is running against itself.
PROVIDED_APIS = frozenset({
    "get_object_pose", "sample_grasp_pose", "goto_pose", "open_gripper",
    "close_gripper", "solve_ik", "move_to_joints", "get_observation",
    "plan_grasp", "point_prompt_molmo", "segment_sam3_text_prompt",
    "segment_sam3_point_prompt", "get_oriented_bounding_box_from_3d_points",
})


def shadowed_apis(code: str) -> set[str]:
    """Provided API names the program redefines as its own function or value."""
    import ast
    try:
        # Inlined rather than imported from .repair, which already imports
        # classify_root_cause from here.
        tree = ast.parse(re.sub(r"^# Code block \d+\s*$", "", code, flags=re.M))
    except SyntaxError:
        return set()
    hit: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if n.name in PROVIDED_APIS:
                hit.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in PROVIDED_APIS:
                    hit.add(t.id)
    return hit


def codes_of(episode: Mapping) -> list[str]:
    out = [(t.get("code") or t.get("content") or "").strip()
           for t in (episode.get("turns") or [])]
    return [c for c in out if c]


#: Printed by the OPENCAP_CALL_ENTRYPOINT patch when it invokes a function.
ENTRYPOINT_MARKER = "invoking never-called entry point"


def _entrypoint_was_called(episode: Mapping) -> bool:
    for turn in episode.get("turns") or []:
        if ENTRYPOINT_MARKER in (turn.get("summary_tail") or ""):
            return True
    return False


def _is_never_invoked(codes: list[str]) -> bool:
    """Every defined function unused, and nothing at top level to run."""
    if not codes:
        return False
    import ast
    src = re.sub(r"^# Code block \d+\s*$", "", codes[-1], flags=re.M)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    defs = {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if not defs:
        return False
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    if defs - called != defs:
        return False
    runnable = [n for n in tree.body
                if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.Import, ast.ImportFrom, ast.ClassDef))]
    return not runnable


def classify_root_cause(episode: Mapping) -> str:
    """Root cause for one episode trace. Requires `turns` to carry stderr."""
    if episode.get("success"):
        return "success"

    text = ""
    for turn in episode.get("turns") or []:
        text += turn.get("summary_tail", "") or ""
    for name, pat in _EXC_PATTERNS:
        if pat.search(text):
            return name

    # Rebinding the injected APIs means the program never reached the robot,
    # whatever else it did. Checked before the reward heuristic because the
    # resulting reward is meaningless -- nothing moved.
    _codes = codes_of(episode)
    if _codes and shadowed_apis(_codes[-1]):
        return "mock_shadowing"

    # A program sealed inside an uncalled function does nothing at all. It
    # scores 0 with no exception, so it is indistinguishable from "the robot
    # tried and failed" unless the code is parsed. qwen3-4b hit this on 30/30
    # S4 cube_stack episodes.
    # ...unless the harness itself invoked it. The entry-point arm calls the
    # function at runtime, but the SOURCE still contains an uncalled def, so a
    # code-structure test alone keeps reporting never_invoked for episodes that
    # demonstrably ran (rewards moved). Trust the runtime marker over the AST.
    if _is_never_invoked(codes_of(episode)) and not _entrypoint_was_called(episode):
        return "never_invoked"

    # Repeating one program verbatim across turns is its own failure: the
    # 8B rung emitted byte-identical code on all three of its turns while
    # never writing the lift. Scored as `incomplete`, that reads as "needs
    # longer-horizon planning"; it actually needs to react to feedback at all.
    codes = codes_of(episode)
    if len(codes) >= 2 and len(set(codes)) == 1:
        return "no_revision"

    # No exception found. Either the trace lacks stderr, or the code ran clean.
    label = episode.get("failure_label", "")
    if label in ("api_misuse", "syntax_error") and not text:
        return "unknown_execution"      # rc!=0 but no captured stderr
    reward = float(episode.get("reward") or 0.0)
    if reward < NO_OP_REWARD:
        return "no_op"
    if reward < INCOMPLETE_REWARD:
        return "marginal"
    return "incomplete"


def summarise(episodes: Iterable[Mapping]) -> dict[str, int]:
    from collections import Counter
    return dict(Counter(classify_root_cause(e) for e in episodes))


def family(cause: str) -> str:
    if cause == "success":
        return "success"
    if cause in EXECUTION or cause == "unknown_execution":
        return "execution"
    return "semantic"
