"""What actually went wrong, derived from a verified repair.

The earlier taxonomy in `failures.py` inferred a cause from exception text and
reward thresholds -- a guess about what a program needed. This module works from
evidence instead: for every failing program we now hold a MINIMAL edit that was
re-executed and reached the task ceiling, so the diff between the two IS the
defect, at line granularity.

Every tag below was observed in this corpus and every one is backed by a repair
that verified at 30/30. Nothing here is speculative.

    SYNTAX      the program did not parse
    STRUCTURE   it parsed and could never act: never called, APIs shadowed by
                stubs, gripper calls ordered so the object is never held
    INTERFACE   it acted on the API wrongly: bad object strings, wrong arity,
                duplicate kwargs, names used before binding
    SEMANTIC    it ran and did the wrong thing: wrong object's coordinates, a
                missing terminal action, a wrong formula, a stale pose read
    TRAJECTORY  the actions were right and the motion between them was not:
                z_approach misuse, spurious pre-moves, post-release retreats
    FRAME       the task, the API and the motion were all right and the
                arithmetic was done in the wrong coordinate frame: an offset
                applied in world coordinates instead of rotated into the
                target's frame, or a wxyz quaternion handed to a library that
                takes xyzw

TRAJECTORY is the class the exception-based taxonomy could not see at all --
none of those defects raises, and each costs only a handful of scenes in thirty.

FRAME IS n=4. It was added because five programs with a verified, non-zero
distance repair carried no tag at all, and eight of the ten untagged programs
were >=8B: the taxonomy was silent exactly where the interesting models are.
Only four programs in this corpus attempt a coordinate transform, every one of
them is >=8B and on nut_assembly, and every one gets it wrong. That is a
hypothesis worth a designed run, NOT a rate -- read the family share as "4
programs", and see STATUS.md 2026-09-18 for what it would take to test.
"""
from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass

FAMILY = {
    "syntax": "SYNTAX",
    "never_invoked": "STRUCTURE",
    "mock_shadowing": "STRUCTURE",
    "api_rebound": "STRUCTURE",
    "gripper_order": "STRUCTURE",
    "object_name": "INTERFACE",
    "call_signature": "INTERFACE",
    "unbound_name": "INTERFACE",
    "wrong_target": "SEMANTIC",
    "missing_action": "SEMANTIC",
    "wrong_formula": "SEMANTIC",
    "stale_read": "SEMANTIC",
    "wrong_task": "SEMANTIC",
    "missing_coverage": "SEMANTIC",
    "approach_param": "TRAJECTORY",
    "spurious_move": "TRAJECTORY",
    "post_release_retreat": "TRAJECTORY",
    "frame_convention": "FRAME",
}

#: Names the harness injects into the code-execution namespace. A program that
#: rebinds one runs against itself; a program that CALLS one is using the API,
#: not inventing a name.
#:
#: This set was hand-maintained and it was wrong. `goto_home_joint_position` is
#: bound at runtime: the human reference calls it (results/references.json), 21
#: verified nut_assembly repairs call it and reach 30/30 in the official
#: verifier, and the nine model programs that call it raised no NameError in
#: 270 episodes -- while the same detector caught `get_fixed_rigid_transform`
#: on a sibling program in the same task. Omitting it tagged those nine
#: `unbound_name`, which is what put INTERFACE at 69% in the >=12B band.
#:
#: Do not edit this by hand again. scripts/verify_provided_api.py reads the
#: names out of a cap-x checkout's `_init_exec_globals` and fails on a
#: mismatch; it runs in stage 0.
PROVIDED = frozenset({
    "get_object_pose", "sample_grasp_pose", "goto_pose", "open_gripper",
    "close_gripper", "solve_ik", "move_to_joints", "get_observation",
    "goto_home_joint_position",
})

#: Bound only when OPENCAP_PREIMPORT_NUMPY is set, which the baseline arm sets
#: and the numpy_off arm does not (scripts/run_arm.sh). Using `np` without
#: importing it is therefore legal in the corpus everything here is tagged
#: from, and is the entire subject of the A/B in the arm that is not.
PREIMPORTED = frozenset({"np", "numpy"})

_MARK = re.compile(r"^# Code block \d+\s*$", re.M)

#: wxyz <-> xyzw. The privileged API hands back wxyz and scipy takes xyzw; the
#: human reference does the reorder by hand and comments on it, which is a fair
#: sign of how easy it is to miss.
_QUAT_PERM = re.compile(
    r"\[\s*1\s*,\s*2\s*,\s*3\s*,\s*0\s*\]"
    r"|\[\s*3\s*,\s*0\s*,\s*1\s*,\s*2\s*\]"
    r"|np\.roll\s*\([^)]*?,\s*-?1\s*\)")

#: Taking a vector out of one frame and into another.
_FRAME_ROT = re.compile(
    r"\.apply\s*\(|from_quat\s*\(|from_matrix\s*\(|as_matrix\s*\("
    r"|quat_to_rot\s*\(|\.inv\s*\(\s*\)|\.T\s*@")


def _frame_ops(src: str) -> tuple[int, int]:
    """(quaternion reorders, frame rotations) present in a program."""
    txt = _MARK.sub("", src)
    return len(_QUAT_PERM.findall(txt)), len(_FRAME_ROT.findall(txt))


def _lines(src: str) -> list[str]:
    return [l.rstrip() for l in _MARK.sub("", src).splitlines() if l.strip()]


def _code_lines(src: str) -> list[str]:
    return [l for l in _lines(src) if not l.strip().startswith("#")]


def _parses(src: str) -> bool:
    try:
        ast.parse(_MARK.sub("", src))
        return True
    except SyntaxError:
        return False


def _defined_funcs(src: str) -> set[str]:
    try:
        t = ast.parse(_MARK.sub("", src))
    except SyntaxError:
        return set()
    return {n.name for n in ast.walk(t)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _called_names(src: str) -> set[str]:
    try:
        t = ast.parse(_MARK.sub("", src))
    except SyntaxError:
        return set()
    return {n.func.id for n in ast.walk(t)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def _rebound(src: str) -> set[str]:
    try:
        t = ast.parse(_MARK.sub("", src))
    except SyntaxError:
        return set()
    hit = set()
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in PROVIDED:
            hit.add(n.name)
        elif isinstance(n, ast.Assign):
            for tgt in n.targets:
                if isinstance(tgt, ast.Name) and tgt.id in PROVIDED:
                    hit.add(tgt.id)
    return hit


def _loop_depth(src: str) -> int:
    """Max nesting of for/while. A coverage task needs 2; most programs write 0."""
    try:
        t = ast.parse(_MARK.sub("", src))
    except SyntaxError:
        return -1
    best = 0

    def walk(node, d):
        nonlocal best
        for ch in ast.iter_child_nodes(node):
            nd = d + 1 if isinstance(ch, (ast.For, ast.While)) else d
            best = max(best, nd)
            walk(ch, nd)

    walk(t, 0)
    return best


def _assign_rhs_roots(line: str) -> tuple[str, str] | None:
    """(lhs, first identifier on the rhs) for a simple assignment line."""
    m = re.match(r"\s*([A-Za-z_]\w*)\s*=\s*(.+)", line)
    if not m:
        return None
    rhs = re.search(r"[A-Za-z_]\w*", m.group(2))
    return (m.group(1), rhs.group(0)) if rhs else None


def _unpack_arity(line: str) -> int | None:
    """How many names a get_object_pose result is unpacked into, if any.

    Comma counting was catching every edit near the call -- adding
    return_bbox_extent=True changes the comma count without touching arity.
    """
    m = re.search(r"^\s*([\w,\s]+?)\s*=\s*get_object_pose\(", line)
    if not m:
        return None
    return len([p for p in m.group(1).split(",") if p.strip()])


def _z_values(src: str) -> list[str]:
    return re.findall(r"z_approach\s*=\s*([0-9.]+)", src)


@dataclass(frozen=True)
class Defect:
    tag: str
    family: str
    evidence: str


def classify_edits(original: str, repaired: str) -> list[Defect]:
    """Every defect the verified repair had to fix, one tag per distinct cause."""
    out: list[Defect] = []
    seen: set[str] = set()

    def add(tag: str, evidence: str) -> None:
        if tag not in seen:
            seen.add(tag)
            out.append(Defect(tag, FAMILY[tag], evidence[:150]))

    if not _parses(original):
        add("syntax", "original does not parse")

    # --- STRUCTURE ---------------------------------------------------------
    reb_o, reb_r = _rebound(original), _rebound(repaired)
    gone = reb_o - reb_r
    if gone:
        # A stub DEFINITION of an API is a mock; an assignment over it is a rebind.
        defs_o = _defined_funcs(original)
        tag = "mock_shadowing" if gone & defs_o else "api_rebound"
        add(tag, f"rebinds provided API: {', '.join(sorted(gone))}")

    defs_o = _defined_funcs(original)
    if defs_o and not (defs_o & _called_names(original)) and (defs_o & _called_names(repaired)):
        add("never_invoked", f"defines {', '.join(sorted(defs_o))} and never calls it")

    o_txt, r_txt = "\n".join(_lines(original)), "\n".join(_lines(repaired))
    if o_txt.count("close_gripper") and r_txt.count("close_gripper"):
        oi = [i for i, l in enumerate(_code_lines(original)) if "gripper" in l]
        ri = [i for i, l in enumerate(_code_lines(repaired)) if "gripper" in l]
        oseq = [("open" if "open_" in _code_lines(original)[i] else "close") for i in oi]
        rseq = [("open" if "open_" in _code_lines(repaired)[i] else "close") for i in ri]
        if oseq and rseq and oseq != rseq and sorted(oseq) == sorted(rseq):
            add("gripper_order", f"gripper sequence {oseq} -> {rseq}")

    # --- INTERFACE ---------------------------------------------------------
    if re.search(r'["\'](red|green|brown)_\w+["\']', original) and \
            not re.search(r'["\'](red|green|brown)_\w+["\']', repaired):
        add("object_name", "underscored object string, e.g. 'red_cube'")

    sm = difflib.SequenceMatcher(a=_code_lines(original), b=_code_lines(repaired))
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        was = " ".join(_code_lines(original)[i1:i2])
        now = " ".join(_code_lines(repaired)[j1:j2])
        if op == "replace":
            # An assignment whose left side survives but whose right side now
            # starts from a different variable: the program was reading the
            # wrong thing. qwen3.5-4b lifted to `position` (the cube's ORIGINAL
            # pose) where it needed the grasp pose.
            for ow, nw in zip(_code_lines(original)[i1:i2], _code_lines(repaired)[j1:j2]):
                ao, an = _assign_rhs_roots(ow), _assign_rhs_roots(nw)
                if ao and an and ao[0] == an[0] and ao[1] != an[1]:
                    add("wrong_target", f"{ao[0]} read from {ao[1]} -> {an[1]}")
            if re.search(r"position\s*=", was) and "goto_pose" in was and was.count("position") > 1:
                add("call_signature", "duplicate position= alongside a positional arg")
            elif _unpack_arity(was) and _unpack_arity(now) and \
                    _unpack_arity(was) != _unpack_arity(now):
                add("call_signature",
                    f"unpack arity on get_object_pose "
                    f"{_unpack_arity(was)} -> {_unpack_arity(now)}")
            zo, zn = _z_values(was), _z_values(now)
            if zo != zn and (zo or zn):
                add("approach_param", f"z_approach {zo or 'absent'} -> {zn or 'absent'}")
            if re.search(r"\bred_\w*pose|\bred_position", was) and \
                    re.search(r"\bgreen_\w*pose|\bgreen_position", now):
                add("wrong_target", "placement used the red cube's coordinates")
            if re.search(r"[+\-*/]", was) and re.search(r"[+\-*/]", now) and "place" in (was + now):
                add("wrong_formula", "placement height arithmetic")
        if op == "insert":
            if "get_object_pose" in now and "green" in now:
                add("stale_read", "target pose re-read after the lift")
            elif re.search(r"goto_pose|close_gripper|open_gripper", now):
                add("missing_action", f"repair adds: {now.strip()[:60]}")
        if op == "delete":
            if "goto_pose" in was:
                idx = i1 / max(len(_code_lines(original)), 1)
                if idx > 0.75:
                    add("post_release_retreat", "trailing move after releasing")
                else:
                    add("spurious_move", "move to the object's own pose before acting")

    # names used before binding
    try:
        import builtins
        t = ast.parse(_MARK.sub("", original))
        assigned, used = set(), set()
        for n in ast.walk(t):
            if isinstance(n, ast.Name):
                (assigned if isinstance(n.ctx, ast.Store) else used).add(n.id)
            # A def binds its own name AND its parameters; a comprehension or
            # `for` binds its target. Counting those as unbound produced a
            # false positive on every program that defines a function.
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assigned.add(n.name)
                a = n.args
                for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs):
                    assigned.add(arg.arg)
                if a.vararg:
                    assigned.add(a.vararg.arg)
                if a.kwarg:
                    assigned.add(a.kwarg.arg)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                assigned.add(n.name)
            # An import binds a name. Without this the walk reported every
            # scipy import as unbound -- `from scipy.spatial.transform import
            # Rotation as R` made `R` a defect -- and the only thing hiding it
            # was the hand-written np/numpy exclusion below, which happened to
            # cover the one import most programs write.
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for al in n.names:
                    assigned.add((al.asname or al.name).split(".")[0])
        unbound = (used - assigned - PROVIDED - PREIMPORTED
                   - set(dir(builtins)) - {"__name__"})
        if unbound:
            add("unbound_name", f"used before binding: {', '.join(sorted(unbound))}")
    except SyntaxError:
        pass

    # The program did the geometry in the wrong frame.
    #
    # THE RULE HAS A DEGREE OF FREEDOM AND IT MATTERS. "Attempted a transform"
    # is read NARROWLY here: the original must contain rotation machinery
    # (from_quat, as_matrix, .apply, a transpose-multiply). Read widely -- the
    # original computes any bare difference between two position variables that
    # the repair then rotates -- it fires on 8 programs instead of 4, spanning
    # 2B to 32B, and the "only >=8B models attempt this" reading disappears.
    #
    # The narrow rule was kept because it is the literal one, NOT because of
    # the cleaner result, and the cleaner result is therefore not evidence for
    # itself. Anyone citing FRAME against scale must say which rule they used.
    #
    # The guard is `qo or ro`: the ORIGINAL has to have attempted a transform.
    # Without it this fires on 21 of 21 nut_assembly programs and 0 of 49
    # everywhere else -- because every nut_assembly repair adds the same
    # insertion recipe, so the tag would be identifying the task rather than
    # the defect. Seventeen of those programs never attempt the transform at
    # all; for them the missing insertion is already `missing_action`. With the
    # guard it fires on 4 programs, every one of which wrote frame algebra and
    # got it wrong.
    qo, ro = _frame_ops(original)
    qr, rr = _frame_ops(repaired)
    if (qo or ro) and (qr > qo or rr > ro):
        add("frame_convention",
            f"frame algebra attempted and corrected: quaternion reorders "
            f"{qo} -> {qr}, frame rotations {ro} -> {rr}")

    # The repair had to introduce iteration the model never wrote: an area
    # coverage task done as a handful of waypoints.
    do, dr = _loop_depth(original), _loop_depth(repaired)
    if dr > do:
        add("missing_coverage",
            f"repair raises loop nesting {do} -> {dr} (area coverage)")

    if re.search(r"place_position|place_pos", original) and \
            "cube_lifting" not in original and \
            not re.search(r"green", original) and re.search(r"0\.5", original):
        add("wrong_task", "solves pick-and-place on a lifting task")

    return out
