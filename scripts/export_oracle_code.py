"""Export CaP-X's human oracle program for a task into results/references.json.

    python3 scripts/export_oracle_code.py nut_assembly [more tasks...] [--capx DIR]
                                         [--store hf:savoji/capx-oracle-seeded]

WHY THIS EXISTS
---------------
The distance-to-solution metric is defined as the edit distance between a
model's program and a MINIMAL VERIFIED REPAIR, and a repair is written against
the task's ground truth. references.json carries that ground truth for
cube_lifting, cube_stack and spill_wipe -- and not for nut_assembly, because
the task was dropped from the panel before anyone needed it and came back after.

That gap is why 21 of the 70 failing programs in the seeded corpus have no
repair. Their defects are diagnosable from the code alone -- a hallucinated
`get_fixed_rigid_transform`, a grasp on the wrong object, a program that
computes a pose and never moves -- but fixing those moves a program from 0.000
to roughly the 0.35 reward plateau where most of the panel already sits. The
plateau is the insertion, and nobody can write the insertion without knowing
what correct looks like.

So this runs first, it needs no GPU, and it costs seconds.

WHERE THE CODE LIVES
--------------------
Not in the oracle YAML. `use_oracle_code: true` there means "run
`env.oracle_code`", and that attribute is a module-level string constant on the
env class the YAML's `env._target_` names (ORACLE_CODE in
capx/envs/tasks/franka/franka_nut_assembly.py). The class is read with `ast`,
never imported: cap-x's dependencies live in its own venv, and the harness
python must not grow them. `env.cfg.oracle_code` in the YAML, if present,
overrides the class attribute exactly as CodeExecutionEnvBase does.

An oracle episode in the store has no turns -- there is no LLM to record -- so
the store contributes the SCORE (successes/n) and the checkout contributes the
CODE. references.json entries are `{model, tier, successes, n, code}`, the shape
export_references.py writes for the model-derived references; this writes the
same shape with model="oracle".
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opencap.capx_bridge import CORE_TASKS, oracle_config_path  # noqa: E402
from opencap.repair import strip_markers  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REFS = ROOT / "results" / "references.json"
DEFAULT_STORE = "hf:savoji/capx-oracle-seeded"


def _class_oracle_code(module_path: Path, cls: str) -> str:
    """`oracle_code` of class `cls` in `module_path`, resolved without importing."""
    tree = ast.parse(module_path.read_text(), filename=str(module_path))
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    consts[t.id] = node.value.value
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == cls):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "oracle_code"
                       for t in stmt.targets):
                continue
            v = stmt.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                return v.value
            if isinstance(v, ast.Name) and v.id in consts:
                return consts[v.id]
            raise KeyError(f"{cls}.oracle_code in {module_path.name} is not a "
                           f"string constant this can resolve")
        raise KeyError(f"{cls} in {module_path.name} sets no oracle_code")
    raise KeyError(f"class {cls} not found in {module_path}")


def extract(capx: Path, task: str) -> str:
    """The oracle program the task's privileged oracle YAML runs."""
    import yaml
    rel = oracle_config_path(task, privileged=True)
    path = capx / rel
    if not path.exists():
        raise FileNotFoundError(f"{rel} not found under {capx}")
    cfg = yaml.safe_load(path.read_text())
    env = cfg.get("env") or {}
    # A YAML-level override wins, exactly as it does in CodeExecutionEnvBase.
    code = (env.get("cfg") or {}).get("oracle_code")
    if isinstance(code, str) and code.strip():
        return strip_markers(code)
    target = env.get("_target_")
    if not target:
        raise KeyError(f"{rel} names no env._target_")
    mod, cls = target.rsplit(".", 1)
    module_path = capx / (mod.replace(".", "/") + ".py")
    if not module_path.exists():
        raise FileNotFoundError(f"{target}: {module_path} not found")
    code = _class_oracle_code(module_path, cls)
    if not code.strip():
        # Fail loudly rather than write an empty string that would silently
        # become a zero-distance reference.
        raise KeyError(f"{target}.oracle_code is empty")
    return strip_markers(code)


def oracle_score(store_spec: str, task: str) -> tuple[str, int, int] | None:
    """(tier, successes, n) of the oracle cell for `task`, or None if absent."""
    from opencap.store import open_store
    s = open_store(store_spec)
    for cell in s.completed():
        if cell.task == task and cell.model == "oracle":
            eps = list(s.read_cell(cell))
            return cell.tier, sum(1 for e in eps if e.get("success")), len(eps)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", default=["nut_assembly"])
    ap.add_argument("--capx", default=None,
                    help="cap-x checkout (default $CAPX_DIR)")
    ap.add_argument("--store", default=DEFAULT_STORE,
                    help="oracle store the score is read from")
    a = ap.parse_args()
    capx = Path(a.capx or os.environ.get("CAPX_DIR", "/content/cap-x"))
    refs = json.loads(REFS.read_text()) if REFS.exists() else {}

    rc = 0
    for task in (a.tasks or ["nut_assembly"]):
        if task not in CORE_TASKS:
            print(f"[refs] unknown task {task!r}"); rc = 1; continue
        try:
            code = extract(capx, task)
        except Exception as e:
            print(f"[refs] {task}: FAILED -- {type(e).__name__}: {e}"); rc = 1; continue
        try:
            score = oracle_score(a.store, task)
        except Exception as e:
            print(f"[refs] {task}: cannot read {a.store} -- {type(e).__name__}: {e}")
            score = None
        if score is None:
            # A reference without its score is a reference nobody can trust
            # (spill_wipe's model reference scores 29/30, and that matters).
            print(f"[refs] {task}: no oracle cell in {a.store}; run stage 2 first")
            rc = 1; continue
        tier, ok, n = score
        old = refs.get(task)
        if old and old.get("code") != code:
            print(f"[refs] {task}: reference CHANGED ({old.get('model')} {old.get('tier')} "
                  f"{old.get('successes')}/{old.get('n')} -> oracle); any distance "
                  f"already computed against the old one is stale")
        refs[task] = {"model": "oracle", "tier": tier, "successes": ok, "n": n,
                      "code": code}
        print(f"[refs] {task}: {len(code.splitlines())} lines, oracle {tier} {ok}/{n}")
    REFS.write_text(json.dumps(refs, indent=2))
    print(f"[refs] wrote {REFS.relative_to(ROOT)} -- tasks: {sorted(refs)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
