"""Optionally invoke a defined-but-never-called entry point.

WHY THIS EXISTS
---------------
At the low-level tiers (S3/S4) the prompt reads like a coding task, and models
respond in kind: qwen3-4b wrapped a complete, competent pick-and-stack program
-- IK solve, gripper control, placement pose -- inside

    def pick_and_stack():
        ...

and never called it. On all 30 S4 cube_stack episodes. The module binds a name
and exits, so the robot never moves: reward 0.000, NO exception, nothing in the
score or the stderr to distinguish it from "the model tried and failed".

That matters for the headline number. A(m) = SR(S1) - SR(S4) is meant to
measure the abstraction cliff -- what the scaffold is worth. If a chunk of the
S4 floor is really "wrote a function, forgot to call it", then A(m) is partly
measuring OUR prompt, not the model's grasp of low-level control.

Gated so the SAME build runs both conditions:

    OPENCAP_CALL_ENTRYPOINT=1   -> after exec, call a zero-arg function if the
                                 module defined one and nothing else ran
    unset                     -> stock CaP-X behaviour

Deliberately conservative. It fires ONLY when:
  * the code parsed and executed without raising, and
  * exactly one zero-argument function was newly defined, and
  * that function is never called anywhere in the source.

One candidate only: with two, picking one is a guess, and a guess inside the
executor would be indistinguishable from a capability difference in the
results. Ambiguous cases stay unrun and keep scoring 0, which biases the
measured effect DOWN -- the safe direction.

Idempotent. Applied from bootstrap_colab.sh.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

MARKER = "# opencap: optional entry-point invocation"

ANCHOR = "                exec(code, self._exec_globals, self._exec_globals)"

INJECT = '''
                # opencap: optional entry-point invocation -- see
                # scripts/patch_call_entrypoint.py. Off by default so stock
                # behaviour is preserved; the A/B difference measures how much
                # of the S4 floor is "forgot to call it" rather than an
                # inability to do low-level control.
                import os as _os
                if _os.environ.get("OPENCAP_CALL_ENTRYPOINT"):
                    try:
                        import ast as _ast
                        _tree = _ast.parse(code)
                        _defs = [n for n in _tree.body
                                 if isinstance(n, _ast.FunctionDef)
                                 and not n.args.args
                                 and not n.args.posonlyargs
                                 and not n.args.kwonlyargs]
                        _called = {n.func.id for n in _ast.walk(_tree)
                                   if isinstance(n, _ast.Call)
                                   and isinstance(n.func, _ast.Name)}
                        _runnable = [n for n in _tree.body
                                     if not isinstance(n, (_ast.FunctionDef,
                                                           _ast.AsyncFunctionDef,
                                                           _ast.Import,
                                                           _ast.ImportFrom,
                                                           _ast.ClassDef))]
                        _cands = [d for d in _defs if d.name not in _called]
                        if len(_cands) == 1 and not _runnable:
                            _fn = self._exec_globals.get(_cands[0].name)
                            if callable(_fn):
                                print("[opencap] invoking never-called entry "
                                      "point: " + _cands[0].name)
                                _fn()
                    except BaseException:
                        traceback.print_exc(file=tee_err)
'''


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/cap-x")
    target = root / "capx/envs/tasks/base.py"
    if not target.exists():
        print(f"  MISSING {target}")
        return 1
    text = target.read_text()
    if MARKER in text:
        print("  already patched   capx/envs/tasks/base.py")
        return 0
    if ANCHOR not in text:
        print("  ANCHOR NOT FOUND -- inspect the exec() call site manually")
        return 1
    # Insert immediately after the exec, still inside the redirect context and
    # the try, so a failure in the invoked function is captured exactly like a
    # failure in top-level code.
    idx = text.index(ANCHOR) + len(ANCHOR)
    target.write_text(text[:idx] + "\n" + INJECT.rstrip("\n") + text[idx:])
    print("  patched           capx/envs/tasks/base.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
