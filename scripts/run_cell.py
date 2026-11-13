"""Run one experiment cell end to end and commit it, atomically.

    python3 scripts/run_cell.py --model qwen3-1.7b --tier S1 --task cube_stack \
        --capx /content/cap-x --store local:/content/corpus

Order of operations matters. The cell is only published after
`results.cross_check()` confirms our per-trial parse agrees with CaP-X's own
summary -- a wrong cell is worse than a missing one, because `completed()`
would count it as done and never re-run it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opencap import config as cfg
from opencap.capx_bridge import LaunchPlan
from opencap.cells import Cell
from opencap.models import BY_KEY
from opencap.results import (MAX_TIMEOUT_FRACTION, classify_failure,
                           cross_check, is_complete, parse_trials,
                           resolved_output_dir, timeout_fraction)
from opencap.schema import Episode
from opencap.store import open_store

# Long jobs are launched detached with stdout redirected to a file, where
# Python block-buffers by default -- the log stays empty until exit, so a
# multi-hour run shows no progress and a crash shows no context. Line
# buffering costs nothing and makes the log readable while it runs.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass



def _read_stats(path: str | None, since: float) -> dict:
    """Aggregate the shim's per-request stats for this cell.

    Requests are not individually attributable to trials -- CaP-X does not
    surface the response id -- so we summarise the window instead. That is
    enough for H4, which asks whether confidence tracks success at the *cell*
    level, and it costs nothing to collect.
    """
    if not path or not Path(path).exists():
        return {}
    lps, empties, n = [], 0, 0
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("t", 0) < since:
                continue
            n += 1
            if r.get("logprob_mean") is not None:
                lps.append(r["logprob_mean"])
            empties += bool(r.get("empty_content"))
    if not n:
        return {}
    return {
        "n_requests": n,
        "logprob_mean": sum(lps) / len(lps) if lps else None,
        "logprob_min": min(lps) if lps else None,
        "empty_content_rate": empties / n,
    }


#: port -> the service's own log basename. The status code for an input error
#: exists only in the SERVICE's log: the SAM3 client's post_with_retries
#: swallows it and reports "failed after N retries".
SERVICE_LOGS = {"8114": "sam3", "8115": "graspnet", "8116": "pyroki",
                "8122": "molmo"}


def _log_dir() -> Path:
    return Path(os.environ.get("OPENCAP_LOGDIR", "/data/capx-logs"))


def _service_log_marks() -> dict[str, int]:
    """Current size of every perception log, keyed by port.

    Taken before the cell runs. A missing log marks 0, which makes the later
    read fall back to the whole file -- conservative in the right direction,
    since more context can only make the guard refuse more often.
    """
    marks: dict[str, int] = {}
    for port, name in SERVICE_LOGS.items():
        f = _log_dir() / f"{name}_server.log"
        try:
            marks[port] = f.stat().st_size
        except OSError:
            marks[port] = 0
    return marks


def _log_since(port: str, marks: dict[str, int]) -> str:
    """Only what this cell appended to that service's log.

    If the file shrank it was rotated or truncated mid-cell, and the offset is
    meaningless; returning "" then makes the caller refuse rather than classify
    from a window it cannot trust.
    """
    name = SERVICE_LOGS.get(port)
    if not name:
        return ""
    f = _log_dir() / f"{name}_server.log"
    start = marks.get(port, 0)
    try:
        size = f.stat().st_size
        if size < start:
            return ""
        with f.open("rb") as fh:
            fh.seek(start)
            return fh.read().decode("utf-8", errors="ignore")
    except OSError:
        return ""


# Cached: the episode records both the turns and their COUNT, and both come
# from this read, so it is called twice per trial.
@lru_cache(maxsize=4096)
def _read_turns(run_dir: Path, trial: int) -> list[dict]:
    """Pull the generated code and stderr for one trial into the trace.

    The shards were storing `turns: []` -- scalars only, no model output. That
    makes the "forensic source of truth" unable to answer the one question
    forensics exist for: what did the model actually write? Output directories
    live on ephemeral pod storage, so once a runtime is reaped the evidence is
    gone unless it was captured here.
    """
    import glob
    out = []
    for d in sorted(glob.glob(str(run_dir / f"trial_{trial:02d}_*"))):
        p = Path(d)
        rec: dict = {"dir": p.name}
        code = p / "code.py"
        if code.exists():
            rec["code"] = code.read_text()[:20000]
        summ = p / "summary.txt"
        if summ.exists():
            txt = summ.read_text()
            # keep the tail, which holds stderr and the reward line
            rec["summary_tail"] = txt[-4000:]
        out.append(rec)
    return out


def _host_kind() -> str:
    """Derive the host from the GPU rather than trusting an env var.

    An explicit OPENCAP_HOST_KIND is honoured, but the default is inferred:
    a hard-coded value in a wrapper script once stamped A100 cells as
    colab-l4, which would have silently corrupted the host-effect control.
    """
    explicit = os.environ.get("OPENCAP_HOST_KIND", "").strip()
    if explicit:
        return explicit
    name = _gpu_name().upper()
    if "H200" in name:
        return "h200"
    if "H100" in name:
        return "h100"
    if "A100" in name:
        return "colab-a100"
    if "L4" in name:
        return "colab-l4"
    if "T4" in name:
        return "colab-t4"
    return name.lower().replace(" ", "-") or "unknown"


def _gpu_name() -> str:
    """Name of the card this run is pinned to.

    nvidia-smi reads NVML, which ignores CUDA_VISIBLE_DEVICES and always
    enumerates from physical index 0, so an unqualified query returns whatever
    card happens to be first on the box. That fed `host_kind`, which is the
    label the entire host-effect control is grouped by -- a field that must not
    be right only by accident. `OPENCAP_GPU` names our card explicitly.
    """
    cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    gpu = os.environ.get("OPENCAP_GPU", "").strip()
    if gpu:
        cmd[1:1] = ["-i", gpu]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else ""
    except Exception:                                  # noqa: BLE001
        return ""


def _harness_commit() -> str:
    """Git revision of this harness, recorded onto every episode.

    Was a dead field in the Colab corpus (empty on all 1080 episodes), so no
    trace there can say which version of the scripts produced it.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:                                  # noqa: BLE001
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--capx",
                    default=os.environ.get("CAPX_DIR", "/content/cap-x"))
    ap.add_argument("--store", required=True)
    ap.add_argument("--server-url", default=None)
    ap.add_argument("--trials", type=int, default=cfg.STATES_PER_TASK)
    ap.add_argument("--num-workers", type=int, default=12)

    # Output dirs are keyed by model/tier/task only, so two runs of the same
    # cell under DIFFERENT conditions collide -- the A/B arms wrote 30 trial
    # dirs each into one directory and the parse saw a mixture. Arms must set
    # OPENCAP_SCRATCH to separate paths.
    ap.add_argument("--scratch",
                    default=os.environ.get("OPENCAP_SCRATCH", "/content/outputs"))
    args = ap.parse_args()

    spec = BY_KEY[args.model]
    cell = Cell(args.model, args.tier, args.task)
    store = open_store(args.store)

    if cell in store.completed():
        print(f"[skip] {cell} already in corpus")
        return 0

    # KNOWN-UNCOLLECTABLE CELLS. Checked HERE, before the stale-dir cleanup
    # below, because that cleanup deletes the very trial summaries the skip
    # list is regenerated from -- check after it and the list erases its own
    # evidence the first time it is used.
    #
    # These cells call point_prompt_molmo, which is bound to 127.0.0.1:8122;
    # nothing declares that port and capx/serving ships no launcher for it.
    # At temperature 0 the program is identical every attempt, so the refusal
    # is identical too: 17 of 17 reproduced on the 2026-09-30 resume, ~20 min
    # each. Skipping changes NOTHING about the corpus -- run_cell.py already
    # refuses them -- only whether a resume spends three hours re-deriving it.
    if not os.environ.get("OPENCAP_IGNORE_MOLMO_SKIP"):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from molmo_skiplist import load as _molmo_skips
            _skips = _molmo_skips()
        except Exception:            # a missing/malformed list must not block a run
            _skips = set()
        if (args.model, args.tier, args.task) in _skips:
            print(f"[skip] {cell} needs :8122 (molmo), which has no backend in "
                  f"this checkout -- refused identically on every previous "
                  f"attempt. Set OPENCAP_IGNORE_MOLMO_SKIP=1 to try anyway, or "
                  f"clear the list once the port is resolved.")
            return 0

    endpoints = cfg.Endpoints.from_env()
    run_name = f"{args.model}__{args.tier}__{args.task}"
    requested = Path(args.scratch) / run_name

    # A cell must start from an EMPTY output directory, for the reason
    # verify_repairs.run_one already guards: CaP-X names trial dirs by outcome
    # (trial_NN_sandboxrc_X_reward_Y_taskcompleted_Z), so a re-run into a
    # populated dir overwrites same-outcome trials and adds differing ones, and
    # the parse reads a mixture of two runs.
    #
    # The verifier had this guard and the sweep did not. It bit twice in Phase
    # 7: an interrupted sampled cell left four trial dirs behind, and the
    # qwen3.5-27B draw refused on throughput left thirty -- WITH an
    # aaa_done_flag, so is_complete() returned true on the wreckage and only a
    # hand `rm -rf` kept the re-run clean. A resume is exactly when this
    # happens, and a resume is exactly when nobody is watching.
    import shutil
    for stale in (requested, resolved_output_dir(requested, spec.repo)):
        if stale.exists():
            print(f"[clean] removing stale output dir {stale}")
            shutil.rmtree(stale)

    plan = LaunchPlan(
        task=args.task, tier=args.tier,
        model_repo=spec.repo,
        server_url=args.server_url or endpoints.agent_llm,
        vdm_model_repo=cfg.VDM_MODEL, vdm_server_url=endpoints.vdm,
        output_dir=str(requested),
        trials=args.trials, num_workers=args.num_workers,
    )

    stats_path = os.environ.get("OPENCAP_STATS_FILE")
    stats_since = time.time()
    env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    # --no-sync: `uv run` re-resolves cap-x's lockfile on every invocation,
    # which reaches the network. On 2026-09-11 a GitHub 500 on the flash-attn
    # wheel killed a cell that way -- an upstream outage recorded as a failed
    # experiment. The environment is already synced by bootstrap; re-resolving
    # per cell buys nothing and makes every cell depend on github.com being up.
    cmd = ["uv", "run", "--no-sync", "capx/envs/launch.py", *plan.argv()]
    print("[run]", " ".join(cmd))
    # Snapshot each perception log's SIZE before the cell runs, so the
    # classification below can read only what this cell appended.
    #
    # Why by byte offset rather than by timestamp: the guard it feeds used to
    # read a fixed 200 KB tail, and both logs carry thousands of older lines --
    # sam3_server.log alone holds 150 500s and graspnet 90 'Grasp planning
    # failed'. Any rule over that tail answers about some earlier request. The
    # offset needs no timestamp parsing and no agreement about log format,
    # which is what makes it safe to rely on.
    log_marks = _service_log_marks()
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=args.capx, env=env,
                          capture_output=True, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-3000:], proc.stderr[-3000:], sep="\n")
        print(f"[fail] launch exited {proc.returncode}")
        return proc.returncode

    # CaP-X rewrites the path: /x/run -> /x/<model_slug>/run
    run_dir = resolved_output_dir(requested, spec.repo)
    if not is_complete(run_dir):
        print(f"[fail] no done flag in {run_dir}")
        return 1

    # An API server that never came up takes every trial down and looks like a
    # model failure. Checked AFTER the run because launch.py starts the servers
    # itself -- the earlier pre-run port check refused every cell, since the
    # port is legitimately closed until launch.py opens it.
    # THIS USED TO FIRE ONLY WHEN EVERY TRIAL FAILED, which left the partial
    # case silent -- the same hole verify_repairs.run_one closed on 2026-09-10
    # and this copy never did. It matters more here than there. The S3/S4
    # reduced API is perception-backed (segment_sam3_*, detect_object_owlvit,
    # point_prompt_molmo), so whether a trial reaches a vision server depends
    # on whether the generated program happens to call one. A cell where ten
    # of thirty trials call it and die would commit with twenty real results
    # and ten environment failures scored as model defects -- straight into
    # the failure MIX, which is the thing the tier grid exists to measure.
    #
    # Count the affected trials directly and refuse on any of them.
    # A DEAD PORT IS NOT THE ONLY WAY TO LOSE A SERVICE. 2026-09-29:
    # qwen3-1.7b S3 cube_lifting COMMITTED with 4 of 30 trials dead on
    #
    #   RuntimeError: Failed to communicate with GraspNet service at
    #   http://127.0.0.1:8115: 500 Server Error
    #
    # and the server log gives the reason: "Grasp planning failed: CUDA out of
    # memory. Tried to allocate 1.83 GiB ... 1.68 GiB is free", with a
    # neighbour holding 77 GB of the card. The request after it returned 200
    # and 198 grasps, so the service was alive the whole time -- it just could
    # not allocate. A live service returning 500 contains neither "Connection
    # refused" nor "Max retries exceeded", so the first version of this guard
    # passed it straight through and four environment failures entered the
    # corpus as model defects. Exactly the hole this guard was written to
    # close, one error string over.
    #
    # Refuse on ANY of these, not a majority.
    #
    # SUPERSEDED 2026-10-27. This used to read "do not try to sort an OOM 500
    # from a bad-input 500 here: the client-side summary does not carry the
    # reason". The premise is right and the conclusion was wrong: the summary
    # does not carry it, but the SERVICE's log does, and five A4 draws were
    # thrown away over the difference. The loss is biased, not random -- a
    # program that sends an unusable request is a failing program, so refusing
    # those cells removes exactly the defects the corpus is for. _classify_5xx
    # asks the service, over this cell's window only, and still refuses on
    # "unknown".
    # A 4xx IS NOT A LOST SERVICE -- IT IS A MODEL DEFECT, AND THIS GUARD USED
    # TO DISCARD IT. 2026-10-01, once molmo was finally served, programs began
    # reaching the molmo -> point -> segment_sam3_point_prompt stage and SAM3
    # started answering 422 Unprocessable Entity. The cause was the generated
    # program, not the service:
    #
    #     point_prompt = point_prompt_molmo(rgb_image, "brown spill")
    #     if "point_coords" in point_prompt:        # dict is keyed by OBJECT
    #         x, y = point_prompt["point_coords"]   # NAME, so never runs
    #     ...
    #     segment_sam3_point_prompt(rgb_image, (x, y))   # x, y still None
    #
    # SAM3 was healthy throughout -- a hand-built request with real coords
    # returns 200 and three masks. Refusing that cell throws away a textbook
    # api_misuse defect, which is the opposite error from the one this guard
    # exists to prevent and just as damaging to the tag mix.
    #
    # THE TRIAL SUMMARY CANNOT TELL THEM APART. A 500 surfaces as "500 Server
    # Error", but the SAM3 client's post_with_retries swallows the status and
    # reports only "failed after N retries", so the 422 is invisible on the
    # client side. The status code exists only in the SERVICE's own log, so
    # that is where this looks.
    SERVICE_LOST = (
        "Connection refused",
        "Max retries exceeded",
        "500 Server Error",             # live service, genuinely failed
        "502 Server Error", "503 Server Error", "504 Server Error",
    )
    #: Ambiguous on the client side: could be a timeout (environment) or a
    #: rejected payload (model). Resolved against the service log below.
    AMBIGUOUS = "failed after"

    #: A 5xx the service raised while HEALTHY, about the request it was given.
    #: These strings are the service's own description of a malformed payload
    #: (2026-10-21: a program passed a box where a point belongs and SAM3
    #: answered 'Point prompt inference failed: Sizes of tensors must match').
    #: A program that sends an unusable request has a defect; refusing the cell
    #: throws the defect away, and the refused cells are precisely the failing
    #: programs, so the loss is biased rather than random.
    BAD_INPUT = ("Point prompt inference failed",
                 "Sizes of tensors must match",
                 "Grasp planning failed")
    #: A 5xx the service raised because IT was in trouble. Nothing about the
    #: model can be concluded from a trial that hit one.
    SERVICE_SICK = ("CUDA out of memory", "Tried to allocate",
                    "No such file or directory", "Address already in use")

    def _classify_5xx(port: str) -> str:
        """'model' | 'environment' | 'unknown', from the service's OWN log.

        Scoped to what THIS CELL appended (log_marks, taken before the run).
        The previous version read a fixed 200 KB tail, and both logs carry
        thousands of older lines -- 150 500s in sam3_server.log, 90 'Grasp
        planning failed' in graspnet's -- so any rule over that tail answered
        about somebody else's request. With the window right, the question is
        answerable; with the window wrong, no pattern could have fixed it.

        Three states, not two, and 'unknown' is the common one. A window with
        no 5xx in it at all means the failure is not visible from here, and
        that is NOT evidence the model was at fault.
        """
        win = _log_since(port, log_marks)
        if not win:
            return "unknown"
        # A service that was sick stays sick for the purposes of this cell,
        # whatever else is in the window. Checked FIRST, deliberately: the
        # first version of this guard let four CUDA-OOM 500s into the corpus
        # as model defects, and that error is worse than the opposite one.
        if any(k in win for k in SERVICE_SICK):
            return "environment"
        # It must also have been ANSWERING. A window that holds no 2xx is a
        # window in which the service may have been dead the whole time.
        import re as _re
        if not _re.search(r'" 2\d\d ', win):
            return "unknown"
        if any(k in win for k in BAD_INPUT):
            return "model"
        return "unknown"

    def _service_rejected_us(port: str) -> bool:
        """Did that service refuse OUR request while otherwise healthy?

        4xx means the server was alive and rejected the payload. A 5xx can
        mean the same thing -- see _classify_5xx -- or that the service itself
        failed, and the two are not distinguishable from the client side.
        """
        import re as _re
        win = _log_since(port, log_marks)
        if not win:
            return False
        if any(k in win for k in SERVICE_SICK):
            return False
        if _re.search(r'" 4\d\d ', win):
            return True
        return _classify_5xx(port) == "model"

    _t = parse_trials(run_dir)
    n_unreachable = 0
    n_rejected = 0
    why: dict[str, int] = {}
    for _d in sorted(run_dir.glob("trial_*")):
        _f = _d / "summary.txt"
        if not _f.exists():
            continue
        try:
            _txt = _f.read_text()
        except OSError:
            continue
        hit = [s for s in SERVICE_LOST if s in _txt]
        if hit:
            # A 5xx is not self-explanatory. The service raises one both when
            # the PAYLOAD was unusable (a model defect: the program passed a
            # box where a point belongs) and when IT was in trouble (an
            # environment failure: CUDA OOM with a neighbour holding 77 GB).
            # The client-side summary carries neither reason, so ask the
            # service's own log, over this cell's window only.
            #
            # Owner's decision 2026-10-27: a bad-input 5xx from a service that
            # was answering 200s in the same cell IS a model defect. Five A4
            # draws were refused on exactly that and the loss is biased --
            # the refused cells are the failing programs. 'unknown' still
            # refuses, which keeps the default conservative.
            port = (re.findall(r"127\.0\.0\.1:(\d{4})", _txt) or [""])[0]
            if (any(h.startswith(("500", "502", "503", "504")) for h in hit)
                    and port and _classify_5xx(port) == "model"):
                n_rejected += 1
                continue
            n_unreachable += 1
            why[hit[0]] = why.get(hit[0], 0) + 1
            continue
        if AMBIGUOUS in _txt and "Failed to communicate with" in _txt:
            ports = re.findall(r"127\.0\.0\.1:(\d{4})", _txt)
            if ports and _service_rejected_us(ports[0]):
                n_rejected += 1          # model's fault: keep the cell
            else:
                n_unreachable += 1
                why["timeout/no reply"] = why.get("timeout/no reply", 0) + 1

    if n_rejected and not n_unreachable:
        print(f"[ok-ish] {n_rejected} of {len(_t)} trials had a perception "
              f"service REJECT the request while it was otherwise healthy: "
              f"a 4xx, or a bad-input 5xx raised in a window where the same "
              f"service answered 2xx. That is the generated program sending a "
              f"bad payload, i.e. a "
              f"model defect, so the cell is KEPT.")
    if n_unreachable:
        detail = ", ".join(f"{k!r} x{v}" for k, v in sorted(why.items()))
        print(f"[fail] a perception service did not answer during "
              f"{n_unreachable} of {len(_t)} trials ({detail}) -- environment "
              f"failure, not a model result. A reduced-API tier needs its "
              f"vision servers up AND able to allocate; a partial loss here "
              f"would enter the corpus as a model defect.")
        return 1

    # A timed-out request yields an empty program, reward 0.0 and no code --
    # indistinguishable in the corpus from a model that answered badly. Four
    # cells committed that way on 2026-09-10 before this check existed, each
    # reading 0/30 with ~25 of 30 scenes never attempted. Refuse rather than
    # publish a number nobody can attribute; the cell simply re-runs.
    _tf, _nto, _ntr = timeout_fraction(run_dir)
    if _tf > MAX_TIMEOUT_FRACTION:
        print(f"[fail] {_nto}/{_ntr} trials timed out waiting on the engine "
              f"({_tf:.0%} > {MAX_TIMEOUT_FRACTION:.0%}) -- refusing to commit. "
              f"This is a throughput failure, not a model result: lower "
              f"--num-workers so requests do not queue past CaP-X's read "
              f"timeout, then re-run.")
        return 1

    ok, msg = cross_check(run_dir)
    if not ok:
        print(f"[fail] cross-check disagrees with CaP-X summary: {msg}")
        return 1

    # Cross-check only proves the parse matches CaP-X's own summary -- both
    # agree on a truncated run. A cell killed mid-flight once committed 5
    # episodes instead of 30, and `completed()` would then SKIP it forever,
    # leaving a short cell silently mixed into the corpus.
    n_parsed = len(parse_trials(run_dir))
    if n_parsed != args.trials:
        print(f"[fail] incomplete cell: {n_parsed}/{args.trials} trials -- "
              f"refusing to commit (was the run interrupted?)")
        return 1

    trials = parse_trials(run_dir)
    stats = _read_stats(stats_path, stats_since)
    episodes = [
        Episode(
            episode_id=f"{run_name}__{t.trial}__{uuid.uuid4().hex[:8]}",
            model=args.model, tier=args.tier, task=args.task,
            state_id=t.trial, wave=cell.wave,
            success=t.success,
            # sandbox_rc separates "could not write the code" from "wrote code
            # that ran and did the wrong thing" -- the S3/S4 cut that tells
            # api-misuse apart from spatial-reasoning failure.
            failure_label=classify_failure(t),
            reward=t.reward,
            wall_seconds=wall / max(len(trials), 1),
            weight_revision=spec.repo,
            # Serving stack and harness revision. Both are confounded with the
            # host unless written down, and both were empty in the Colab
            # corpus -- see STATUS.md, "Trace fields that are DEAD".
            vllm_version=os.environ.get("OPENCAP_VLLM_VERSION", ""),
            harness_commit=_harness_commit(),
            host_gpu=_gpu_name(),
            host_kind=_host_kind(),
            logprob_mean=stats.get("logprob_mean"),
            logprob_min=stats.get("logprob_min"),
            turns=_read_turns(run_dir, t.trial),
            # DEAD FIELD, fixed 2026-10-27. `turns_used` is declared in the
            # schema with a default of 0 and was never assigned by anything,
            # so it read 0 in all 2490 M1 episodes -- and HANDOFF told the pod
            # to "report turns_used above all: if the loop never engaged, the
            # success numbers are a re-run of S1". Taken literally that kills
            # A3 on a field nobody wrote. The loop DID engage: len(turns) is
            # >1 in 785 of those 2490 episodes. Derived from the same read, so
            # the two cannot disagree again.
            turns_used=len(_read_turns(run_dir, t.trial)),
        ).to_trace()
        for t in trials
    ]
    store.write_cell(cell, episodes, verified=ok)
    n_ok = sum(t.success for t in trials)
    lp = stats.get("logprob_mean")
    print(f"[ok] {cell}: {n_ok}/{len(trials)} = {n_ok/len(trials):.3f} "
          f"in {wall:.1f}s ({msg})"
          + (f" | logprob {lp:.3f}" if lp is not None else "")
          + (f" | empty {stats['empty_content_rate']:.0%}"
             if stats.get("empty_content_rate") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
