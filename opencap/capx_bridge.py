"""Mapping from our tier grid onto CaP-X's actual config files and CLI.

Verified against capgym/cap-x @ 53e9966 on 2026-09-04 by a full listing of
env_configs/ (an earlier pass enumerated only guessed directory names and
wrongly concluded there were 5 task families -- there are 7, matching the
paper's "across 7 tasks and 12 models").

Repo-specific facts the paper does not give you:

  * Task slugs are `cube_lifting` (not cube_lift) and `nut_assembly`
    (the paper's "Peg Insertion").
  * Config naming is not uniform. `two_arm_handover` drops the
    `franka_robosuite_` prefix, and it ships **no M4 config**.
  * The two-arm tasks use `_reduced` for the single-turn low-level tiers
    (S3/S4) but keep `_reduced_api` in the M4 name.
  * `human_oracle_code/robosuite/` ships oracle programs for all 7 tasks in
    bare and privileged variants -- the ground truth for Gate 0.

Rather than infer names with clever rules, the table is explicit and
`scripts/verify_configs.py` checks every entry against a real checkout.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config as cfg

#: The 7 Robosuite task families in env_configs/, verified by full listing.
CORE_TASKS = ("cube_lifting", "cube_stack", "cube_restack", "spill_wipe",
              "nut_assembly", "two_arm_lift", "two_arm_handover")

#: Tier -> config filename suffix. `""` is the bare (S2) config.
_SUFFIX = {
    "S1": "_privileged",
    "S2": "",
    "S3": "_reduced_api",
    "S4": "_reduced_api_exampleless",
    "M1": "_multiturn",
    "M2": "_multiturn_vf",             # visual feedback = raw RGB, needs a VLM
    "M3": "_multiturn_vdm",            # visual differencing module -> text
    "M4": "_multiturn_vdm_reduced_api",
}

#: Per-task deviations from the default naming.
_PREFIX = {t: "franka_robosuite_" for t in CORE_TASKS}
_PREFIX["two_arm_handover"] = ""

#: Oracle (human expert) programs, used to calibrate the harness in Gate 0.
_ORACLE_PREFIX = {t: "franka_robosuite_" for t in CORE_TASKS}
_ORACLE_PREFIX["two_arm_handover"] = ""


def oracle_config_path(task: str, privileged: bool = True) -> str:
    """Human-written reference program for `task`.

    Deterministic and LLM-free, so it isolates sim + scoring from the agent --
    which makes it a far better Gate 0 anchor than any published scalar.
    """
    if task not in CORE_TASKS:
        raise ValueError(f"unknown task {task!r}")
    suffix = "_privileged_oracle" if privileged else "_oracle"
    return (f"env_configs/human_oracle_code/robosuite/"
            f"{_ORACLE_PREFIX[task]}{task}{suffix}.yaml")

#: two_arm_lift and two_arm_handover use `_reduced` for the SINGLE-TURN low-level
#: configs (S3/S4) but keep `_reduced_api` for M4. Caught by verify_configs.py --
#: the obvious "shorten everywhere" rule produces a path that does not exist.
_SHORT_REDUCED = {"two_arm_lift", "two_arm_handover"}
_SHORT_REDUCED_TIERS = {"S3", "S4"}

#: (task, tier) pairs the repo does not ship.
MISSING: frozenset[tuple[str, str]] = frozenset({
    ("two_arm_handover", "M4"),  # only a _multiturn_vdm_legacy variant exists
})

#: (task, tier) pairs the repo ships but CaP-X deliberately did not evaluate.
#: Fig. 17: "for Re-Stack, with a simple task prompt, without visual input, we
#: do not observe model writing code to first check the initial condition ...
#: Therefore, we only evaluated all models for the multi-turn setup."
#: Scoring these would read as model failure when it is a task-design artifact.
EXCLUDED_BY_DESIGN: frozenset[tuple[str, str]] = frozenset(
    {("cube_restack", t) for t in ("S1", "S2", "S3", "S4")}
)


def config_path(task: str, tier: str) -> str:
    if task not in CORE_TASKS:
        raise ValueError(f"unknown task {task!r}")
    if tier not in _SUFFIX:
        raise ValueError(f"unknown tier {tier!r}")
    if (task, tier) in MISSING:
        raise KeyError(f"cap-x ships no {tier} config for {task}")

    suffix = _SUFFIX[tier]
    if task in _SHORT_REDUCED and tier in _SHORT_REDUCED_TIERS:
        suffix = suffix.replace("_reduced_api", "_reduced")
    return f"env_configs/{task}/{_PREFIX[task]}{task}{suffix}.yaml"


def has_config(task: str, tier: str) -> bool:
    """Runnable in our grid: the config exists AND CaP-X evaluated that pairing."""
    return (task, tier) not in MISSING and (task, tier) not in EXCLUDED_BY_DESIGN


@dataclass(frozen=True)
class LaunchPlan:
    """A fully-specified CaP-X invocation for one cell.

    Every field that CaP-X would otherwise default is pinned here, because
    several of those defaults would quietly break our controls.
    """

    task: str
    tier: str
    model_repo: str
    server_url: str
    vdm_model_repo: str
    vdm_server_url: str
    output_dir: str
    trials: int = cfg.STATES_PER_TASK
    num_workers: int = 12

    def argv(self) -> list[str]:
        a = [
            "--config-path", config_path(self.task, self.tier),
            "--server-url", self.server_url,
            "--model", self.model_repo,
            # CaP-X defaults to temperature 1.0; we need greedy decoding.
            "--temperature", str(cfg.TEMPERATURE),
            # CaP-X defaults to 2048*10 = 20480 per response.
            "--max-tokens", str(cfg.MAX_NEW_TOKENS),
            # Fixed across the whole grid (reasoning-mode parity control).
            "--reasoning-effort", cfg.REASONING_EFFORT,
            "--total-trials", str(self.trials),
            "--num-workers", str(self.num_workers),
            "--output-dir", self.output_dir,
            # tyro parses booleans as True/False -- lowercase is rejected.
            # Sec. 08: no raw frames. The VDM's text is what the agent consumed.
            "--record-video", "False",
            # CaP-Agent0's test-time compute is a *separate* scaffold axis and
            # must stay off in the baseline grid.
            "--use-parallel-ensemble", "False",
            "--use-multimodel", "False",
        ]
        if self.tier in ("M3", "M4"):
            # Pinning the VDM is essential: CaP-X defaults it to a frontier
            # closed model, which would make M3/M4 measure "small agent plus
            # huge perception module" rather than the scaffold itself.
            a += ["--visual-differencing-model", self.vdm_model_repo,
                  "--visual-differencing-model-server-url", self.vdm_server_url]
        return a
