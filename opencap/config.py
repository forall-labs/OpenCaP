"""Frozen experimental constants and endpoint configuration.

Every value in CONTEXT is a control from the roadmap (Sec. 05). Changing one
invalidates comparability with runs already in the corpus, so they live here as
module-level constants rather than CLI flags.
"""
from __future__ import annotations

import os as _os

import os
from dataclasses import dataclass, asdict
from pathlib import Path as _Path


def _load_dotenv() -> None:
    """Put the repo-root `.env` into the environment, without overriding it.

    Why this is here rather than left to the caller: on 2026-10-27 the local
    HF token had been rotated, `huggingface_hub` fell back to a stale cached
    one, and `condition_report.py` printed "not collected (store absent)" for
    an ablation that was 70/70 complete. The token was sitting in `.env` the
    whole time. The hub answers 401 identically for "private, not yours" and
    "does not exist", so a missing credential turns every private store into a
    null result -- which is the single most dangerous shape of failure in this
    project.

    `store_exists` now refuses to answer rather than guessing, so that
    particular confusion cannot recur. This closes the other half: the token is
    loaded wherever the project is imported from, so a report run without
    `set -a && . ./.env` works instead of refusing.

    Deliberately narrow. Only the repo-root file, only `KEY=value` lines, and
    an existing environment variable always wins -- a shell that exports a
    different token is making a choice, and silently overriding it would be its
    own version of this bug.
    """
    f = _Path(__file__).resolve().parent.parent / ".env"
    try:
        text = f.read_text()
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in _os.environ:
            _os.environ[k] = v


_load_dotenv()

# --- Context budget (roadmap Sec. 05, locked 2026-09-03) -------------------
# 32,768 is the common floor across all four spines. Qwen3 is 32K below 8B,
# Gemma 3 is 32K at 1B; capping everyone here stops large models winning on
# window size rather than capability.
CONTEXT_WINDOW = 32_768
GENERATION_RESERVE = 2_048
PROMPT_BUDGET = CONTEXT_WINDOW - GENERATION_RESERVE  # 30,720
MAX_TURNS = 10          # multi-turn tiers only; uniform (turn budget is itself
                        # a scaffold variable -- see I(m) in Sec. 04)
#: Per turn. A FROZEN CONTROL, and artifact #15 is what it costs: 33.4% of
#: requests from the <=2B band hit this cap and 0% from >=12B, so the cap
#: varies along the same axis as every by-band result. A program cut here
#: enters the store indistinguishable from one the model finished.
#:
#: Gated, not edited, for the same reason as TEMPERATURE: raising the constant
#: would change the condition every future cell is collected under, silently,
#: and the two conditions would be indistinguishable in the corpus. The
#: raised-cap arm opts in per process and writes to its own store, which
#: store.py enforces.
DEFAULT_MAX_NEW_TOKENS = 2_048
MAX_NEW_TOKENS = int(_os.environ.get("OPENCAP_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS))


def is_bigcap() -> bool:
    """True when this process collects under a raised token cap."""
    return MAX_NEW_TOKENS != DEFAULT_MAX_NEW_TOKENS
KEEP_RECENT_TURNS = 3   # never evicted by truncation

# --- Decoding --------------------------------------------------------------
#: Greedy. DELIBERATE, and the reasoning matters for how results are read.
#:
#: A robot policy has to be trustworthy, and in most deployments that means
#: deterministic: the same scene must produce the same program. Greedy decoding
#: IS the deployment condition, so a greedy score is a property of the artifact
#: that would actually run -- not an estimate of a latent distribution over
#: programs.
#:
#: What that buys, and what it costs. At temperature 0 a cell yields exactly
#: ONE program (measured: 1.00 unique program per failing cell), evaluated over
#: 30 scenes. So n=30 for the ENVIRONMENT and n=1 for the MODEL.
#:   * valid  -- "this deployed policy succeeds on k of 30 scenes", and every
#:               failure-analysis claim, which is about the program we have;
#:   * weak   -- "model A is better than model B", which rests on one draw
#:               each. The cross-host accident gave a second draw on 17 cells:
#:               mean change 3.1 of 30 scenes, but with a tail that includes a
#:               full 30 -> 0 flip.
#: A sampled arm (k programs at temperature > 0, to put an error bar on the
#: point estimates) is deferred to spare budget, not rejected. Until it runs,
#: cross-model rankings from this corpus carry that caveat.
#:
#: Note the seed is INERT at temperature 0: argmax has no RNG. It matters only
#: when TEMPERATURE is raised, which is what the sampled arm does.
#:
#: THE SAMPLED ARM, AND WHY THIS IS AN ENV GATE AND NOT AN EDIT.
#: Raising the constant in this file would change the condition every future
#: greedy cell is collected under, silently, for every arm -- and the baseline
#: and the sampled arm would become indistinguishable in the corpus. So the
#: default stays frozen at 0.0 and a sampled run opts in per process:
#:
#:     OPENCAP_TEMPERATURE=0.7 OPENCAP_SAMPLE_INDEX=3 ...
#:
#: Both land in context_manifest(), so every trace can prove which condition it
#: ran under, and store.py REFUSES to write a sampled cell into a greedy store
#: or a greedy cell into a sampled one. Those three together are the guard: a
#: number in a file is not one.
DEFAULT_TEMPERATURE = 0.0
TEMPERATURE = float(_os.environ.get("OPENCAP_TEMPERATURE", DEFAULT_TEMPERATURE))

#: Which draw this process is. Distinct per sample so the k draws of a cell are
#: k different RNG streams rather than k copies of one.
SAMPLE_INDEX = int(_os.environ.get("OPENCAP_SAMPLE_INDEX", "0"))
SEED = SAMPLE_INDEX


def is_sampled() -> bool:
    """True when this process is collecting a sampled draw, not a greedy one."""
    return TEMPERATURE > 0.0

# --- Measured noise floors (do not treat smaller effects as real) -----------
# Greedy decoding is not bit-reproducible on this stack. Measured on
# qwen3-4b / S1 / cube_stack, same host, same seeds:
#
#   micro-batching 8 :  27, 22          -> spread 5 trials  (~17 pts)
#   micro-batching 1 :  28, 27          -> spread 1 trial   (~3 pts)
#   + no prefix cache,
#     eager mode     :  28, 27          -> spread 1 trial   (unchanged)
#
# Disabling OUR batching bought the whole improvement; vLLM's prefix caching
# and CUDA graphs bought none. The residual is floating-point non-associativity
# in GPU kernels, which no flag removes. So determinism is unavailable and the
# honest response is to publish the floor and refuse to interpret below it.
BATCH_MAX = 1                    # harness default: reproducibility over speed
GENERATION_NOISE_TRIALS = 1      # out of STATES_PER_TASK, at BATCH_MAX=1
GENERATION_NOISE_FRACTION = GENERATION_NOISE_TRIALS / 30  # ~0.033

# Engine state contaminated later cells: the SAME cell scored 27/30 on a fresh
# engine and 0/30 on one that had already served a different cell. The
# mechanism is vLLM's AUTOMATIC PREFIX CACHING -- a warm cache takes a
# different code path, and our prompts share a large fixed prefix (task text
# plus API docs), so every cell after the first was contaminated.
#
#   prefix cache ON :  fresh 27/30, reused  0/30
#   prefix cache OFF:  fresh 28/30, reused 28/30
#
# Disabling it removes the effect entirely, so a fresh engine per cell is NOT
# required and the ~2min/cell reload is avoided. It is ON by default in vLLM,
# so any multi-task evaluation sharing one server is exposed.
DISABLE_PREFIX_CACHING = True
FRESH_ENGINE_PER_CELL = False
REASONING_EFFORT = "medium"  # fixed across the grid; several families ship a
                             # thinking switch and CaP-X exposes it per-run.

# --- The VDM is a second model (discovered 2026-09-04) ---------------------
# M3/M4 call a *separate* visual-differencing model. CaP-X defaults it to a
# frontier closed model, which would entangle scaffold value with VDM
# capability -- a 0.6B agent paired with a frontier VDM is not measuring what
# we claim. Pin one open-weight VDM for every cell in the grid.
VDM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# --- Sampling design -------------------------------------------------------
STATES_PER_TASK = 30    # the unit of pairing; frozen once in P1
LIBERO_STATES_PER_TASK = 10


@dataclass(frozen=True)
class Endpoints:
    """Every service is addressed by URL, never a hard-coded localhost port.

    This is what makes topology T1 (2x H100, one box) and T2 (two tunnelled
    Colab instances) differ by config alone.
    """

    agent_llm: str = "http://127.0.0.1:8110/v1/chat/completions"  # vLLM serves
    # under /v1; CaP-X's own default omits it because they front it with a proxy.
    vdm: str = "http://127.0.0.1:8111/chat/completions"
    sam3: str = "http://127.0.0.1:8114"      # cuda; Wave 2 only
    graspnet: str = "http://127.0.0.1:8115"  # Wave 2 only
    pyroki: str = "http://127.0.0.1:8116"    # IK; needed by every tier, S1 included

    @classmethod
    def from_env(cls) -> "Endpoints":
        return cls(
            agent_llm=os.environ.get("OPENCAP_AGENT_LLM", cls.agent_llm),
            vdm=os.environ.get("OPENCAP_VDM", cls.vdm),
            sam3=os.environ.get("OPENCAP_SAM3", cls.sam3),
            graspnet=os.environ.get("OPENCAP_GRASPNET", cls.graspnet),
            pyroki=os.environ.get("OPENCAP_PYROKI", cls.pyroki),
        )

    def as_dict(self) -> dict:
        return asdict(self)


def context_manifest() -> dict:
    """Recorded into every run config so a trace can prove which policy it ran under."""
    return {
        "context_window": CONTEXT_WINDOW,
        "generation_reserve": GENERATION_RESERVE,
        "prompt_budget": PROMPT_BUDGET,
        "max_turns": MAX_TURNS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "keep_recent_turns": KEEP_RECENT_TURNS,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "sample_index": SAMPLE_INDEX,
        "sampled": is_sampled(),
        "bigcap": is_bigcap(),
        "reasoning_effort": REASONING_EFFORT,
        "vdm_model": VDM_MODEL,
    }
