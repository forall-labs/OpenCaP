"""Episode record schema (roadmap Sec. 08).

Two representations, split by purpose:
  * the full trace   -> JSONL+zstd, forensic source of truth
  * flat scalar row  -> Parquet results table, what analysis actually reads

Deliberately absent: raw RGB frames. At ~1MB/episode they would be ~34GB.
We keep the VDM's *text* output (what the agent actually consumed), the frame
hash, and (seed, state_id) for deterministic regeneration.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

FailureLabel = Literal[
    "success",
    "syntax_error",
    "api_misuse",        # bad signature / nonexistent primitive
    "timeout",
    "no_op",             # ran clean, nothing moved
    "physical_failure",  # executed, wrong outcome
    "gave_up",
    "context_overflow",  # pinned content alone exceeded the budget
]

#: Columns of results/episodes.parquet. Order is stable; append only.
RESULT_COLUMNS = (
    "episode_id", "model", "tier", "task", "state_id", "wave",
    "success", "failure_label", "reward",
    "turns_used", "tokens_in", "tokens_out", "gpu_seconds", "wall_seconds",
    "truncation_fired", "turn_pairs_dropped",
    "logprob_mean", "logprob_min",
    "weight_revision", "vllm_version", "harness_commit",
    "host_gpu", "host_kind",
    # --- appended 2026-09-28, artifact #15 -------------------------------
    # `truncation_fired` is OURS and means the prompt was truncated to fit the
    # context budget. It says nothing about the GENERATION being cut at
    # max_new_tokens, and the two were conflated for the life of the project:
    # 33.4% of requests from the <=2B band stopped at the 2,048-token cap and
    # every one of them entered the corpus with tokens_out = 0,
    # truncation_fired false, and a no_op or api_misuse label -- identical to a
    # program the model finished and got wrong.
    #
    # `finish_reason` is the engine's own word for why generation stopped
    # ("stop" | "length"), and `completion_tokens` is what the shim already
    # records per request in stats_<model>.jsonl. Carrying them here is the
    # difference between an artifact found and an artifact fixed: without them
    # every future collection repeats it silently.
    "finish_reason", "completion_tokens",
)


@dataclass
class Episode:
    episode_id: str
    model: str
    tier: str
    task: str
    state_id: int
    wave: int

    success: bool = False
    failure_label: FailureLabel = "physical_failure"
    #: Graded task reward. At the floor, binary success is uninformative --
    #: every cell reads 0/30 -- but reward separates "did nothing" from "nearly
    #: had it", which is the only signal available below the success threshold.
    reward: float = 0.0

    turns_used: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    gpu_seconds: float = 0.0
    wall_seconds: float = 0.0

    truncation_fired: bool = False
    turn_pairs_dropped: int = 0

    #: Why GENERATION stopped, from the engine: "stop" (the model ended) or
    #: "length" (it hit max_new_tokens and the program has no ending). Distinct
    #: from truncation_fired, which is about the PROMPT. Empty on episodes
    #: collected before 2026-09-28, which is most of the corpus -- absence here
    #: means unknown, never "stop".
    finish_reason: str = ""
    #: Tokens the engine actually generated. tokens_out has been 0 for every
    #: episode in the corpus, so this is the first record of generation length
    #: that reaches the store.
    completion_tokens: int = 0

    logprob_mean: float | None = None
    logprob_min: float | None = None

    weight_revision: str = ""
    vllm_version: str = ""
    harness_commit: str = ""

    # Cells will be produced on more than one kind of machine -- small rungs on
    # Colab L4/A100, the 27B+ rungs and GPT-OSS-120B on rented H100s. If any
    # host-dependent behaviour exists (numerics, batch size, driver), it would
    # confound model size with host, because size determines where a rung runs.
    # Recording the host is what makes that checkable rather than invisible.
    host_gpu: str = ""
    host_kind: str = ""   # colab-l4 | colab-a100 | h100 | ...

    # Full trace -- kept only in the JSONL shard, never in the Parquet table.
    turns: list[dict[str, Any]] = field(default_factory=list)
    final_state: dict[str, Any] = field(default_factory=dict)

    def to_trace(self) -> dict:
        return asdict(self)

    def to_row(self) -> dict:
        d = asdict(self)
        return {k: d[k] for k in RESULT_COLUMNS}


def rows_to_parquet(rows: list[dict], path: str) -> None:
    """~34k flat rows is a few MB and makes the HF dataset viewer work."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist([{k: r.get(k) for k in RESULT_COLUMNS} for r in rows])
    pq.write_table(table, path, compression="zstd")
