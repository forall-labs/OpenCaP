"""The model ladder (roadmap Sec. 01).

Four within-family spines plus three calibration anchors. Within-family sweeps
are the design: six sizes on one recipe is a scaling curve; six sizes from six
labs is a leaderboard.

`params_b` here are LABELS. True parameter counts are read from each model's
config.json in P0 and written to manifest/models.json -- the x-axis uses the
true counts, never the marketing name (Qwen3.5-27B is 28B actual).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Spine = Literal["A", "B", "C", "D", "E", "anchor", "control"]


@dataclass(frozen=True)
class ModelSpec:
    key: str                 # short slug used in trace paths
    repo: str                # HF repo id
    params_b: float          # label size, in billions
    vision: bool
    precision: str           # bf16 | fp8 | mxfp4
    spine: Spine
    note: str = ""
    extra_vllm_args: tuple[str, ...] = field(default_factory=tuple)
    #: vLLM's own KV figure at 32K, filled in once a model has been served.
    measured_kv_gb: float | None = None

    #: KV cache GB per billion params at our fixed 32,768-token context,
    #: calibrated against vLLM's own figure for Qwen3-8B (4.5 GiB / 8B).
    #: The earlier formula used a flat +6GB headroom and ignored KV entirely,
    #: which wrongly placed 8B on an L4 -- it loads, then dies with
    #: "0.85 GiB available, 4.5 GiB needed". KV is not headroom; at 32K it can
    #: exceed the weights.
    KV_GB_PER_B_AT_32K = 0.56

    @property
    def kv_gb(self) -> float:
        """KV cache at CONTEXT_WINDOW. Shrinking context to fit is not an
        option -- the 32K cap is the control that makes the curve comparable.

        Prefer a MEASURED value: KV scales with layers x kv_heads, not with
        parameter count. With GQA the kv_head count is often constant across a
        family, so KV grows only with depth -- vLLM reports 4.5 GiB for
        Qwen3-8B (36 layers) and just 5.0 GiB for Qwen3-14B (40 layers), where
        a per-parameter estimate predicts 7.8 GiB. The estimate is a safe
        upper bound for unmeasured models but badly overshoots large ones.
        """
        if self.measured_kv_gb is not None:
            return self.measured_kv_gb
        return self.params_b * self.KV_GB_PER_B_AT_32K

    @property
    def min_vram_gb(self) -> float:
        """Weights + KV at 32K + activations/CUDA graphs."""
        bytes_per_param = {"bf16": 2.0, "fp8": 1.1, "mxfp4": 0.55}[self.precision]
        return self.params_b * bytes_per_param + self.kv_gb + 2.0

    def fits(self, vram_gb: float) -> bool:
        return self.min_vram_gb <= vram_gb

    @property
    def tensor_parallel(self) -> int:
        # GPT-OSS-120B: MXFP4 weights (~60GB) leave only ~13GB KV cache on one
        # 80GB card, which throttles concurrency badly at 32K context. Split it.
        return 2 if self.key == "gpt-oss-120b" else 1

    def vllm_args(self, port: int = 8000) -> list[str]:
        from .config import CONTEXT_WINDOW

        args = [
            "--model", self.repo,
            "--port", str(port),
            "--max-model-len", str(CONTEXT_WINDOW),
            "--seed", "0",
        ]
        if self.precision == "fp8":
            args += ["--quantization", "fp8"]
        elif self.precision == "mxfp4":
            args += ["--quantization", "mxfp4"]
        if self.tensor_parallel > 1:
            args += ["--tensor-parallel-size", str(self.tensor_parallel)]
        args += list(self.extra_vllm_args)
        return args


# Instruction-tuned checkpoints only, never Base -- mixing tuning status would
# confound the curve.
LADDER: tuple[ModelSpec, ...] = (
    #: Precision policy. The 27B/32B rungs were marked fp8 because an 80GB H100
#: holds them at fp8 (47-55GB) but not at bf16 (71-84GB). On a 141GB H200 they
#: fit at bf16 with room to spare, so the whole ladder stays ONE precision and
#: the fp8-vs-bf16 confound never enters the curve. That also removes
#: qwen3-8b-fp8 from the critical path -- a precision control is only needed
#: if a rung on the curve is actually quantised.
#: If you run these on an 80GB card, switch them back to "fp8" AND run the
#: qwen3-8b-fp8 control before pooling them with the bf16 rungs.

    # --- Spine A: resolution. Six rungs, one recipe. -----------------------
    ModelSpec("qwen3-0.6b",  "Qwen/Qwen3-0.6B",  0.6, False, "bf16", "A", "floor"),
    ModelSpec("qwen3-1.7b",  "Qwen/Qwen3-1.7B",  1.7, False, "bf16", "A"),
    ModelSpec("qwen3-4b",    "Qwen/Qwen3-4B",    4.0, False, "bf16", "A"),
    ModelSpec("qwen3-8b",    "Qwen/Qwen3-8B",    8.0, False, "bf16", "A", "dev model",
              measured_kv_gb=4.5),
    ModelSpec("qwen3-14b",   "Qwen/Qwen3-14B",  14.0, False, "bf16", "A",
              measured_kv_gb=5.0),
    ModelSpec("qwen3-32b",   "Qwen/Qwen3-32B",  32.0, False, "bf16",  "A", "ceiling; bf16 starves KV cache"),

    # --- Spine B: modality. The only family giving text+vision on one recipe,
    #     which is what makes the M2-vs-M3 contrast measurable. --------------
    ModelSpec("qwen3vl-2b",  "Qwen/Qwen3-VL-2B-Instruct",   2.0, True, "bf16", "B", "enables M2"),
    ModelSpec("qwen3vl-4b",  "Qwen/Qwen3-VL-4B-Instruct",   4.0, True, "bf16", "B"),
    ModelSpec("qwen3vl-8b",  "Qwen/Qwen3-VL-8B-Instruct",   8.0, True, "bf16", "B"),
    ModelSpec("qwen3vl-32b", "Qwen/Qwen3-VL-32B-Instruct", 32.0, True, "bf16",  "B", "ceiling"),

    # --- Spine C: family control. Clean dense counts, all multimodal.
    #     Gemma 4 rejected: its E2B/E4B are *effective* params, ambiguous axis.
    ModelSpec("gemma3-4b",   "google/gemma-3-4b-it",   4.0, True, "bf16", "C"),
    ModelSpec("gemma3-12b",  "google/gemma-3-12b-it", 12.0, True, "bf16", "C"),
    ModelSpec("gemma3-27b",  "google/gemma-3-27b-it", 27.0, True, "bf16",  "C"),

    # --- Spine D: generation control. Rungs do NOT line up with Spine A;
    #     only 4B<->4B matches exactly. Compare curves, not points. ----------
    ModelSpec("qwen35-0.8b", "Qwen/Qwen3.5-0.8B",  0.8, False, "bf16", "D", "near qwen3-0.6b"),
    ModelSpec("qwen35-2b",   "Qwen/Qwen3.5-2B",    2.0, False, "bf16", "D", "near qwen3-1.7b"),
    ModelSpec("qwen35-4b",   "Qwen/Qwen3.5-4B",    4.0, False, "bf16", "D", "EXACT match to qwen3-4b"),
    ModelSpec("qwen35-9b",   "Qwen/Qwen3.5-9B",    9.0, False, "bf16", "D", "near qwen3-8b"),
    ModelSpec("qwen35-27b",  "Qwen/Qwen3.5-27B",  27.0, False, "bf16",  "D", "28B actual; ceiling"),

    # --- FP8 variants: Colab cannot serve 14B at bf16 with a 32K context
    #     (needs 5.0 GiB KV, has 2.2 GiB). Quantising is the only way to reach
    #     the rung on this hardware, but precision would then be confounded
    #     with size. 14B turned out to be reachable at bf16 after raising
    #     gpu_memory_utilization, so these are NOT spine rungs -- two entries
    #     at one size would corrupt the curve they were meant to extend.
    #     They stay as a CONTROL, still needed because qwen3-32b is FP8 on
    #     H100: merging that rung into a bf16 curve is only justified if this
    #     control shows no precision effect. --------------------------------
    ModelSpec("qwen3-8b-fp8",  "Qwen/Qwen3-8B",   8.0, False, "fp8", "control",
              "precision control: pair with qwen3-8b bf16 before trusting "
              "any FP8 rung on the curve"),

    # --- Anchors: CaP-X published numbers, used to calibrate the harness ----
    ModelSpec("gpt-oss-20b",  "openai/gpt-oss-20b",  20.0, False, "mxfp4", "anchor", "Gate 0 target"),
    ModelSpec("gpt-oss-120b", "openai/gpt-oss-120b", 120.0, False, "mxfp4", "anchor", "anchor + Project C teacher"),
    ModelSpec("qwen25-coder-7b", "Qwen/Qwen2.5-Coder-7B-Instruct", 7.0, False, "bf16", "anchor", "CaP-RL model"),

    # --- Spine E: Qwen2.5-Coder, a code-trained ladder -----------------------
    # Added after Coder-7B scored 29/30 on spill_wipe -- a task every Qwen3
    # rung failed 0/30, 14B included -- by writing the depth-2 nested loop none
    # of them produce. That makes training data, not parameter count, the
    # variable that matters for this capability. These rungs locate where it
    # appears: if 1.5B nests, a 1.5B model beats a 14B on the task.
    ModelSpec("qwen25-coder-1.5b", "Qwen/Qwen2.5-Coder-1.5B-Instruct", 1.5, False, "bf16", "E", "code ladder"),
    ModelSpec("qwen25-coder-3b",   "Qwen/Qwen2.5-Coder-3B-Instruct",   3.0, False, "bf16", "E", "code ladder"),
)

BY_KEY = {m.key: m for m in LADDER}


def spine(name: Spine) -> list[ModelSpec]:
    return [m for m in LADDER if m.spine == name]


def vision_models() -> list[ModelSpec]:
    return [m for m in LADDER if m.vision]


#: Serving VRAM by host. Colab Pro gives L4 and A100; H100s are rented later
#: for the rungs that do not fit.
#: Usable serving VRAM = nameplate x gpu_memory_utilization (0.90), which is
#: what vLLM is actually allowed to allocate. An earlier version used near-
#: nameplate figures and wrongly predicted qwen3-14b would fit at 0.80 util --
#: it was 1GB short. The budget is set by the serving config, not the card.
GPU_MEMORY_UTILIZATION = 0.90
HOSTS = {"colab-l4": 23.0 * GPU_MEMORY_UTILIZATION,      # 20.7
         "colab-a100": 40.0 * GPU_MEMORY_UTILIZATION,    # 36.0
         "h100": 80.0 * GPU_MEMORY_UTILIZATION,          # 72.0
         # 141GB nameplate. Every rung in the ladder fits, including
         # gpt-oss-120b without its tensor-parallel split -- but see
         # OVERLAP_RUNG: reaching a rung is not the same as being able to
         # pool it with cells collected elsewhere.
         "h200": 141.0 * GPU_MEMORY_UTILIZATION}         # 126.9


def runnable_on(host: str) -> list[ModelSpec]:
    return [m for m in LADDER if m.fits(HOSTS[host])]


def needs_h100() -> list[ModelSpec]:
    """Rungs Colab cannot serve -- deferred to rented H100s.

    Because size determines host, host is confounded with size unless at least
    one OVERLAP rung is run on both. See analysis.host_effect().
    """
    return [m for m in LADDER if not m.fits(HOSTS["colab-a100"])]


#: Run this rung on Colab *and* on H100 to measure the host effect.
#: Chosen as the largest model that fits comfortably on both.
OVERLAP_RUNG = "qwen3-8b"
