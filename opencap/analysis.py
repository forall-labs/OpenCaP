"""Metrics from roadmap Sec. 04.

The statistical design is the point of this module, not the arithmetic.
CaP-X evaluates single-seed and reports no variance; our claim to a better
corpus rests entirely on doing this part properly.

Two rules run through everything here:

  * **Paired states are the unit.** Every model sees the same frozen initial
    states, so comparisons resample *states*, never episodes independently.
    An unpaired 95% interval on 150 episodes is roughly +/-8pp -- wide enough
    to hide every effect we care about. Paired tests see the same shift with
    far more power because the state-to-state variance cancels.
  * **Trend, not law.** Six rungs cannot support a power law. We fit a slope
    on log10(params), report it with a bootstrap interval, and say "trend".
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

Z95 = 1.959963984540054


# --------------------------------------------------------------------------
# proportions
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


def wilson(successes: int, n: int, z: float = Z95) -> Interval:
    """Wilson score interval.

    Preferred over the normal approximation because success rates here sit
    near 0 and 1 often -- at the bottom of the ladder especially -- where the
    Wald interval produces bounds outside [0, 1].
    """
    if n == 0:
        return Interval(0.0, 0.0, 1.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------
# paired comparison
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class McNemar:
    b: int          # a succeeded, b failed
    c: int          # a failed, b succeeded
    p_value: float

    @property
    def discordant(self) -> int:
        return self.b + self.c


def mcnemar(a: Mapping[int, bool], b: Mapping[int, bool]) -> McNemar:
    """Exact two-sided McNemar over states present in both cells.

    Concordant pairs carry no information about which cell is better, so only
    discordant ones enter -- that is exactly why pairing buys power.
    """
    shared = set(a) & set(b)
    n01 = sum(1 for s in shared if a[s] and not b[s])
    n10 = sum(1 for s in shared if not a[s] and b[s])
    n = n01 + n10
    if n == 0:
        return McNemar(n01, n10, 1.0)
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return McNemar(n01, n10, min(1.0, 2 * tail))


# --------------------------------------------------------------------------
# scaling trend
# --------------------------------------------------------------------------
def _ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


@dataclass(frozen=True)
class Trend:
    slope: float
    lo: float
    hi: float
    n_rungs: int

    @property
    def significant(self) -> bool:
        """Interval excludes zero."""
        return (self.lo > 0) or (self.hi < 0)

    def __str__(self) -> str:
        star = "" if self.significant else "  (spans zero)"
        return (f"slope {self.slope:+.4f} per decade "
                f"[{self.lo:+.4f}, {self.hi:+.4f}], {self.n_rungs} rungs{star}")


def scaling_trend(
    per_model: Mapping[str, Mapping[int, bool]],
    params_b: Mapping[str, float],
    n_boot: int = 2000,
    seed: int = 0,
) -> Trend:
    """Slope of success rate against log10(params), with a paired bootstrap.

    `per_model` maps model key -> {state_id: success}. Resampling is over
    STATES, held common across models, so the interval reflects the paired
    design rather than treating each model's episodes as independent.
    """
    models = [m for m in per_model if m in params_b]
    if len(models) < 2:
        return Trend(0.0, 0.0, 0.0, len(models))
    xs = [math.log10(params_b[m]) for m in models]

    def slope_for(states: Sequence[int]) -> float:
        ys = []
        for m in models:
            obs = per_model[m]
            hits = [obs[s] for s in states if s in obs]
            ys.append(sum(hits) / len(hits) if hits else 0.0)
        return _ols_slope(xs, ys)

    all_states = sorted(set().union(*(set(per_model[m]) for m in models)))
    point = slope_for(all_states)

    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        draw = [rng.choice(all_states) for _ in all_states]
        boots.append(slope_for(draw))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots)) - 1]
    return Trend(point, lo, hi, len(models))


# --------------------------------------------------------------------------
# the three contrasts that carry the hypotheses
# --------------------------------------------------------------------------
#: name -> (high tier, low tier, hypothesis)
CONTRASTS = {
    "abstraction_gap": ("S1", "S4", "H1"),   # scaffolding as substitute
    "interaction_gain": ("M1", "S3", "H2"),  # scaffolding as multiplier
    "perception_subst": ("M3", "M2", "H3"),  # VDM vs native vision
}


def contrast(
    cells: Mapping[tuple[str, str], Mapping[int, bool]],
    model: str,
    name: str,
) -> float | None:
    """A(m), I(m) or P(m) for one model, over states both tiers ran."""
    hi_t, lo_t, _ = CONTRASTS[name]
    hi = cells.get((model, hi_t))
    lo = cells.get((model, lo_t))
    if not hi or not lo:
        return None
    shared = set(hi) & set(lo)
    if not shared:
        return None
    return (sum(hi[s] for s in shared) - sum(lo[s] for s in shared)) / len(shared)


def contrast_trend(
    cells: Mapping[tuple[str, str], Mapping[int, bool]],
    params_b: Mapping[str, float],
    name: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> Trend:
    """Does the contrast grow or shrink with model size?

    A negative slope on `abstraction_gap` supports H1 (scaffolding substitutes
    for scale); a positive slope on `interaction_gain` supports H2 (it
    multiplies). This function is the paper's headline figure.
    """
    hi_t, lo_t, _ = CONTRASTS[name]
    models = sorted({m for (m, t) in cells if t in (hi_t, lo_t)} & set(params_b))
    models = [m for m in models if (m, hi_t) in cells and (m, lo_t) in cells]
    if len(models) < 2:
        return Trend(0.0, 0.0, 0.0, len(models))

    xs = [math.log10(params_b[m]) for m in models]
    shared_by_model = {
        m: sorted(set(cells[(m, hi_t)]) & set(cells[(m, lo_t)])) for m in models
    }
    pool = sorted(set().union(*shared_by_model.values()))

    def slope_for(states: Sequence[int]) -> float:
        ys = []
        for m in models:
            hi, lo = cells[(m, hi_t)], cells[(m, lo_t)]
            use = [s for s in states if s in hi and s in lo]
            ys.append(
                (sum(hi[s] for s in use) - sum(lo[s] for s in use)) / len(use)
                if use else 0.0
            )
        return _ols_slope(xs, ys)

    rng = random.Random(seed)
    boots = sorted(slope_for([rng.choice(pool) for _ in pool]) for _ in range(n_boot))
    return Trend(slope_for(pool), boots[int(0.025 * len(boots))],
                 boots[int(0.975 * len(boots)) - 1], len(models))


# --------------------------------------------------------------------------
# compute-matched frontier
# --------------------------------------------------------------------------
def inference_flops(active_params_b: float, tokens: int) -> float:
    """~2 * N * T forward-pass proxy. Portable and reviewer-checkable, unlike
    wall-clock GPU-seconds which are hardware-specific. Report both; where they
    disagree, the disagreement is itself a finding."""
    return 2.0 * active_params_b * 1e9 * tokens


def pareto_front(points: Iterable[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Cheapest cell at each success level: maximise success, minimise cost.

    Answers the practitioner's question directly -- at a fixed inference
    budget, buy parameters or buy scaffold turns?
    """
    ordered = sorted(points, key=lambda p: (p[0], -p[1]))
    front: list[tuple[float, float, str]] = []
    best = -math.inf
    for cost, success, label in ordered:
        if success > best:
            front.append((cost, success, label))
            best = success
    return front


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def host_effect(rows: Iterable[Mapping], model: str, tier: str) -> McNemar | None:
    """Same model+tier run on two hosts: is the difference real?

    The ladder is split across machines by VRAM -- small rungs on Colab, 27B+
    on rented H100s -- so host is perfectly correlated with size unless at
    least one OVERLAP rung is run on both. This compares that rung's paired
    outcomes across hosts; a significant result means the two halves of the
    curve are not directly comparable and need a host offset.
    """
    by_host: dict[str, dict[int, bool]] = defaultdict(dict)
    for r in rows:
        if r["model"] == model and r["tier"] == tier and r.get("host_kind"):
            by_host[r["host_kind"]][hash((r["task"], r["state_id"])) & 0x7FFFFFFF] = bool(r["success"])
    if len(by_host) < 2:
        return None
    a, b = (by_host[k] for k in sorted(by_host)[:2])
    return mcnemar(a, b)


def cells_from_rows(rows: Iterable[Mapping]) -> dict[tuple[str, str], dict[int, bool]]:
    """Episode rows -> {(model, tier): {state_id: success}}, pooled over tasks.

    Pooling is what gives 150 episodes per cell (5 tasks x 30 states at the
    time of writing); state_id stays unique per task via the task prefix.
    """
    out: dict[tuple[str, str], dict[int, bool]] = defaultdict(dict)
    for r in rows:
        key = (r["model"], r["tier"])
        state = (r["task"], r["state_id"])
        out[key][hash(state) & 0x7FFFFFFF] = bool(r["success"])
    return dict(out)
