"""Experiment cells: the unit of work, of commit, and of resume.

A cell is (model, tier, task) = STATES_PER_TASK episodes. Cells are atomic:
interrupted mid-cell, the partial shard is discarded and the cell re-runs.
That bounds worst-case loss at 30 episodes and removes any need for sub-cell
resume logic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from .models import LADDER, ModelSpec

# --- Tiers (CaP-X definitions, reused unchanged so our numbers stay
#     comparable to theirs) --------------------------------------------------
# The wave split is about whether the MODEL needs a perception GPU of its own
# (a VLM), NOT about whether the tier needs perception SERVICES. Read as the
# latter it is false for most of wave 1: S3, S4 and M1 all declare SAM3,
# Contact-GraspNet and pyroki in their configs. That misreading cost A3 7h20m
# on 2026-10-15. The authority on what a tier needs is its own config ---
# scripts/verify_servers.py <task> <tier> reads it.
WAVE1_TIERS = ("S1", "S3", "S4", "M1")
WAVE2_TIERS = ("S2", "M2", "M3", "M4")
ALL_TIERS = WAVE1_TIERS + WAVE2_TIERS

#: M2 hands the model raw RGB, so it needs native multimodality. Every other
#: tier is reachable by text-only models -- S2's perception services and M3's
#: visual differencing module both return text.
VISION_ONLY_TIERS = frozenset({"M2"})

TIER_WAVE = {t: 1 for t in WAVE1_TIERS} | {t: 2 for t in WAVE2_TIERS}

# --- Tasks -----------------------------------------------------------------
# Corrected 2026-09-04 against the repo: env_configs/ ships 5 task families,
# not the 7 the paper's task list implied. See capx_bridge for the details.
from .capx_bridge import CORE_TASKS, has_config

_SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True, order=True)
class Cell:
    model: str
    tier: str
    task: str

    def __post_init__(self) -> None:
        for field_name in ("model", "tier", "task"):
            value = getattr(self, field_name)
            if field_name == "tier":
                if value not in ALL_TIERS:
                    raise ValueError(f"unknown tier {value!r}")
            elif not _SLUG.match(value):
                raise ValueError(f"{field_name} {value!r} is not a safe path slug")

    @property
    def wave(self) -> int:
        return TIER_WAVE[self.tier]

    def path(self) -> str:
        """Cell key encoded in the path -- this is why the file tree IS the manifest."""
        return (
            f"traces/wave={self.wave}/model={self.model}"
            f"/tier={self.tier}/task={self.task}/episodes.jsonl.zst"
        )

    @classmethod
    def from_path(cls, path: str) -> "Cell | None":
        m = re.match(
            r"traces/wave=\d+/model=(?P<model>[^/]+)/tier=(?P<tier>[^/]+)"
            r"/task=(?P<task>[^/]+)/episodes\.jsonl\.zst$",
            path,
        )
        if not m:
            return None
        try:
            return cls(m["model"], m["tier"], m["task"])
        except ValueError:
            return None


def is_runnable(model: ModelSpec, tier: str) -> bool:
    """Text-only models cannot run M2; Spine D stays in Wave 1.

    Control models are excluded from the sweep grid entirely -- they are run
    deliberately, against a specific comparison, not swept as rungs.
    """
    if model.spine == "control":
        return False
    if tier in VISION_ONLY_TIERS and not model.vision:
        return False
    if model.spine == "D" and TIER_WAVE[tier] == 2:
        return False
    return True


def cell_exists(task: str, tier: str) -> bool:
    """Not every (task, tier) pair is shipped -- two_arm_handover has no M4."""
    return has_config(task, tier)


def grid(wave: int, models: list[ModelSpec] | None = None,
         tasks: tuple[str, ...] = CORE_TASKS) -> Iterator[Cell]:
    """Enumerate every cell in a wave, skipping combinations that cannot run."""
    tiers = WAVE1_TIERS if wave == 1 else WAVE2_TIERS
    for model in models or LADDER:
        for tier in tiers:
            if not is_runnable(model, tier):
                continue
            for task in tasks:
                if not cell_exists(task, tier):
                    continue
                yield Cell(model.key, tier, task)


def sweep_order(wave: int, **kw) -> list[Cell]:
    """Model-major ordering.

    Load a model once and run every tier/task inside it. Iterating the other way
    makes model loading dominate wall-clock.
    """
    return sorted(grid(wave, **kw), key=lambda c: (c.model, c.tier, c.task))
