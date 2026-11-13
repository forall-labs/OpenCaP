"""Cell-atomic trace storage with Hugging Face sync (roadmap Sec. 08).

Design commitments:
  * Commit granularity is one cell, never a timer. `CommitScheduler` is
    deliberately not used -- it commits on a clock, which is the coupling we
    are avoiding.
  * The file tree is the manifest. Completion is derived from the repo's file
    listing, so there is no second source of truth to drift.
  * Writes are atomic: build a temp file, verify it parses, then publish.
    A crash mid-cell leaves nothing behind and the cell simply re-runs.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence

from .cells import Cell

try:  # optional: only needed for the HF backend
    import zstandard as zstd
except ImportError:  # pragma: no cover - exercised on pods, not locally
    zstd = None


#: zstd frame magic. Shards are named .jsonl.zst, so the bytes must match the
#: name -- an earlier "graceful fallback" silently wrote plain JSONL under a
#: .zst extension when the library was missing, producing a corpus that only
#: decoded on the machine that wrote it. The extension must not lie.
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _require_zstd(action: str):
    if zstd is None:
        raise RuntimeError(
            f"cannot {action}: the `zstandard` package is missing. It is a hard "
            "dependency (see pyproject); install it rather than writing a "
            "corpus that other machines cannot read."
        )
    return zstd


def _compress(payload: bytes) -> bytes:
    return _require_zstd("write a shard").ZstdCompressor(level=10).compress(payload)


def _decompress(payload: bytes) -> bytes:
    if not payload.startswith(_ZSTD_MAGIC):
        return payload            # legacy uncompressed shard; read it anyway
    return _require_zstd("read a compressed shard").ZstdDecompressor().decompress(payload)


def serialise(episodes: Sequence[dict]) -> bytes:
    """JSONL -> zstd. Traces are the forensic source of truth; the Parquet
    results table is what analysis reads."""
    buf = io.StringIO()
    for ep in episodes:
        buf.write(json.dumps(ep, ensure_ascii=False, sort_keys=True))
        buf.write("\n")
    return _compress(buf.getvalue().encode("utf-8"))


def deserialise(blob: bytes) -> list[dict]:
    text = _decompress(blob).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class TraceStore(ABC):
    """Backend-agnostic corpus. Local for development, HF for real runs."""

    @abstractmethod
    def list_paths(self) -> Iterable[str]: ...

    @abstractmethod
    def _put(self, path: str, blob: bytes, message: str) -> None: ...

    @abstractmethod
    def _get(self, path: str) -> bytes: ...

    # -- public API ---------------------------------------------------------
    def completed(self) -> set[Cell]:
        """One call, and a fresh pod knows exactly what is left to run."""
        found = set()
        for path in self.list_paths():
            cell = Cell.from_path(path)
            if cell is not None:
                found.add(cell)
        return found

    def remaining(self, planned: Sequence[Cell]) -> list[Cell]:
        done = self.completed()
        return [c for c in planned if c not in done]

    def write_cell(self, cell: Cell, episodes: Sequence[dict],
                   verified: bool = True) -> None:
        if not episodes:
            raise ValueError(f"refusing to publish an empty cell: {cell}")
        if not verified:
            # results.cross_check() disagreed with CaP-X's own summary, so the
            # per-trial parse is untrustworthy. A wrong cell in the corpus is
            # worse than a missing one -- completed() would never re-run it.
            raise ValueError(f"refusing to publish unverified cell: {cell}")
        blob = serialise(episodes)
        # Verify before publishing. A shard that cannot be read back is worse
        # than a missing one, because `completed()` would count it as done.
        if len(deserialise(blob)) != len(episodes):
            raise RuntimeError(f"round-trip verification failed for {cell}")
        repo = getattr(self, "repo_id", None)
        if repo:
            _check_not_frozen(repo)
            _check_decode_matches_store(repo)
            _check_cap_matches_store(repo)
        self._put(cell.path(), blob,
                  f"{cell.model} {cell.tier} {cell.task} ({len(episodes)} episodes)")

    def read_cell(self, cell: Cell) -> list[dict]:
        return deserialise(self._get(cell.path()))


class LocalStore(TraceStore):
    """Filesystem backend. Used for tests and for the sim-side scratch dir."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list_paths(self):
        for p in self.root.rglob("episodes.jsonl.zst"):
            yield str(p.relative_to(self.root))

    def _put(self, path: str, blob: bytes, message: str) -> None:
        dest = self.root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".partial")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(blob)
            os.replace(tmp, dest)  # atomic within a filesystem
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _get(self, path: str) -> bytes:
        return (self.root / path).read_bytes()


class HFStore(TraceStore):
    """Hugging Face dataset repo. Free tier is sufficient -- 100GB private
    against a corpus that lands under 2GB."""

    def __init__(self, repo_id: str, token: str | None = None, private: bool = True):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.api = HfApi(token=token or os.environ.get("HF_TOKEN"))
        self.api.create_repo(repo_id, repo_type="dataset",
                             private=private, exist_ok=True)
        self._cache: list[str] | None = None

    def list_paths(self):
        if self._cache is None:
            self._cache = self.api.list_repo_files(self.repo_id, repo_type="dataset")
        return self._cache

    def refresh(self) -> None:
        self._cache = None

    def _put(self, path: str, blob: bytes, message: str) -> None:
        # One commit per cell -- the "log by experiment, not by clock" rule.
        # Retried once: a dropped upload on a reaped runtime is common enough
        # that failing the whole cell for it would waste the compute already
        # spent producing it.
        last = None
        for attempt in range(2):
            try:
                self.api.upload_file(
                    path_or_fileobj=blob,
                    path_in_repo=path,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=f"cell: {message}",
                )
                break
            except Exception as exc:                   # noqa: BLE001
                last = exc
                if attempt == 1:
                    raise RuntimeError(
                        f"upload failed for {path}: {exc!r}"
                    ) from last
        if self._cache is not None and path not in self._cache:
            self._cache.append(path)

    def _get(self, path: str) -> bytes:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(self.repo_id, path, repo_type="dataset")
        return Path(local).read_bytes()


#: Stores that are finished and must not be written to again. A cell's shard
#: path is (wave, model, tier, task) -- re-running any cell into the same store
#: REPLACES its shard in place, with no version history and nothing to compare
#: against afterwards. That is the right behaviour for a corpus still being
#: collected and exactly the wrong one for a corpus other analyses already
#: reference.
#:
#: capx-ab-on is frozen because it is the published baseline: the prompt-fix
#: pairing, the host gate and every reported number are measured against it. A
#: re-run that "just refreshes" a cell would silently move a reference point.
#: Re-runs go to a NEW store and are compared before anything is promoted.
#: RETIRED 2026-09-11. Every one of these contains Colab-collected cells, and
#: artifact #12 established that the generated program itself changes across
#: GPU architectures in half of all cells. They are kept as a historical record
#: and must not be written to, read as a baseline, or pooled with the
#: single-host corpus. Their replacements carry the -h200 suffix.
FROZEN_STORES = frozenset({
    # Colab-era: cross-host, and artifact #12 showed the generated program
    # itself changes across GPU architectures in half of all cells.
    "savoji/capx-ab-on",
    "savoji/capx-ab-off",
    "savoji/capx-promptfix",
    "savoji/capx-entrypoint",
    "savoji/capx-oracle",
    "savoji/capx-scaling-traces",  # also unusable: 0/1170 stderr captured
    # Single-host but UNSEEDED. Artifact #13: the seeding patch reached only
    # robosuite_cubes.py, so cube_lifting, spill_wipe and nut_assembly drew a
    # random scene on every run. These stores stay READABLE and their cell
    # totals are valid unpaired estimates -- but trial 7 in one run is not
    # trial 7 in another, so no state-level pairing is available on two of
    # their three tasks, and nothing may be written to them.
    "savoji/capx-ab-on-h200",
    "savoji/capx-ab-off-h200",
    "savoji/capx-promptfix-h200",
    "savoji/capx-entrypoint-h200",
    "savoji/capx-oracle-h200",
    "savoji/capx-ctrl-batch-h200",
    "savoji/capx-ctrl-prefix-h200",
    "savoji/capx-ctrl-replicate-h200",
})

#: Retired store -> its single-host replacement, so tooling can say what to use
#: instead of merely refusing.
SUPERSEDED_BY = {
    "savoji/capx-ab-on": "savoji/capx-ab-on-seeded",
    "savoji/capx-ab-off": "savoji/capx-ab-off-seeded",
    "savoji/capx-promptfix": "savoji/capx-promptfix-seeded",
    "savoji/capx-entrypoint": "savoji/capx-entrypoint-seeded",
    "savoji/capx-oracle": "savoji/capx-oracle-seeded",
    "savoji/capx-ab-on-h200": "savoji/capx-ab-on-seeded",
    "savoji/capx-ab-off-h200": "savoji/capx-ab-off-seeded",
    "savoji/capx-promptfix-h200": "savoji/capx-promptfix-seeded",
    "savoji/capx-entrypoint-h200": "savoji/capx-entrypoint-seeded",
    "savoji/capx-oracle-h200": "savoji/capx-oracle-seeded",
}


#: A sampled cell and a greedy cell of the same (wave, model, tier, task) share
#: a shard path, so one would REPLACE the other in place and the corpus would
#: hold a mixture that nothing downstream could separate. Sampled stores are
#: named apart and the two conditions refuse to cross.
SAMPLED_MARKER = "-sampled"
#: Same reasoning for the token cap: a raised-cap cell and a standard-cap cell
#: of the same (wave, model, tier, task) share a shard path, and artifact #15
#: is precisely that the two are indistinguishable once written.
BIGCAP_MARKER = "-bigcap"


def _check_decode_matches_store(repo_id: str) -> None:
    from . import config as _cfg
    sampled_store = SAMPLED_MARKER in repo_id
    if _cfg.is_sampled() and not sampled_store:
        raise PermissionError(
            f"refusing to write a SAMPLED cell (temperature "
            f"{_cfg.TEMPERATURE}) into {repo_id}, which is a greedy store.\n"
            f"A sampled and a greedy cell share a shard path, so this would "
            f"replace a greedy cell in place and leave a corpus that mixes "
            f"two decode conditions with no way to tell them apart.\n"
            f"Use a store whose name contains '{SAMPLED_MARKER}'."
        )
    if sampled_store and not _cfg.is_sampled():
        raise PermissionError(
            f"refusing to write a GREEDY cell into {repo_id}, which is a "
            f"sampled store. Set OPENCAP_TEMPERATURE, or write elsewhere.\n"
            f"An unset OPENCAP_TEMPERATURE in one shell of a multi-shell run is "
            f"exactly how this would happen."
        )


def _check_cap_matches_store(repo_id: str) -> None:
    from . import config as _cfg
    bigcap_store = BIGCAP_MARKER in repo_id
    if _cfg.is_bigcap() and not bigcap_store:
        raise PermissionError(
            f"refusing to write a RAISED-CAP cell (max_new_tokens "
            f"{_cfg.MAX_NEW_TOKENS}) into {repo_id}, which is a standard-cap "
            f"store.\nArtifact #15 is that a cut program and a finished one "
            f"are indistinguishable once written; mixing the two caps in one "
            f"store makes that permanent.\nUse a store whose name contains "
            f"'{BIGCAP_MARKER}'."
        )
    if bigcap_store and not _cfg.is_bigcap():
        raise PermissionError(
            f"refusing to write a STANDARD-CAP cell into {repo_id}, which is a "
            f"raised-cap store. Set OPENCAP_MAX_NEW_TOKENS, or write elsewhere."
        )


def _check_not_frozen(repo_id: str) -> None:
    if repo_id in FROZEN_STORES and os.environ.get("OPENCAP_ALLOW_FROZEN") != "1":
        raise PermissionError(
            f"{repo_id} is frozen: it is a published corpus that existing "
            f"analyses are measured against, and writing a cell REPLACES its "
            f"shard in place.\n"
            f"Use {SUPERSEDED_BY.get(repo_id, repo_id + '-h200')} instead.\n"
            f"If you genuinely mean to overwrite, set OPENCAP_ALLOW_FROZEN=1."
        )


class StoreUnreachable(RuntimeError):
    """The hub could not be asked, so no claim about collection is possible.

    This exists because the two answers are not symmetric. "The store is
    there" is self-evident; "the store is NOT there" is a claim about what was
    collected, and on 2026-10-27 a dead local token made every private store
    answer 401 -- identical to never created -- and `condition_report.py`
    printed "not collected (store absent)" for a B1 that was 70/70 complete.
    An infrastructure failure that prints as a negative result is the exact
    artifact class this project exists to catalogue, so the no answer is now
    only given when the hub gave it.
    """


#: True once whoami has succeeded, the exception once it has failed, else None.
#: Cached so a report over seven stores does not call whoami seven times.
_WHOAMI: object | None = None


def store_exists(spec: str) -> bool:
    """Does this store exist, WITHOUT creating it?

    `HFStore.__init__` calls create_repo(exist_ok=True), so merely opening a
    store for reading brings it into existence. That is deliberate for a
    collector -- a run should not die because nobody made the repo first --
    and it is a trap for everything else:

      * a read-only report that opens seven stores creates seven empty repos
        (2026-10-15: scripts/condition_report.py did exactly that while being
        tested, before the ablation had run a single cell);
      * and an empty repo is indistinguishable from "collected, zero cells" to
        the next reader. That confusion is already on the record -- the three
        `-seeded` arm repos existed and were empty, and the register had to be
        taught to print PENDING instead of a verdict over 0 cells.

    Any analysis path should use this first and say "not collected" rather
    than conjuring the thing it is looking for.

    It raises `StoreUnreachable` rather than answering when the hub could not
    be asked. A False here is a claim about the collection, so it is only
    returned when the hub actually answered "no" -- see that exception.
    """
    kind, _, rest = spec.partition(":")
    if kind == "local":
        return Path(rest).exists()
    if kind == "hf":
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        try:
            if api.repo_exists(rest, repo_type="dataset"):
                return True
        except Exception as e:  # network down, hub 5xx, anything
            raise StoreUnreachable(
                f"cannot ask the hub about {spec}: {type(e).__name__}: {e}"
            ) from e
        # A private store we are not authorised to see answers with the SAME
        # 401 as one that was never created, so "absent" is only believable
        # once we know the credentials work.
        _check_credentials(spec)
        return False
    raise ValueError(f"unknown store spec {spec!r}; use local:PATH or hf:REPO")


def _check_credentials(spec: str) -> None:
    """Raise StoreUnreachable unless the hub recognises our token."""
    global _WHOAMI
    if _WHOAMI is True:
        return
    if isinstance(_WHOAMI, Exception):
        raise StoreUnreachable(_unreachable_msg(spec, _WHOAMI)) from _WHOAMI
    from huggingface_hub import HfApi
    try:
        HfApi(token=os.environ.get("HF_TOKEN")).whoami()
    except Exception as e:
        _WHOAMI = e
        raise StoreUnreachable(_unreachable_msg(spec, e)) from e
    _WHOAMI = True


def _unreachable_msg(spec: str, e: Exception) -> str:
    return (
        f"{spec} did not answer and the hub does not recognise our "
        f"credentials ({type(e).__name__}: {e}).\n"
        f"Every PRIVATE store would read as absent, which is NOT the same as "
        f"uncollected -- the hub returns 401 for both. Refusing to report.\n"
        f"Fix the token (`hf auth login`, or set HF_TOKEN) and re-run. The "
        f"project's token was flagged for rotation on 2026-10-04, so a dead "
        f"local token is the expected cause."
    )


def open_store(spec: str) -> TraceStore:
    """`local:/path/to/dir` or `hf:user/repo`."""
    kind, _, rest = spec.partition(":")
    if kind == "local":
        return LocalStore(rest)
    if kind == "hf":
        return HFStore(rest)
    raise ValueError(f"unknown store spec {spec!r}; use local:PATH or hf:REPO")
