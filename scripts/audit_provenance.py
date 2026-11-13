"""Assert a store is single-host and fully attributed. Exit 1 if not.

    python3 scripts/audit_provenance.py hf:savoji/capx-ab-on-seeded [more...]

Every number this project published before 2026-09-11 was measured across a
host boundary that we now know changes the generated program in half the cells
(STATUS.md, artifact #12). A corpus that mixes hosts is not a corpus, and one
that cannot name its host cannot be checked. This is the gate that makes
"single hardware" an enforced property rather than an intention.

Fails a store that:
  * contains cells from more than one host_kind;
  * contains any cell with no host_kind at all;
  * contains any episode with no vllm_version or harness_commit.

Oracle episodes (model == "oracle") are LLM-free: no serving stack ran, so an
empty vllm_version is the correct record there, not a gap. They still need a
host and a harness commit.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opencap.store import open_store  # noqa: E402


def audit(spec: str) -> int:
    store = open_store(spec)
    cells = sorted(store.completed())
    if not cells:
        print(f"{spec}: empty")
        return 0
    hosts, gpus = Counter(), Counter()
    no_vllm = no_commit = llm_free = 0
    total = 0
    for c in cells:
        for e in store.read_cell(c):
            total += 1
            hosts[e.get("host_kind") or "<missing>"] += 1
            gpus[e.get("host_gpu") or "<missing>"] += 1
            if e.get("model") == "oracle":
                llm_free += 1
            else:
                no_vllm += not e.get("vllm_version")
            no_commit += not e.get("harness_commit")

    print(f"\n=== {spec}: {len(cells)} cells, {total} episodes ===")
    print(f"  host_kind : {dict(hosts)}")
    print(f"  host_gpu  : {dict(gpus)}")
    print(f"  missing vllm_version  : {no_vllm}"
          + (f"  ({llm_free} oracle episodes are LLM-free: no serving "
             f"stack to name)" if llm_free else ""))
    print(f"  missing harness_commit: {no_commit}")

    problems = []
    real = {h for h in hosts if h != "<missing>"}
    if len(real) > 1:
        problems.append(f"MIXED HOSTS: {sorted(real)} -- cells are not comparable")
    if hosts.get("<missing>"):
        problems.append(f"{hosts['<missing>']} episodes have no host_kind")
    if no_vllm:
        problems.append(f"{no_vllm} episodes cannot name their serving stack")
    if no_commit:
        problems.append(f"{no_commit} episodes cannot name the harness revision")
    for p in problems:
        print(f"  [FAIL] {p}")
    if not problems:
        print(f"  [ok] single host ({next(iter(real))}), fully attributed")
    return len(problems)


def main() -> int:
    specs = sys.argv[1:] or ["hf:savoji/capx-ab-on-seeded"]
    return 1 if sum(audit(s) for s in specs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
