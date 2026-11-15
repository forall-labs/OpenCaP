"""Build website/static/explorer.json, the data behind the project page's failure explorer.

    python3 website/tools/build_explorer.py

Joins every tagged failing program to its code: the 70 S1 programs of
results/tags_s1_hand.json and the 154 S3/S4 programs of results/tags_reduced.json
that carry a repair level (the same samples as Figure 1 and Table 1 on the page).
Evidence strings come from the tag files (S1: results/tags_seeded.json, where the
tag matches). For each tag, `names` lists the identifiers and quoted strings its
evidence names that also occur in the code, so the page can mark where they appear.
It marks names, not faulty lines: a missing action, for one, has no line to point at.
Programs longer than MAX_LINES are cut for the page (`cut` counts the dropped lines).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from opencap.editclass import FAMILY  # noqa: E402
from opencap.models import BY_KEY  # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "website" / "static" / "explorer.json"
MAX_LINES = 90

code = {}
for f in ["failures.json", "failures_s3.json", "failures_s4.json", "failures_seeded.json", "failures_frame_s3.json"]:
    for r in json.loads((RES / f).read_text()):
        code.setdefault(r["pid"], r)

s1 = json.loads((RES / "tags_s1_hand.json").read_text())
red = json.loads((RES / "tags_reduced.json").read_text())
seeded = {r["pid"]: r for r in json.loads((RES / "tags_seeded.json").read_text())}


def model_name(key: str) -> str:
    repo = BY_KEY[key].repo.split("/")[-1]
    repo = re.sub(r"-Instruct$|-it$", "", repo)
    m = re.match(r"gemma-3-(\d+)b", repo)
    return f"Gemma 3 {m.group(1)}B" if m else repo.replace("-", " ").replace("Qwen2.5 Coder", "Qwen2.5-Coder").replace("Qwen3 VL", "Qwen3-VL")


TOKEN = re.compile(r"'([^']{2,40})'|\"([^\"]{2,40})\"|\b([A-Za-z_][A-Za-z0-9_]*(?:_[A-Za-z0-9_]+|\())")
STOP = {"repair_adds", "the", "and"}
NUMERIC = re.compile(r"^[-+\d.,\s\[\]]+$")  # a bare value such as '0.0' would mark every zero in the program


def names_for(evidence: str, src: str) -> list[str]:
    """Identifiers and quoted strings the evidence names that also occur in the code."""
    out = []
    for m in TOKEN.finditer(evidence or ""):
        tok = (m.group(1) or m.group(2) or m.group(3) or "").rstrip("(")
        if len(tok) >= 3 and tok.lower() not in STOP and not NUMERIC.match(tok) and tok in src and tok not in out:
            out.append(tok)
    return out


rows = []
for tier_name, table in (("S1", s1), ("S3/S4", red)):
    for pid, r in table.items():
        if tier_name == "S3/S4" and r.get("level") is None:
            continue
        c = code[pid]
        src_lines = c["code"].rstrip("\n").split("\n")
        cut = max(0, len(src_lines) - MAX_LINES)  # long programs are cut for the page; the repo has them whole
        src_lines = src_lines[:MAX_LINES]
        src = "\n".join(src_lines)
        ev = (r.get("evidence") or (seeded.get(pid) or {}).get("evidence") or {})
        tags = []
        for t in r["tags"]:
            if t not in FAMILY:  # over_dense_path etc. are not causes (see cross_tier_report.py)
                continue
            e = ev.get(t, "")
            tags.append({"tag": t, "family": FAMILY[t], "evidence": e, "names": names_for(e, src)})
        if not tags:
            continue
        model = r["models"][0]
        rows.append({
            "pid": pid,
            "tier": r.get("tier") or c["tier"],
            "task": r["task"],
            "model": model_name(model),
            "size": BY_KEY[model].params_b,
            "tags": tags,
            "code": src,
            "cut": cut,
        })

rows.sort(key=lambda x: (x["tier"], x["task"], x["size"], x["pid"]))
OUT.write_text(json.dumps(rows, separators=(",", ":")))
by = {}
for r in rows:
    by[r["tier"]] = by.get(r["tier"], 0) + 1
print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} programs {by}, "
      f"{sum(bool(t['names']) for r in rows for t in r['tags'])} of {sum(len(r['tags']) for r in rows)} tags name something in the code, "
      f"{OUT.stat().st_size // 1024} KB")
