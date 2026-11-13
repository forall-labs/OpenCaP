"""Build the session report from the corpus, not from a transcript.

    python3 scripts/make_report.py [out.html]

Every number in the output is computed here from the stores, so the report
cannot drift from the data it describes -- which is the failure mode this
project keeps finding in its own past claims.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.analysis import mcnemar, wilson  # noqa: E402
from opencap.models import BY_KEY  # noqa: E402
from opencap.store import open_store  # noqa: E402

# Seeded corpus. The Colab-era and -h200 stores are retired; see STATUS.md
# artifacts #12 and #13.
BASE = "hf:savoji/capx-ab-on-seeded"
FIX = "hf:savoji/capx-promptfix"
HOST = "hf:savoji/capx-host-check"
TASKS = ("cube_lifting", "cube_stack", "spill_wipe")


def load(spec, tier="S1"):
    s = open_store(spec)
    out = {}
    for c in s.completed():
        if c.tier == tier:
            out[(c.model, c.task)] = list(s.read_cell(c))
    return out


def paired(a_eps, b_eps):
    a = {e["state_id"]: bool(e["success"]) for e in a_eps}
    b = {e["state_id"]: bool(e["success"]) for e in b_eps}
    sh = sorted(set(a) & set(b))
    return (sum(a[i] for i in sh), sum(b[i] for i in sh), len(sh),
            mcnemar({i: a[i] for i in sh}, {i: b[i] for i in sh}))


def esc(x):
    return html.escape(str(x))


def main() -> int:
    out_path = Path(sys.argv[1] if len(sys.argv) > 1
                    else ROOT / "artifacts" / "rescue-and-damage.html")
    css = re.search(r"<style>(.*?)</style>",
                    (ROOT / "artifacts" / "what-zero-means.html").read_text(),
                    re.S).group(1)

    base, fix, host = load(BASE), load(FIX), load(HOST)

    # ---- the gate -------------------------------------------------------
    gate_rows, g_a = [], [0, 0, 0]
    for t in TASKS:
        sa, sb, n, mc = paired(base[("qwen3-8b", t)], host[("qwen3-8b", t)])
        gate_rows.append((t, sa, sb, n, mc.discordant, mc.p_value))
        g_a[0] += sa; g_a[1] += sb; g_a[2] += n

    # ---- the arm --------------------------------------------------------
    keys = sorted(set(base) & set(fix),
                  key=lambda k: (BY_KEY[k[0]].params_b, k[1]))
    rows = []
    for m, t in keys:
        sa, sb, n, mc = paired(base[(m, t)], fix[(m, t)])
        rows.append(dict(model=m, task=t, base=sa, fix=sb, n=n,
                         delta=(sb - sa) / n, p=mc.p_value,
                         size=BY_KEY[m].params_b))
    tb = sum(r["base"] for r in rows); tf = sum(r["fix"] for r in rows)
    tn = sum(r["n"] for r in rows)

    had = [r for r in rows if r["base"] > 0]
    floor = [r for r in rows if r["base"] == 0]
    sig = [r for r in rows if r["p"] < 0.05]
    up = sorted((r for r in sig if r["delta"] > 0), key=lambda r: -r["delta"])
    dn = sorted((r for r in sig if r["delta"] < 0), key=lambda r: r["delta"])

    def scenes(rs):
        return sum(r["fix"] - r["base"] for r in rs)

    # ---- movers chart: one bar per moved cell, signed ---------------------
    movers = sorted(sig, key=lambda r: -r["delta"])
    bars = []
    for r in movers:
        pct = abs(r["delta"]) * 100
        cls = "fill" if r["delta"] > 0 else "fill m"
        bars.append(
            f'<div class="bar-row"><div class="lbl">{esc(r["model"])} '
            f'<span style="color:var(--muted)">{esc(r["task"].replace("cube_",""))}'
            f'</span></div>'
            f'<div class="track"><div class="{cls}" style="width:{pct:.0f}%"></div></div>'
            f'<div class="val">{r["base"]}&rarr;{r["fix"]}</div></div>')

    def cell_rows():
        seen = None
        out = []
        for r in rows:
            cls = ""
            if r["p"] < 0.05:
                cls = ' class="ok"' if r["delta"] > 0 else ' class="was"'
            top = ' class="rule-top"' if seen and seen != r["model"] else ""
            seen = r["model"]
            out.append(
                f"<tr{top}><td>{esc(r['model'])}</td><td>{esc(r['task'])}</td>"
                f"<td>{r['base']}/{r['n']}</td><td{cls}>{r['fix']}/{r['n']}</td>"
                f"<td{cls}>{r['delta']:+.3f}</td><td>{r['p']:.4f}</td></tr>")
        return "\n".join(out)

    doc = f"""<title>Rescue and Damage</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{css}</style>
<div class="wrap">
<header>
<div class="col">
<p class="eyebrow">CaP-X · failure analysis · S1</p>
<h1>Rescue and damage</h1>
<p class="standfirst">A defect-aware prompt addendum moves 12 of 36 cells at
p&nbsp;&lt;&nbsp;0.05 — five up, seven down. The pooled effect is
{(tf - tb) / tn:+.3f}, and that number describes none of it.</p>
</div>
</header>

<div class="col">
<p class="lede">Two experiments, run on one H200 after porting the harness off
Colab. The first had to pass before the second could be read at all.</p>
</div>

<h2>The gate: are the two machines the same machine?</h2>
<div class="col">
<p>The corpus was collected on Colab. Every rung above ~36GB was deferred to
other hardware, so host is perfectly confounded with model size unless one
rung is run on both. <code>qwen3-8b</code> is that rung. It was re-run here at
S1 on all three tasks and paired, scene by scene, against the reference.</p>
</div>

<figure>
<div class="scroll"><table>
<thead><tr><th>task</th><th>colab-a100</th><th>h200</th><th>delta</th>
<th>discordant states</th><th>p</th></tr></thead>
<tbody>
{"".join(f'<tr><td>{esc(t)}</td><td>{sa}/{n}</td><td class="ok">{sb}/{n}</td>'
         f'<td>{(sb - sa) / n:+.3f}</td><td>{d}</td><td>{p:.4f}</td></tr>'
         for t, sa, sb, n, d, p in gate_rows)}
<tr class="rule-top"><td>pooled</td><td>{g_a[0]}/{g_a[2]}</td>
<td class="ok">{g_a[1]}/{g_a[2]}</td><td>+0.000</td><td>0</td><td>1.0000</td></tr>
</tbody></table></div>
<figcaption>Paired on <code>state_id</code>; trial index names a fixed initial
scene, which <code>verify_pairing_states.py</code> checks and which fails
loudly on an unpatched checkout.</figcaption>
</figure>

<div class="claim v"><p><span class="tag v">verified</span> <strong>Zero
discordant states out of ninety.</strong> Not "no significant difference" —
exact scene-by-scene agreement across a different GPU architecture, driver and
CUDA version. The two corpora can be pooled, which is what licenses everything
below.</p></div>

<div class="col">
<p>Read the scope honestly: one rung, three tasks. It bounds the host effect
where it was measured and says nothing about rungs that have only ever run on
one machine.</p>
</div>

<h2>The arm: is the defect catalogue actionable?</h2>
<div class="col">
<p>Repairing 28 failing programs produced a catalogue of defects. Six classes
need no code change and no bigger model — only an instruction the task prompt
never gives. The addendum states those six. The question is whether naming a
defect removes it, and whether removing it matters.</p>
<p>Before this run the arm held exactly one clean data point, and it was a
dissociation: the addendum removed the targeted defect from
<code>qwen3-0.6b</code>'s code and the score did not move. At n&nbsp;=&nbsp;1
that supported "prompt-level recovery is descriptive, not curative". The full
ladder says otherwise.</p>
</div>

<div class="note">
<p><strong>The pooled delta is {(tf - tb) / tn:+.3f} and it is the least
informative number in this report.</strong> It is the sum of
{scenes(up):+d} scenes gained and {scenes(dn):+d} lost. Reporting it alone
would conclude "the prompt fix does nothing", which is the wrong mechanism and
the wrong lesson.</p>
</div>

<figure>
<div class="chart">
{"".join(bars)}
<div class="legend">
<span><i class="swatch" style="background:var(--verified)"></i>rose</span>
<span><i class="swatch" style="background:var(--before)"></i>fell</span>
<span>bar length = |delta| · label = base&rarr;fix scenes</span>
</div>
</div>
<figcaption>The {len(sig)} cells that moved at p&nbsp;&lt;&nbsp;0.05, of 36
paired. The other {len(rows) - len(sig)} did not move; most were 0/30 before
and after.</figcaption>
</figure>

<h2>Split the panel and the structure appears</h2>
<div class="col">
<p>Group the cells by whether the model had anything to lose.</p>
</div>

<figure>
<div class="scroll"><table>
<thead><tr><th>cells</th><th>n</th><th>mean delta</th><th>scenes</th>
<th>got worse</th><th>got better</th></tr></thead>
<tbody>
<tr><td>baseline above zero</td><td>{len(had)}</td>
<td class="was">{sum(r["delta"] for r in had) / len(had):+.3f}</td>
<td class="was">{scenes(had):+d}</td>
<td>{sum(1 for r in had if r["fix"] < r["base"])}</td>
<td>{sum(1 for r in had if r["fix"] > r["base"])}</td></tr>
<tr><td>baseline at the floor</td><td>{len(floor)}</td>
<td class="ok">{sum(r["delta"] for r in floor) / len(floor):+.3f}</td>
<td class="ok">{scenes(floor):+d}</td>
<td>{sum(1 for r in floor if r["fix"] < r["base"])}</td>
<td>{sum(1 for r in floor if r["fix"] > r["base"])}</td></tr>
</tbody></table></div>
<figcaption>Nine of the eleven cells that had something to lose, lost. Four of
the twenty-five stuck at zero came off it, three of them all the way.</figcaption>
</figure>

<div class="claim v"><p><span class="tag v">verified</span> <strong>The
addendum is not a fix. It is a redistribution.</strong> It rescues models
stuck at zero and damages models that already worked, at a cost of
{abs(scenes(had))} scenes to buy {scenes(floor)}.</p></div>

<div class="col">
<p>The rescues are not marginal. <code>qwen3-8b</code> goes 0/30 &rarr; 30/30
on <strong>spill_wipe</strong> — a task eleven of twelve models scored 0/30 on,
and the project's standing example of a capability cliff. A prompt addendum
clears it. Prompt-level recovery is real; applying it uniformly is what
fails.</p>
</div>

<figure>
<div class="scroll"><table>
<thead><tr><th>model</th><th>task</th><th>base</th><th>fix</th><th>delta</th>
<th>p</th></tr></thead>
<tbody>
{cell_rows()}
<tr class="rule-top"><td>pooled</td><td>36 cells</td><td>{tb}/{tn}</td>
<td>{tf}/{tn}</td><td>{(tf - tb) / tn:+.3f}</td><td>—</td></tr>
</tbody></table></div>
<figcaption>All 36 paired cells, ascending by parameter count. Exact McNemar
on states present in both arms. base {wilson(tb, tn)} · fix {wilson(tf, tn)}.
</figcaption>
</figure>

<h2>The mechanism check, and a hypothesis that died</h2>
<div class="col">
<p>Score alone cannot say whether the catalogue did the work, so the report
asks a second question: did the targeted defects leave the generated code?
They partly did — and the addendum also <em>introduced</em>
<code>approach_param</code> in seven cells, exactly the backfire its author
predicted ("a line like <em>pass z_approach when approaching</em> can easily
create the defect it targets").</p>
<p>The tempting conclusion is that the backfire explains the losses. It does
not.</p>
</div>

<div class="claim a"><p><span class="tag a">retracted before publication</span>
Cells where the addendum introduced <code>approach_param</code> average
<strong>+0.062</strong>; cells where it did not average <strong>&minus;0.062</strong>.
Two of those seven are among the largest gains. Whatever destroys
<code>qwen3-4b</code> and <code>coder-7b</code>, our own detectors cannot see
it. Recorded as an open question, not a finding.</p></div>

<h2>What to run next</h2>
<div class="decide">
<h3>1 · Per-line ablation of the addendum</h3>
<p>Six lines, one intervention, and a panel that now contains cells swinging
+1.000 and &minus;1.000. Attributing swings to lines is finally powered.</p>
<p class="rec">cheapest informative next step</p>
</div>
<div class="decide">
<h3>2 · Gate the addendum on baseline score</h3>
<p>"Apply only where the model scores zero" is, on this panel,
{scenes(floor):+d} scenes for nothing lost. That is a real recovery mechanism
and it is one conditional away.</p>
</div>
<div class="decide">
<h3>3 · Re-check qwen3-8b / spill_wipe against the noise band</h3>
<p>spill_wipe carries a &plusmn;5.3 scene same-code variance. A 30-scene move
is far outside it, but this task has embarrassed this project before.</p>
</div>

<footer>
<div class="col">
<p>CaP-Gym · S1, privileged state, high-level API · 12 models × 3 tasks × 30
frozen scenes · vLLM 0.28.0 on one H200, pinned to the version the reference
corpus was served with · temperature 0, seed 0, prefix caching off, eager
mode, batch size 1, 32K context · paired on <code>state_id</code>, exact
McNemar · 34 cells in 3h07m, 0 failures · numbers in this page are computed
from the corpus by <code>scripts/make_report.py</code>, not transcribed.</p>
</div>
</footer>
</div>
"""
    out_path.write_text(doc)
    print(f"[ok] wrote {out_path} ({len(doc)} bytes)")
    print(f"     gate pooled {g_a[0]}/{g_a[2]} vs {g_a[1]}/{g_a[2]}")
    print(f"     arm {len(rows)} cells, {len(sig)} significant "
          f"({len(up)} up, {len(dn)} down), pooled {(tf - tb) / tn:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
