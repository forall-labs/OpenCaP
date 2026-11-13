import html
fam = [  # family, S1 (pct, lo, hi), S3/S4 all (pct, lo, hi), verified pct, Fisher p
 ("FRAME",      (4,1,12),   (47,40,55), 48, "≈1e-11", "Camera→world, quaternion order, homogeneous coordinates"),
 ("INTERFACE",  (26,17,37), (60,52,67), 52, "≈3e-6",  "Misread API returns and arguments, wrong prompts, unbound names"),
 ("TRAJECTORY", (47,36,59), (19,14,26), 23, "≈5e-5",  "Mostly S1's z_approach argument, which the reduced API lacks"),
 ("STRUCTURE",  (29,19,40), (35,28,43), 33, "n.s.",   "Missing/extra actions, never-called code, gripper order"),
 ("SEMANTIC",   (94,86,98), (95,90,97), 93, "1.0",    "Wrong target or formula; near-universal at every tier"),
]
bands = ["≤2B", "3–4B", "7–9B", "≥12B"]
by_band = {"S1": [4.17, 2.70, 1.54, 1.54], "S3": [4.10, 4.61, 3.53, 3.46], "S4": [3.20, 3.82, 3.56, 3.21]}
tasks = [("cube_lifting", 0, 37, 20, 44), ("cube_stack", 0, 50, 40, 64), ("nut_assembly", 14, 44, 29, 86), ("spill_wipe", 0, 59, 16, 49)]

# ---- family chart: paired horizontal bars with CI whiskers, one scale 0..100
W, rowh, left, right, top = 720, 54, 150, 40, 30
plotw = W - left - right
def x(p): return left + plotw * p / 100
H = top + rowh * len(fam) + 30
svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Share of failed programs with at least one defect in each family, S1 versus S3/S4" class="chart">']
for t in range(0, 101, 25):
    svg.append(f'<line x1="{x(t):.1f}" y1="{top-8}" x2="{x(t):.1f}" y2="{H-24}" class="grid"/>')
    svg.append(f'<text x="{x(t):.1f}" y="{H-8}" class="axis" text-anchor="middle">{t}%</text>')
for i, (name, s1, s3, ver, p, _) in enumerate(fam):
    y0 = top + i * rowh
    svg.append(f'<text x="{left-12}" y="{y0+21}" class="lbl" text-anchor="end">{name}</text>')
    for j, (v, cls) in enumerate(((s1, "s1"), (s3, "s34"))):
        yy = y0 + 4 + j * 18
        svg.append(f'<rect x="{left}" y="{yy}" width="{x(v[0])-left:.1f}" height="13" class="bar {cls}"/>')
        svg.append(f'<line x1="{x(v[1]):.1f}" y1="{yy+6.5}" x2="{x(v[2]):.1f}" y2="{yy+6.5}" class="ci"/>')
        svg.append(f'<text x="{min(x(v[2])+6, W-4):.1f}" y="{yy+11}" class="val">{v[0]}%</text>')
svg.append('</svg>')
fam_svg = "\n".join(svg)

# ---- band chart: dot plot, defects per program, one scale 0..5
W2, H2, l2, r2, t2, b2 = 720, 300, 60, 30, 20, 50
pw, ph = W2 - l2 - r2, H2 - t2 - b2
def xb(i): return l2 + pw * (i + 0.5) / len(bands)
def yb(v): return t2 + ph * (1 - v / 5.0)
s = [f'<svg viewBox="0 0 {W2} {H2}" role="img" aria-label="Mean defect tags per failed program by model size, S1, S3 and S4" class="chart">']
for v in range(0, 6):
    s.append(f'<line x1="{l2}" y1="{yb(v):.1f}" x2="{W2-r2}" y2="{yb(v):.1f}" class="grid"/>')
    s.append(f'<text x="{l2-10}" y="{yb(v)+4:.1f}" class="axis" text-anchor="end">{v}</text>')
for i, b in enumerate(bands):
    s.append(f'<text x="{xb(i):.1f}" y="{H2-22}" class="axis" text-anchor="middle">{b}</text>')
for tier, cls, dx in (("S1", "s1", -14), ("S3", "s3", 0), ("S4", "s4", 14)):
    pts = [(xb(i) + dx, yb(v)) for i, v in enumerate(by_band[tier])]
    s.append('<polyline points="' + " ".join(f"{a:.1f},{c:.1f}" for a, c in pts) + f'" class="line {cls}"/>')
    for (a, c), v in zip(pts, by_band[tier]):
        s.append(f'<circle cx="{a:.1f}" cy="{c:.1f}" r="5" class="dot {cls}"><title>{tier} {v:.2f}</title></circle>')
s.append(f'<text x="{l2}" y="{H2-4}" class="axis">model size band (parameters)</text>')
s.append('</svg>')
band_svg = "\n".join(s)

fam_rows = "\n".join(
    f"<tr><th scope=row>{n}</th><td>{a[0]}% <span class=ci-t>[{a[1]}–{a[2]}]</span></td>"
    f"<td>{b[0]}% <span class=ci-t>[{b[1]}–{b[2]}]</span></td><td>{v}%</td><td>{p}</td><td class=note>{html.escape(d)}</td></tr>"
    for n, a, b, v, p, d in fam)
task_rows = "\n".join(f"<tr><th scope=row><code>{t}</code></th><td>{f1}%</td><td>{f2}%</td><td>{i1}%</td><td>{i2}%</td></tr>" for t, f1, f2, i1, i2 in tasks)

import importlib.util as _u, sys as _sys
_sys.path.insert(0, ".")
_sp = _u.spec_from_file_location("_pmp", "scripts/per_model_profile.py"); _pmp = _u.module_from_spec(_sp); _sp.loader.exec_module(_pmp)
from opencap.models import BY_KEY as _BY
_prof = _pmp.profiles()
def _cell(p, k): return f"{p[k]}/{p['n']}" if p["n"] else "–"
def _tp(p): return f"{p['tags']/p['n']:.2f}" if p["n"] else "–"
model_rows = "\n".join(
    f"<tr><th scope=row><code>{m}</code></th><td>{_BY[m].params_b:g}B</td><td>{_tp(_prof[m]['S1'])}</td><td>{_tp(_prof[m]['S3S4'])}</td>"
    f"<td>{_cell(_prof[m]['S1'],'FRAME')}</td><td>{_cell(_prof[m]['S3S4'],'FRAME')}</td></tr>"
    for m in sorted(_prof, key=lambda m: (_BY[m].params_b, m)))
page = open("artifacts/cross_tier_defects_template.html").read()
page = page.replace("{{FAM_SVG}}", fam_svg).replace("{{BAND_SVG}}", band_svg).replace("{{FAM_ROWS}}", fam_rows).replace("{{TASK_ROWS}}", task_rows).replace("{{MODEL_ROWS}}", model_rows)
open("artifacts/cross-tier-defects.html", "w").write(page)
print("built", len(page))
