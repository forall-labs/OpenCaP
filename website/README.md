# Project page

A static page with no build step: `index.html` plus `static/`. It is published to GitHub Pages
by `.github/workflows/pages.yml` on every push to `main` that touches this folder (enable Pages in
the repository settings with **GitHub Actions** as the source).

Serve the folder, since the program explorer loads its data with `fetch`:

```bash
python -m http.server --directory website 8000    # then open http://localhost:8000
```

The charts are drawn in HTML from the numbers of `scripts/cross_tier_report.py` (Tables 1 and 2
of `docs/WRITEUP.md`), so they follow the light and dark themes.

The program explorer (Figure 3) reads `static/explorer.json`, built from the tag and failure files in
`results/` by

```bash
python3 website/tools/build_explorer.py
```

The program in the hero is `results/tags_reduced.json` pid `f9a22ee2b2bd`, with comments and blank
lines removed; its marks are set by hand in `index.html` from that program's three tags.

`static/fl-page.css` and `static/fl-page.js` are the interactive layer shared by the Forall Labs
project pages (OpenCaP, AutoPolish, CALVIN-MAX): the hero with a visual beside the title, the sticky
section bar, the compare slider, figure zoom and the hover states. The same two files ship in each
repository, so edit them in one and copy them to the others.
