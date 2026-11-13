# Project page

A static page with no build step: `index.html` plus `static/`. It is published to GitHub Pages
by `.github/workflows/pages.yml` on every push to `main` that touches this folder (enable Pages in
the repository settings with **GitHub Actions** as the source).

Open `index.html` directly in a browser, or serve the folder:

```bash
python -m http.server --directory website 8000    # then open http://localhost:8000
```

The charts are drawn in HTML from the numbers of `scripts/cross_tier_report.py` (Tables 1 and 2
of `docs/WRITEUP.md`), so they follow the light and dark themes.
