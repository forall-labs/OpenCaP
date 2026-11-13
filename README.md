<div align="center">

# OpenCaP

### How open-weight models fail at writing robot code, and the smallest fix for each failure

**Sajjad Pakdamansavoji**<br>
Forall Labs

[![Project page](https://img.shields.io/badge/Project-page-0072B2.svg)](https://forall-labs.github.io/OpenCaP/)
[![Write-up](https://img.shields.io/badge/Write--up-docs%2FWRITEUP.md-555.svg)](docs/WRITEUP.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgray.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)

</div>

---

Open-weight language models can write Code-as-Policies programs that control a robot, and most
of those programs fail. **OpenCaP** asks what each failure actually is. Twenty-one open-weight
models (0.6B to 32B, including Qwen3, Qwen3-VL, Qwen3.5 and Gemma 3) write programs for four
manipulation tasks in [CaP-X](https://github.com/capgym/cap-x) at three levels of robot API. For
every program that fails, we find the **smallest edit that makes it work**, keeping the model's own
code, and tag each cause that edit had to fix.

## Highlights

- **Lowering the API changes what fails.** With high-level helpers (S1), coordinate-frame errors
  appear in 4% of failing programs. With only perception, IK and the gripper (S3/S4) they appear in
  **47%**, and misread API calls rise from 26% to **60%**.
- **The scale advantage disappears.** At S1, larger models' failing programs carry fewer defects.
  At S3/S4 every size band sits at 3.2 to 4.6, and the ≥12B models carry about **twice** their S1
  count (1.54 against 3.33). This holds for all 15 models of 3B and up.
- **With high-level helpers, the causes are mostly not robotic.** At S1, failures come from an
  uncalled function, a shadowed primitive, a missing final action or the wrong object's coordinates.
  Semantic slips (wrong target, wrong formula) appear in over 90% of failing programs at every level.

## How it works

1. **Run the grid.** 21 models × 4 tasks (cube lifting, cube stacking, nut assembly, spill wiping) ×
   3 API tiers, 30 fixed scenes per cell, greedy decoding through vLLM: 7,560 trials.
   - **S1** gives the model high-level helpers such as `get_object_pose` and `goto_pose`.
   - **S3** removes them: only perception (SAM3, Molmo, Contact-GraspNet), `solve_ik`,
     `move_to_joints` and the gripper.
   - **S4** also removes the worked example.
2. **Repair minimally.** Each failing program gets the smallest edit that makes it pass, keeping the
   model's code. A repair is accepted when no scene fails in all *k* draws, where *k* is calibrated
   per task on the task's own reference program.
3. **Tag the cause.** Each repair is tagged with one tag per distinct cause, in six families
   (SYNTAX, STRUCTURE, INTERFACE, SEMANTIC, TRAJECTORY, FRAME).

The tags were produced by a language model (Claude Opus 5) reading each program against its verified
repair, and checked by an independent model rater from a different family (GPT-6 Astra) on a blind
60-program sample: per-family κ 0.82 for FRAME and 0.87 for INTERFACE. No human has rated the
packet; [`artifacts/second_rater/`](artifacts/second_rater/) is ready for one. See
[the write-up](docs/WRITEUP.md) for the method, all tables, checks and limitations.

## Installation

```bash
git clone https://github.com/forall-labs/OpenCaP && cd OpenCaP
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # analysis and reports (CPU)
pip install -e ".[serve]"        # adds vLLM to collect new trials (GPU)
```

Collecting new trials also needs a [CaP-X](https://github.com/capgym/cap-x) checkout and its
simulators; `scripts/bootstrap_server.sh` clones, patches and verifies one.

## Reproducing the results

Every number in the write-up is regenerated from the files in `results/` on a CPU:

```bash
python scripts/cross_tier_report.py --no-truncated              # Table 1: what fails, by family
python scripts/cross_tier_report.py --by-band --no-truncated    # Table 2: defects per program by model size
python scripts/per_model_profile.py                             # per-model profiles
python scripts/score_second_rater.py artifacts/second_rater/answers_codex.csv   # rater agreement
python scripts/claims_register.py                               # every claim, re-derived from data
python -m unittest discover -s tests
```

The raw per-episode traces were collected to private Hugging Face stores and are not part of this
release; the tags, repairs, verdicts and references they produced are in `results/`.

## Repository layout

```
opencap/        the harness: models, cells, trace store, result parsing, statistics, CLI
scripts/        collection (run_*.sh, serve_vllm.sh), repair verification, and every report
tests/          controls and statistics, not only code (Wilson and McNemar against published values)
results/        tags, minimal repairs, verification verdicts and references
artifacts/      the cross-tier report page and the second-rater packet
docs/           WRITEUP.md, the method and results
website/        the project page (GitHub Pages)
```

## Citation

```bibtex
@misc{pakdamansavoji2026opencap,
  title        = {OpenCaP: How Open-Weight Models Fail at Writing Robot Code},
  author       = {Pakdamansavoji, Sajjad},
  year         = {2026},
  howpublished = {\url{https://github.com/forall-labs/OpenCaP}}
}
```

## License

The code is released under the [MIT License](LICENSE). CaP-X and the models it runs keep their own
licenses.

## Acknowledgements

OpenCaP is built on [CaP-X](https://github.com/capgym/cap-x). We thank its authors for releasing the
benchmark, and the authors of Qwen3, Qwen3-VL, Qwen3.5, Gemma 3, vLLM, SAM3, Molmo and
Contact-GraspNet for their models and tools.
