#!/usr/bin/env bash
# One-command pod rebuild. Every line here is something we learned the hard way
# on 4 Sep; see comments. Takes ~10 min on a fresh Colab runtime.
set -uo pipefail

CAPX="${CAPX_DIR:-/content/cap-x}"

echo "== uv =="
# The astral install script does not land on Colab; pip does.
pip install -q uv

echo "== cap-x =="
# NOT --recurse-submodules: that pulls b1k (BEHAVIOR-1K) and verl trees we
# never use. But uv resolves the whole lockfile, so every path in
# [tool.uv.sources] must EXIST even when its extra is not installed -- which
# makes all of these mandatory at sync time.
[ -d "$CAPX/.git" ] || git clone -q https://github.com/capgym/cap-x "$CAPX"
cd "$CAPX" || exit 1
for m in sam3 contact_graspnet_pytorch verl curobo robosuite \
         LIBERO-PRO libero_dependencies/robosuite; do
  git submodule update --init --depth 1 "capx/third_party/$m" >/dev/null 2>&1
done

echo "== sync =="
# Never pipe uv sync into tail without pipefail -- the pipe swallows the exit
# code and a failed sync reports success.
set -o pipefail
if uv sync --extra robosuite > /content/sync.log 2>&1; then
  echo "[ok] cap-x venv ready"
else
  echo "[FAIL] uv sync; tail:"; tail -20 /content/sync.log; exit 1
fi

echo "== vllm =="
# vLLM appears in cap-x's pyproject only behind the molmo/verl extras, pinned
# to CONFLICTING versions (0.15.0 vs 0.11.0). Installing it into cap-x's venv
# would corrupt the sim environment; it is a separate service reached by URL.
if ! python3 -c 'import vllm' 2>/dev/null; then
  pip install -q vllm > /content/vllm_install.log 2>&1 || {
    echo "[FAIL] vllm install"; tail -10 /content/vllm_install.log; exit 1; }
fi
# vLLM upgrades torch to a newer CUDA than Colab's preinstalled torchaudio,
# and the mismatch aborts `vllm serve` at import time. Text serving does not
# need torchaudio.
pip uninstall -y -q torchaudio 2>/dev/null || true
python3 -c 'import vllm;print("[ok] vllm",vllm.__version__)'
# NOTE: `vllm serve` is unusable here -- it loads the engine then never binds.
# scripts/vllm_shim.py serves the engine directly; see its docstring.

echo "== patch =="
# CaP-X accepts seed= but never passes it to the placement sampler, so scenes
# are random every run and trial indices are not comparable. Must be applied
# on every fresh checkout, before any cell is run.
cd /content/capx-scaling 2>/dev/null && python3 -u scripts/patch_seeding.py "$CAPX" || exit 1
# Inert unless OPENCAP_PREIMPORT_NUMPY is set; lets one build run both arms of
# the instruction-following vs capability comparison.
python3 -u scripts/patch_exec_globals.py "$CAPX" || exit 1
python3 -u scripts/patch_prompt_fix.py "$CAPX" || exit 1

echo "== verify =="
cd /content/capx-scaling 2>/dev/null && {
  python3 -u scripts/verify_configs.py  "$CAPX" || exit 1
  python3 -u scripts/verify_controls.py "$CAPX" || exit 1
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "BOOTSTRAP_OK"
