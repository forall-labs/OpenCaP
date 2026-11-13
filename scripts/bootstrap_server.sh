#!/usr/bin/env bash
# One-command rebuild on a bare-metal multi-GPU Linux box.
#
# The server counterpart to bootstrap_colab.sh. Same steps in the same order;
# the differences are all environmental and each one below cost time to find.
# Idempotent -- safe to re-run.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

VENV="${OPENCAP_VLLM_VENV:-/data/venvs/vllm028}"

echo "== vllm venv =="
# vLLM lives in its OWN venv, never in cap-x's: cap-x pins vllm 0.15.0 and
# 0.11.0 behind its molmo/verl extras, and it must not be installed into the
# system conda base either -- that base is shared with other users on this box.
#
# Pinned to 0.28.0, the version the Colab corpus was served with. The host
# gate is supposed to measure the host; upgrading the engine at the same time
# would fold a second change into the one number that decides whether the two
# corpora can be pooled.
if ! "$OPENCAP_VLLM_PYTHON" -c 'import vllm' 2>/dev/null; then
  uv venv "$VENV" --python 3.12
  uv pip install --python "$VENV/bin/python" \
      "vllm==0.28.0" "huggingface_hub>=0.26" "zstandard>=0.22" "pyarrow>=16" || exit 1
fi
"$OPENCAP_VLLM_PYTHON" -c 'import vllm;print("[ok] vllm",vllm.__version__)' || exit 1

echo "== cap-x =="
# NOT --recurse-submodules: that pulls b1k and the LIBERO tree we never use.
# But uv resolves the whole lockfile, so every path in [tool.uv.sources] must
# EXIST even when its extra is not installed -- which makes these mandatory.
[ -d "$CAPX_DIR/.git" ] || git clone https://github.com/capgym/cap-x "$CAPX_DIR"
cd "$CAPX_DIR" || exit 1
for m in sam3 contact_graspnet_pytorch verl curobo robosuite \
         LIBERO-PRO libero_dependencies/robosuite; do
  git submodule update --init --depth 1 "capx/third_party/$m" >/dev/null 2>&1
done

echo "== sync =="
# Never pipe uv sync into tail without pipefail -- the pipe swallows the exit
# code and a failed sync reports success.
set -o pipefail
if uv sync --extra robosuite > "$OPENCAP_LOGDIR/sync.log" 2>&1; then
  echo "[ok] cap-x venv ready"
else
  echo "[FAIL] uv sync; tail:"; tail -20 "$OPENCAP_LOGDIR/sync.log"; exit 1
fi

echo "== patch =="
cd "$OPENCAP_DIR" || exit 1
python3 -u scripts/patch_seeding.py       "$CAPX_DIR" || exit 1
python3 -u scripts/patch_exec_globals.py  "$CAPX_DIR" || exit 1
python3 -u scripts/patch_prompt_fix.py    "$CAPX_DIR" || exit 1

echo "== verify =="
python3 -u scripts/verify_configs.py  "$CAPX_DIR" || exit 1
python3 -u scripts/verify_controls.py "$CAPX_DIR" || exit 1
# The precondition every paired statistic rests on. Not optional, and it is
# cheap -- no LLM, no GPU to speak of. It fails loudly on an unpatched
# checkout (60 observation fields differ across same-seed resets), which is
# what makes it a real check rather than a reassuring one.
python3 -u scripts/verify_pairing_states.py "$CAPX_DIR" 5 || exit 1

echo "== tests =="
python3 -m unittest discover -s tests -q || exit 1

nvidia-smi -i "$OPENCAP_GPU" --query-gpu=index,name,memory.total --format=csv,noheader
echo "BOOTSTRAP_OK"
