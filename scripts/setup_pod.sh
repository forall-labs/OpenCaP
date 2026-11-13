#!/usr/bin/env bash
# Bootstrap a pod (2x H100 box, or a Colab instance) for Phase 0.
set -euo pipefail

REPO_DIR="${OPENCAP_DIR:-$HOME/capx-scaling}"
CAPX_DIR="${CAPX_DIR:-$HOME/cap-x}"

# LIFECYCLE WARNING: /content dies with the Colab runtime, so `colabctl down`
# destroys this checkout and its venv. Rebuilding costs ~10 minutes. Keep the
# runtime up for a whole working session; only `down` at the end.

echo "== deps =="
pip install -q -U "huggingface_hub[hf_transfer]" zstandard pyarrow vllm
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "== CaP-X =="
# Do NOT use --recurse-submodules: that pulls b1k (BEHAVIOR-1K) and the LIBERO
# tree, neither of which this project uses, and it is slow and large.
#
# pyproject.toml [tool.uv.sources] makes four submodules UNCONDITIONAL deps --
# sam3, contact_graspnet_pytorch, verl, curobo -- so `uv sync` fails without
# them even though sam3/graspnet are only *used* in Wave 2. robosuite sits
# behind its extra. b1k (BEHAVIOR-1K) is the one we can genuinely skip.
# uv resolves the whole lockfile, so an editable path must EXIST even when its
# extra is not being installed. That makes every path in [tool.uv.sources]
# mandatory at sync time, LIBERO included -- discovered the hard way.
REQUIRED_SUBMODULES="
capx/third_party/sam3
capx/third_party/contact_graspnet_pytorch
capx/third_party/verl
capx/third_party/curobo
capx/third_party/robosuite
capx/third_party/LIBERO-PRO
capx/third_party/libero_dependencies/robosuite
"
if [ ! -d "$CAPX_DIR/.git" ]; then
  git clone https://github.com/capgym/cap-x "$CAPX_DIR"
fi
cd "$CAPX_DIR"
for m in $REQUIRED_SUBMODULES; do
  git submodule update --init --depth 1 "$m"
done

# uv picks up Colab's CPython 3.10 automatically; cap-x requires 3.10.
# NOTE: never pipe uv sync into tail without pipefail -- the pipe swallows the
# exit code and a failed sync reports success.
set -o pipefail
uv sync --extra robosuite

echo "== token =="
: "${HF_TOKEN:?export HF_TOKEN (on Colab: google.colab.userdata, never inline in a cell)}"
python3 - <<'PY'
from huggingface_hub import HfApi
print("[hf] authenticated as", HfApi().whoami()["name"])
PY

echo "== sanity =="
cd "$REPO_DIR" && python3 -m unittest discover -s tests -q
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "[ok] pod ready"
