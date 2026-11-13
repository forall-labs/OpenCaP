#!/usr/bin/env bash
# Save/restore the build caches so a reaped runtime costs ~3 min, not ~12.
#
# Colab reaps runtimes aggressively -- observed lifetimes of 30-60 min against
# a 12-minute bootstrap, so a quarter to a third of every session is rebuild.
# Over a 500-cell Wave 1 that is the single largest avoidable cost.
#
# We cache the PACKAGE CACHES (~/.cache/uv, ~/.cache/pip) and the git checkout,
# not the installed venvs. Venvs embed absolute paths and are fragile to move;
# a warm package cache makes `uv sync` and `pip install` mostly no-ops while
# staying a supported code path.
#
# Transfer goes via HF (the runtime pulls at ~89 MB/s) rather than through the
# local machine (~19 MB/s via the contents API).
#
#   scripts/env_cache.sh save    <hf-repo>
#   scripts/env_cache.sh restore <hf-repo>
set -uo pipefail

ACTION="${1:?usage: env_cache.sh save|restore HF_REPO}"
REPO="${2:?}"
CAPX="${CAPX_DIR:-/content/cap-x}"
TAR=/content/envcache.tar.zst

: "${HF_TOKEN:?export HF_TOKEN first (Colab: google.colab.userdata)}"

case "$ACTION" in
  save)
    echo "[cache] packing uv/pip caches + cap-x checkout"
    tar --zstd -cf "$TAR" \
        -C / \
        --exclude='*/.venv' \
        --exclude='*/outputs' \
        root/.cache/uv root/.cache/pip \
        "${CAPX#/}" 2>/dev/null
    ls -lh "$TAR"
    python3 - "$REPO" "$TAR" <<'PY'
import sys
from huggingface_hub import HfApi
repo, path = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_file(path_or_fileobj=path, path_in_repo="envcache.tar.zst",
                repo_id=repo, repo_type="dataset",
                commit_message="env cache")
print("[cache] uploaded")
PY
    ;;
  restore)
    echo "[cache] fetching"
    python3 - "$REPO" "$TAR" <<'PY'
import shutil, sys
from huggingface_hub import hf_hub_download
repo, dest = sys.argv[1], sys.argv[2]
try:
    p = hf_hub_download(repo, "envcache.tar.zst", repo_type="dataset")
    shutil.copy(p, dest)
    print("[cache] downloaded")
except Exception as exc:                              # noqa: BLE001
    print(f"[cache] MISS ({exc.__class__.__name__}) -- falling back to a cold build")
    raise SystemExit(3)
PY
    rc=$?
    [ $rc -ne 0 ] && exit $rc
    tar --zstd -xf "$TAR" -C / && echo "[cache] restored; run bootstrap_colab.sh now (it will hit warm caches)"
    ;;
  *) echo "unknown action $ACTION"; exit 2;;
esac
