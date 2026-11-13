#!/usr/bin/env bash
# Bare-metal server environment (8x H200). Source this, do not execute it:
#
#     source scripts/env_server.sh
#
# The Colab port is entirely a matter of these variables -- see HANDOFF §6.2.
# Nothing below changes a frozen control; the controls live in opencap/config.py.

# --- Paths (§6.2). /content does not exist here. --------------------------
export OPENCAP_DIR="${OPENCAP_DIR:-/data/code/playground/capx-scaling}"
export CAPX_DIR="${CAPX_DIR:-/data/cap-x}"
export OPENCAP_LOGDIR="${OPENCAP_LOGDIR:-/data/capx-logs}"
# OPENCAP_SCRATCH is deliberately NOT set here: it must differ per arm. Two arms
# sharing one scratch directory made the parse read a mixture of both (§8).
# Each launch sets its own.

# --- GPU ------------------------------------------------------------------
# This box is shared. We own GPU 2 and nothing else; GPUs 0/3/4/5 carry other
# people's jobs. run_sweep.sh turns this into CUDA_VISIBLE_DEVICES *and* into
# the `-i` argument of every nvidia-smi query, which is the part that does not
# follow automatically -- NVML ignores CUDA_VISIBLE_DEVICES.
export OPENCAP_GPU="${OPENCAP_GPU:-2}"
# Pin HERE, not only in run_sweep.sh. Gate 0 and the repair verifier launch
# the simulator and the IK server directly, never through run_sweep.sh, so
# with only OPENCAP_GPU set their CUDA work landed on physical GPU 0 -- another
# user's card. Everything that sources this file is now pinned.
export CUDA_VISIBLE_DEVICES="$OPENCAP_GPU"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-$OPENCAP_GPU}"

# --- Interpreters ---------------------------------------------------------
# Three environments, deliberately separate:
#   this shell's python3  -- the harness (huggingface_hub, zstandard)
#   OPENCAP_VLLM_PYTHON     -- the engine only
#   cap-x/.venv (uv run)  -- the simulator only
# cap-x pins vLLM 0.15.0/0.11.0 behind its molmo/verl extras, so installing an
# engine into its venv would corrupt the sim environment.
#
# vLLM is pinned to 0.28.0: that is the version the Colab corpus was served
# with (§6.1). The host-effect gate is meant to measure the HOST, so the
# serving stack is held fixed across it rather than upgraded at the same time.
export OPENCAP_VLLM_PYTHON="${OPENCAP_VLLM_PYTHON:-/data/venvs/vllm028/bin/python}"

# --- Caches (root filesystem has ~56GB free; /shared has terabytes) --------
# Set UNCONDITIONALLY, not with `:-`. This box exports a system-wide
# HF_HOME=/shared/huggingface, a cache shared with other users: its top level
# is world-writable, but each model directory is owned by whoever pulled it
# first. models--Qwen--Qwen3-8B is owned by another account, so a download
# into it dies with EACCES partway through -- after the readiness gate has
# already started its clock. A `:-` default would silently inherit that.
export HF_HOME=/data/hf_cache
# That path is the POD's. Sourced on a workstation it is unwritable, and the
# failure surfaces twenty frames deep as
#   OSError: [Errno 30] Read-only file system: '/shared'
# from inside hf_hub_download, which reads like a corrupt store rather than a
# misconfigured cache. Say so once, here, and carry on: overriding it would
# change where the pod caches 51 GB of weights.
if [ ! -w "$(dirname "$HF_HOME")" ] && [ ! -w "$HF_HOME" ] 2>/dev/null; then
  echo "[env] NOTE: HF_HOME=$HF_HOME is not writable here (this is the pod's"
  echo "[env] path). Any hub DOWNLOAD will fail with 'Read-only file system'."
  echo "[env] For local analysis, set HF_HOME=~/.cache/huggingface first, or"
  echo "[env] run the report directly without sourcing this file -- the Python"
  echo "[env] entry points load .env on their own (opencap/config.py)."
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-/data/uv-cache}"
# huggingface_hub 1.x pulls large files over Xet by default. On this box that
# transport stalls mid-file and never resumes -- Qwen3-8B's third shard sat at
# 3.12 of 3.96 GB with the process alive and no bytes moving for 5 minutes,
# and a restarted download stuck at 0 bytes. The plain HTTP path completes the
# same file in about a minute. A stalled weight download is not harmless: it
# burns the engine-readiness window and the sweep reports a timeout.
export HF_HUB_DISABLE_XET=1

# --- Token: read from a file, never passed on a command line (§0.2) -------
# Two sources, hf_token.txt first for the pod, then .env for a workstation.
# An already-exported HF_TOKEN always wins: a shell that set one deliberately
# must not be overridden.
#
# .env was added 2026-10-27. Before it, the local token had been rotated,
# huggingface_hub silently fell back to a stale cached one, and
# condition_report.py printed "not collected (store absent)" for an ablation
# that was 70/70 complete -- the hub answers 401 identically for "private, not
# yours" and "does not exist", so a missing credential turns every private
# store into a null result. opencap/config.py loads the same file for Python
# entry points that never source this script.
if [ -z "${HF_TOKEN:-}" ] && [ -r "$OPENCAP_DIR/hf_token.txt" ]; then
  HF_TOKEN="$(tr -d '\n\r ' < "$OPENCAP_DIR/hf_token.txt")"
  export HF_TOKEN
fi
# OPENCAP_DIR defaults to the POD's checkout path, so on a workstation it names
# a directory that does not exist. Fall back to this script's own parent, which
# is right in both places.
_OPENCAP_SELF="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2>/dev/null && pwd)"
for _ENVF in "$OPENCAP_DIR/.env" "${_OPENCAP_SELF:-.}/.env"; do
  [ -n "${HF_TOKEN:-}" ] && break
  [ -r "$_ENVF" ] || continue
  # Only HF_TOKEN, and only the value: no `source`, so a stray command in
  # .env is not executed by every script that sets up an environment.
  HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' "$_ENVF" \
              | head -1 | tr -d '\n\r "'"'"'')"
  [ -n "$HF_TOKEN" ] && export HF_TOKEN
done
unset _ENVF _OPENCAP_SELF

# --- Serving ---------------------------------------------------------------
export PORT="${PORT:-8110}"
# /tmp is shared with other users on this box; keep our own lock.
export OPENCAP_PILOT_LOCK="${OPENCAP_PILOT_LOCK:-$OPENCAP_LOGDIR/opencap_pilot.lock}"

# --- Frozen determinism controls, restated so a launch log shows them -----
export OPENCAP_BATCH_MAX="${OPENCAP_BATCH_MAX:-1}"

# Serving pool: 0.90 of 143GB. This was 0.40 while GPU 2 was shared with
# another account's job; 57GB cannot hold the 27B/32B rungs (71-84GB at
# bf16), and a multi-day detached run must not find that out at the 32B
# rung. 0.90 is also the value the corpus was collected under.
#
# 0.90 exists because a 40GB A100 had to be squeezed to fit qwen3-14b bf16 at
# a 32K context. Here the same rung needs ~33GB of a 143GB card, so 0.90 would
# reserve 96GB nobody uses -- on a box where another account's job appeared on
# this very card twice while we were setting up. Holding a pool that large for
# hours would OOM them; being pushed out by them would kill the sweep.
#
# This is NOT one of the frozen controls (those live in opencap/config.py) and
# it does not touch the numbers: the pool sets KV cache size, which bounds
# concurrency, and OPENCAP_BATCH_MAX=1 means exactly one sequence is ever in
# flight. 57GB still leaves every S1 rung a KV cache larger than the A100
# baseline had in total.
export OPENCAP_GPU_UTIL="${OPENCAP_GPU_UTIL:-0.90}"
# OPENCAP_WORKERS is left at run_sweep.sh's default of min(nproc, 5). This box
# has 240 vCPUs and could run far more sim workers, but the reference corpus
# was collected at 5 (Colab A100, nproc=12) and every new cell here is paired
# against it. Throughput is not worth another difference in the comparison.

mkdir -p "$OPENCAP_LOGDIR"
echo "[env] gpu=$OPENCAP_GPU capx=$CAPX_DIR logs=$OPENCAP_LOGDIR"
echo "[env] vllm python=$OPENCAP_VLLM_PYTHON"
# Report presence ONLY. `${HF_TOKEN:-MISSING}` prints the token when it is
# set -- this line leaked one into a log exactly once, which is once too many.
if [ -n "${HF_TOKEN:-}" ]; then echo "[env] hf token: present"; else echo "[env] hf token: MISSING"; fi
