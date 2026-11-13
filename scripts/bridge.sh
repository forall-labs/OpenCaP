#!/usr/bin/env bash
# Topology T2: two Colab instances that must talk to each other.
#
# Colab gives no inbound ports, so both instances tunnel out to this machine
# via colab-gpu-bridge (SSH over cloudflared). This machine is the rendezvous
# hub: -L pulls A's vLLM port here, -R pushes it into B.
#
#     Colab A (vLLM :8000) --> local :8000 --> Colab B (localhost:8000)
#
# B then points OPENCAP_AGENT_LLM at its own localhost and needs no other change.
#
# Usage: scripts/bridge.sh <colabA-host> <colabB-host> [PORT]
set -euo pipefail

A="${1:?usage: bridge.sh COLAB_A COLAB_B [PORT]}"
B="${2:?usage: bridge.sh COLAB_A COLAB_B [PORT]}"
PORT="${3:-8000}"

echo "[bridge] pulling ${A}:${PORT} to local:${PORT}"
ssh -N -L "${PORT}:127.0.0.1:${PORT}" "$A" &
A_PID=$!
trap 'kill $A_PID 2>/dev/null || true' EXIT
sleep 2

echo "[bridge] pushing local:${PORT} into ${B}:${PORT}"
echo "[bridge] on ${B}, set: export OPENCAP_AGENT_LLM=http://127.0.0.1:${PORT}/v1"
ssh -N -R "${PORT}:127.0.0.1:${PORT}" "$B"
