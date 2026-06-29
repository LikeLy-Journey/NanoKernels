#!/usr/bin/env bash
# Sync the NanoKernels repo from local Mac to the remote A100 node.
# Remote has no rsync, so we use a tar-over-SSH pipe (only needs tar both ends).
# Usage: scripts/sync.sh
set -euo pipefail

# macOS: don't emit AppleDouble (._*) files into the tarball
export COPYFILE_DISABLE=1

# --- config ---
SOCK="${SSH_CTL_SOCK:-$HOME/.mcp-ctl-366765122.sock}"
HOST="${REMOTE_HOST:-366765122}"
REMOTE_DIR="${REMOTE_DIR:-/mnt/bn/jianglielin-yg/codes/NanoKernels_v0.1}"

# repo root = parent of this script's dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "[sync] local : $LOCAL_DIR"
echo "[sync] remote: $HOST:$REMOTE_DIR"

# tar the repo (excluding build/profile artifacts) and unpack on the remote
tar czf - -C "$LOCAL_DIR" \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.o' \
  --exclude 'build' \
  --exclude 'vadd_cpu' \
  --exclude 'vadd_cuda' \
  --exclude '*.ncu-rep' \
  --exclude '*.nsys-rep' \
  --exclude '*.sqlite' \
  . \
| ssh -o ControlPath="$SOCK" "$HOST" "mkdir -p '$REMOTE_DIR' && tar xzf - -C '$REMOTE_DIR' --warning=no-unknown-keyword"

echo "[sync] done."
