#!/usr/bin/env bash
# Fetch the all-MiniLM-L6-v2 ONNX export that burn-import compiles at build time.
# The file is large (~86 MB) and gitignored, so each machine fetches it once.
set -euo pipefail

DEST="$(dirname "$0")/../crates/embed-burn/artifacts/model.onnx"
URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"

mkdir -p "$(dirname "$DEST")"
if [ -f "$DEST" ]; then
    echo "already present: $DEST"
    exit 0
fi
echo "downloading $URL"
if command -v curl >/dev/null 2>&1; then
    curl -fSL -o "$DEST" "$URL"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$DEST" "$URL"
else
    echo "error: need curl or wget to fetch the model" >&2
    exit 1
fi
echo "saved to $DEST"
