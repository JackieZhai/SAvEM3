#!/usr/bin/env bash
# Create AC4 teacher-feature symlinks for SAvEM3 training.
# Teacher H5 files are not tracked in Git; place them in data/AC3-AC4_new/ first.
# The SAvEM3 config reads from data/AC3-AC4/ by default.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/data/AC3-AC4_new"
DST="$ROOT/data/AC3-AC4"
mkdir -p "$DST"
for name in AC4_features.h5 AC4_embeddings.h5; do
    if [ -f "$SRC/$name" ]; then
        ln -sfn "../AC3-AC4_new/$name" "$DST/$name"
        echo "link $name"
    else
        echo "skip $name (not found: $SRC/$name)" >&2
    fi
done
