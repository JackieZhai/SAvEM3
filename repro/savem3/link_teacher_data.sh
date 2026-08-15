#!/usr/bin/env bash
# 为 SAvEM3 训练准备 AC4 教师特征符号链接。
# 教师特征 h5 不入 git；请先放到 data/AC3-AC4_new/。
# savem3 config 默认从 data/AC3-AC4/ 读取。
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
        echo "skip $name（未找到 $SRC/$name）" >&2
    fi
done
