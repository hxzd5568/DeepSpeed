#!/bin/bash
# Apply hxzd5568's 6 commits (casync engine + fixes) to an installed deepspeed
#
# Usage:
#   bash apply_hxzd5568_patches.sh                          # auto-detect from pip
#   bash apply_hxzd5568_patches.sh /path/to/deepspeed/pkg   # specify site-packages dir
#
# The script requires the DeepSpeed git repo to generate the patch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- determine installed deepspeed location ---
if [ $# -ge 1 ]; then
    SITE_PACKAGES="$1"
else
    SITE_PACKAGES=$(python3 -c "import deepspeed, os; print(os.path.dirname(os.path.dirname(deepspeed.__file__)))" 2>/dev/null)
    if [ -z "$SITE_PACKAGES" ]; then
        echo "ERROR: cannot locate installed deepspeed. Run: bash $0 /path/to/site-packages"
        exit 1
    fi
fi

echo "[INFO] deepspeed installed at: $SITE_PACKAGES"

# --- generate patch from git repo ---
BASE_COMMIT=$(git -C "$REPO_ROOT" rev-parse cde7b599^)
TARGET_COMMIT=632f16c6
PATCH_FILE=$(mktemp -t deepspeed_hxzd5568_patch.XXXXXX)

echo "[INFO] generating patch from $BASE_COMMIT..$TARGET_COMMIT ..."
git -C "$REPO_ROOT" diff "$BASE_COMMIT".."$TARGET_COMMIT" > "$PATCH_FILE"

echo "[INFO] applying patch (dry-run first)..."
# dry-run
if patch -d "$SITE_PACKAGES" -p1 --dry-run < "$PATCH_FILE"; then
    echo "[INFO] dry-run OK, applying..."
    patch -d "$SITE_PACKAGES" -p1 < "$PATCH_FILE"
    echo "[INFO] done. 5 files patched:"
    echo "        deepspeed/datastates/config.py"
    echo "        deepspeed/runtime/checkpoint_engine/casync_checkpoint_engine.py (new)"
    echo "        deepspeed/runtime/checkpoint_engine/datastates_checkpoint_engine.py"
    echo "        deepspeed/runtime/checkpoint_engine/torch_checkpoint_engine.py"
    echo "        deepspeed/runtime/checkpoint_engine/utils.py"
else
    echo "[ERROR] dry-run failed. Installed deepspeed may already be patched or incompatible."
    rm -f "$PATCH_FILE"
    exit 1
fi

rm -f "$PATCH_FILE"
