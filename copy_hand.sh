#!/bin/bash
# Apply hxzd5568's changes (casync engine + two-stage load + state_dict_factory)
# by direct copy from a local, non-git DeepSpeed tree (e.g. extracted from a zip).
#
# Usage:
#   bash apply_hxzd5568_copy.sh                          # auto-detect src via pip-installed location? no -- src must be given
#   bash apply_hxzd5568_copy.sh /path/to/DeepSpeed-develop /path/to/site-packages
#
# Notes:
#   - SRC must contain a top-level `deepspeed/` directory (the zip's extracted root).
#   - TGT must be the site-packages directory that contains `deepspeed/`.
#   - Files are backed up as <file>.bak.<timestamp> before being overwritten.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default SRC: one level up from scripts/ (i.e. repo/extracted zip root)
DEFAULT_SRC="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ $# -ge 1 ]; then
    SRC_ROOT="$1"
else
    SRC_ROOT="$DEFAULT_SRC"
fi

if [ $# -ge 2 ]; then
    SITE_PACKAGES="$2"
else
    SITE_PACKAGES=$(python3 -c "import deepspeed, os; print(os.path.dirname(os.path.dirname(deepspeed.__file__)))" 2>/dev/null || true)
    if [ -z "$SITE_PACKAGES" ]; then
        echo "ERROR: cannot locate installed deepspeed. Run: bash $0 <src_root> /path/to/site-packages"
        exit 1
    fi
fi

# --- sanity checks ---
if [ ! -d "$SRC_ROOT/deepspeed" ]; then
    echo "ERROR: $SRC_ROOT/deepspeed not found. Pass the extracted DeepSpeed source root as arg 1."
    exit 1
fi
if [ ! -d "$SITE_PACKAGES/deepspeed" ]; then
    echo "ERROR: $SITE_PACKAGES/deepspeed not found. Pass the site-packages dir as arg 2."
    exit 1
fi

echo "[INFO] source     : $SRC_ROOT/deepspeed"
echo "[INFO] destination: $SITE_PACKAGES/deepspeed"

# --- files to sync (relative to deepspeed/) ---
FILES=(
    "datastates/config.py"
    "runtime/checkpoint_engine/casync_checkpoint_engine.py"
    "runtime/checkpoint_engine/checkpoint_engine.py"
    "runtime/checkpoint_engine/datastates_checkpoint_engine.py"
    "runtime/checkpoint_engine/torch_checkpoint_engine.py"
    "runtime/checkpoint_engine/utils.py"
    "runtime/engine.py"
    "runtime/state_dict_factory.py"
)

STAMP=$(date +%Y%m%d_%H%M%S)

# --- preflight: verify all source files exist ---
missing=0
for rel in "${FILES[@]}"; do
    if [ ! -f "$SRC_ROOT/deepspeed/$rel" ]; then
        echo "[MISSING] $SRC_ROOT/deepspeed/$rel"
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo "ERROR: some source files are missing; aborting."
    exit 1
fi

# --- backup + copy ---
for rel in "${FILES[@]}"; do
    src="$SRC_ROOT/deepspeed/$rel"
    dst="$SITE_PACKAGES/deepspeed/$rel"
    dst_dir="$(dirname "$dst")"
    mkdir -p "$dst_dir"

    if [ -f "$dst" ]; then
        cp -p "$dst" "$dst.bak.$STAMP"
        echo "[BACKUP] $dst -> $dst.bak.$STAMP"
    else
        echo "[NEW]    $dst"
    fi
    cp -p "$src" "$dst"
    echo "[COPY]   $rel"
done

echo "[INFO] done. ${#FILES[@]} files synced."
echo "       To roll back: for f in $SITE_PACKAGES/deepspeed/**/*.bak.$STAMP; do mv \"\$f\" \"\${f%.bak.$STAMP}\"; done"
