#!/bin/bash
# Incrementally rebuild Kira after patching src/kira/trivial_sym.cpp with the
# KIRA_NO_SYMMETRY env-var guard in skip_symmetry(). Uses the existing
# build/build.ninja (flags + absolute conda-compiler path already baked in), so
# only trivial_sym.cpp is recompiled and kira is relinked. The new binary is
# backward-compatible (no env var => identical to the original).
set -e
KDIR=/het/p4/dshih/jet_images-deep_learning/IBPreduction/kira
export PATH=/cms/base/Miniconda/miniconda/bin:$PATH
cd "$KDIR"
echo "ninja: $(command -v ninja)"
echo "rebuilding kira (incremental) ..."
ninja -C build
echo "=== build done ==="
ls -l build/src/kira/kira
echo "=== install patched binary as kira-3.1 (orig backed up at kira-3.1.orig.bak) ==="
cp build/src/kira/kira "$KDIR/kira-3.1"
ls -l "$KDIR/kira-3.1"
echo "=== sanity: --version ==="
"$KDIR/kira-3.1" --version
