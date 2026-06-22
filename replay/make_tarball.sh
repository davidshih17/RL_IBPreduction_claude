#!/bin/bash
# Build the distributable tarball for the standalone pentagonbox (8,5) replay.
# Ships ONLY the user-facing files (program + two data files + README) inside a
# single top-level directory so it untars cleanly:
#
#   sailir_pentagonbox_85_replay/
#       replay.py
#       reduction_cache.pkl
#       topology_pentagonbox.json
#       README.md
#
# Dev-only files (build_bundle.py, _recon_inspect.py, _crosscheck_vs_gold.py,
# logs/, this script) are deliberately excluded.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PKG=sailir_pentagonbox_85_replay
STAGE="$HERE/$PKG"
TARBALL="$HERE/${PKG}.tar.gz"

rm -rf "$STAGE" "$TARBALL"
mkdir -p "$STAGE"
cp "$HERE/replay.py" "$HERE/reduction_cache.pkl" \
   "$HERE/topology_pentagonbox.json" "$HERE/README.md" "$STAGE/"
chmod +x "$STAGE/replay.py"   # ship with the executable bit so ./replay.py works

# Deterministic, portable tar (sorted names, no per-user metadata noise).
tar --sort=name --owner=0 --group=0 --numeric-owner \
    -czf "$TARBALL" -C "$HERE" "$PKG"
rm -rf "$STAGE"

echo "built: $TARBALL"
ls -l "$TARBALL"
echo ""
echo "contents:"
tar -tzf "$TARBALL"
