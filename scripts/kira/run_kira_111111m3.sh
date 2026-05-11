#!/bin/bash
# Reduce the single integral I[1,1,1,1,1,1,-3] with Kira -- useful for a quick
# end-to-end smoke test.
# Requires KIRA and FERMATPATH (see run_benchmark_s7.sh).
set -euo pipefail

if [[ -z "${KIRA:-}" || -z "${FERMATPATH:-}" ]]; then
    echo "ERROR: set KIRA and FERMATPATH env vars before running." >&2
    exit 1
fi
export FERMATPATH

WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$WORKDIR"
"$KIRA" --parallel=1 jobs_111111m3.yaml
