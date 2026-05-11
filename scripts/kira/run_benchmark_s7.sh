#!/bin/bash
# Run Kira on the benchmark integrals for s=7 (the most memory-intensive ones).
# Requires:
#   KIRA       -- path to the Kira binary (e.g. /opt/kira/bin/kira)
#   FERMATPATH -- path to the Fermat binary
# Optionally:
#   WORKDIR    -- directory containing jobs_benchmark.yaml + integrals-*.* files
#                 (default: this script's directory)
set -euo pipefail

if [[ -z "${KIRA:-}" || -z "${FERMATPATH:-}" ]]; then
    echo "ERROR: set KIRA and FERMATPATH env vars before running." >&2
    echo "  e.g. export KIRA=/opt/kira/bin/kira" >&2
    echo "       export FERMATPATH=/opt/fermat/fer64" >&2
    exit 1
fi
export FERMATPATH

WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$WORKDIR"

# Run each s=7 integral. The yaml is identical between runs except for the
# integrals-* input file it points to.
for spec in 10.7.4 11.7.5 12.7.6 13.7.7; do
    if [[ ! -f "integrals-${spec}" ]]; then
        echo "WARNING: integrals-${spec} not found, skipping" >&2
        continue
    fi
    echo "=== Running Kira on integrals-${spec} ==="
    cp -f "integrals-${spec}" integrals
    /usr/bin/time -v "$KIRA" --parallel=1 jobs_benchmark.yaml 2>&1 | tee "kira_bench_${spec}.log"
    rm -f integrals
done

echo "Done. Logs in $WORKDIR/kira_bench_*.log"
