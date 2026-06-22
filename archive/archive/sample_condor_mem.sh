#!/bin/bash
# Sample a single job's RSS from condor_q every N seconds, log to file.
# Usage: sample_condor_mem.sh <cluster_id> <out_file> [interval_sec=120]
set -e
CLUSTER=$1
OUT=$2
INTERVAL=${3:-120}

echo "# epoch cluster size_mb" > $OUT
while true; do
    # condor_q output: pos 8 is SIZE in MB
    LINE=$(condor_q -nobatch $CLUSTER 2>/dev/null | grep "^$CLUSTER" || echo "")
    if [ -z "$LINE" ]; then
        echo "$(date +%s) $CLUSTER DONE" >> $OUT
        break
    fi
    SIZE=$(echo "$LINE" | awk '{print $8}')
    echo "$(date +%s) $CLUSTER $SIZE" >> $OUT
    sleep $INTERVAL
done
