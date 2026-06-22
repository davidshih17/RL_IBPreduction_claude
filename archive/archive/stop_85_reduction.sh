#!/bin/bash
# STOP the (8,5) reduction:
#   1. Snapshot all 45 currently-running (8,5) workers (cluster.proc + memory + integral)
#      to a manifest file BEFORE killing — so we can resubmit later.
#   2. SIGTERM the orchestrator (hierarchical_reduction.py) so it stops resubmitting.
#   3. SIGTERM the track_long_jobs.sh tracker if running.
#   4. condor_rm the 45 (8,5) cluster IDs (explicit list, NOT a username-wide rm).
#   5. Verify all gone (ps aux + condor_q).
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
OUT=$BASE/results/pentagonbox_8_5_delta/logs
TS=$(date +%Y%m%d_%H%M%S)
MANIFEST=$OUT/stop_manifest_${TS}.txt

echo "[$(date +%T)] === STOP (8,5) reduction ==="
echo "manifest: $MANIFEST"

# 1. Snapshot all currently-running (8,5) workers BEFORE killing
echo "[$(date +%T)] snapshotting running (8,5) workers..."
condor_q dshih -af ClusterId ProcId MemoryUsage RequestMemory JobStartDate Args 2>&1 \
  | awk -v now=$(date +%s) '
      /pentagonbox_8_5_delta/ {
        mem=$3; req=$4; start=$5
        integ = $0
        gsub(/.*--integral=./, "", integ); gsub(/. --output.*/, "", integ)
        name = $0
        gsub(/.*results\/async_/, "async_", name); gsub(/\.pkl .*/, "", name)
        runh = (start == "undefined") ? 0 : (now - start) / 3600.0
        printf "%s.%s\t%d\t%s\t%.2f\t%s\t%s\n", $1, $2, mem, req, runh, name, integ
      }' \
  | tee $MANIFEST > /dev/null
n=$(wc -l < $MANIFEST)
echo "  $n workers snapshotted to manifest"

# 2. Kill orchestrator
echo "[$(date +%T)] killing orchestrator (hierarchical_reduction.py)..."
ORCH_PIDS=$(ps -fu dshih 2>/dev/null | grep "hierarchical_reduction.*pentagonbox_8_5_delta" | grep -v grep | awk '{print $2}')
if [ -n "$ORCH_PIDS" ]; then
  echo "  orchestrator PIDs: $ORCH_PIDS"
  echo "$ORCH_PIDS" | xargs -r kill -TERM
  sleep 2
  # Force-kill if still alive
  STILL=$(ps -fu dshih 2>/dev/null | grep "hierarchical_reduction.*pentagonbox_8_5_delta" | grep -v grep | awk '{print $2}')
  if [ -n "$STILL" ]; then
    echo "  still alive after SIGTERM, sending SIGKILL: $STILL"
    echo "$STILL" | xargs -r kill -KILL
  fi
else
  echo "  no orchestrator running"
fi

# 3. Kill track_long_jobs.sh tracker if running
echo "[$(date +%T)] killing track_long_jobs.sh tracker..."
TRK_PIDS=$(ps -fu dshih 2>/dev/null | grep -E "track_long_jobs|long_jobs\.log" | grep -v grep | grep -v "stop_85_reduction" | awk '{print $2}')
if [ -n "$TRK_PIDS" ]; then
  echo "  tracker PIDs: $TRK_PIDS"
  echo "$TRK_PIDS" | xargs -r kill -TERM
fi

# 4. condor_rm by explicit cluster.proc list from manifest
echo "[$(date +%T)] condor_rm explicit cluster IDs..."
CLUSTERS=$(awk '{print $1}' $MANIFEST | tr '\n' ' ')
echo "  to remove: $CLUSTERS"
if [ -n "$CLUSTERS" ]; then
  condor_rm $CLUSTERS
fi

# 5. Verify
sleep 3
echo "[$(date +%T)] verifying..."
echo "--- ps aux orchestrator ---"
ps -fu dshih 2>/dev/null | grep -E "hierarchical_reduction|track_long_jobs" | grep -v grep || echo "  (none)"
echo "--- condor_q (8,5) workers ---"
REMAINING=$(condor_q dshih -af ClusterId Args 2>&1 | grep "pentagonbox_8_5_delta" | wc -l)
echo "  remaining (8,5) jobs in queue: $REMAINING"
echo
echo "[$(date +%T)] DONE. Manifest of killed workers: $MANIFEST"
