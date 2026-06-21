#!/bin/bash
# Lightweight watcher: every 60s, condor_release any HELD round3 worker job.
# The 8-core affinity pin confines each job to 8 cores, but at large scale a rare
# (~1%) transient during the 8-way enumerate fork-spawn can momentarily read >8
# CPU and Condor holds the job. Releasing re-runs it (the transient almost never
# recurs), so no target is stranded and the orchestrator eventually reaps it.
# Runs until the orchestrator finishes (reduction.pkl appears) or it is killed.
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
R=$BASE/results/pentagonbox_8_5_v6_round3
LOG=$R/logs/release_watcher.log
CONSTRAINT='regexp("pentagonbox_8_5_v6_round3", Args) && JobStatus == 5'
echo "[release-watcher] start $(date)" >> $LOG
while true; do
  # list held round3 jobs (cluster + cpu-hold reason) before releasing
  HELD=$(condor_q dshih -constraint "$CONSTRAINT" -af ClusterId HoldReason 2>/dev/null)
  if [ -n "$HELD" ]; then
    echo "[$(date +%H:%M:%S)] releasing held round3 jobs:" >> $LOG
    echo "$HELD" >> $LOG
    condor_release -constraint "$CONSTRAINT" >> $LOG 2>&1
  fi
  # stop when the reduction is done
  if [ -f $R/reduction.pkl ]; then
    echo "[release-watcher] reduction.pkl present — exiting $(date)" >> $LOG
    break
  fi
  sleep 60
done
