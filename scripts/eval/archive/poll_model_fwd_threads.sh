#!/bin/bash
# Wait for the n-threads=4 profiled 74 to reach step 40, then compare the
# model_fwd bucket (avg over steps 28-40) against the n-threads=1 run.
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
T1=$BASE/results/probe_74_v7_prof/probe.out      # n-threads 1 (already done)
T4=$BASE/results/probe_74_v7_prof_t4/probe.out   # n-threads 4
LOG=$BASE/results/probe_74_v7_prof_t4/model_fwd_threads.txt

for i in $(seq 1 160); do
    n=$(grep -c "PROF step=" "$T4" 2>/dev/null || echo 0)
    if [ "$n" -ge 41 ]; then break; fi
    sleep 15
done

# avg model_fwd + t_step over steps 28-40
avg() {
  awk '
    /^\[v6 step/ { for(i=1;i<=NF;i++) if($i~/t_step=/){split($i,a,"=");gsub("s","",a[2]);cur=a[2]+0} ; st=$3+0 }
    /PROF step=/ { split($2,a,"="); ps=a[2]+0; inr=(ps>=28 && ps<=40); if(inr){n++; ts+=cur} }
    /model_fwd=/ && inr { for(i=1;i<=NF;i++) if($i~/model_fwd=/){split($i,a,"=");mf+=a[2]+0} }
    END{ if(n==0){print "n=0"; exit} printf "n=%d  t_step=%.2fs  model_fwd=%.2fs (%.0f%% of step)\n", n, ts/n, mf/n, 100*mf/ts }' "$1"
}

{
  echo "=== model_fwd scaling: n-threads 1 vs 4 (74, steps 28-40) ==="
  echo "n-threads=1 : $(avg "$T1")"
  echo "n-threads=4 : $(avg "$T4")"
} > "$LOG" 2>&1
cat "$LOG"
echo "DONE poll_model_fwd_threads"
