#!/bin/bash
# Poll the profiled 74 run until it reaches step 40, then dump the PROF
# breakdown lines for steps 30-40 (the depth where t_step is large).
OUT=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/probe_74_v7_prof/probe.out
LOG=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/probe_74_v7_prof/poll_summary.txt
for i in $(seq 1 120); do
    n=$(grep -c "PROF step=" "$OUT" 2>/dev/null || echo 0)
    if [ "$n" -ge 40 ]; then
        break
    fi
    sleep 15
done
{
    echo "=== reached $(grep -c 'PROF step=' "$OUT") PROF steps ==="
    echo "=== per-step PROF breakdown, steps 28-40 ==="
    awk '/^\[v6 step/ {ln=$0} /PROF step=/{split($2,a,"="); s=a[2]+0; if(s>=28 && s<=40){print ln; print} } /^    P1:|^    P2:|^    P3:/{if(s>=28 && s<=40)print}' "$OUT"
} > "$LOG" 2>&1
echo "DONE poll_74_prof"
