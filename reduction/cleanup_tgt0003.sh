#!/bin/bash
# Remove ONLY the prematurely-launched tgt0003 reduction (meta false-positive).
# Surgical: the tgt0003 orchestrator PID (guarded by its unique integral) and its
# Condor workers (identified by '/tgt0003_w7_6/' in their --output path). The
# (7,6) run pentagonbox_7_6_cache_from_85_75 and everything else are untouched.
set -x
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2

# 0. confirm the buggy meta is not running (it already exited)
ps -p 2541926 -o pid,cmd --no-headers 2>/dev/null && echo "WARN: meta 2541926 still alive" || echo "meta 2541926 confirmed gone"

# 1. kill the tgt0003 orchestrator -- ONLY if PID 2603189 really is tgt0003
if ps -p 2603189 -o cmd --no-headers 2>/dev/null | grep -q -- '--integral 1,1,1,1,1,1,1,-1,-3,-1,-1'; then
    kill 2603189
    echo "killed tgt0003 orchestrator 2603189"
else
    echo "ABORT: PID 2603189 is not the tgt0003 orchestrator"
fi
sleep 3

# 2. condor_rm ONLY tgt0003 workers (grep on the tgt0003 output path; cannot match (7,6))
IDS=$(condor_q -nobatch 2>/dev/null | grep onestep_worker_v7 | grep '/tgt0003_w7_6/work/results/' | awk '{print $1}')
echo "tgt0003 workers to remove: $(echo $IDS | wc -w)"
if [ -n "$IDS" ]; then condor_rm $IDS; fi
sleep 3

# 3. delete the run dir (incl. target.txt, so a future meta re-queues rank 3 cleanly)
rm -rf $BASE/results/meta_reduce/tgt0003_w7_6

# 4. verify aftermath
echo "=== (7,6) orchestrator still alive (MUST be yes) ==="
ps -p 2350902 -o pid,etime --no-headers 2>&1
echo "=== remaining SAILIR workers by run dir ==="
condor_q -nobatch 2>/dev/null | grep onestep_worker_v7 | grep -oP '/\K[^/ ]+(?=/work/results/)' | sort | uniq -c
echo "=== tgt0003 dir gone? ==="
ls -d $BASE/results/meta_reduce/tgt0003_w7_6 2>&1
