#!/bin/bash
# Wait until all 4 pentagonbox regression arms have written reduction.pkl,
# then run the reference comparison. Polls every 2 min; also exits (with the
# current partial comparison) if any orchestrator dies without producing output.
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
ARMS="gate_small_regress m1_regress m2_regress m3_regress"
while true; do
    done_n=0
    alive=0
    for t in $ARMS; do
        [ -f $BASE/results/ab_symmetry/$t/design1/reduction.pkl ] && done_n=$((done_n+1))
        pgrep -f "output $BASE/results/ab_symmetry/$t/design1/reduction.pkl" > /dev/null && alive=$((alive+1))
    done
    [ $done_n -eq 4 ] && break
    if [ $alive -eq 0 ] && [ $done_n -lt 4 ]; then
        echo "WARNING: no orchestrators alive but only $done_n/4 outputs present"
        break
    fi
    sleep 120
done
PYTHONUNBUFFERED=1 $PY $BASE/reduction/cmp_regress_probes.py
