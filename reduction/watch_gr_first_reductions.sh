#!/bin/bash
# Wait for the 3 first gravity reductions; run the FIRE-oracle comparison when
# all have finished (or report partial state if orchestrators die).
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
ARMS="g885 g127 g893"
while true; do
    done_n=0; alive=0
    for t in $ARMS; do
        [ -f $BASE/results/gr_reduce/$t/reduction.pkl ] && done_n=$((done_n+1))
        pgrep -f "output $BASE/results/gr_reduce/$t/reduction.pkl" > /dev/null && alive=$((alive+1))
    done
    [ $done_n -eq 3 ] && break
    if [ $alive -eq 0 ] && [ $done_n -lt 3 ]; then
        echo "WARNING: no orchestrators alive but only $done_n/3 outputs present"
        break
    fi
    sleep 180
done
SAILIR_TOPOLOGY=gravity3L SAILIR_SECTOR_RANK=1 PYTHONUNBUFFERED=1 \
    $PY $BASE/reduction/cmp_gr_vs_oracle.py g885 g127 g893
