#!/bin/bash
# Poll until all listed A/B tags have BOTH baseline+design1 reduction.pkl, then exit
# (the harness re-invokes on exit). Args: TAGS... ; env TIMEOUT_ITERS (default 270 = 45min).
ROOT=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/ab_symmetry
TAGS="$@"
ITERS=${TIMEOUT_ITERS:-270}
for i in $(seq 1 $ITERS); do
    done=1
    for t in $TAGS; do
        if [ ! -f $ROOT/$t/baseline/reduction.pkl ] || [ ! -f $ROOT/$t/design1/reduction.pkl ]; then
            done=0
        fi
    done
    if [ $done -eq 1 ]; then
        echo "ALL DONE after $((i*10))s"
        exit 0
    fi
    sleep 10
done
echo "TIMEOUT after $((ITERS*10))s"
