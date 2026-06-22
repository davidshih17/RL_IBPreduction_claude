#!/bin/bash
# Wait for the 74 streamA probe to write its first checkpoint (step 50), then
# (1) validate the streamed checkpoint is loadable, (2) confirm the run passed
# the checkpoint without crashing, (3) show MEMBD around step 50.
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
OUT=$BASE/results/probe_74_v7_streamA
LOG=$OUT/streamA_validation.txt

for i in $(seq 1 80); do
    # ckpt written at step 50; run must be at >=51 to prove it survived
    n=$(grep -oE "step +[0-9]+\] beam=" "$OUT/probe.out" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
    if [ -n "$n" ] && [ "$n" -ge 51 ] && [ -s "$OUT/ckpt.pkl" ]; then break; fi
    # bail if the job died
    if ! condor_q dshih -af ClusterId 2>/dev/null | grep -q 1657385 && [ ! -s "$OUT/ckpt.pkl" ]; then
        echo "job left queue before first ckpt" ; break
    fi
    sleep 15
done

{
    echo "=== 74 streamA: passed first checkpoint? ==="
    echo "last step: $(grep -E 'step +[0-9]+\] beam=' "$OUT/probe.out" 2>/dev/null | tail -1 | grep -oE 'step +[0-9]+')"
    echo "ckpt.pkl: $(ls -la "$OUT/ckpt.pkl" 2>/dev/null | awk '{print $5" bytes"}')"
    echo ""
    echo "=== streamed checkpoint validation ==="
    $PY $BASE/scripts/eval/validate_streamed_ckpt.py "$OUT/ckpt.pkl" 2>&1
    echo ""
    echo "=== MEMBD around the step-50 checkpoint (spike check) ==="
    grep -E "MEMBD step (4[0-9]|5[0-9])\]" "$OUT/probe.out" 2>/dev/null | grep -oE "step [0-9]+\] peak_rss=[0-9]+MB rss=[0-9]+MB"
} > "$LOG" 2>&1
cat "$LOG"
echo "DONE poll_streamA_validate"
