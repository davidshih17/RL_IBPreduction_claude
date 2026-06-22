#!/bin/bash
# Wait for the 74 and 84 streamA probes to finish, then compare their result.pkl
# (masters + coeffs) against the v6 baselines -> bit-identical gate (d).
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
CMP=$BASE/scripts/eval/compare_reduction_results.py
LOG=$BASE/results/probe_74_v7_streamA/bitident.txt

R74=$BASE/results/probe_74_v7_streamA/result.pkl
R84=$BASE/results/probe_84_v7_streamA/result.pkl
B74=$BASE/results/probe_74_v6_btabu_C0/result.pkl
B84=$BASE/results/probe_84_v6_btabu_C0/result.pkl

for i in $(seq 1 240); do
    if [ -s "$R74" ] && [ -s "$R84" ]; then break; fi
    sleep 20
done

{
    echo "=== bit-identical gate (d): streamA result.pkl vs v6 baseline ==="
    echo ""
    echo "######## 74 (streamA vs probe_74_v6_btabu_C0) ########"
    $PY $CMP "$B74" "$R74" 2>&1 | grep -E "masters|coeff|IDENTICAL|only in"
    echo ""
    echo "######## 84 (streamA vs probe_84_v6_btabu_C0) ########"
    $PY $CMP "$B84" "$R84" 2>&1 | grep -E "masters|coeff|IDENTICAL|only in"
} > "$LOG" 2>&1
cat "$LOG"
echo "DONE poll_streamA_bitident"
