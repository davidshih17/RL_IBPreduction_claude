#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
R=$B/results/probe_74_v7_seedmerge/result.pkl
for i in $(seq 1 300); do [ -s "$R" ] && grep -q "PROF2 DONE" $B/results/prof2/prof2.out 2>/dev/null && break; sleep 30; done
echo "=== seed/delta merge: bit-identical + timing ==="
echo "bit-identical vs v6: $($PY $B/scripts/eval/compare_reduction_results.py $B/results/probe_74_v6_btabu_C0/result.pkl $R 2>&1 | grep -oE 'FINAL REDUCTION IDENTICAL: \w+')"
echo "t_total: seedmerge=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_74_v7_seedmerge/probe.out|tail -1)  contains=792.2s  (host $(grep -oE 'host: <[0-9.]+' $B/results/probe_74_v7_seedmerge/probe.log|head -1))"
echo ""
echo "=== re-profile (current state: substitute_one+contains+seedmerge+cache) ==="
sed -n '/=== cProfile summary/,$p' $B/results/prof2/analysis.txt 2>/dev/null | head -26
echo DONE
