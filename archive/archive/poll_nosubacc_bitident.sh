#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
CMP=$B/scripts/eval/compare_reduction_results.py
R74=$B/results/probe_74_v7_nosubacc/result.pkl
R84=$B/results/probe_84_v7_nosubacc/result.pkl
for i in $(seq 1 200); do
  [ -s "$R74" ] && [ -s "$R84" ] && break
  sleep 20
done
LOG=$B/results/probe_74_v7_nosubacc/bitident.txt
{
  echo "=== sub_accum-removed vs v6 baseline (masters+coeffs) ==="
  echo "## 74"; $PY $CMP $B/results/probe_74_v6_btabu_C0/result.pkl $R74 2>&1 | grep -E "masters|coeff|IDENTICAL"
  echo "## 84"; $PY $CMP $B/results/probe_84_v6_btabu_C0/result.pkl $R84 2>&1 | grep -E "masters|coeff|IDENTICAL"
} > "$LOG" 2>&1
cat "$LOG"
echo DONE
