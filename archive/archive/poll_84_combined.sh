#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
R=$B/results/probe_84_v7_combined/result.pkl
for i in $(seq 1 600); do [ -s "$R" ] && break; sleep 30; done
LOG=$B/results/probe_84_v7_combined/verify.txt
{
  echo "=== 84 COMBINED (raw-strip fix + Stage 3b): bit-identical vs v6 ==="
  $PY $B/scripts/eval/compare_reduction_results.py \
      $B/results/probe_84_v6_btabu_C0/result.pkl $R 2>&1 | grep -E "masters|coeff|IDENTICAL"
  echo ""
  echo "wall-clock: combined84=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_84_v7_combined/probe.out | tail -1)  v6_84=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_84_v6_btabu_C0/probe.out 2>/dev/null | tail -1)"
  echo "peak_rss combined84=$(grep -oE 'peak_rss=[0-9]+MB' $B/results/probe_84_v7_combined/probe.out | tail -1)"
} > "$LOG" 2>&1
cat "$LOG"; echo DONE
