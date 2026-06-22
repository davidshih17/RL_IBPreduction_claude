#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
R=$B/results/probe_74_v7_cython/result.pkl
for i in $(seq 1 300); do [ -s "$R" ] && break; sleep 30; done
LOG=$B/results/probe_74_v7_cython/verify.txt
{
  echo "=== 74 with Cython substitute_one kernel ==="
  echo "bit-identical vs v6: $($PY $B/scripts/eval/compare_reduction_results.py $B/results/probe_74_v6_btabu_C0/result.pkl $R 2>&1 | grep -oE 'FINAL REDUCTION IDENTICAL: \w+')"
  echo "host: $(grep -oE 'executing on host: <[0-9.]+' $B/results/probe_74_v7_cython/probe.log|head -1)"
  echo "t_total: cython=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_74_v7_cython/probe.out|tail -1)  cache2/numpy=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_74_v7_cache2/probe.out|tail -1)  combined=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_74_v7_combined/probe.out|tail -1)"
} > "$LOG" 2>&1
cat "$LOG"; echo DONE
