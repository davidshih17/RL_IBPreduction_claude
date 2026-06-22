#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
R=$B/results/probe_74_v7_p1b/result.pkl
for i in $(seq 1 300); do [ -s "$R" ] && break; sleep 30; done
LOG=$B/results/probe_74_v7_p1b/verify.txt
{
  echo "=== Cython enumerate Phase-1b loop (1 thread) ==="
  echo "bit-identical vs v6: $($PY $B/scripts/eval/compare_reduction_results.py $B/results/probe_74_v6_btabu_C0/result.pkl $R 2>&1 | grep -oE 'FINAL REDUCTION IDENTICAL: \w+')"
  echo "t_total: p1b=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_74_v7_p1b/probe.out|tail -1)  contains(prev)=792.2s"
  echo "host: p1b=$(grep -oE 'host: <[0-9.]+' $B/results/probe_74_v7_p1b/probe.log|head -1)  contains=192.168.1.109"
} > "$LOG" 2>&1
cat "$LOG"; echo DONE
