#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
R74=$B/results/probe_74_v7_cache/result.pkl; R84=$B/results/probe_84_v7_cache/result.pkl
for i in $(seq 1 600); do [ -s "$R74" ] && [ -s "$R84" ] && break; sleep 30; done
LOG=$B/results/probe_74_v7_cache/cache.txt
{
  echo "=== _master_sector_tuples cache fix: bit-identical + wall-clock ==="
  echo "## 74"
  $PY $B/scripts/eval/compare_reduction_results.py $B/results/probe_74_v6_btabu_C0/result.pkl $R74 2>&1 | grep -E "IDENTICAL"
  echo "   cache:    host=$(grep -oE 'host: <[0-9.]+' $B/results/probe_74_v7_cache/probe.log|head -1)  t_total=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_74_v7_cache/probe.out|tail -1)"
  echo "   pre-cache(combined): host=$(grep -oE 'host: <[0-9.]+' $B/results/probe_74_v7_combined/probe.log|head -1)  t_total=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_74_v7_combined/probe.out|tail -1)"
  echo "## 84"
  $PY $B/scripts/eval/compare_reduction_results.py $B/results/probe_84_v6_btabu_C0/result.pkl $R84 2>&1 | grep -E "IDENTICAL"
  echo "   cache:    host=$(grep -oE 'host: <[0-9.]+' $B/results/probe_84_v7_cache/probe.log|head -1)  t_total=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_84_v7_cache/probe.out|tail -1)"
  echo "   pre-cache(combined): host=$(grep -oE 'host: <[0-9.]+' $B/results/probe_84_v7_combined/probe.log|head -1)  t_total=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_84_v7_combined/probe.out|tail -1)"
  echo "   v6_84 ref: t_total=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_84_v6_btabu_C0/probe.out 2>/dev/null|tail -1)"
} > "$LOG" 2>&1
cat "$LOG"; echo DONE
