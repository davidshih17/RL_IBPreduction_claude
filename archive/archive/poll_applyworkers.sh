#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
R1=$B/results/probe_74_v7_aw1/result.pkl; R8=$B/results/probe_74_v7_aw8/result.pkl
for i in $(seq 1 360); do [ -s "$R1" ] && [ -s "$R8" ] && break; sleep 20; done
LOG=$B/results/probe_74_v7_aw8/applyworkers.txt
{
  echo "=== apply-workers: bit-identical vs v6 + throughput ==="
  for d in aw1 aw8; do
    echo "## $d"
    $PY $B/scripts/eval/compare_reduction_results.py \
        $B/results/probe_74_v6_btabu_C0/result.pkl $B/results/probe_74_v7_$d/result.pkl 2>&1 | grep -E "IDENTICAL"
    echo "   host=$(grep -oE 'executing on host: <[0-9.]+' $B/results/probe_74_v7_$d/probe.log 2>/dev/null | head -1)  t_total=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_74_v7_$d/probe.out | tail -1)"
  done
  echo ""
  t1=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_74_v7_aw1/probe.out | tail -1 | grep -oE '[0-9.]+')
  t8=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_74_v7_aw8/probe.out | tail -1 | grep -oE '[0-9.]+')
  echo "speedup workers=8 vs workers=1: $(awk "BEGIN{printf \"%.2fx\", $t1/$t8}") (total wall-clock; apply is only part of per-step)"
} > "$LOG" 2>&1
cat "$LOG"; echo DONE
