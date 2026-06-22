#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
R=$B/results/probe_74_v7_packedrs/result.pkl
for i in $(seq 1 300); do [ -s "$R" ] && break; sleep 20; done
LOG=$B/results/probe_74_v7_packedrs/verify.txt
{
  echo "=== packed-rs (Stage 3b): bit-identical vs v6 ==="
  $PY $B/scripts/eval/compare_reduction_results.py \
      $B/results/probe_74_v6_btabu_C0/result.pkl $R 2>&1 | grep -E "masters|coeff|IDENTICAL"
  echo ""
  echo "=== memory: peak_rss + rss at matched steps, packed-rs vs preeval (dict rs) ==="
  printf "  %-6s %-12s %-12s\n" step "packedrs" "preeval(dict)"
  for s in 20 40 60 80; do
    gp(){ grep -E "MEMBD step $s\]" $B/results/$1/probe.out 2>/dev/null | grep -oE "peak_rss=[0-9]+MB rss=[0-9]+MB" | head -1; }
    printf "  %-6s %-12s %-12s\n" "$s" "$(gp probe_74_v7_packedrs)" "$(gp probe_74_v7_preeval)"
  done
  echo ""
  echo "wall-clock: packedrs=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_74_v7_packedrs/probe.out | tail -1)  preeval=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_74_v7_preeval/probe.out | tail -1)"
} > "$LOG" 2>&1
cat "$LOG"; echo DONE
