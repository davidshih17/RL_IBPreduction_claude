#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
R=$B/results/probe_74_v7_contains/result.pkl
for i in $(seq 1 300); do [ -s "$R" ] && break; sleep 30; done
LOG=$B/results/probe_74_v7_contains/verify.txt
{
  echo "=== Cython contains_sorted (Phase-1b id-membership) ==="
  echo "bit-identical vs v6: $($PY $B/scripts/eval/compare_reduction_results.py $B/results/probe_74_v6_btabu_C0/result.pkl $R 2>&1 | grep -oE 'FINAL REDUCTION IDENTICAL: \w+')"
  echo "t_total: contains=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_74_v7_contains/probe.out|tail -1)  cython2(substitute only)=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_74_v7_cython2/probe.out|tail -1)  cache2(numpy)=976.6s"
  echo "host: contains=$(grep -oE 'host: <[0-9.]+' $B/results/probe_74_v7_contains/probe.log|head -1)  cython2=$(grep -oE 'host: <[0-9.]+' $B/results/probe_74_v7_cython2/probe.log|head -1)"
  echo ""
  echo "memory inuse(LIVE)/free vs cython2 (same kernels + contains):"
  printf "%-5s | %-30s | %-30s\n" step "contains" "cython2"
  for s in 20 40 60 80; do
    g(){ grep -E "MEMBD step $s\]" $B/results/$1/probe.out 2>/dev/null | head -1 | grep -oE "inuse=[0-9]+ free=[0-9]+|peak_rss=[0-9]+MB"|tr '\n' ' '; }
    printf "%-5s | %-30s | %-30s\n" "$s" "$(g probe_74_v7_contains)" "$(g probe_74_v7_cython2)"
  done
} > "$LOG" 2>&1
cat "$LOG"; echo DONE
