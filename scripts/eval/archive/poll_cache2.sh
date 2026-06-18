#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
R=$B/results/probe_74_v7_cache2/result.pkl
for i in $(seq 1 300); do [ -s "$R" ] && break; sleep 30; done
LOG=$B/results/probe_74_v7_cache2/verify.txt
{
  echo "=== 74 clean tree (combined + sector cache, apply-pool reverted) ==="
  echo "bit-identical vs v6: $($PY $B/scripts/eval/compare_reduction_results.py $B/results/probe_74_v6_btabu_C0/result.pkl $R 2>&1 | grep -oE 'FINAL REDUCTION IDENTICAL: \w+')"
  echo "host: $(grep -oE 'executing on host: <[0-9.]+' $B/results/probe_74_v7_cache2/probe.log|head -1)"
  echo "t_total: cache2=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_74_v7_cache2/probe.out|tail -1)  combined=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_74_v7_combined/probe.out|tail -1)"
  echo ""
  echo "memory (inuse=LIVE / free=reclaimable / peak_rss) vs combined:"
  printf "%-5s | %-38s | %-38s\n" step "cache2 (combined+cache)" "combined"
  for s in 20 40 60 80; do
    g(){ grep -E "MEMBD step $s\]" $B/results/$1/probe.out 2>/dev/null | head -1 | grep -oE "inuse=[0-9]+ free=[0-9]+|peak_rss=[0-9]+MB"|tr '\n' ' '; }
    printf "%-5s | %-38s | %-38s\n" "$s" "$(g probe_74_v7_cache2)" "$(g probe_74_v7_combined)"
  done
} > "$LOG" 2>&1
cat "$LOG"; echo DONE
