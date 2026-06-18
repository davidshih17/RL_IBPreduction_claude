#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
CMP=$B/scripts/eval/compare_reduction_results.py
R74=$B/results/probe_74_v7_minw/result.pkl; R84=$B/results/probe_84_v7_minw/result.pkl
for i in $(seq 1 300); do [ -s "$R74" ] && [ -s "$R84" ] && break; sleep 20; done
LOG=$B/results/probe_74_v7_minw/verify.txt
{
  echo "=== min_w12 (no-churn) bit-identical vs v6 ==="
  echo "## 74"; $PY $CMP $B/results/probe_74_v6_btabu_C0/result.pkl $R74 2>&1 | grep -E "masters|coeff|IDENTICAL"
  echo "## 84"; $PY $CMP $B/results/probe_84_v6_btabu_C0/result.pkl $R84 2>&1 | grep -E "masters|coeff|IDENTICAL"
  echo ""
  echo "=== peak_rss @ matched steps: minw vs nosubacc vs rawstrip(strip-after) (74) ==="
  printf "  %-6s %-10s %-10s %-10s %-12s\n" step minw nosubacc rawstrip "minw glibc_free"
  for s in 10 20 30 40 50 60 70 80; do
    getp(){ grep -E "MEMBD step $s\]" $B/results/$1/probe.out 2>/dev/null | grep -oE "peak_rss=[0-9]+MB" | grep -oE "[0-9]+"; }
    getf(){ grep -E "MEMBD step $s\]" $B/results/$1/probe.out 2>/dev/null | grep -oE "free=[0-9]+" | grep -oE "[0-9]+"; }
    m=$(getp probe_74_v7_minw); n=$(getp probe_74_v7_nosubacc); r=$(getp probe_74_v7_rawstrip); mf=$(getf probe_74_v7_minw)
    printf "  %-6s %-10s %-10s %-10s %-12s\n" "$s" "${m:--}MB" "${n:--}MB" "${r:--}MB" "${mf:--}MB"
  done
} > "$LOG" 2>&1
cat "$LOG"; echo DONE
