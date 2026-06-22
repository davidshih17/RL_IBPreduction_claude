#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
R=$B/results/probe_74_v7_combined/result.pkl
for i in $(seq 1 360); do [ -s "$R" ] && break; sleep 20; done
LOG=$B/results/probe_74_v7_combined/verify.txt
{
  echo "=== COMBINED (raw-strip FIX + Stage 3b packed-rs): bit-identical vs v6 ==="
  $PY $B/scripts/eval/compare_reduction_results.py \
      $B/results/probe_74_v6_btabu_C0/result.pkl $R 2>&1 | grep -E "masters|coeff|IDENTICAL"
  echo ""
  echo "=== peak_rss / rss / glibc_free: combined vs nosubacc (full baseline) ==="
  printf "%-5s | %-32s | %-32s\n" step "COMBINED (fix+3b)" "nosubacc (full, no strip, no 3b)"
  for s in 20 40 60 80; do
    g(){ grep -E "MEMBD step $s\]" $B/results/$1/probe.out 2>/dev/null | head -1 | grep -oE "peak_rss=[0-9]+MB rss=[0-9]+MB|free=[0-9]+" | tr '\n' ' '; }
    printf "%-5s | %-32s | %-32s\n" "$s" "$(g probe_74_v7_combined)" "$(g probe_74_v7_nosubacc)"
  done
  host=$(grep -oE "executing on host: <[0-9.]+" $B/results/probe_74_v7_combined/probe.log 2>/dev/null | head -1)
  echo ""
  echo "combined host: $host (nosubacc was .25)"
  echo "wall-clock: combined=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_74_v7_combined/probe.out | tail -1)"
} > "$LOG" 2>&1
cat "$LOG"; echo DONE
