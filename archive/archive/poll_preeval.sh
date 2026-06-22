#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
R=$B/results/probe_74_v7_preeval/result.pkl
for i in $(seq 1 300); do [ -s "$R" ] && break; sleep 20; done
LOG=$B/results/probe_74_v7_preeval/verify.txt
{
  echo "=== strip-before-eval: bit-identical vs v6 ==="
  $PY $B/scripts/eval/compare_reduction_results.py \
      $B/results/probe_74_v6_btabu_C0/result.pkl $R 2>&1 | grep -E "masters|coeff|IDENTICAL"
  echo ""
  echo "=== wall-clock (this run vs earlier same-host runs) ==="
  tt(){ grep -oE "t_total=[0-9.]+s" $B/results/$1/probe.out 2>/dev/null | tail -1; }
  echo "  preeval (STRIP, before-eval): $(tt probe_74_v7_preeval)"
  echo "  nosubacc (full, host .25)   : $(tt probe_74_v7_nosubacc)"
  echo "  minw (weight-strip, host .25): $(tt probe_74_v7_minw)"
  host=$(grep -oE "executing on host: <[0-9.]+" $B/results/probe_74_v7_preeval/probe.log 2>/dev/null | head -1)
  echo "  preeval host: $host"
} > "$LOG" 2>&1
cat "$LOG"; echo DONE
