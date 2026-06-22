#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PO=$B/results/probe_74_v7_preeval2/probe.out
for i in $(seq 1 300); do grep -q "MEMBD step 80\]" "$PO" 2>/dev/null && break; sleep 20; done
LOG=$B/results/probe_74_v7_preeval2/mem.txt
{
  host=$(grep -oE "executing on host: <[0-9.]+" $B/results/probe_74_v7_preeval2/probe.log 2>/dev/null | head -1)
  echo "preeval2 host: $host  (nosubacc/preeval were .25)"
  echo ""
  echo "peak_rss / rss / glibc_free at matched steps:"
  printf "%-5s | %-30s | %-30s | %-30s\n" step "preeval2 (FIXED strip)" "preeval (old strip)" "nosubacc (full)"
  for s in 20 40 60 80; do
    g(){ grep -E "MEMBD step $s\]" $B/results/$1/probe.out 2>/dev/null | head -1 | grep -oE "peak_rss=[0-9]+MB rss=[0-9]+MB|free=[0-9]+" | tr '\n' ' '; }
    printf "%-5s | %-30s | %-30s | %-30s\n" "$s" "$(g probe_74_v7_preeval2)" "$(g probe_74_v7_preeval)" "$(g probe_74_v7_nosubacc)"
  done
} > "$LOG" 2>&1
cat "$LOG"; echo DONE
