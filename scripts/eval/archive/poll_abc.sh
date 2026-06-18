#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
for i in $(seq 1 200); do grep -q "ABC DONE" $B/results/abc/abc.out 2>/dev/null && break; sleep 30; done
echo "=== sector cache ON vs OFF — SAME condor node, only diff = the cache ==="
echo "host: $(grep 'RUNNING ON HOST' $B/results/abc/abc.out 2>/dev/null)"
echo "metric = peak_rss / rss / glibc_inuse(LIVE) / glibc_free(reclaimable)"
printf "%-5s | %-42s | %-42s\n" step "cache ON" "cache OFF"
for s in 20 40 60 80; do
  g(){ grep -E "MEMBD step $s\]" $B/logs/abc_$1.log 2>/dev/null | head -1 | grep -oE "peak_rss=[0-9]+MB rss=[0-9]+MB|inuse=[0-9]+ free=[0-9]+" | tr '\n' ' '; }
  printf "%-5s | %-42s | %-42s\n" "$s" "$(g cacheON)" "$(g cacheOFF)"
done
echo ""
echo "t_total: ON=$(grep -oE 't_total=[0-9.]+s' $B/logs/abc_cacheON.log|tail -1)  OFF=$(grep -oE 't_total=[0-9.]+s' $B/logs/abc_cacheOFF.log|tail -1)"
echo DONE
