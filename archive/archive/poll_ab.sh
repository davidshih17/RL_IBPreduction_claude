#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
for i in $(seq 1 120); do grep -q "AB DONE" $B/logs/ab_driver.log 2>/dev/null && break; sleep 15; done
echo "=== sector cache ON vs OFF, SAME machine, all else identical ==="
echo "    metric = peak_rss / rss / glibc_inuse(LIVE) / glibc_free(reclaimable)"
printf "%-5s | %-40s | %-40s\n" step "cache ON" "cache OFF"
for s in 15 25 35 45; do
  g(){ grep -E "MEMBD step $s\]" $B/logs/ab_$1.log 2>/dev/null | head -1 | grep -oE "peak_rss=[0-9]+MB rss=[0-9]+MB|inuse=[0-9]+ free=[0-9]+"|tr '\n' ' '; }
  printf "%-5s | %-40s | %-40s\n" "$s" "$(g cacheON)" "$(g cacheOFF)"
done
echo ""
echo "t_total: ON=$(grep -oE 't_total=[0-9.]+s' $B/logs/ab_cacheON.log|tail -1)  OFF=$(grep -oE 't_total=[0-9.]+s' $B/logs/ab_cacheOFF.log|tail -1)"
echo DONE
