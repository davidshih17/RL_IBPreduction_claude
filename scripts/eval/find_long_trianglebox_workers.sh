#!/bin/bash
# Find trianglebox workers with the most steps. Uses tail -n 200 (last steps
# + peak memory line) to avoid full-file scans.
OUT=${1:-logs/find_long_trianglebox_workers.txt}
mkdir -p $(dirname $OUT)
> $OUT
echo "Scanning $(find /het/p4/dshih/scratch/ibp_async_mixed_bench_v13_*/logs -name '*.out' 2>/dev/null | wc -l) workers..." | tee -a $OUT

find /het/p4/dshih/scratch/ibp_async_mixed_bench_v13_*/logs -name "*.out" 2>/dev/null \
  | xargs -P 32 -I {} bash -c '
    f="{}"
    # last `Step N:` line in tail
    laststep=$(tail -n 60 "$f" 2>/dev/null | grep -oE "^Step [0-9]+" | tail -1 | awk "{print \$2}")
    if [ -n "$laststep" ] && [ "$laststep" -ge 50 ]; then
      mem=$(grep -oE "peak memory: [0-9.]+ MB" "$f" 2>/dev/null | tail -1)
      printf "%6s  %s  %s\n" "$laststep" "$mem" "$f"
    fi
  ' >> $OUT

echo "" | tee -a $OUT
echo "=== Top 30 by step count ===" | tee -a $OUT
sort -n -r $OUT | head -30 | tee -a $OUT
