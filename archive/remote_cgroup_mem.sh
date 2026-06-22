# Read the JOB CGROUP's physical memory (what HTCondor MemoryUsage reflects) and
# a fork child's Private_Dirty (to see if the fork pool breaks COW). Handles
# cgroup v1 and v2; tries the namespaced root and the resolved path.
echo "=== cgroup membership ==="
cat /proc/self/cgroup
echo ""
echo "=== cgroup v2 (namespaced root /sys/fs/cgroup) ==="
echo -n "memory.current: "; cat /sys/fs/cgroup/memory.current 2>/dev/null || echo "n/a"
echo -n "memory.peak:    "; cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo "n/a"
echo ""
echo "=== cgroup v1 (namespaced root) ==="
echo -n "usage_in_bytes:      "; cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo "n/a"
echo -n "max_usage_in_bytes:  "; cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes 2>/dev/null || echo "n/a"
echo ""
echo "=== locate condor memory cgroup files on host ==="
for f in $(find /sys/fs/cgroup -path '*condor*' \( -name memory.current -o -name memory.peak -o -name memory.usage_in_bytes -o -name memory.max_usage_in_bytes \) 2>/dev/null | head -20); do
  echo "$f = $(cat $f 2>/dev/null)"
done
echo ""
echo "=== process tree RSS + Private_Dirty (COW-break check) ==="
MAIN=$(ps -eo pid,rss,comm 2>/dev/null | grep -i python | sort -k2 -n | tail -1 | awk '{print $1}')
echo "main python PID=$MAIN"
# children of main (the fork pool, if alive)
for P in $MAIN $(pgrep -P $MAIN 2>/dev/null); do
  pd=$(grep -E '^Private_Dirty' /proc/$P/smaps_rollup 2>/dev/null | awk '{print $2}')
  rss=$(grep -E '^VmRSS' /proc/$P/status 2>/dev/null | awk '{print $2}')
  pss=$(grep -E '^Pss' /proc/$P/smaps_rollup 2>/dev/null | awk '{print $2}')
  echo "PID $P: VmRSS=${rss:-?}kB  Pss=${pss:-?}kB  Private_Dirty=${pd:-?}kB"
done
