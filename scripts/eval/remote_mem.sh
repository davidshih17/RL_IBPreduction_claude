echo "=== uname / host ==="
hostname
echo "=== python processes (RSS,VSZ in KB) ==="
ps -eo pid,ppid,rss,vsz,etime,comm | grep -iE "PID|python" || ps aux | grep -i python | grep -v grep
P=$(ps -eo pid,rss,comm | grep -i python | sort -k2 -n | tail -1 | awk '{print $1}')
echo "biggest python PID=$P"
echo "=== /proc/$P/status (VmRSS=real resident, VmHWM=peak) ==="
grep -E "VmRSS|VmHWM|VmData|VmSize|VmSwap" /proc/$P/status 2>/dev/null
echo "=== /proc/$P/smaps_rollup (Rss/Pss/Anonymous breakdown) ==="
cat /proc/$P/smaps_rollup 2>/dev/null
