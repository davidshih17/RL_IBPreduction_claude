#!/bin/bash
# Read the REAL resident memory of a running Condor job (vs the unreliable
# condor_q MemoryUsage) by condor_ssh_to_job-ing into the sandbox and reading
# ps RSS + /proc/<pid>/smaps_rollup of the worker's python process.
# Usage: ssh_probe_mem.sh <ClusterId.ProcId>
JOB=${1:?need job id like 1668475.0}
echo "=== job $JOB: identity ==="
condor_q "${JOB%%.*}" -af ClusterId RemoteHost MemoryUsage 'ServerTime-JobCurrentStartDate' Args 2>/dev/null | cut -c1-300
echo ""
echo "=== REAL memory inside the sandbox (ps RSS in KB, smaps_rollup) ==="
timeout 90 condor_ssh_to_job "$JOB" bash -c '
echo "--- python processes (RSS,VSZ in KB) ---"
ps -eo pid,ppid,rss,vsz,etime,comm 2>/dev/null | grep -iE "PID|python"
PID=$(ps -eo pid,rss,comm 2>/dev/null | grep -i python | sort -k2 -n | tail -1 | awk "{print \$1}")
echo "--- biggest python PID=$PID ---"
grep -E "VmRSS|VmHWM|VmData|VmSize" /proc/$PID/status 2>/dev/null
echo "--- smaps_rollup ---"
cat /proc/$PID/smaps_rollup 2>/dev/null
' 2>&1
