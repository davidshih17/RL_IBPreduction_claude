#!/bin/bash
# Append a per-snapshot line: timestamp, count of jobs running > 3h,
# plus median/max runtime, and a per-priority-bucket histogram.
# Polls every 5 min.

LOG=${1:-/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/pentagonbox_8_5_delta/logs/long_jobs.log}
THRESHOLD_SEC=${2:-10800}  # 3h
INTERVAL=${3:-300}         # 5 min

while true; do
    now=$(date +%s)
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    # Pull running jobs only; JobStartDate=undefined when still idle.
    snap=$(condor_q dshih -af JobStartDate JobPrio 2>&1 | awk -v now=$now -v thr=$THRESHOLD_SEC '
        $1 != "undefined" {
            rt = now - $1
            if (rt > thr) {
                count++
                if (rt > max) max = rt
                rts[count] = rt
                # priority bucket = floor(prio / 1_000_000) = level
                lvl = int($2 / 1000000)
                bucket[lvl]++
            }
        }
        END {
            if (count == 0) { print "n=0"; exit }
            n = asort(rts)
            mid = n % 2 == 0 ? (rts[n/2] + rts[n/2+1]) / 2 : rts[(n+1)/2]
            printf "n=%d median=%ds max=%ds", count, int(mid), int(max)
            for (lvl in bucket) printf " L%d=%d", lvl, bucket[lvl]
            print ""
        }
    ')
    echo "$ts $snap" >> "$LOG"
    sleep $INTERVAL
done
