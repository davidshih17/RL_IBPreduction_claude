#!/bin/bash
# Find logs of completed workers whose total runtime <= TOTAL_MAX seconds
# AND whose final P4 aux time is >= P4_MIN seconds, indicating they
# reached the slow-aux regime. These are candidate benchmark targets.

LOGDIR=${1:-results/pentagonbox_8_5_delta/work/logs}
TOTAL_MAX=${2:-3600}
P4_MIN=${3:-5}

echo "logs in $LOGDIR with SUCCESS, total<=${TOTAL_MAX}s, final_p4_aux>=${P4_MIN}s"
echo "total_s | last_step | P4_aux_last | file"
echo "---"

for f in "$LOGDIR"/*.out; do
    total=$(grep -oE "SUCCESS in [0-9.]+s" "$f" | head -1 | sed 's/SUCCESS in //;s/s$//')
    [ -z "$total" ] && continue
    total_int=${total%.*}
    [ "$total_int" -gt "$TOTAL_MAX" ] && continue

    last_p4=$(grep -E "^Step [0-9]+:" "$f" | tail -1 | grep -oE "aux=[0-9.]+" | sed 's/aux=//')
    [ -z "$last_p4" ] && continue
    p4_int=${last_p4%.*}
    [ "$p4_int" -lt "$P4_MIN" ] && continue

    last_step=$(grep -E "^Step [0-9]+:" "$f" | tail -1 | grep -oE "Step [0-9]+" | sed 's/Step //')
    echo "${total}s | step ${last_step} | aux=${last_p4}s | $(basename $f)"
done | sort -n -k1
