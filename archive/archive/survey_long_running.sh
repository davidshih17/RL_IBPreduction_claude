#!/bin/bash
# Survey all currently-running dshih jobs and report any running > 60 min.
# Output goes to a log file (stdout/stderr combined).
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
LOGDIR=$BASE/results/pentagonbox_8_5_v3/logs
LOGFILE=$LOGDIR/long_running_survey_$(date +%Y%m%d_%H%M%S).log

mkdir -p $LOGDIR

{
  echo "=== Long-running job survey ($(date)) ==="
  echo "Threshold: 60 min run time (excludes queue wait)"
  echo

  # Pull ClusterId, JobStatus (2=R, 1=I, 5=H), JobStartDate, RequestCpus,
  # and the integral from the arguments string.
  # Sort by JobStartDate ascending so the oldest runners come first.
  condor_q dshih -af:t ClusterId JobStatus JobStartDate RequestCpus Args \
    | awk -F'\t' -v now=$(date +%s) '
        BEGIN {
          OFS = "\t"
          print "cluster_id", "run_min", "cpus", "integral"
        }
        $2 == 2 && $3 != "undefined" && $3 != "0" {
          run_sec = now - $3
          run_min = run_sec / 60
          if (run_min < 60) next

          # Extract --integral=... or --integral ... from Args
          integral = "?"
          if (match($5, /--integral=?[^ \t]*/)) {
            integral = substr($5, RSTART, RLENGTH)
            sub(/.*--integral=?/, "", integral)
            gsub(/['\'']/, "", integral)
          }
          print $1, sprintf("%.1f", run_min), $4, integral
        }
      ' \
    | sort -t$'\t' -k2 -n -r

  echo
  echo "=== Total dshih running/idle/held counts ==="
  condor_q dshih -totals
} 2>&1 | tee $LOGFILE

echo
echo "Survey written to: $LOGFILE"
