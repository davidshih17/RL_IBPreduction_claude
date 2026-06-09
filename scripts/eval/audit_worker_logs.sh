#!/bin/bash
# Run the full worker-log audit in background, unbuffered, output to log.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
LOGFILE=$BASE/results/pentagonbox_8_5_delta/logs/audit_$(date +%Y%m%d_%H%M%S).log

echo "Writing audit to: $LOGFILE"
PYTHONUNBUFFERED=1 nohup $PYTHON -u $BASE/scripts/eval/audit_worker_logs.py "$@" > $LOGFILE 2>&1 &
echo "PID: $!"
echo "tail -f $LOGFILE"
