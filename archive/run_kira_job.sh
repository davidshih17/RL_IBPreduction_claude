#!/bin/bash
# Generic Kira 3.1 runner. The work dir must already contain jobs.yaml and
# target_integrals; this stages config/ (integralfamilies.yaml + kinematics.yaml)
# and runs Kira in the background, unbuffered, into a single log.
# Usage: run_kira_job.sh <work_dir>
set -e
if [ -z "$1" ]; then echo "usage: $0 <work_dir>" >&2; exit 1; fi
mkdir -p "$1"
WORK="$(cd "$1" && pwd)"   # absolute, so the post-cd log redirect is correct
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
KIRA=/het/p4/dshih/jet_images-deep_learning/IBPreduction/kira/kira-3.1
export FERMATPATH=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/fire/FIRE7/extra/fuel/extra/ferl64/fer64
[ -x "$KIRA" ] || { echo "ERROR: kira not at $KIRA" >&2; exit 1; }
[ -x "$FERMATPATH" ] || { echo "ERROR: fermat not at $FERMATPATH" >&2; exit 1; }
[ -f "$WORK/jobs.yaml" ] || { echo "ERROR: $WORK/jobs.yaml missing" >&2; exit 1; }
mkdir -p "$WORK/config"
cp $BASE/topology_input/pentagonbox/integralfamilies.yaml "$WORK/config/"
cp $BASE/topology_input/pentagonbox/kinematics.yaml "$WORK/config/"
cd "$WORK"
echo "running kira in $WORK ; FERMATPATH=$FERMATPATH"
nohup $KIRA jobs.yaml > "$WORK/kira_run.log" 2>&1 &
echo "kira launched PID=$!"
echo "  log: $WORK/kira_run.log"
