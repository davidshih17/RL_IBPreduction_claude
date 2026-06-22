#!/bin/bash
# Run Kira 3.1 to reduce TA[1,0,0,0,0,1,0,1,0,0,0] -- the corner integral both
# paper-masters-only SAILIR runs got stuck on (sector 161). Kira applies sector
# symmetries that SAILIR's IBP+LI beam search lacks, so this tests whether the
# stuck corner really does reduce to the 61-master basis.
#
# Work dir already contains jobs.yaml + target_integrals; this script stages the
# config/ (integralfamilies.yaml + kinematics.yaml) and runs Kira in background,
# unbuffered, into a single log.
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
WORK=$BASE/results/kira_reduce_161
KIRA=/het/p4/dshih/jet_images-deep_learning/IBPreduction/kira/kira-3.1
export FERMATPATH=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/fire/FIRE7/extra/fuel/extra/ferl64/fer64

if [ ! -x "$KIRA" ]; then echo "ERROR: kira not found at $KIRA" >&2; exit 1; fi
if [ ! -x "$FERMATPATH" ]; then echo "ERROR: fermat not found at $FERMATPATH" >&2; exit 1; fi

mkdir -p $WORK/config
cp $BASE/topology_input/pentagonbox/integralfamilies.yaml $WORK/config/
cp $BASE/topology_input/pentagonbox/kinematics.yaml $WORK/config/

cd $WORK
echo "running kira in $WORK ; FERMATPATH=$FERMATPATH"
nohup $KIRA jobs.yaml > $WORK/kira_run.log 2>&1 &
echo "kira launched PID=$!"
echo "  log: $WORK/kira_run.log"
