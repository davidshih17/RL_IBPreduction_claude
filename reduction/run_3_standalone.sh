#!/bin/bash
# ============================================================================
# Standalone reductions for the 3 covered_by_cache list_TA integrals that the
# meta wrongly pruned (their one-step key was cached, but their products only
# cancelled IN-CONTEXT inside tgt0091 / tgt0092 / v7_fresh -- they do NOT reduce
# standalone). Each gets a full orchestrator, reusing the CLEAN combined cache
# (--resume-from _clean_cache) so only the genuinely-missing products are
# computed.
#   extra0001_w6_4 : TA[1,1,1,1,1,1,0,0,-4,0,0]
#   extra0002_w6_3 : TA[1,1,1,0,1,1,0,1,0,-3,0]
#   extra0003_w6_3 : TA[1,1,1,0,1,1,0,1,0,-2,-1]
# Production config: pentagonbox + --no-paper-masters-only + v7 1-cpu + 1000-wide
# + stragglers off + --resume. Separate work-dirs -> no conflict, no duplication.
# ============================================================================
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
CACHE=$BASE/results/meta_reduce/_burst_cache   # intact & clean; _clean_cache was lost to the NFS glitch (genuinely-missing products compute fresh either way)

names=(extra0001_w6_4 extra0002_w6_3 extra0003_w6_3)
ints=("1,1,1,1,1,1,0,0,-4,0,0" "1,1,1,0,1,1,0,1,0,-3,0" "1,1,1,0,1,1,0,1,0,-2,-1")

for i in 0 1 2; do
  name=${names[$i]}
  INTEGRAL=${ints[$i]}
  RUNDIR=$BASE/results/meta_reduce/$name
  mkdir -p $RUNDIR/logs $RUNDIR/work/logs $RUNDIR/work/results
  nohup env PYTHONUNBUFFERED=1 $PY -u $BASE/reduction/hierarchical_reduction.py \
      --topology $TOPOLOGY \
      --integral="$INTEGRAL" \
      --output $RUNDIR/reduction.pkl \
      --work-dir $RUNDIR/work \
      --resume-from $CACHE \
      --model-checkpoint $MODEL \
      --beam_width 40 --max_steps 1000000 --prime 1009 \
      --no-paper-masters-only \
      --use-v7-worker --v7-cpus 1 --worker-memory-gb 4 \
      --straggler-timeout 1000000000 --straggler2-timeout 1000000000 \
      --check-interval 5 --max-concurrent 1000 --resume \
    > $RUNDIR/logs/hierarchical.log 2>&1 &
  echo "launched $name ($INTEGRAL) PID=$! -> $RUNDIR/logs/hierarchical.log"
  disown
done
