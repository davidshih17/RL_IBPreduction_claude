#!/bin/bash
# ============================================================================
# RECOMMENDED fresh hierarchical reduction launcher (canonical GitHub config).
#
# Reduces a single pentagonbox top integral to the UNIQUE 62-master IBP+LI
# basis. The orchestrator runs in-process (a poller: it enumerates the start,
# submits one Condor worker per non-master, reaps results, and chains weight
# levels down to the masters).
#
# SETTINGS = the production `pentagonbox_8_5_v7_fresh` run (1-CPU workers fanned
# out up to 1000-wide; NO os.fork / fork pool; torch single-threaded; flat 4 GB
# request; straggler escalation disabled; --resume for crash recovery) EXCEPT
# for exactly two flags:
#     --topology   pentagonbox_nosym   (62-master IBP+LI-only basis, not 61)
#     --paper-masters-only             (reduce to that basis)
# so the reduction terminates at the *unique* master basis instead of leaving an
# overcomplete, path-dependent corner basis. v7_fresh round-1 itself used plain
# `pentagonbox` + `--no-paper-masters-only` (see launch_hier_85_v7_fresh.sh).
#
# To reduce a DIFFERENT integral: change INTEGRAL_STR and OUTDIR below.
# To resume from / reuse the (8,5) cache: add
#     --resume-from $BASE/results/pentagonbox_8_5_v7_fresh
# ============================================================================
set -e
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2

# ---- target: args $1=integral $2=outdir, else these defaults ----
INTEGRAL_STR="${1:-1,1,1,1,1,1,1,1,-5,0,0}"           # 11-int target (8 props + 3 ISPs)
OUTDIR="${2:-$BASE/results/my_reduction}"             # output dir (shared /het NFS, NOT /tmp)
# (greedy sector symmetry: prefix the call with SAILIR_SYMMETRY=1 — the
#  orchestrator forwards --symmetry to every worker. See README §11.)
# ------------------------------------------

PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox_nosym

# work-dir MUST be on shared NFS so Condor workers on other nodes can read it.
mkdir -p $OUTDIR/logs $OUTDIR/work/logs $OUTDIR/work/results

set -x
PYTHONUNBUFFERED=1 $PYTHON -u $BASE/reduction/hierarchical_reduction.py \
    --topology $TOPOLOGY \
    --integral="$INTEGRAL_STR" \
    --output $OUTDIR/reduction.pkl \
    --work-dir $OUTDIR/work \
    --model-checkpoint $MODEL \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --paper-masters-only \
    --use-v7-worker \
    --v7-cpus 1 \
    --worker-memory-gb 4 \
    --straggler-timeout 1000000000 \
    --straggler2-timeout 1000000000 \
    --check-interval 5 \
    --max-concurrent 1000 \
    --resume \
  > $OUTDIR/logs/hierarchical.log 2>&1 &
ORCH_PID=$!
set +x
echo "reduction orchestrator launched PID=$ORCH_PID"
echo "  log:     $OUTDIR/logs/hierarchical.log"
echo "  workdir: $OUTDIR/work"
echo "  result:  $OUTDIR/reduction.pkl"
