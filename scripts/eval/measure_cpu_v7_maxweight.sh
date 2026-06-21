#!/bin/bash
# Measure steady-state process-tree CPU (cores used) of v7 at 8 threads/8 workers
# with the total-weight + maxweight config, to find WHY it overshoots request=8.
# Samples main python's utime+stime+cutime+cstime (includes reaped fork workers)
# over a 20s window once past step 10.
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
SCRIPT=$BASE/scripts/eval/beam_search_v7.py
CLK=$(getconf CLK_TCK)
ARGS="--topology $BASE/topology_input/pentagonbox --model $MODEL --integral=0,1,1,1,1,1,1,1,-4,0,0 \
  --ckpt-every 9999 --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 \
  --max-steps 40 --max-actions 900 --beam-sort weight --no-paper-masters-only \
  --prime 1009 --n-threads 8 --n-workers 8 --device cpu --model-batch-chunk 8"

cpu_of() { awk '{print $14+$15+$16+$17}' /proc/$1/stat 2>/dev/null; }

run_case() {
  local label="$1"; shift
  local O=$BASE/results/cpumeas_$label; mkdir -p $O
  env PYTHONUNBUFFERED=1 SAILIR_END_OF_STEP_TRIM=1 SAILIR_TABU_CAP=0 \
      SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1 SAILIR_SUCCESS_TOTAL=1 \
      SAILIR_BEAM_TOTAL=1 "$@" \
      $PYTHON -u $SCRIPT $ARGS --output $O/r.pkl > $O/run.log 2>&1 &
  local PID=$!
  for i in $(seq 1 90); do
    grep -qE "step +10\]" $O/run.log 2>/dev/null && break
    kill -0 $PID 2>/dev/null || break
    sleep 1
  done
  local t0=$(cpu_of $PID); local w0=$(date +%s.%N)
  sleep 20
  local t1=$(cpu_of $PID); local w1=$(date +%s.%N)
  kill $PID 2>/dev/null; pkill -P $PID 2>/dev/null; wait $PID 2>/dev/null
  local cores=$(echo "scale=2; ($t1-$t0)/$CLK/($w1-$w0)" | bc -l 2>/dev/null)
  printf "  %-20s cores_used=%s  (request would be 8)\n" "$label" "${cores:-?}"
}

echo "=== v7 8/8 maxweight CPU (CLK_TCK=$CLK) ==="
run_case maxweight_capsON   SAILIR_ACTION_SELECT=maxweight
run_case maxweight_interop1 SAILIR_ACTION_SELECT=maxweight SAILIR_CAP_INTEROP=1
run_case first900_capsON    SAILIR_ACTION_SELECT=first900
