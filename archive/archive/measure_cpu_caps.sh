#!/bin/bash
# Verify the 8/8 CPU-overshoot mechanism + the thread-cap fix. Runs probe 74 at
# 8 threads / 8 workers for ~50 steps, samples the process-tree CPU in steady
# state (main python's utime+stime+cutime+cstime -> includes reaped fork
# workers) over a 15s window = cores used. Compares caps OFF vs ON, and reports
# avg t_step so we see the model isn't slowed by the caps.
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox
SCRIPT=$BASE/scripts/eval/beam_search_v7.py
CLK=$(getconf CLK_TCK)
ARGS="--topology $TOPOLOGY --model $MODEL --integral=0,1,1,1,1,1,1,1,-4,0,0 \
  --ckpt-every 9999 --tabu --no-exprkeyed --iraws-keep-first 50 --beam-width 40 \
  --max-steps 60 --max-actions 900 --beam-sort weight --no-paper-masters-only \
  --prime 1009 --n-threads 8 --n-workers 8 --device cpu --model-batch-chunk 8"

cpu_of() {  # $1 = pid -> total cpu ticks (self + reaped children)
  awk '{print $14+$15+$16+$17}' /proc/$1/stat 2>/dev/null
}

run_case() {  # $1 = label, $2... = env assignments
  local label="$1"; shift
  local O=$BASE/results/cpucap_$label; mkdir -p $O
  env PYTHONUNBUFFERED=1 SAILIR_END_OF_STEP_TRIM=1 SAILIR_TABU_CAP=0 \
      SAILIR_STRIP_RAWS=1 SAILIR_PACKED_RS=1 "$@" \
      $PYTHON -u $SCRIPT $ARGS --output $O/r.pkl > $O/run.log 2>&1 &
  local PID=$!
  # wait for steady state (step >= 8)
  for i in $(seq 1 60); do
    grep -qE "step +8\]" $O/run.log 2>/dev/null && break
    kill -0 $PID 2>/dev/null || break
    sleep 1
  done
  local t0=$(cpu_of $PID); local w0=$(date +%s.%N)
  sleep 15
  local t1=$(cpu_of $PID); local w1=$(date +%s.%N)
  kill $PID 2>/dev/null; wait $PID 2>/dev/null
  local cores=$(echo "scale=2; ($t1-$t0)/$CLK/($w1-$w0)" | bc -l 2>/dev/null)
  local tstep=$(grep -oE "t_step=[0-9.]+s" $O/run.log | tail -8 | grep -oE "[0-9.]+" | \
                awk '{s+=$1;n++} END{if(n)printf "%.2f",s/n}')
  printf "  %-12s cores_used=%-6s avg_t_step=%ss\n" "$label" "${cores:-?}" "${tstep:-?}"
}

echo "=== 8/8 CPU usage: caps OFF vs ON (CLK_TCK=$CLK) ==="
run_case caps_off
run_case caps_on OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
echo "(request was 8 cores; >8 => overshoot)"
