#!/bin/bash
# Assemble the model_fwd thread-scaling curve (1/4/8/16) over steps 28-40.
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
declare -A RUN=(
  [1]=$BASE/results/probe_74_v7_prof/probe.out
  [4]=$BASE/results/probe_74_v7_prof_t4/probe.out
  [8]=$BASE/results/probe_74_v7_prof_t8/probe.out
  [16]=$BASE/results/probe_74_v7_prof_t16/probe.out
)
LOG=$BASE/results/probe_74_v7_prof_t16/model_fwd_curve.txt

# wait until both t8 and t16 reach >=41 PROF steps
for i in $(seq 1 200); do
    n8=$(grep -c "PROF step=" "${RUN[8]}" 2>/dev/null || echo 0)
    n16=$(grep -c "PROF step=" "${RUN[16]}" 2>/dev/null || echo 0)
    if [ "$n8" -ge 41 ] && [ "$n16" -ge 41 ]; then break; fi
    sleep 15
done

avg() {
  awk '
    /^\[v6 step/ { for(i=1;i<=NF;i++) if($i~/t_step=/){split($i,a,"=");gsub("s","",a[2]);cur=a[2]+0} }
    /PROF step=/ { split($2,a,"="); ps=a[2]+0; inr=(ps>=28 && ps<=40); if(inr){n++; ts+=cur} }
    /model_fwd=/ && inr { for(i=1;i<=NF;i++) if($i~/model_fwd=/){split($i,a,"=");mf+=a[2]+0} }
    END{ if(n==0){printf "n=0"; exit} printf "%.2f %.2f", mf/n, ts/n }' "$1"
}

{
  echo "=== model_fwd thread-scaling (74, steps 28-40) ==="
  printf "%-9s %-12s %-12s %-10s %-10s\n" "threads" "model_fwd" "t_step" "mf_speedup" "step_speedup"
  read mf1 ts1 <<< "$(avg "${RUN[1]}")"
  for K in 1 4 8 16; do
    read mf ts <<< "$(avg "${RUN[$K]}")"
    if [ -z "$mf" ] || [ "$mf" = "n=0" ]; then printf "%-9s (no data)\n" "$K"; continue; fi
    mfsp=$(awk -v a="$mf1" -v b="$mf" 'BEGIN{printf "%.2fx", a/b}')
    tssp=$(awk -v a="$ts1" -v b="$ts" 'BEGIN{printf "%.2fx", a/b}')
    eff=$(awk -v a="$mf1" -v b="$mf" -v k="$K" 'BEGIN{printf "%.0f%%", 100*(a/b)/k}')
    printf "%-9s %-12s %-12s %-10s %-10s eff=%s\n" "$K" "${mf}s" "${ts}s" "$mfsp" "$tssp" "$eff"
  done
} > "$LOG" 2>&1
cat "$LOG"
echo "DONE poll_model_fwd_curve"
