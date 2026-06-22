#!/bin/bash
# Wait for the v6 profiled run to reach step 40, then average the per-step
# PROF buckets over steps 28-40 for BOTH v6 and v7 and print a side-by-side diff.
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
V6=$BASE/results/probe_74_v6_prof/probe.out
V7=$BASE/results/probe_74_v7_prof/probe.out
LOG=$BASE/results/probe_74_v6_prof/v6_vs_v7_buckets.txt

for i in $(seq 1 160); do
    n=$(grep -c "PROF step=" "$V6" 2>/dev/null || echo 0)
    if [ "$n" -ge 40 ]; then break; fi
    sleep 15
done

# awk: average t_step, model_fwd, apply, gv(enumerate), aux(PhaseA/B) over steps 28-40
avg() {
  awk '
    /^\[v6 step/ { for(i=1;i<=NF;i++) if($i~/t_step=/){split($i,a,"=");gsub("s","",a[2]);cur_ts=a[2]+0}; cur_step=$3+0 }
    /PROF step=/ { split($2,a,"="); ps=a[2]+0;
                   if(ps>=28 && ps<=40){ inrange=1; n++; ts+=cur_ts } else inrange=0 }
    /aux=/ && inrange { for(i=1;i<=NF;i++){ if($i~/^aux=/){split($i,a,"=");aux+=a[2]+0}; if($i~/^gv=/){split($i,a,"=");gv+=a[2]+0} } }
    /model_fwd=/ && inrange { for(i=1;i<=NF;i++) if($i~/model_fwd=/){split($i,a,"=");mf+=a[2]+0} }
    /apply=/ && inrange { for(i=1;i<=NF;i++) if($i~/^apply=/){split($i,a,"=");ap+=a[2]+0} }
    END{ if(n==0){print "no steps"; exit}
         printf "%d %.2f %.2f %.2f %.2f %.2f\n", n, ts/n, mf/n, ap/n, gv/n, aux/n }' "$1"
}

{
  echo "=== avg per-step over steps 28-40 (n_steps t_step model_fwd apply enumerate phaseA) ==="
  echo "v6: $(avg "$V6")"
  echo "v7: $(avg "$V7")"
  echo ""
  echo "columns: n  t_step  model_fwd  apply  enumerate(gv)  phaseA/B(aux)   [seconds]"
} > "$LOG" 2>&1
cat "$LOG"
echo "DONE poll_v6_vs_v7_prof"
