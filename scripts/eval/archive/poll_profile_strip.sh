#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
for i in $(seq 1 240); do grep -q "ALL PROFILES DONE" $B/logs/profile_strip_driver.log 2>/dev/null && break; sleep 15; done
echo "=== per-step t_step: STRIP=1 (stripped) vs STRIP=0 (full), same machine back-to-back ==="
for t in strip1 strip0; do
  grep -oE "step +[0-9]+\] beam=.*t_step=[0-9.]+s t_total=[0-9.]+s" $B/logs/profile_${t}.log 2>/dev/null \
    | sed -E 's/step +([0-9]+)\].*t_step=([0-9.]+)s t_total=([0-9.]+)s/\1 \2 \3/' > /tmp/$t.tt
done
printf "%-6s %-9s %-9s\n" step STRIP1 STRIP0
join -a1 -e'-' -o '0,1.2,2.2' /tmp/strip1.tt /tmp/strip0.tt 2>/dev/null
echo ""
echo "final t_total: STRIP1=$(tail -1 /tmp/strip1.tt | awk '{print $3}')s  STRIP0=$(tail -1 /tmp/strip0.tt | awk '{print $3}')s"
echo ""
echo "=== top cumulative time: STRIP=1 .prof ==="
$PY -c "import pstats; pstats.Stats('$B/results/profile_strip/strip1.prof').sort_stats('cumulative').print_stats(18)" 2>/dev/null | sed -n '6,30p'
echo "=== top cumulative time: STRIP=0 .prof ==="
$PY -c "import pstats; pstats.Stats('$B/results/profile_strip/strip0.prof').sort_stats('cumulative').print_stats(18)" 2>/dev/null | sed -n '6,30p'
echo DONE
