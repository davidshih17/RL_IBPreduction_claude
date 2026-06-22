#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
Roff=$B/results/probe_84_v7_packoff/result.pkl; Ron=$B/results/probe_84_v7_packon/result.pkl
for i in $(seq 1 600); do [ -s "$Roff" ] && [ -s "$Ron" ] && break; sleep 30; done
LOG=$B/results/probe_84_v7_packon/packcmp.txt
{
  echo "=== packed-rs ON vs OFF at 84 (both raw-strip fix), launched together ==="
  for d in packoff packon; do
    echo "  $d host: $(grep -oE 'executing on host: <[0-9.]+' $B/results/probe_84_v7_$d/probe.log 2>/dev/null | head -1)  t_total=$(grep -oE 't_total=[0-9.]+s' $B/results/probe_84_v7_$d/probe.out | tail -1)"
  done
  echo ""
  for d in packoff packon; do
    grep -oE "step +[0-9]+\] beam=.*t_step=[0-9.]+s" $B/results/probe_84_v7_$d/probe.out 2>/dev/null \
      | sed -E 's/step +([0-9]+)\].*t_step=([0-9.]+)s/\1 \2/' > /tmp/84$d.tt
  done
  echo "per-step t_step (early + deep, where rs is large):"
  printf "  %-5s %-9s %-9s %-7s\n" step packOFF packON "ratio"
  join -a1 -e'-' -o '0,1.2,2.2' /tmp/84packoff.tt /tmp/84packon.tt 2>/dev/null | \
    awk 'NR<=8 || NR%30==0 {r=($3>0&&$2>0)?$3/$2:0; printf "  %-5s %-9s %-9s %.2f\n",$1,$2,$3,r}'
} > "$LOG" 2>&1
cat "$LOG"; echo DONE
