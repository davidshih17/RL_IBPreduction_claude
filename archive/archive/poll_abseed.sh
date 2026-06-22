#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
for i in $(seq 1 200); do grep -q "ABSEED DONE" $B/results/abseed/abseed.out 2>/dev/null && break; sleep 30; done
echo "=== OLD vs NEW delta — SAME node, back-to-back, only the delta form differs ==="
echo "host: $(grep 'HOST:' $B/results/abseed/abseed.out|head -1)"
echo "t_total: OLD=$(grep -oE 't_total=[0-9.]+s' $B/results/abseed/OLDdelta.out|tail -1)  NEW=$(grep -oE 't_total=[0-9.]+s' $B/results/abseed/NEWdelta.out|tail -1)"
echo ""
for d in OLDdelta NEWdelta; do
  grep -oE "step +[0-9]+\] beam=.*t_total=[0-9.]+s" $B/results/abseed/$d.out 2>/dev/null \
    | sed -E 's/step +([0-9]+)\].*t_total=([0-9.]+)s/\1 \2/' > /tmp/abs_$d.tt
done
echo "cumulative t_total at matched steps:"
printf "%-5s %-10s %-10s %-8s\n" step OLD NEW "NEW-OLD"
join /tmp/abs_OLDdelta.tt /tmp/abs_NEWdelta.tt 2>/dev/null | awk 'NR%10==0{printf "%-5s %-10s %-10s %+.1f\n",$1,$2,$3,$3-$2}'
echo DONE
