#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
for i in $(seq 1 200); do grep -q "ABSEEDREV DONE" $B/results/abseed_rev/abseed_rev.out 2>/dev/null && break; sleep 30; done
echo "############ DECISIVE: code effect vs order/environment ############"
echo ""
echo "FORWARD run (OLD first, NEW second), node $(grep -oE 'HOST: \S+' $B/results/abseed/abseed.out|head -1):"
echo "   OLD(1st)=$(grep -oE 't_total=[0-9.]+s' $B/results/abseed/OLDdelta.out|tail -1)   NEW(2nd)=$(grep -oE 't_total=[0-9.]+s' $B/results/abseed/NEWdelta.out|tail -1)"
echo ""
echo "REVERSE run (NEW first, OLD second), node $(grep -oE 'HOST: \S+' $B/results/abseed_rev/abseed_rev.out|head -1):"
echo "   NEW(1st)=$(grep -oE 't_total=[0-9.]+s' $B/results/abseed_rev/NEWdelta.out|tail -1)   OLD(2nd)=$(grep -oE 't_total=[0-9.]+s' $B/results/abseed_rev/OLDdelta.out|tail -1)"
echo ""
echo "INTERPRETATION:"
echo "  - if NEW slower in BOTH orders -> CODE (NEW genuinely slower)"
echo "  - if the 2nd run is slower in BOTH -> ORDER/environment (not the code)"
echo ""
echo "REVERSE per-step (NEW-1st vs OLD-2nd):"
for d in NEWdelta OLDdelta; do
  grep -oE "step +[0-9]+\] beam=.*t_total=[0-9.]+s" $B/results/abseed_rev/$d.out 2>/dev/null \
    | sed -E 's/step +([0-9]+)\].*t_total=([0-9.]+)s/\1 \2/' > /tmp/rev_$d.tt
done
printf "%-5s %-10s %-10s %-8s\n" step NEW1st OLD2nd "NEW-OLD"
join /tmp/rev_NEWdelta.tt /tmp/rev_OLDdelta.tt 2>/dev/null | awk 'NR%10==0{printf "%-5s %-10s %-10s %+.1f\n",$1,$2,$3,$2-$3}'
echo DONE
