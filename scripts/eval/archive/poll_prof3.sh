#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
for i in $(seq 1 120); do grep -q "PROF3 DONE" $B/results/prof3/prof3.out 2>/dev/null && break; sleep 20; done
echo "host: $(grep -oE 'HOST: \S+' $B/results/prof3/prof3.out|head -1)"
echo ""
echo "############ PARALLELIZABILITY: phase breakdown (last 3 profiled steps) ############"
grep -E "step +[0-9]+\] beam=|P1:|P2:|P3:" $B/results/prof3/phase.out 2>/dev/null | tail -12
echo ""
echo "############ TOP TARGET: cProfile bucket + top functions ############"
sed -n '/=== cProfile summary/,$p' $B/results/prof3/analysis.txt 2>/dev/null | head -26
echo DONE
