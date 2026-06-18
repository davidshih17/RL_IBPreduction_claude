#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
for i in $(seq 1 120); do grep -q "PROF2 DONE" $B/results/prof2/prof2.out 2>/dev/null && break; sleep 20; done
echo "=== current-state profile (post substitute_one Cython + sector cache) ==="
sed -n '/=== cProfile summary/,$p' $B/results/prof2/analysis.txt 2>/dev/null | head -40
echo DONE
