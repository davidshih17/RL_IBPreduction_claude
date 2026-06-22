#!/bin/bash
B=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
for i in $(seq 1 120); do grep -q "ALL PROF DONE" $B/logs/prof_driver.log 2>/dev/null && break; sleep 15; done
echo "###### (1) PHASE BREAKDOWN — last profiled step (deepest) ######"
grep -E "step +[0-9]+\] beam=|P1:|P2:|P3:" $B/logs/prof_phase.log 2>/dev/null | tail -8
echo ""
echo "###### size counters (equation/array sizes) ######"
grep -oE "cu_size_max=[0-9]+|rs_max=[0-9]+|n_iraws_total=[0-9]+|n_valid_total=[0-9]+|enum_calls=[0-9]+|apply_calls=[0-9]+" $B/logs/prof_phase.log 2>/dev/null | tail -8
echo ""
echo "###### (2) cProfile bucket summary + top functions ######"
sed -n '/=== cProfile summary/,$p' $B/logs/prof_cfn.log 2>/dev/null
echo DONE
