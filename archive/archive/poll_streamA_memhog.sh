#!/bin/bash
# Wait for memhog_streamA (1657387) + longrunner_streamA (1657388) to finish,
# then: (d) bit-identical result, (c) memory peak vs the old (non-streamed) run.
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PY=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python
CMP=$BASE/scripts/eval/compare_reduction_results.py
LOG=$BASE/results/memhog_v7_streamA/streamA_final.txt

RM=$BASE/results/memhog_v7_streamA/result.pkl       # memhog streamA
RL=$BASE/results/longrunner_v7_streamA/result.pkl   # longrunner streamA
BM=$BASE/results/memhog_v7/result.pkl               # memhog v7 (already ==v6)
BL=$BASE/results/probe_longrunner_v6/result.pkl     # longrunner v6 baseline

for i in $(seq 1 1400); do
    if [ -s "$RM" ] && [ -s "$RL" ]; then break; fi
    sleep 30
done

membd_peak() { grep -oE "peak_rss=[0-9]+MB" "$1" 2>/dev/null | grep -oE "[0-9]+" | sort -rn | head -1; }
log_peak() { grep "MemoryUsage of job (MB)" "$1" 2>/dev/null | grep -oE ": +[0-9]+ " | grep -oE "[0-9]+" | sort -rn | head -1; }

{
    echo "=== memhog streamA: bit-identical (vs memhog_v7, already ==v6) ==="
    $PY $CMP "$BM" "$RM" 2>&1 | grep -E "masters|coeff|IDENTICAL"
    echo ""
    echo "=== longrunner streamA: bit-identical (vs probe_longrunner_v6) ==="
    $PY $CMP "$BL" "$RL" 2>&1 | grep -E "masters|coeff|IDENTICAL"
    echo ""
    echo "=== MEMORY: memhog OLD (non-streamed) vs streamA ==="
    echo "  condor_history MemoryUsage high-water (catches the ckpt spike):"
    echo "    OLD memhog_v7 (1654925): $(condor_history 1654925 -af MemoryUsage 2>/dev/null | head -1) MB"
    echo "    streamA memhog (1657387): $(condor_history 1657387 -af MemoryUsage 2>/dev/null | head -1) MB"
    echo "  MEMBD peak_rss max (steady-state, misses spike):"
    echo "    OLD memhog_v7: $(membd_peak $BASE/results/memhog_v7/probe.out) MB"
    echo "    streamA memhog: $(membd_peak $BASE/results/memhog_v7_streamA/probe.out) MB"
    echo "  probe.log MemoryUsage sample max:"
    echo "    streamA memhog: $(log_peak $BASE/results/memhog_v7_streamA/probe.log) MB"
} > "$LOG" 2>&1
cat "$LOG"
echo "DONE poll_streamA_memhog"
