#!/bin/bash
# Collect steps-to-success for the 4x5 action-select experiment.
ROOT=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2/results/actionselect_exp
declare -A BASE=( [74]=81 [84]=182 [long]=968 [memhog]=720 )
printf "%-8s | %-9s %-9s %-9s %-9s %-9s\n" "probe" first900 last900 maxweight shortest sumweight
printf -- "---------+--------------------------------------------------------\n"
for probe in 74 84 long memhog; do
  printf "%-8s |" "$probe(${BASE[$probe]})"
  for strat in first900 last900 maxweight shortest sumweight; do
    O=$ROOT/${probe}_${strat}
    out=$O/probe.out
    if grep -q "SUCCESS" "$out" 2>/dev/null; then
      pl=$(grep -oE "path_len=[0-9]+" "$out" | tail -1 | grep -oE "[0-9]+")
      printf " %-9s" "${pl}✓"
    elif [ -f "$out" ]; then
      st=$(grep -oE "step +[0-9]+\]" "$out" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
      # job still in queue?
      q=$(condor_q -nobatch 2>/dev/null | grep -c "${probe}_${strat}\|SAILIR_ACTION_SELECT=${strat}.*${probe}" 2>/dev/null)
      printf " %-9s" "@${st:-0}"
    else
      printf " %-9s" "idle"
    fi
  done
  echo ""
done
echo ""
echo "queue: $(condor_q 2>/dev/null | tail -1)"
