#!/bin/bash
# v8 probe-74 action-select comparison at max-actions=5000, beam 40, 300 steps,
# now with per-step pre-cap valid-action logging (nvalid_max/nvalid_tot/capped).
# Strategies: maxweight (total-weight), sumweight, shortest.
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
TEMPLATE=$BASE/results/probe_74_v8_mwtotal/probe.sub
for strat in maxweight sumweight shortest; do
    D=$BASE/results/probe_74_v8_${strat}_5k
    mkdir -p $D
    sed -e "s#probe_74_v8_mwtotal#probe_74_v8_${strat}_5k#g" \
        -e "s#SAILIR_ACTION_SELECT=maxweight#SAILIR_ACTION_SELECT=${strat}#g" \
        $TEMPLATE > $D/probe.sub
    condor_submit $D/probe.sub
done
