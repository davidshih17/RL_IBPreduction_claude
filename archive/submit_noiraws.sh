#!/bin/bash
# v8 probe-74: maxweight (total-weight) + max-actions 5000 + NO iraws-keep-first
# (full indirect-relation set). iraws-keep-first was a MEMORY bound, so request
# more memory and watch RSS. beam 40, 300 steps.
set -u
BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
SRC=$BASE/results/probe_74_v8_maxweight_5k/probe.sub
D=$BASE/results/probe_74_v8_mw5k_noiraws
mkdir -p $D
sed -e "s#probe_74_v8_maxweight_5k#probe_74_v8_mw5k_noiraws#g" \
    -e "s/ --iraws-keep-first 50//g" \
    -e "s/request_memory = 32GB/request_memory = 48GB/g" \
    $SRC > $D/probe.sub
condor_submit $D/probe.sub
echo "--- verify flag removed ---"
grep -oE "iraws[^ ]* [0-9]+|max-actions [0-9]+|request_memory = [0-9]+GB" $D/probe.sub