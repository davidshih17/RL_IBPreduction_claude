#!/bin/bash
# Quick import-only smoke test for delta_beam_search + delta_onestep_worker.
# No Condor — just verify the modules load cleanly on the login node.
set -e

BASE=/het/p4/dshih/jet_images-deep_learning/SAILIR_phase2
PYTHON=/het/p4/dshih/jet_images-deep_learning/RL_MIR_IBP/conda_env/bin/python

mkdir -p $BASE/results/delta_smoke
LOG=$BASE/results/delta_smoke/import_test.log

cd $BASE
PYTHONUNBUFFERED=1 $PYTHON -c "
import sys
sys.path.insert(0, 'scripts/eval')
from sailir.delta_beam_search import (
    DeltaState, beam_search_delta, _apply_action, _sort_key, _dedup_key_rs,
)
print('OK: delta_beam_search imports')
import importlib.util
spec = importlib.util.spec_from_file_location('delta_onestep_worker',
    'scripts/eval/delta_onestep_worker.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print('OK: delta_onestep_worker imports')
" > $LOG 2>&1

cat $LOG
