#!/bin/bash
# Resume hierarchical_reduction_v14.py from checkpoint
cd /home/shih/work/IBPreduction

CHECKPOINT_DIR="checkpoints/reduction_111111m3"
OUTPUT="results/reduction_111111m3_v14.pkl"
LOGFILE="logs/hierarchical_v14_111111m3_resumed.log"

echo "Resuming hierarchical reduction v14 from checkpoint"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Output: $OUTPUT"
echo "  Log: $LOGFILE"

nohup python -u scripts/eval/hierarchical_reduction_v14.py \
    --resume "$CHECKPOINT_DIR" \
    --output "$OUTPUT" \
    --beam_width 20 \
    --max_steps 500 \
    --filter_mode subsector \
    --n_workers 16 \
    > "$LOGFILE" 2>&1 &

echo "Started with PID $!"
echo "Monitor with: tail -f $LOGFILE"
