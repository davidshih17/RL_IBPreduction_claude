#!/bin/bash
# Run hierarchical_reduction_v14.py with checkpointing
cd /home/shih/work/IBPreduction

INTEGRAL="1,1,1,1,1,1,-3"
OUTPUT="results/reduction_111111m3_v14.pkl"
CHECKPOINT_DIR="checkpoints/reduction_111111m3"
LOGFILE="logs/hierarchical_v14_111111m3.log"

echo "Running hierarchical reduction v14 with checkpointing"
echo "  Integral: $INTEGRAL"
echo "  Output: $OUTPUT"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Log: $LOGFILE"

nohup python -u scripts/eval/hierarchical_reduction_v14.py \
    --integral "$INTEGRAL" \
    --output "$OUTPUT" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --prime 1009 \
    --beam_width 20 \
    --max_steps 500 \
    --filter_mode subsector \
    --n_workers 16 \
    > "$LOGFILE" 2>&1 &

echo "Started with PID $!"
echo "Monitor with: tail -f $LOGFILE"
