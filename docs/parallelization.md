# Async Parallel IBP Reduction with Condor

This document describes our approach to parallelizing IBP reduction across Condor CPU nodes, achieving ~10x speedup over sequential execution.

## Overview

The key insight is that IBP reduction can be parallelized at the **one-step level**: each reduction step takes an integral and produces a linear combination of simpler integrals. These one-step reductions are independent and can be distributed across multiple workers.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Main Controller                               │
│  (hierarchical_reduction_async.py)                              │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Work Queue   │  │ Memoization  │  │  Straggler   │          │
│  │ (non-masters)│  │    Cache     │  │  Detection   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Submit Condor jobs
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Condor Cluster                              │
│                                                                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │Worker 1 │  │Worker 2 │  │Worker 3 │  │Worker N │   ...      │
│  │(1 CPU)  │  │(1 CPU)  │  │(1 CPU)  │  │(8 CPU)  │            │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘            │
│                                                                  │
│  Each worker: reduce_integral_onestep_worker.py                 │
│  - Takes one integral                                           │
│  - Runs beam search for one weight level                        │
│  - Returns reduced expression                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Async Work Distribution

Unlike level-synchronized approaches, our async method:
- Submits all pending non-masters as Condor jobs immediately
- Doesn't wait for all jobs at a level to complete
- Processes results as they arrive and submits new work

This keeps workers saturated and avoids bottlenecks from slow integrals.

### 2. Memoization Cache

Many integrals appear multiple times during reduction (from different parent integrals). The memoization cache:
- Stores: `integral → reduced expression`
- Avoids redundant reductions when stragglers produce already-cached results
- Achieved ~55,000 cache hits for the hard integral (saving significant compute)

### 3. Straggler Detection and Handling

Some integrals take much longer than others (especially in lower sectors with high dot products). Our straggler handling:

1. **Detection**: Jobs running longer than `--straggler-timeout` (default 30 min)
2. **Kill**: Original single-CPU job is terminated via `condor_rm`
3. **Resubmit**: New job with more CPUs (`--straggler-cpus`, default 8)
4. **Parallel beam search**: Worker uses multiprocessing to evaluate beam candidates faster

This prevents a single slow integral from blocking overall progress.

### 4. Paper-Masters-Only Mode

The `--paper-masters-only` flag restricts reduction to exactly the 16 master integrals from arXiv:2502.05121 (equation 2.5):
- No corner integrals in uncovered sectors
- Cleaner final result matching the paper's basis
- Slightly more reduction work but consistent output

## Performance Results

### Test Integral: I[3,2,1,3,2,2,-6]

| Approach | Time | Speedup |
|----------|------|---------|
| Sequential (V14) | ~20 hours | 1x |
| Async parallel (broken straggler handling) | 222 min | 5.4x |
| Async parallel (fixed straggler handling) | **115 min** | **~10x** |

### Statistics from successful run:
- Total jobs submitted: 21,096
- Stragglers resubmitted: 24
- Total reduction steps: 131,769
- Cache size: 21,072 entries
- Cache hits: 55,075
- Final result: 16 paper masters

## Usage

### Basic Usage

```bash
python -u scripts/eval/hierarchical_reduction_async.py \
    --integral 3,2,1,3,2,2,-6 \
    --output results/reduction.pkl \
    --work-dir /scratch/ibp_async \
    --paper-masters-only
```

### With Custom Straggler Settings

```bash
python -u scripts/eval/hierarchical_reduction_async.py \
    --integral 3,2,1,3,2,2,-6 \
    --output results/reduction.pkl \
    --work-dir /scratch/ibp_async \
    --paper-masters-only \
    --straggler-timeout 1800 \   # 30 minutes
    --straggler-cpus 8           # Resubmit with 8 CPUs
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--integral` | required | Starting integral (comma-separated indices) |
| `--output` | required | Output pickle file |
| `--work-dir` | required | Working directory for Condor files |
| `--paper-masters-only` | False | Reduce to 16 paper masters only |
| `--straggler-timeout` | 1800 | Seconds before job is considered straggler |
| `--straggler-cpus` | 8 | CPUs for resubmitted straggler jobs |
| `--beam_width` | 20 | Beam width for search |
| `--max_steps` | 10^15 | Max steps per integral (effectively unlimited) |
| `--prime` | 1009 | Prime for modular arithmetic |
| `--check-interval` | 5 | Seconds between job status checks |

## Implementation Details

### Worker Script

Each Condor job runs `reduce_integral_onestep_worker.py`:
1. Loads the ML model
2. Receives one integral to reduce
3. Runs beam search until weight decreases by one level
4. Returns the reduced expression as a pickle file

### Job Submission

Jobs are submitted via dynamically generated Condor submit files:
```
universe = vanilla
executable = /path/to/python
arguments = -u reduce_integral_onestep_worker.py --integral='...' ...
request_cpus = 1  (or 8 for stragglers)
request_memory = 4GB  (or 32GB for stragglers)
```

### Straggler Handling Bug Fix

An important bug was discovered and fixed: the cluster ID parsing was incorrectly extracting "1" from "1 job(s) submitted to cluster 91346" instead of the actual cluster ID. This caused `condor_rm` to fail silently, leaving orphan jobs running. The fix uses regex: `r'cluster\s+(\d+)'` to correctly extract the ID after "cluster".

## Requirements

- **HTCondor**: Cluster with available CPU nodes
- **Shared filesystem**: Work directory must be accessible from all nodes
- **Python environment**: Same environment on all nodes with torch, numpy

## Future Improvements

Potential enhancements:
1. **GPU workers**: Use GPU nodes for faster beam search
2. **Adaptive timeouts**: Learn typical reduction times per sector level
3. **Priority scheduling**: Prioritize higher-weight integrals
4. **Checkpoint/resume**: Save cache state for crash recovery
