#!/usr/bin/env python3
"""
Async hierarchical reduction with memoization.

Strategy:
1. Maintain a global expression and a cache of reduction results
2. Submit jobs for ALL non-master integrals not in cache or pending
3. As results come in, cache them and apply substitutions to the expression
4. Keep workers saturated - no artificial synchronization points
5. Cache hits avoid redundant work when stragglers produce already-reduced integrals

Usage:
    python hierarchical_reduction_async.py \
        --integral 1,1,1,1,1,1,-3 \
        --output results/reduction_async.pkl \
        --work-dir /scratch/ibp_async
"""

import sys
import argparse
import pickle
import time
import subprocess
import os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))

from ibp_env import IBPEnvironment, set_prime, set_paper_masters_only, is_master, weight, PRIME
from beam_search_classifier_v4_target import get_sector_mask

REPO_DIR = Path(__file__).parent.parent.parent.resolve()
PYTHON_PATH = "/het/p4/dshih/conda_envs/rl_dilogs/bin/python"


def integral_to_str(integral):
    """Convert integral tuple to string for filenames."""
    return '_'.join(str(x) for x in integral)


def apply_substitutions(expr, cache, prime):
    """Apply all cached substitutions to an expression until no more apply.

    This recursively substitutes any integral that's in the cache with its
    reduced form, until only masters and un-cached integrals remain.
    """
    changed = True
    iterations = 0
    while changed:
        changed = False
        iterations += 1
        new_expr = {}
        for integral, coeff in expr.items():
            if coeff == 0:
                continue
            if integral in cache:
                # Substitute: integral -> cache[integral]
                for sub_int, sub_coeff in cache[integral].items():
                    if sub_coeff == 0:
                        continue
                    new_expr[sub_int] = (new_expr.get(sub_int, 0) + coeff * sub_coeff) % prime
                changed = True
            else:
                new_expr[integral] = (new_expr.get(integral, 0) + coeff) % prime
        expr = {k: v for k, v in new_expr.items() if v != 0}

        # Safety check
        if iterations > 10000:
            print(f"WARNING: apply_substitutions exceeded 10000 iterations")
            break

    return expr


def get_non_masters(expr):
    """Get all non-master integrals in an expression."""
    return {i for i, c in expr.items() if c != 0 and not is_master(i)}


def create_condor_submit(work_dir, integral, job_name, output_file,
                         model_checkpoint, beam_width, max_steps, prime,
                         paper_masters_only=False, cpus=1):
    """Create a Condor submit file for a single-integral one-step reduction."""

    integral_str = ','.join(str(x) for x in integral)
    paper_masters_flag = ' --paper-masters-only' if paper_masters_only else ''
    n_workers_flag = f' --n_workers {cpus}' if cpus > 1 else ''
    memory = 4 * cpus  # Scale memory with CPUs

    # Use one-step worker - reduces by one weight level then returns
    # The async loop will cache and resubmit as needed
    submit_content = f"""universe = vanilla
executable = {PYTHON_PATH}
arguments = -u {REPO_DIR}/scripts/eval/reduce_integral_onestep_worker.py --integral='{integral_str}' --output {output_file} --model-checkpoint {model_checkpoint} --beam_width {beam_width} --max_steps {max_steps} --prime {prime} --device cpu -v{paper_masters_flag}{n_workers_flag}
output = {work_dir}/logs/{job_name}.out
error = {work_dir}/logs/{job_name}.err
log = {work_dir}/logs/{job_name}.log
request_cpus = {cpus}
request_memory = {memory}GB
request_disk = 1GB
+JobFlavour = "workday"
queue
"""

    submit_file = work_dir / f'{job_name}.sub'
    with open(submit_file, 'w') as f:
        f.write(submit_content)

    return submit_file


def submit_condor_job(submit_file):
    """Submit a Condor job and return the cluster ID."""
    try:
        result = subprocess.run(
            ['condor_submit', str(submit_file)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            # Parse output like: "1 job(s) submitted to cluster 91346."
            # Look for "cluster" followed by the ID (not "1 job(s)" which comes before)
            import re
            match = re.search(r'cluster\s+(\d+)', result.stdout, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    except Exception as e:
        print(f"Error submitting job: {e}")
        return None


def check_job_status(cluster_ids):
    """Check status of Condor jobs. Returns set of completed cluster IDs."""
    if not cluster_ids:
        return set()

    try:
        result = subprocess.run(
            ['condor_q'] + list(cluster_ids) + ['-format', '%d\n', 'ClusterId'],
            capture_output=True, text=True, timeout=30
        )
        running = set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()
        return set(cluster_ids) - running
    except Exception as e:
        print(f"Error checking job status: {e}")
        return set()


def main():
    parser = argparse.ArgumentParser(description='Async hierarchical reduction with memoization')
    parser.add_argument('--integral', type=str, required=True,
                        help='Starting integral indices (comma-separated)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output pickle file for reduction result')
    parser.add_argument('--work-dir', type=str, required=True,
                        help='Working directory for intermediate files')
    parser.add_argument('--model-checkpoint', type=str,
                        default=str(REPO_DIR / 'checkpoints/classifier_v5/best_model.pt'),
                        help='Path to model checkpoint')
    parser.add_argument('--beam_width', type=int, default=20,
                        help='Beam width for action search')
    parser.add_argument('--max_steps', type=int, default=10**15,
                        help='Max steps per integral (effectively unlimited)')
    parser.add_argument('--prime', type=int, default=1009,
                        help='Prime for modular arithmetic')
    parser.add_argument('--check-interval', type=int, default=5,
                        help='Seconds between job status checks')
    parser.add_argument('--max-concurrent', type=int, default=10000,
                        help='Maximum concurrent Condor jobs')
    parser.add_argument('--paper-masters-only', action='store_true',
                        help='Reduce to paper masters only (no corner integrals)')
    parser.add_argument('--straggler-timeout', type=int, default=1800,
                        help='Seconds before a job is considered a straggler (default 30 min)')
    parser.add_argument('--straggler-cpus', type=int, default=8,
                        help='CPUs to allocate when resubmitting stragglers')
    parser.add_argument('--dry-run', action='store_true',
                        help='Create submit files but do not submit')
    args = parser.parse_args()

    print('='*70)
    print('Async Hierarchical Reduction with Memoization')
    print('='*70)
    print(f'Config:')
    for k, v in vars(args).items():
        print(f'  {k}: {v}')
    print()

    # Setup
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / 'logs').mkdir(exist_ok=True)
    (work_dir / 'results').mkdir(exist_ok=True)

    set_prime(args.prime)
    if args.paper_masters_only:
        set_paper_masters_only(True)

    env = IBPEnvironment()

    # Parse starting integral
    starting_integral = tuple(int(x) for x in args.integral.split(','))
    print(f'Starting integral: I{list(starting_integral)}')
    print(f'Starting weight: {weight(starting_integral)}')
    print()

    # State
    expr = {starting_integral: 1}  # Current expression (linear combo of integrals)
    cache = {}  # integral -> reduced expression (memoization)
    pending = {}  # integral -> (cluster_id, output_file, submit_time, cpus)
    straggler_integrals = set()  # integrals that have been resubmitted as stragglers

    total_jobs = 0
    total_steps = 0
    cache_hits = 0
    stragglers_resubmitted = 0
    start_time = time.time()

    iteration = 0
    while True:
        iteration += 1

        # Apply all cached substitutions to the expression
        old_size = len(expr)
        expr = apply_substitutions(expr, cache, args.prime)

        # Count cache hits (integrals that were substituted)
        # This is approximate - we count how many integrals were removed

        # Find non-masters that need reduction
        non_masters = get_non_masters(expr)

        # Integrals to submit: non-masters not in cache and not pending
        to_submit = non_masters - set(cache.keys()) - set(pending.keys())

        # Limit concurrent jobs
        available_slots = args.max_concurrent - len(pending)
        if available_slots < len(to_submit):
            # Prioritize by weight (higher weight first - they're more likely to produce useful cache entries)
            to_submit = sorted(to_submit, key=lambda i: (-weight(i)[0], -weight(i)[1]))
            to_submit = set(to_submit[:available_slots])

        # Submit new jobs
        newly_submitted = 0
        for integral in to_submit:
            job_name = f"async_{total_jobs}_{integral_to_str(integral)}"
            output_file = work_dir / 'results' / f'{job_name}.pkl'

            submit_file = create_condor_submit(
                work_dir, integral, job_name, output_file,
                args.model_checkpoint, args.beam_width, args.max_steps, args.prime,
                args.paper_masters_only, cpus=1
            )

            if args.dry_run:
                pending[integral] = (None, output_file, time.time(), 1)
                total_jobs += 1
                newly_submitted += 1
            else:
                cluster_id = submit_condor_job(submit_file)
                if cluster_id:
                    pending[integral] = (cluster_id, output_file, time.time(), 1)
                    total_jobs += 1
                    newly_submitted += 1
                else:
                    print(f"  WARNING: Failed to submit I{list(integral)}")

        if newly_submitted > 0:
            print(f"[Iter {iteration}] Submitted {newly_submitted} jobs")

        # Check if done
        if not pending and not to_submit:
            print(f"\n[Iter {iteration}] All done!")
            break

        if args.dry_run:
            print(f"[Iter {iteration}] Dry run - stopping after job creation")
            break

        # Wait a bit
        time.sleep(args.check_interval)

        # Check for stragglers (jobs running too long) and resubmit with more CPUs
        current_time = time.time()
        for integral, (cluster_id, output_file, submit_time, cpus) in list(pending.items()):
            job_runtime = current_time - submit_time
            # Only resubmit as straggler if: running too long, not already a straggler, and using single CPU
            if (job_runtime > args.straggler_timeout and
                integral not in straggler_integrals and
                cpus == 1):
                # Kill the slow job
                if cluster_id:
                    try:
                        kill_result = subprocess.run(['condor_rm', str(cluster_id)], capture_output=True, text=True, timeout=10)
                        if kill_result.returncode == 0:
                            print(f"  Killed straggler job {cluster_id}", flush=True)
                        else:
                            # Job might have already finished on its own
                            print(f"  Job {cluster_id} already gone (completed or removed)", flush=True)
                    except Exception as e:
                        print(f"  WARNING: condor_rm {cluster_id} exception: {e}", flush=True)
                else:
                    print(f"  WARNING: No cluster_id for straggler I{list(integral)}", flush=True)

                # Remove old output file if it exists (partial)
                if output_file.exists():
                    try:
                        output_file.unlink()
                    except:
                        pass

                # Resubmit with more CPUs
                job_name = f"straggler_{total_jobs}_{integral_to_str(integral)}"
                new_output_file = work_dir / 'results' / f'{job_name}.pkl'

                submit_file = create_condor_submit(
                    work_dir, integral, job_name, new_output_file,
                    args.model_checkpoint, args.beam_width, args.max_steps, args.prime,
                    args.paper_masters_only, cpus=args.straggler_cpus
                )

                new_cluster_id = submit_condor_job(submit_file)
                if new_cluster_id:
                    pending[integral] = (new_cluster_id, new_output_file, time.time(), args.straggler_cpus)
                    straggler_integrals.add(integral)
                    stragglers_resubmitted += 1
                    total_jobs += 1
                    print(f"  STRAGGLER: Resubmitted I{list(integral)} with {args.straggler_cpus} CPUs "
                          f"(was running {job_runtime/60:.1f} min)")
                else:
                    print(f"  WARNING: Failed to resubmit straggler I{list(integral)}")

        # Check for completed jobs
        completed_integrals = []
        for integral, (cluster_id, output_file, submit_time, cpus) in list(pending.items()):
            if output_file.exists():
                # Load result
                try:
                    with open(output_file, 'rb') as f:
                        result = pickle.load(f)

                    if result.get('success'):
                        # Cache the reduction: integral -> result expression
                        result_expr = result.get('final_expr', result.get('expr', {}))
                        cache[integral] = result_expr
                        steps = result.get('steps', 0)
                        total_steps += steps

                        # Count new non-masters introduced
                        new_non_masters = get_non_masters(result_expr)
                        cached_count = len(new_non_masters & set(cache.keys()))
                        cache_hits += cached_count

                        # Get sector and level info
                        sector = tuple(get_sector_mask(integral))
                        level = sum(sector)
                        w = weight(integral)

                        print(f"  Completed I{list(integral)} (L{level} w={w[:2]}): {len(result_expr)} terms, "
                              f"{steps} steps, {len(new_non_masters)} non-masters "
                              f"({cached_count} cached)")
                    else:
                        # Failed - cache as identity (no reduction possible)
                        cache[integral] = {integral: 1}
                        print(f"  Failed I{list(integral)} - caching as identity")

                    completed_integrals.append(integral)

                except Exception as e:
                    print(f"  Error loading result for I{list(integral)}: {e}")
                    # Don't cache - will retry

        # Remove completed from pending
        for integral in completed_integrals:
            del pending[integral]

        # Status update with level breakdown
        masters_count = sum(1 for i, c in expr.items() if c != 0 and is_master(i))
        non_masters_count = len(non_masters)

        # Count non-masters by level
        level_counts = {}
        for i in non_masters:
            level = sum(get_sector_mask(i))
            level_counts[level] = level_counts.get(level, 0) + 1
        level_str = " ".join(f"L{l}:{c}" for l, c in sorted(level_counts.items(), reverse=True))

        print(f"[Iter {iteration}] {masters_count} masters, {non_masters_count} non-masters ({level_str}) | "
              f"Pending: {len(pending)} | Cache: {len(cache)} | Hits: {cache_hits}")

    # Final substitution
    expr = apply_substitutions(expr, cache, args.prime)

    # Final report
    elapsed = time.time() - start_time
    masters = {i: c for i, c in expr.items() if c != 0 and is_master(i)}
    non_masters = {i: c for i, c in expr.items() if c != 0 and not is_master(i)}

    print()
    print('='*70)
    print('ASYNC REDUCTION COMPLETE')
    print('='*70)
    print(f'Time: {elapsed:.1f}s ({elapsed/60:.1f} min)')
    print(f'Total jobs submitted: {total_jobs}')
    print(f'Stragglers resubmitted: {stragglers_resubmitted}')
    print(f'Total steps: {total_steps}')
    print(f'Cache size: {len(cache)}')
    print(f'Cache hits: {cache_hits}')
    print()
    print(f'Final expression: {len(masters)} masters, {len(non_masters)} non-masters')

    if non_masters:
        print(f'\nWARNING: {len(non_masters)} non-masters remaining!')
        for integral, coeff in sorted(non_masters.items(), key=lambda x: (-weight(x[0])[0], x[0])):
            print(f'  I{list(integral)} coeff={coeff} weight={weight(integral)[:2]}')
    else:
        print('\nSUCCESS! All integrals reduced to masters.')

    print(f'\nFinal masters:')
    for integral, coeff in sorted(masters.items(), key=lambda x: (-weight(x[0])[0], x[0])):
        print(f'  I{list(integral)} coeff={coeff} weight={weight(integral)[:2]}')

    # Save result
    result = {
        'final_expr': expr,
        'cache': cache,
        'total_jobs': total_jobs,
        'total_steps': total_steps,
        'elapsed_time': elapsed,
        'cache_hits': cache_hits,
    }
    with open(args.output, 'wb') as f:
        pickle.dump(result, f)

    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
