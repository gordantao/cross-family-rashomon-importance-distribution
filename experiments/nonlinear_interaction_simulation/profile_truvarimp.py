#!/usr/bin/env python3
"""
Profile cross-family RID's TruVaRImp candidate search on one simulated cell.

Two probes, both on the exact production path the simulation uses
(``run_cross_family_rid_truvarimp`` / ``run_cross_family_rid``, same
``RID_MODEL_CONFIGS``, same DGP seeds):

1. ``single`` -- a few bootstraps in-process (n_jobs=1) under cProfile, with
   BLAS/OpenMP threads capped at --blas-threads. Reports where one
   bootstrap's time goes: real candidate fits vs. GP refits vs. acquisition
   vs. VI computation. Answers "is TruVaRImp intrinsically slow?".
2. ``parallel`` -- one wave of bootstraps (--n-jobs workers) once per
   --inner-threads setting, timing the wall clock and recording how many
   BLAS threads each worker actually got. ``env`` means "inherit
   OMP_NUM_THREADS etc. from the environment", which is what the production
   .sl does (it exports them as SLURM_CPUS_PER_TASK); an integer forces that
   many threads per worker via joblib's ``inner_max_num_threads``. Answers
   "does thread oversubscription across workers explain the slowdown?".

Usage:
    python profile_truvarimp.py --probe single --bootstraps 2
    python profile_truvarimp.py --probe parallel --n-jobs 64 --inner-threads 1 env
    sbatch profile_truvarimp.sl
"""

import argparse
import cProfile
import io
import json
import logging
import os
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from joblib import Parallel, delayed, parallel_config
from joblib.externals.loky import get_reusable_executor
from threadpoolctl import threadpool_info, threadpool_limits

from rid import add_logging_arguments, configure_logging, silence_warnings

import run_nonlinear_interaction_simulation as base
import run_nonlinear_interaction_simulation_truvarimp as tv

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "results" / "profile_truvarimp")

# (label, file suffix, function name) -> cumulative time pulled out of cProfile.
# Nested rows are indented to show they're contained in the row above.
PHASE_FUNCTIONS = [
    ("cross-family RID total", "rid/core.py", "compute_rid_cross_family"),
    ("  candidate search (all phases)", "rid/truvarimp.py", "run_joint_truvarimp"),
    ("    evaluate (fit + score + GP refit)", "rid/truvarimp.py", "evaluate"),
    ("      GP refit", "rid/truvarimp.py", "_refit_gp"),
    ("        GaussianProcessRegressor.fit", "gaussian_process/_gpr.py", "fit"),
    ("        lookahead tables", "rid/truvarimp.py", "_precompute_lookahead_tables"),
    ("    acquisition (best_candidate_and_score)", "rid/truvarimp.py", "best_candidate_and_score"),
    ("    update_partition", "rid/truvarimp.py", "update_partition"),
    ("    harvest", "rid/truvarimp.py", "_harvest_predicted_rashomon"),
    ("  grid candidate training", "rid/models.py", "train"),
    ("  VI: sub_mr", "rid/core.py", "vi_sub_mr"),
    ("  VI: loco", "rid/core.py", "vi_loco"),
    ("  VI: iloco", "rid/core.py", "vi_iloco"),
]


def blas_thread_counts():
    """``{internal_api: num_threads}`` for every BLAS/OpenMP pool loaded in this process."""

    return {info["internal_api"]: info["num_threads"] for info in threadpool_info()}


def _worker_blas_threads():
    # Force the BLAS libraries to load in the worker before inspecting them.
    import numpy as np

    np.dot(np.ones((2, 2)), np.ones((2, 2)))
    import sklearn.gaussian_process  # noqa: F401

    return os.getpid(), blas_thread_counts()


def build_cell(dgp, beta, rep, sample_size, noise_std, base_seed, beta_grid):
    """Same dataset ``run_simulation_study`` builds for (dgp, beta, rep)."""

    dgp_index = list(base.ALL_SIMULATORS).index(dgp)
    beta_index = beta_grid.index(beta)
    seed = base_seed + dgp_index * 1_000_000 + beta_index * 10_000 + rep
    dataset = base.ALL_SIMULATORS[dgp](n=sample_size, beta=beta, noise_std=noise_std, seed=seed)
    X = dataset[base.feature_columns_from_dataset(dataset)]
    y = dataset["Y"].astype(int)
    return X, y


def run_cross_family(args, X, y, n_bootstraps, n_jobs):
    if args.candidate_search == "truvarimp_joint":
        return tv.run_cross_family_rid_truvarimp(
            X,
            y,
            n_bootstraps=n_bootstraps,
            total_budget=args.truvarimp_total_budget,
            epsilon=args.epsilon,
            family_balance_mode="unweighted",
            n_jobs=n_jobs,
            rid_metric=args.rid_metric,
        )
    return base.run_cross_family_rid(
        X,
        y,
        n_bootstraps=n_bootstraps,
        n_models_per_class=args.models_per_class,
        epsilon=args.epsilon,
        family_balance_mode="unweighted",
        n_jobs=n_jobs,
        rid_metric=args.rid_metric,
    )


def phase_table(stats, n_bootstraps):
    by_name = {}
    for (filename, _, funcname), (_, n_calls, _, cumtime, _) in stats.stats.items():
        for label, suffix, target in PHASE_FUNCTIONS:
            if funcname == target and filename.replace(os.sep, "/").endswith(suffix):
                calls, seconds = by_name.get(label, (0, 0.0))
                by_name[label] = (calls + n_calls, seconds + cumtime)

    total = by_name.get(PHASE_FUNCTIONS[0][0], (0, 0.0))[1] or 1.0
    lines = [f"{'phase':<46} {'calls':>8} {'total s':>10} {'s/bootstrap':>12} {'% total':>8}"]
    for label, _, _ in PHASE_FUNCTIONS:
        if label not in by_name:
            continue
        calls, seconds = by_name[label]
        lines.append(
            f"{label:<46} {calls:>8} {seconds:>10.2f} {seconds / n_bootstraps:>12.2f} "
            f"{100.0 * seconds / total:>7.1f}%"
        )
    return "\n".join(lines)


def probe_single(args, X, y):
    logger.info(f"[single] {args.bootstraps} bootstraps in-process, blas_threads={args.blas_threads}")
    profiler = cProfile.Profile()
    with threadpool_limits(limits=args.blas_threads):
        logger.info(f"[single] BLAS pools: {blas_thread_counts()}")
        start = time.perf_counter()
        profiler.enable()
        run_cross_family(args, X, y, n_bootstraps=args.bootstraps, n_jobs=1)
        profiler.disable()
        elapsed = time.perf_counter() - start

    prof_path = os.path.join(args.output_dir, f"single_{args.candidate_search}.prof")
    profiler.dump_stats(prof_path)

    top = io.StringIO()
    stats = pstats.Stats(profiler, stream=top)
    stats.sort_stats("cumulative").print_stats(args.top)

    table = phase_table(stats, args.bootstraps)
    report = (
        f"candidate_search={args.candidate_search} bootstraps={args.bootstraps} "
        f"blas_threads={args.blas_threads} wall={elapsed:.1f}s "
        f"({elapsed / args.bootstraps:.1f}s/bootstrap)\n\n{table}\n\n{top.getvalue()}"
    )
    report_path = os.path.join(args.output_dir, f"single_{args.candidate_search}.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)

    logger.info(f"[single] wall={elapsed:.1f}s ({elapsed / args.bootstraps:.1f}s/bootstrap)")
    logger.info("[single] phase breakdown:\n" + table)
    logger.info(f"[single] full report -> {report_path}; snakeviz/pstats -> {prof_path}")
    return {"wall_seconds": elapsed, "seconds_per_bootstrap": elapsed / args.bootstraps}


def probe_parallel(args, X, y):
    n_bootstraps = args.bootstraps_parallel or args.n_jobs
    results = {}
    for setting in args.inner_threads:
        inner = None if setting == "env" else int(setting)
        # Fresh worker pool per setting, so workers pick up the new thread env.
        get_reusable_executor().shutdown(wait=True)
        with parallel_config(backend="loky", inner_max_num_threads=inner):
            worker_threads = Parallel(n_jobs=args.n_jobs)(
                delayed(_worker_blas_threads)() for _ in range(args.n_jobs)
            )
            per_worker = sorted({json.dumps(counts, sort_keys=True) for _, counts in worker_threads})
            logger.info(
                f"[parallel inner_threads={setting}] n_jobs={args.n_jobs}, "
                f"BLAS threads per worker: {per_worker}"
            )
            start = time.perf_counter()
            run_cross_family(args, X, y, n_bootstraps=n_bootstraps, n_jobs=args.n_jobs)
            elapsed = time.perf_counter() - start
        logger.info(
            f"[parallel inner_threads={setting}] {n_bootstraps} bootstraps on {args.n_jobs} workers: "
            f"wall={elapsed:.1f}s"
        )
        results[setting] = {
            "n_jobs": args.n_jobs,
            "n_bootstraps": n_bootstraps,
            "wall_seconds": elapsed,
            "blas_threads_per_worker": [json.loads(item) for item in per_worker],
        }
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile cross-family RID's TruVaRImp candidate search on one simulated cell."
    )
    parser.add_argument("--probe", choices=["single", "parallel", "both"], default="both")
    parser.add_argument(
        "--candidate-search",
        choices=["truvarimp_joint", "grid"],
        default="truvarimp_joint",
        help="Profile TruVaRImp, or the grid path for a baseline (default: truvarimp_joint)",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dgp", choices=list(base.ALL_SIMULATORS), default="chen")
    parser.add_argument("--beta", type=float, default=0.25, help="Must be in the default beta grid")
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=base.DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--noise-std", type=float, default=base.DEFAULT_NOISE_STD)
    parser.add_argument("--base-seed", type=int, default=base.DEFAULT_BASE_SEED)
    parser.add_argument("--epsilon", type=float, default=base.DEFAULT_EPSILON)
    parser.add_argument("--rid-metric", type=str, default="sub_mr")
    parser.add_argument("--truvarimp-total-budget", type=int, default=tv.DEFAULT_TRUVARIMP_TOTAL_BUDGET)
    parser.add_argument(
        "--models-per-class", type=int, default=100, help="Grid path only (default: 100, as in the .sl)"
    )
    parser.add_argument(
        "--bootstraps", type=int, default=2, help="Bootstraps for the single-process probe (default: 2)"
    )
    parser.add_argument(
        "--blas-threads", type=int, default=1, help="BLAS/OpenMP threads for the single probe (default: 1)"
    )
    parser.add_argument("--top", type=int, default=40, help="cProfile rows to keep in the report")
    parser.add_argument(
        "--n-jobs", type=int, default=None, help="Workers for the parallel probe (default: all CPUs)"
    )
    parser.add_argument(
        "--bootstraps-parallel",
        type=int,
        default=None,
        help="Bootstraps for the parallel probe (default: --n-jobs, i.e. exactly one wave)",
    )
    parser.add_argument(
        "--inner-threads",
        nargs="+",
        default=["1", "env"],
        help=(
            "BLAS threads per worker to compare in the parallel probe: integers, or 'env' to "
            "inherit OMP_NUM_THREADS etc. as production does (default: 1 env). Run the cheap "
            "setting first so its number is in the log even if a slow one hits the time limit."
        ),
    )
    add_logging_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    silence_warnings()
    configure_logging(args.log_level, args.log_file)
    if args.n_jobs is None:
        args.n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    os.makedirs(args.output_dir, exist_ok=True)

    thread_env = {
        var: os.environ.get(var)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    }
    logger.info(f"cell: dgp={args.dgp} beta={args.beta} rep={args.rep} n={args.sample_size}")
    logger.info(f"candidate_search={args.candidate_search}, cpu_count={os.cpu_count()}, thread env={thread_env}")

    X, y = build_cell(
        args.dgp, args.beta, args.rep, args.sample_size, args.noise_std, args.base_seed, base.DEFAULT_BETA_GRID
    )

    summary = {"settings": vars(args), "thread_env": thread_env, "cpu_count": os.cpu_count()}
    summary_path = os.path.join(args.output_dir, f"summary_{args.candidate_search}.json")

    def save_summary():
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    if args.probe in ("single", "both"):
        summary["single"] = probe_single(args, X, y)
        save_summary()
    if args.probe in ("parallel", "both"):
        summary["parallel"] = probe_parallel(args, X, y)
        save_summary()
    logger.info(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
