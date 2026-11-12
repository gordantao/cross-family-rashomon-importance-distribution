#!/bin/bash
#SBATCH --job-name=cross-rid-simulation
#SBATCH --output=cross-rid-simulation_%A_%a.out
#SBATCH --error=cross-rid-simulation_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-69
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtao@unc.edu

# -------------------------------------------------------
# SLURM automatically sets:
#   SLURM_CPUS_PER_TASK
# which your Python script can read for NUM_WORKERS.
#
# Grid-vs-TruVaRImp validation study. Real-data experiments use TruVaRImp
# candidate search only; this run checks it against the exhaustive grid over
# the identical shared candidate pool (500 candidates/family, budget 300 fits
# per bootstrap), so the grid's Rashomon set is the exact ground truth.
#
# Job array: one task per (DGP, beta) -- task i runs DGPS[i / 7] at
# BETAS[i % 7], i.e. 12 reps per task. Measured 2026-11-12 (Apple M2, 4
# workers, 1 BLAS thread each, chen beta=1, B=4): per bootstrap, grid ~156s
# (2,475 fits), TruVaRImp ~29s (~290 fits), rid_tree ~0.2s; stepwise + the
# evaluator panel ~25s per cell. At B=500 on 50 workers (10 waves) a cell is
# ~32 min, so one DGP's 84 cells would be ~44h (more on slower cluster cores)
# -- too close to a 48h limit for one task per DGP. Per (DGP, beta) it is
# ~6.5h per task at M2 speed; 24h leaves room for ~3.5x slower cores.
# Peak memory is ~0.3 GB per worker (~15 GB for 50).
# All tasks share one --output-dir: each writes only its own
# per_dgp/{study_results,search_coverage}_<dgp>__beta<beta>.csv, and the
# combined tables/plots are rebuilt from whatever per-(DGP, beta) files exist
# whenever a task finishes.
#
# Timed-out or failed tasks: resubmit with the same arguments -- cells already
# checkpointed are skipped. To rerun specific (DGP, beta) tasks only:
#   sbatch --array=0,9 experiments/nonlinear_interaction_simulation/run_nonlinear_interaction_simulation.sl
# -------------------------------------------------------

set -euo pipefail

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID (array ${SLURM_ARRAY_JOB_ID:-n/a}_${SLURM_ARRAY_TASK_ID:-n/a})"
echo "Node:          $(hostname)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory:        $SLURM_MEM_PER_NODE MB"
echo "Submit dir:    $SLURM_SUBMIT_DIR"
echo "Start time:    $(date)"
echo "========================================"

# --- Clean module environment ---
module purge

# --- Change to the working directory ---
cd "$SLURM_SUBMIT_DIR"

# --- One BLAS/OpenMP thread per process ---
# Parallelism comes from joblib worker processes (n_jobs = SLURM_CPUS_PER_TASK).
# joblib would cap each worker's BLAS pool at cpu_count // n_jobs, but defers
# to these variables when they're set -- setting them to SLURM_CPUS_PER_TASK
# gave every worker its own N-thread pool, i.e. N^2 threads on N cores (a
# ~400x slowdown of TruVaRImp's GP math; see
# experiments/nonlinear_interaction_simulation/profile_truvarimp.py).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- Real-time logging, no warning spam ---
# Unbuffered stdout so progress lines reach the .out file as they happen
# rather than only when Python's 8KB block buffer fills. PYTHONWARNINGS is
# inherited by every joblib worker process, where nearly all sklearn
# ConvergenceWarning/FutureWarning output originates (each script and
# rid.compute_rid* also ignore warnings in-process; this is belt-and-braces).
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

echo "Thread settings:"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"

# --- Use Python directly from your conda environment ---
PYTHON_ENV="/nas/longleaf/home/gtao/.conda/envs/zikry_lab-nonlinear_interaction_simulation/bin/python"

echo "Using python from: $PYTHON_ENV"
$PYTHON_ENV --version

# --- Run the analysis ---
# Per simulated cell: (1) cross-family RID, exhaustive grid over the shared
# pool, (2) cross-family RID, TruVaRImp over the same pool, (3) stepwise
# logistic regression, (4) stepwise random forest, (5) single-family RID on a
# fully enumerated decision-tree Rashomon set. Records ground-truth recovery
# for all five and TruVaRImp's Rashomon coverage / cost / RID agreement vs (1).
# The last 4 DGPs are the redundant-driver diagnostics (chen/friedman with a
# correlated duplicate of the true driver X1 at redundancy_corr 0.95/0.70),
# testing whether methods share importance credit across a redundant pair
# instead of arbitrarily crediting only one.
DGPS=(
	chen friedman monk1 monk3 custom_nonlinear custom_sin_log
	chen_redundant_r095 chen_redundant_r070 friedman_redundant_r095 friedman_redundant_r070
)
# Must match DEFAULT_BETA_GRID: seeds derive from a beta's index in it, so
# --betas subsets reproduce the full-run datasets.
BETAS=(0.25 0.5 1.0 1.5 2.0 3.0 4.0)
DGP="${DGPS[$((SLURM_ARRAY_TASK_ID / ${#BETAS[@]}))]}"
BETA="${BETAS[$((SLURM_ARRAY_TASK_ID % ${#BETAS[@]}))]}"
echo "DGP:           $DGP"
echo "beta:          $BETA"

$PYTHON_ENV experiments/nonlinear_interaction_simulation/run_nonlinear_interaction_simulation.py \
	--output-dir experiments/nonlinear_interaction_simulation/results/nonlinear_interaction_simulation \
	--dgps "$DGP" \
	--betas "$BETA" \
	--bootstraps 500 \
	--n-candidates-per-family 500 \
	--truvarimp-total-budget 300 \
	--num-workers "$SLURM_CPUS_PER_TASK"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
