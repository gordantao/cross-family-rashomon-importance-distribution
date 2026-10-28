#!/bin/bash
#SBATCH --job-name=top40-feature-comparison
#SBATCH --output=top40-feature-comparison_%A_%a.out
#SBATCH --error=top40-feature-comparison_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtao@unc.edu

# -------------------------------------------------------
# SLURM automatically sets:
#   SLURM_CPUS_PER_TASK
# which this Python script can use for parallel workers.
#
# Job array: one task per method (index into METHODS below), each with its own
# 50 cpus / 48h. Every method checkpoints <task>/<method>_result.json when it
# finishes; whichever task finishes last finds all five checkpoints and builds
# the combined outputs (top_feature_overlap.csv, run_settings.json,
# overall_task_summary.csv).
#
# Timed-out or failed tasks: resubmit -- checkpointed methods are skipped. To
# rerun specific methods only:
#   sbatch --array=3,4 experiments/staellert/run_staellert_top40_comparison.sl
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
PYTHON_ENV="/nas/longleaf/home/gtao/.conda/envs/zikry_lab-staellert/bin/python"

echo "Using python from: $PYTHON_ENV"
$PYTHON_ENV --version

# --- Run the top-k comparison analysis (classification-only) ---
# Methods: (0) forward stepwise selection via random forest,
#          (1) forward stepwise selection via logistic regression,
#          (2) single-family RID on a fully enumerated decision-tree Rashomon set,
#          (3) cross-family RID (unweighted), exhaustive grid candidate search,
#          (4) cross-family RID (unweighted), joint TruVaRImp candidate search.
# Results go to a fresh dir: RID now judges Rashomon membership/VI out-of-bag,
# so results in top40_feature_comparison/ (in-sample) are not comparable.
# 100 bootstraps, not 500: Staellert candidate fits average ~1 min, so a grid
# bootstrap is ~500 fit-minutes and 500 bootstraps on 50 cpus would need ~85h
# (TruVaRImp ~50h). At 100: grid ~17h, TruVaRImp ~10h.
METHODS=(
	stepwise_rf stepwise_logreg rid_tree cross_family_unweighted cross_family_unweighted_truvarimp
)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "Method:        $METHOD"

$PYTHON_ENV experiments/staellert/run_staellert_top40_comparison.py \
    --data-dir experiments/staellert/data/staellert_et_al \
    --output-dir experiments/staellert/results/top40_feature_comparison_oob \
    --tasks annotated_phase \
    --methods "$METHOD" \
    --top-k 40 \
    --rid-n-bootstraps 100 \
    --rid-n-models-pool 100 \
    --truvarimp-total-budget 300 \
    --stepwise-n-jobs "$SLURM_CPUS_PER_TASK" \
    --rid-n-jobs "$SLURM_CPUS_PER_TASK"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
