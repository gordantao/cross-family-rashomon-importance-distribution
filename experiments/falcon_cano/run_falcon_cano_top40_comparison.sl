#!/bin/bash
#SBATCH --job-name=falcon-cano-top40-comparison
#SBATCH --output=falcon-cano-top40-comparison_%j.out
#SBATCH --error=falcon-cano-top40-comparison_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtao@unc.edu

# -------------------------------------------------------
# SLURM automatically sets:
#   SLURM_CPUS_PER_TASK
# which this Python script can use for parallel workers.
# -------------------------------------------------------

set -euo pipefail

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory:        $SLURM_MEM_PER_NODE MB"
echo "Submit dir:    $SLURM_SUBMIT_DIR"
echo "Start time:    $(date)"
echo "========================================"

# --- Clean module environment ---
module purge

# --- Change to the repo root (submit this job from the repo root) ---
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
PYTHON_ENV="/nas/longleaf/home/gtao/.conda/envs/zikry_lab-falcon_cano/bin/python"

echo "Using python from: $PYTHON_ENV"
$PYTHON_ENV --version

# --- Run the top-k comparison analysis ---
# Compares: (1) forward stepwise selection via logistic regression,
#           (2) forward stepwise selection via random forest,
#           (3) single-family RID on a fully enumerated decision-tree Rashomon set,
#           (4) cross-family RID (family_balance_mode=unweighted),
#           (5) cross-family RID (family_balance_mode=weighted).
# Every method's selected features are scored with a common held-out CV
# evaluator (logistic regression) for a fair, apples-to-apples comparison.
# (--inject-redundant-duplicate is available as an opt-in diagnostic but is
# NOT used here -- it changes the feature space for every method, which
# would make this run's results incomparable to the standard benchmark. Run
# it as a separate one-off invocation instead, e.g. via sbatch --wrap.)
$PYTHON_ENV experiments/falcon_cano/run_falcon_cano_top40_comparison.py \
    --data experiments/falcon_cano/falcon_cano_featured.csv \
    --output-dir experiments/falcon_cano/results/top40_feature_comparison \
    --top-k 40 \
    --rid-n-bootstraps 500 \
    --rid-n-models-pool 100 \
    --stepwise-n-jobs "$SLURM_CPUS_PER_TASK" \
    --rid-n-jobs "$SLURM_CPUS_PER_TASK"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
