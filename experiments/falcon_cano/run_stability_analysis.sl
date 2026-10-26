#!/bin/bash
#SBATCH --job-name=falcon-cano-stability
#SBATCH --output=falcon-cano-stability_%j.out
#SBATCH --error=falcon-cano-stability_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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

# --- Run the analysis ---
# Tests whether RID's own P(phi>0)-derived ambiguity score predicts genuine
# single-model instability under bootstrap resampling, on the real
# Falcon-Cano bioavailability dataset (no ground truth needed -- unlike the
# nonlinear simulation's redundancy/equivalence metrics).
$PYTHON_ENV experiments/falcon_cano/run_stability_analysis.py \
    --data experiments/falcon_cano/falcon_cano_featured.csv \
    --families Lasso,FullyEnumeratedTree \
    --n-repeats 30 \
    --output-dir experiments/falcon_cano/results/stability_analysis \
    --num-workers "$SLURM_CPUS_PER_TASK"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
