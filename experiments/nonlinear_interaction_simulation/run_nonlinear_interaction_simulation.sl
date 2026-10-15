#!/bin/bash
#SBATCH --job-name=cross-rid-simulation
#SBATCH --output=cross-rid-simulation_%j.out
#SBATCH --error=cross-rid-simulation_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtao@unc.edu

# -------------------------------------------------------
# SLURM automatically sets:
#   SLURM_CPUS_PER_TASK
# which your Python script can read for NUM_WORKERS.
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

# --- Change to the working directory ---
cd "$SLURM_SUBMIT_DIR"

# --- Explicitly set thread counts to match SLURM allocation ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

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
# Benchmarks ground-truth feature recovery across five methods per simulated cell:
# (1) cross-family RID (unweighted), (2) cross-family RID (weighted),
# (3) stepwise logistic regression, (4) stepwise random forest,
# (5) single-family RID on a fully enumerated decision-tree Rashomon set.
# --include-redundant-dgps adds 4 diagnostic DGPs (chen/friedman with a
# correlated duplicate of the true driver X1 at redundancy_corr 0.95/0.70) on
# top of the standard 6, testing whether methods share importance credit
# across a redundant pair instead of arbitrarily crediting only one.
$PYTHON_ENV experiments/nonlinear_interaction_simulation/run_nonlinear_interaction_simulation.py \
	--output-dir experiments/nonlinear_interaction_simulation/results/nonlinear_interaction_simulation \
	--include-redundant-dgps \
	--bootstraps 500 \
	--models-per-class 100 \
	--num-workers "$SLURM_CPUS_PER_TASK"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
