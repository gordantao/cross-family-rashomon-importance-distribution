#!/bin/bash
#SBATCH --job-name=rid-stability-analysis
#SBATCH --output=rid-stability-analysis_%j.out
#SBATCH --error=rid-stability-analysis_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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

echo "Thread settings:"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"

# --- Use Python directly from your conda environment ---
PYTHON_ENV="/nas/longleaf/home/gtao/.conda/envs/zikry_lab-nonlinear_interaction_simulation/bin/python"

echo "Using python from: $PYTHON_ENV"
$PYTHON_ENV --version

# --- Run the analysis ---
# Tests whether RID's own P(phi>0)-derived ambiguity score predicts genuine
# single-model instability under bootstrap resampling: repeatedly refits a
# single Lasso/FullyEnumeratedTree model on bootstrap resamples of a fixed
# dataset, measures per-feature top-k flip rate, and correlates it against
# RID's own reported ambiguity on the same dataset. Run on both a standard
# DGP (chen) and its redundant-driver variant (chen_redundant_r095) -- the
# redundant duplicate of a true driver is the scenario expected to produce
# the sharpest single-model instability.
$PYTHON_ENV experiments/nonlinear_interaction_simulation/run_stability_analysis.py \
    --dgps chen,chen_redundant_r095 \
    --betas 1.0 2.0 \
    --families Lasso,FullyEnumeratedTree \
    --n-repeats 30 \
    --output-dir experiments/nonlinear_interaction_simulation/results/stability_analysis \
    --num-workers "$SLURM_CPUS_PER_TASK"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
