#!/bin/bash
#SBATCH --job-name=falcon-cano-top40-auto-k
#SBATCH --output=falcon-cano-top40-auto-k_%j.out
#SBATCH --error=falcon-cano-top40-auto-k_%j.err
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

# --- Explicitly set thread counts to match SLURM allocation ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Thread settings:"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"

# --- Use Python directly from your conda environment ---
PYTHON_ENV="/nas/longleaf/home/gtao/.conda/envs/zikry_lab-falcon_cano/bin/python"

echo "Using python from: $PYTHON_ENV"
$PYTHON_ENV --version

# --- Run the top-k comparison analysis with elbow-based auto k-selection ---
# Same five methods as run_falcon_cano_top40_comparison.sl, but each method's
# feature count is chosen via elbow detection on its own native signal curve
# instead of a flat top-40 (--top-k auto; see feature_count_selection.py):
# RID-based methods elbow-detect prob_positive (P(phi>0) across the Rashomon
# set, already computed by fitting -- no extra refit), stepwise methods
# elbow-detect their own held-out CV-score-vs-step curve. --rid-metric is
# pinned explicitly to sub_mr (also the script's default) since this pipeline
# has previously had its default silently drift to iloco and back -- see
# CLAUDE.md's rid/core.py section for why iloco isn't a safe drop-in replacement.
$PYTHON_ENV experiments/falcon_cano/run_falcon_cano_top40_comparison.py \
    --data experiments/falcon_cano/falcon_cano_featured.csv \
    --output-dir experiments/falcon_cano/results/top40_feature_comparison_auto_k \
    --top-k auto \
    --top-k-max-sweep 100 \
    --rid-metric sub_mr \
    --rid-n-bootstraps 500 \
    --rid-n-models-pool 50 \
    --stepwise-n-jobs "$SLURM_CPUS_PER_TASK" \
    --rid-n-jobs "$SLURM_CPUS_PER_TASK"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
