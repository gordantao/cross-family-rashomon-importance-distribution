#!/bin/bash
#SBATCH --job-name=falcon-cano-top40-mrmr
#SBATCH --output=falcon-cano-top40-mrmr_%j.out
#SBATCH --error=falcon-cano-top40-mrmr_%j.err
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

# --- Run the top-k comparison analysis with mRMR-reordered RID selection ---
# Same five methods and elbow-based auto-k as
# run_falcon_cano_top40_comparison_auto_k.sl, but RID-based methods
# (rid_tree, cross_family_unweighted, cross_family_weighted) have their
# ranking greedily reordered by mRMR (mrmr_selection.py) before k-selection:
# trades off each feature's RID importance against its redundancy with
# already-picked features, instead of RID's plain marginal-importance sort.
# This targets the specific confound where stepwise selection wins a
# matched-k predictive comparison partly because its own greedy CV-scored
# search avoids redundant features "for free," while RID's marginal ranking
# doesn't. Under --top-k=auto, RID methods elbow-detect a held-out CV-gain
# curve here (not prob_positive -- see --rid-selection's help text).
# --rid-metric is pinned explicitly to sub_mr (also the script's default).
$PYTHON_ENV experiments/falcon_cano/run_falcon_cano_top40_comparison.py \
    --data experiments/falcon_cano/falcon_cano_featured.csv \
    --output-dir experiments/falcon_cano/results/top40_feature_comparison_auto_k_mrmr \
    --top-k auto \
    --top-k-max-sweep 100 \
    --rid-selection mrmr \
    --mrmr-lambda 1.0 \
    --rid-metric sub_mr \
    --rid-n-bootstraps 500 \
    --rid-n-models-pool 50 \
    --stepwise-n-jobs "$SLURM_CPUS_PER_TASK" \
    --rid-n-jobs "$SLURM_CPUS_PER_TASK"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
