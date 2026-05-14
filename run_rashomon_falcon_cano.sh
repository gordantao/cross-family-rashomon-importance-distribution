#!/bin/bash
#SBATCH --job-name=rid-analysis
#SBATCH --output=rid-analysis_%j.out
#SBATCH --error=rid-analysis_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50          # <-- NUM_WORKERS is read from SLURM_CPUS_PER_TASK
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=general        # Uncomment and set your partition
#SBATCH --account=your_account     # Uncomment and set your account/allocation
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtao@unc.edu

# -------------------------------------------------------
# Sync note:
#   --cpus-per-task above automatically sets the env var
#   SLURM_CPUS_PER_TASK, which the Python script reads
#   to set NUM_WORKERS. No manual syncing needed.
# -------------------------------------------------------

set -euo pipefail

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory:        $SLURM_MEM_PER_NODE MB"
echo "Start time:    $(date)"
echo "========================================"

# --- Environment setup ---
# Adjust these to match your cluster's module system / conda location
module purge
module add anaconda/2024.02

# --- Change to the working directory ---
cd "$SLURM_SUBMIT_DIR"

# --- Run the analysis ---
python run_rashomon_falcon_cano.py \
    --data /users/g/t/gtao/falcon_cano_featured.csv \
    --output-dir results \
    --n-bootstraps 500 \
    --epsilon 0.05 \
    --n-models-pool 50

echo "========================================"
echo "End time: $(date)"
echo "========================================"
