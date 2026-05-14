#!/bin/bash
#SBATCH --job-name=rid-nonlinear-sim
#SBATCH --output=rid-nonlinear-sim_%j.out
#SBATCH --error=rid-nonlinear-sim_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --account=your_account
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtao@unc.edu

set -euo pipefail

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory:        $SLURM_MEM_PER_NODE MB"
echo "Start time:    $(date)"
echo "========================================"

module purge
module add anaconda/2024.02

cd "$SLURM_SUBMIT_DIR"

python run_nonlinear_interaction_simulation.py \
    --output-dir results/nonlinear_interaction_simulation \
    --sample-size 400 \
    --repetitions 12 \
    --bootstraps 12 \
    --models-per-class 6 \
    --beta-grid 0.25 0.5 1.0 1.5 2.0 3.0 4.0 \
    --noise-std 1.0 \
    --epsilon 0.05 \
    --family-balance-mode unweighted

echo "========================================"
echo "End time: $(date)"
echo "========================================"
