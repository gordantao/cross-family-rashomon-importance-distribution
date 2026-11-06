#!/bin/bash
#SBATCH --job-name=rashomon-family-diag
#SBATCH --output=rashomon-family-diag_%j.out
#SBATCH --error=rashomon-family-diag_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtao@unc.edu

# -------------------------------------------------------
# Re-runs the grid cross-family candidate search on Staellert annotated_phase
# (same splits, grids and seeds as cross_family_unweighted in
# run_staellert_top40_comparison.sl) and records every candidate's OOB losses
# and calibration stats instead of a Rashomon set -- see the script docstring.
#
# One bootstrap per worker, 100 bootstraps on 50 cpus = 2 waves. Models are
# scored and dropped immediately, so memory stays far below the 96G the grid
# comparison task needs. Each bootstrap checkpoints
# results/rashomon_family_diagnostics/per_bootstrap/b<NNN>_*; resubmit after a
# timeout and finished bootstraps are skipped.
#
# Afterwards, copy results/rashomon_family_diagnostics/ back and run locally:
#   python experiments/staellert/diagnose_rashomon_family_losses.py plot
# -------------------------------------------------------

set -euo pipefail

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Start time:    $(date)"
echo "========================================"

module purge
cd "$SLURM_SUBMIT_DIR"

# One BLAS/OpenMP thread per joblib worker (see run_staellert_top40_comparison.sl).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

PYTHON_ENV="/nas/longleaf/home/gtao/.conda/envs/zikry_lab-staellert/bin/python"
$PYTHON_ENV --version

$PYTHON_ENV experiments/staellert/diagnose_rashomon_family_losses.py collect \
    --data-dir experiments/staellert/data/staellert_et_al \
    --output-dir experiments/staellert/results/rashomon_family_diagnostics \
    --n-bootstraps 100 \
    --n-models-pool 100 \
    --n-jobs "$SLURM_CPUS_PER_TASK"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
