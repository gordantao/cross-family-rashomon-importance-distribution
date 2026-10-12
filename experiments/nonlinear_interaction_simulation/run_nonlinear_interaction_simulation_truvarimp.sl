#!/bin/bash
#SBATCH --job-name=cross-rid-simulation-truvarimp
#SBATCH --output=cross-rid-simulation-truvarimp_%j.out
#SBATCH --error=cross-rid-simulation-truvarimp_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=250
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtao@unc.edu

# -------------------------------------------------------
# SLURM automatically sets:
#   SLURM_CPUS_PER_TASK
# which your Python script can read for NUM_WORKERS.
#
# NOTE: --cpus-per-task=250 requests 250 cores on a SINGLE node
# (--ntasks=1). Every other .sl in this repo runs at 50 -- confirm a
# Longleaf partition actually has a node with >=250 cores (e.g. `sinfo -N -l`)
# before submitting, or this job will sit PENDING indefinitely rather than
# erroring.
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
# Same ground-truth-recovery study as run_nonlinear_interaction_simulation.sl,
# except cross_family_unweighted/cross_family_weighted populate their
# per-bootstrap candidate pool via joint TruVaRImp (CASH-style active search,
# rid/truvarimp.py) instead of exhaustively refitting every grid point.
# rid_tree (single-family RID) has no candidate_search option and stays on
# the grid path, controlled by --models-per-class as before.
$PYTHON_ENV experiments/nonlinear_interaction_simulation/run_nonlinear_interaction_simulation_truvarimp.py \
	--output-dir experiments/nonlinear_interaction_simulation/results/nonlinear_interaction_simulation_truvarimp \
	--include-redundant-dgps \
	--bootstraps 500 \
	--truvarimp-total-budget 300 \
	--models-per-class 100 \
	--num-workers "$SLURM_CPUS_PER_TASK"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
