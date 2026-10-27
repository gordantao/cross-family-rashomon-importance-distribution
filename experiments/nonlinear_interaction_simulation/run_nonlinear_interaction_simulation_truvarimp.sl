#!/bin/bash
#SBATCH --job-name=cross-rid-simulation-truvarimp
#SBATCH --output=cross-rid-simulation-truvarimp_%A_%a.out
#SBATCH --error=cross-rid-simulation-truvarimp_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-9
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtao@unc.edu

# -------------------------------------------------------
# SLURM automatically sets:
#   SLURM_CPUS_PER_TASK
# which your Python script can read for NUM_WORKERS.
#
# Job array: one task per DGP (index into DGPS below), same layout as
# run_nonlinear_interaction_simulation.sl. This used to be one 250-cpu/200G
# job, which (a) was ~400x slower than necessary from BLAS oversubscription
# (fixed below) and (b) sat pending for lack of a node with 250 free cores.
# 50 cpus/32G per task is the configuration the base array schedules
# immediately and runs stably at (640MB/core).
#
# Timed-out or failed tasks: resubmit with the same arguments -- cells already
# checkpointed are skipped. To rerun specific DGPs only:
#   sbatch --array=0,3 experiments/nonlinear_interaction_simulation/run_nonlinear_interaction_simulation_truvarimp.sl
# -------------------------------------------------------

set -euo pipefail

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID (array ${SLURM_ARRAY_JOB_ID:-n/a}_${SLURM_ARRAY_TASK_ID:-n/a})"
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
PYTHON_ENV="/nas/longleaf/home/gtao/.conda/envs/zikry_lab-nonlinear_interaction_simulation/bin/python"

echo "Using python from: $PYTHON_ENV"
$PYTHON_ENV --version

# --- Run the analysis ---
# Same ground-truth-recovery study as run_nonlinear_interaction_simulation.sl,
# except cross_family_unweighted populates its
# per-bootstrap candidate pool via joint TruVaRImp (CASH-style active search,
# rid/truvarimp.py) instead of exhaustively refitting every grid point.
# rid_tree (single-family RID) has no candidate_search option and stays on
# the grid path, controlled by --models-per-class as before.
DGPS=(
	chen friedman monk1 monk3 custom_nonlinear custom_sin_log
	chen_redundant_r095 chen_redundant_r070 friedman_redundant_r095 friedman_redundant_r070
)
DGP="${DGPS[$SLURM_ARRAY_TASK_ID]}"
echo "DGP:           $DGP"

$PYTHON_ENV experiments/nonlinear_interaction_simulation/run_nonlinear_interaction_simulation_truvarimp.py \
	--output-dir experiments/nonlinear_interaction_simulation/results/nonlinear_interaction_simulation_truvarimp \
	--dgps "$DGP" \
	--bootstraps 500 \
	--truvarimp-total-budget 300 \
	--models-per-class 100 \
	--num-workers "$SLURM_CPUS_PER_TASK"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
