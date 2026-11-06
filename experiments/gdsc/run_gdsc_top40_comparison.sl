#!/bin/bash
#SBATCH --job-name=gdsc-top40-comparison
#SBATCH --output=gdsc-top40-comparison_%A_%a.out
#SBATCH --error=gdsc-top40-comparison_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-24
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtao@unc.edu

# -------------------------------------------------------
# Job array: one (task, method) pair per array index -- 5 GDSC drug tasks x 5
# methods = 25 indices, index = task_idx * 5 + method_idx. Each pair
# checkpoints <task>/<method>_result.json; whichever pair finishes a task last
# builds that task's combined outputs (top_feature_overlap.csv,
# biomarker_recovery.csv, run_settings.json), and overall_task_summary.csv is
# rebuilt from every finished task each time.
#
# Prerequisites (once, on the cluster): the conda env, the raw data, and the
# processed panel.
#   conda env create -f experiments/gdsc/environment.yml   # -> zikry_lab-gdsc
#   bash experiments/gdsc/download_gdsc_data.sh
#   /nas/longleaf/home/gtao/.conda/envs/zikry_lab-gdsc/bin/python experiments/gdsc/prepare_gdsc_data.py
#
# Memory: 96G, matching the Staellert grid cross-family task that OOM'd at 32G
# (each of 50 workers holds a bootstrap's full ~520-model candidate pool).
#
# Timed-out or failed pairs: resubmit -- checkpointed pairs are skipped. To
# rerun specific pairs only, e.g. every cross_family_unweighted (method 3):
#   sbatch --array=3,8,13,18,23 experiments/gdsc/run_gdsc_top40_comparison.sl
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

module purge
cd "$SLURM_SUBMIT_DIR"

# One BLAS/OpenMP thread per joblib worker (see
# experiments/staellert/run_staellert_top40_comparison.sl for why).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

PYTHON_ENV="/nas/longleaf/home/gtao/.conda/envs/zikry_lab-gdsc/bin/python"
echo "Using python from: $PYTHON_ENV"
$PYTHON_ENV --version

TASKS=(dabrafenib trametinib nutlin3a lapatinib talazoparib)
METHODS=(
	stepwise_rf stepwise_logreg rid_tree cross_family_unweighted cross_family_unweighted_truvarimp
)
TASK="${TASKS[$((SLURM_ARRAY_TASK_ID / ${#METHODS[@]}))]}"
METHOD="${METHODS[$((SLURM_ARRAY_TASK_ID % ${#METHODS[@]}))]}"
echo "Task:          $TASK"
echo "Method:        $METHOD"

$PYTHON_ENV experiments/gdsc/run_gdsc_top40_comparison.py \
    --data-dir experiments/gdsc/data/processed \
    --output-dir experiments/gdsc/results/top40_feature_comparison \
    --tasks "$TASK" \
    --methods "$METHOD" \
    --top-k 40 \
    --rid-n-bootstraps 100 \
    --rid-n-models-pool 100 \
    --truvarimp-total-budget 300 \
    --stepwise-n-jobs "$SLURM_CPUS_PER_TASK" \
    --rid-n-jobs "$SLURM_CPUS_PER_TASK"

echo "========================================"
echo "End time: $(date)"
echo "========================================"
