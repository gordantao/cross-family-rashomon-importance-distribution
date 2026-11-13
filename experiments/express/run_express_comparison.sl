#!/bin/bash
#SBATCH --job-name=express-comparison
#SBATCH --output=express-comparison_%A_%a.out
#SBATCH --error=express-comparison_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtao@unc.edu

# -------------------------------------------------------
# Job array: one method per array index (4 methods, one label). Each method
# checkpoints <label>/<method>_result.json; whichever finishes last builds the
# combined outputs (top_feature_overlap.csv, answer_key_recovery.csv,
# run_settings.json), and overall_summary.csv is rebuilt from every finished
# label each time.
#
# Evaluation: the 279 patients are split once (--holdout-fraction 0.2,
# --holdout-seed 42; every row of a test patient is a test row). The label
# (per-combo z-score of log CR), gene selection, correlation filter and
# standardization are fit on the train patients; every method selects on the
# train rows (patient-grouped inner CV / grouped RID bootstrap) and the
# evaluator panel scores on the test patients' rows.
#
# Prerequisites (once, on the cluster): the conda env, the raw data, and the
# processed tables. The env does NOT exist on Longleaf yet -- create it first:
#   conda env create -f experiments/express/environment.yml   # -> zikry_lab-express
#   bash experiments/express/download_express_data.sh         # Table S9 must be copied in by hand
#   /nas/longleaf/home/gtao/.conda/envs/zikry_lab-express/bin/python experiments/express/prepare_express_data.py
#
# Secondary label: rerun with LABEL=cr_lt_075 (sbatch --export=ALL,LABEL=cr_lt_075 ...);
# results go to a separate <label>/ subdirectory of the same output dir.
#
# Timed-out or failed methods: resubmit -- checkpointed methods are skipped. To
# rerun one method only, e.g. cross_family_unweighted_truvarimp (method 3):
#   sbatch --array=3 experiments/express/run_express_comparison.sl
#
# Memory: 128G. cross_family_unweighted_truvarimp OOM'd at 32G and peaked at
# ~70G RSS across its 50 workers (job 9043384); the other methods need far less.
#
# Extra arguments: EXTRA_ARGS is appended to the python command line, and a
# repeated flag overrides the one above (argparse keeps the last). E.g. profile
# where cross_family_unweighted_truvarimp's per-bootstrap time goes, with 4
# bootstraps into a separate output dir (never the real run's checkpoints):
#   EXTRA_ARGS="--rid-n-bootstraps 4 --profile cprofile --output-dir experiments/express/results/express_profile" \
#     sbatch --array=3 --export=ALL experiments/express/run_express_comparison.sl
# -> <output-dir>/<label>/<method>_profile.csv (one row per bootstrap) and
#    <method>_profile/bootstrap_<b>.prof (python -m pstats <file>). --profile
#    timers skips the cProfile overhead and writes only the CSV.
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

PYTHON_ENV="/nas/longleaf/home/gtao/.conda/envs/zikry_lab-express/bin/python"
echo "Using python from: $PYTHON_ENV"
$PYTHON_ENV --version

LABEL="${LABEL:-zscore_mean}"
METHODS=(
	stepwise_rf stepwise_logreg rid_tree cross_family_unweighted_truvarimp
)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
echo "Label:         $LABEL"
echo "Method:        $METHOD"
EXTRA_ARGS="${EXTRA_ARGS:-}"
echo "Extra args:    ${EXTRA_ARGS:-none}"

$PYTHON_ENV experiments/express/run_express_comparison.py \
    --data-dir experiments/express/data/processed \
    --output-dir experiments/express/results/express_comparison \
    --label "$LABEL" \
    --holdout-fraction 0.2 \
    --holdout-seed 42 \
    --methods "$METHOD" \
    --top-k 40 \
    --n-top-variance-genes 500 \
    --rid-n-bootstraps 100 \
    --truvarimp-total-budget 300 \
    --stepwise-n-jobs "$SLURM_CPUS_PER_TASK" \
    --rid-n-jobs "$SLURM_CPUS_PER_TASK" \
    $EXTRA_ARGS  # unquoted on purpose: split into separate arguments

echo "========================================"
echo "End time: $(date)"
echo "========================================"
