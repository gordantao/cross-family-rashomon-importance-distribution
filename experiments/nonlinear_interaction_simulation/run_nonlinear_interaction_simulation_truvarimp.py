#!/usr/bin/env python3
"""
Cross-family RID simulation study runner, TruVaRImp candidate-search variant.

Identical ground-truth-recovery study to ``run_nonlinear_interaction_simulation.py``
(same DGPs, same four methods, same metrics/output schema, same per-DGP
checkpointing/resume), except cross-family RID (``cross_family_unweighted``)
populates its per-bootstrap candidate pool via joint TruVaRImp
(``candidate_search="truvarimp_joint"``, see rid/truvarimp.py) instead of
exhaustively refitting every point of each family's discretized grid.
``rid_tree`` (single-family RID) is unaffected -- ``compute_rid``/
``RashomonImportanceDistribution`` has no ``candidate_search`` option, so it
stays on the grid path exactly as before.

Reuses everything else (DGP simulators, study loop, checkpointing, metrics,
stepwise ranking, plotting, output schema) directly from ``run_nonlinear_interaction_simulation`` rather
than duplicating it, so results from the two scripts stay directly comparable
column-for-column; only the cross-family RID call and the CLI knob that
controls its budget differ.

Usage:
    python run_nonlinear_interaction_simulation_truvarimp.py
    sbatch run_nonlinear_interaction_simulation_truvarimp.sl
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rid import (
    CrossFamilyRashomonImportanceDistribution,
    add_logging_arguments,
    configure_logging,
    silence_warnings,
)

import run_nonlinear_interaction_simulation as base

logger = logging.getLogger(__name__)

DEFAULT_TRUVARIMP_TOTAL_BUDGET = 300
DEFAULT_OUTPUT_DIR = str(
    Path(__file__).resolve().parent / "results" / "nonlinear_interaction_simulation_truvarimp"
)


def run_cross_family_rid_truvarimp(
    X,
    y,
    *,
    n_bootstraps,
    total_budget,
    epsilon,
    family_balance_mode,
    n_jobs=1,
    rid_metric="sub_mr",
):
    estimator = CrossFamilyRashomonImportanceDistribution(
        model_configs=base.RID_MODEL_CONFIGS,
        epsilon=epsilon,
        n_bootstraps=n_bootstraps,
        vi_metrics=(rid_metric,),
        performance_metrics=base.RID_PERFORMANCE_METRICS,
        family_balance_mode=family_balance_mode,
        n_jobs=n_jobs,
        candidate_search="truvarimp_joint",
        truvarimp_config={"total_budget": total_budget},
    )

    estimator.fit(X, y)

    ranked_features = [feature for feature, _ in estimator.rank_features(rid_metric)]
    return ranked_features


def build_method_runners_truvarimp(
    X,
    y,
    cv,
    scoring,
    *,
    n_bootstraps,
    truvarimp_total_budget,
    n_models_per_class,
    epsilon,
    stepwise_logreg_C,
    stepwise_rf_n_estimators,
    random_state,
    n_jobs=1,
    rid_metric="sub_mr",
):
    """``base.build_method_runners`` with cross-family RID swapped to TruVaRImp search."""

    method_runners = base.build_method_runners(
        X,
        y,
        cv,
        scoring,
        n_bootstraps=n_bootstraps,
        n_models_per_class=n_models_per_class,
        epsilon=epsilon,
        stepwise_logreg_C=stepwise_logreg_C,
        stepwise_rf_n_estimators=stepwise_rf_n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        rid_metric=rid_metric,
    )
    method_runners["cross_family_unweighted"] = lambda: run_cross_family_rid_truvarimp(
        X,
        y,
        n_bootstraps=n_bootstraps,
        total_budget=truvarimp_total_budget,
        epsilon=epsilon,
        family_balance_mode="unweighted",
        n_jobs=n_jobs,
        rid_metric=rid_metric,
    )
    return method_runners


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ground-truth feature recovery across four methods (cross-family RID "
            "unweighted, stepwise logistic regression, stepwise random forest, and "
            "single-family tree RID) on simulated nonlinear-interaction datasets, with "
            "cross-family RID using joint TruVaRImp candidate search (candidate_search='truvarimp_joint') instead of exhaustive grid refitting."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for result tables and plot",
    )
    parser.add_argument(
        "--sample-size", type=int, default=base.DEFAULT_SAMPLE_SIZE, help="Sample size per simulation run"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=base.DEFAULT_REPETITIONS,
        help="Repetitions per beta per simulation",
    )
    parser.add_argument(
        "--bootstraps", type=int, default=base.DEFAULT_BOOTSTRAPS, help="RID bootstraps per fit"
    )
    parser.add_argument(
        "--truvarimp-total-budget",
        type=int,
        default=DEFAULT_TRUVARIMP_TOTAL_BUDGET,
        help=(
            "Shared per-bootstrap real-fit budget across all families for cross-family RID "
            "(overrides rid.truvarimp.DEFAULT_TRUVARIMP_CONFIG's total_budget; "
            "default: 300). Replaces --models-per-class for that method -- "
            "candidate_search='truvarimp_joint' ignores n_models_per_class entirely."
        ),
    )
    parser.add_argument(
        "--models-per-class",
        type=int,
        default=base.DEFAULT_MODELS_PER_CLASS,
        help=(
            "Candidate models per grid-point per bootstrap for rid_tree (single-family RID) "
            "only -- compute_rid has no candidate_search option, so the tree family always "
            "uses the grid path regardless of --truvarimp-total-budget."
        ),
    )
    parser.add_argument(
        "--beta-grid",
        type=float,
        nargs="+",
        default=base.DEFAULT_BETA_GRID,
        help="List of beta values used in latent score beta*signal+epsilon",
    )
    parser.add_argument(
        "--noise-std", type=float, default=base.DEFAULT_NOISE_STD, help="Noise standard deviation for epsilon"
    )
    parser.add_argument(
        "--base-seed", type=int, default=base.DEFAULT_BASE_SEED, help="Base random seed"
    )
    parser.add_argument(
        "--epsilon", type=float, default=base.DEFAULT_EPSILON, help="Rashomon epsilon"
    )
    parser.add_argument(
        "--rid-metric",
        type=str,
        default="sub_mr",
        help=(
            "RID variable-importance metric used by cross_family_unweighted and "
            "rid_tree (default: sub_mr; also available: loco, coef, iloco)."
        ),
    )
    parser.add_argument(
        "--stepwise-cv-splits",
        type=int,
        default=base.DEFAULT_STEPWISE_CV_SPLITS,
        help="CV folds for the forward-stepwise comparison methods (default: 5)",
    )
    parser.add_argument(
        "--stepwise-logreg-C",
        type=float,
        default=base.DEFAULT_STEPWISE_LOGREG_C,
        help="Inverse regularization strength for the stepwise logistic regression method (default: 1.0)",
    )
    parser.add_argument(
        "--stepwise-rf-n-estimators",
        type=int,
        default=base.DEFAULT_STEPWISE_RF_N_ESTIMATORS,
        help="Random forest trees for the stepwise random-forest method (default: 100)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=base.DEFAULT_RANDOM_STATE,
        help="Random state for stepwise CV folds and models (default: 42)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel workers (defaults to SLURM_CPUS_PER_TASK)",
    )
    parser.add_argument(
        "--include-redundant-dgps",
        action="store_true",
        default=False,
        help=(
            "Also run the redundant-driver DGP variants (REDUNDANT_SIMULATORS): "
            "chen/friedman with a correlated duplicate of the true driver X1 at "
            "redundancy_corr 0.95 and 0.70, on top of the standard DGPs. Diagnostic "
            "addition, off by default (default: False). Ignored when --dgps is given."
        ),
    )
    base.add_dgps_argument(parser)
    add_logging_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    silence_warnings()
    configure_logging(args.log_level, args.log_file)
    n_jobs = base.resolve_num_workers(args.num_workers)

    simulators = base.select_simulators(args.dgps, args.include_redundant_dgps)

    study_settings = {
        "candidate_search": "truvarimp_joint",
        "sample_size": args.sample_size,
        "repetitions_per_beta": args.repetitions,
        "bootstraps": args.bootstraps,
        "truvarimp_total_budget": args.truvarimp_total_budget,
        "models_per_class_rid_tree_only": args.models_per_class,
        "beta_grid": args.beta_grid,
        "noise_std": args.noise_std,
        "base_seed": args.base_seed,
        "epsilon": args.epsilon,
        "rid_metric": args.rid_metric,
        "stepwise_cv_splits": args.stepwise_cv_splits,
        "stepwise_logreg_C": args.stepwise_logreg_C,
        "stepwise_rf_n_estimators": args.stepwise_rf_n_estimators,
        "random_state": args.random_state,
        "num_workers": n_jobs,
        "methods_compared": list(base.METHODS),
    }
    base.prepare_output_dir(args.output_dir, study_settings)

    logger.info("=" * 70)
    logger.info("Running nonlinear interaction simulation study (TruVaRImp candidate search)")
    logger.info(f"output_dir={args.output_dir}")
    logger.info(f"sample_size={args.sample_size}")
    logger.info(f"repetitions={args.repetitions}")
    logger.info(f"bootstraps={args.bootstraps}")
    logger.info(f"truvarimp_total_budget={args.truvarimp_total_budget}")
    logger.info(f"models_per_class (rid_tree only)={args.models_per_class}")
    logger.info(f"beta_grid={args.beta_grid}")
    logger.info(f"noise_std={args.noise_std}")
    logger.info(f"epsilon={args.epsilon}")
    logger.info(f"rid_metric={args.rid_metric}")
    logger.info(f"stepwise_cv_splits={args.stepwise_cv_splits}")
    logger.info(f"stepwise_logreg_C={args.stepwise_logreg_C}")
    logger.info(f"stepwise_rf_n_estimators={args.stepwise_rf_n_estimators}")
    logger.info(f"methods_compared={list(base.METHODS)}")
    logger.info(f"simulations_included={list(simulators.keys())}")
    logger.info(f"n_jobs={n_jobs}")
    logger.info("=" * 70)

    started = time.time()

    base.run_simulation_study(
        simulators,
        beta_grid=args.beta_grid,
        sample_size=args.sample_size,
        n_repetitions=args.repetitions,
        noise_std=args.noise_std,
        base_seed=args.base_seed,
        stepwise_cv_splits=args.stepwise_cv_splits,
        random_state=args.random_state,
        make_method_runners=lambda X, y, cv, scoring: build_method_runners_truvarimp(
            X,
            y,
            cv,
            scoring,
            n_bootstraps=args.bootstraps,
            truvarimp_total_budget=args.truvarimp_total_budget,
            n_models_per_class=args.models_per_class,
            epsilon=args.epsilon,
            stepwise_logreg_C=args.stepwise_logreg_C,
            stepwise_rf_n_estimators=args.stepwise_rf_n_estimators,
            random_state=args.random_state,
            n_jobs=n_jobs,
            rid_metric=args.rid_metric,
        ),
        output_dir=args.output_dir,
    )
    output_paths = base.write_combined_outputs(args.output_dir)

    elapsed = time.time() - started
    logger.info("Saved outputs:")
    for path in [os.path.join(args.output_dir, base.PER_DGP_SUBDIR), *output_paths]:
        logger.info(f"  {path}")
    logger.info(f"Finished in {elapsed:.1f}s ({elapsed / 60.0:.1f} min)")


if __name__ == "__main__":
    main()
