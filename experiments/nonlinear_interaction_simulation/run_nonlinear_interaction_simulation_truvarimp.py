#!/usr/bin/env python3
"""
Cross-family RID simulation study runner, TruVaRImp candidate-search variant.

Identical ground-truth-recovery study to ``run_nonlinear_interaction_simulation.py``
(same DGPs, same five methods, same metrics/output schema), except the two
cross-family RID methods (``cross_family_unweighted``, ``cross_family_weighted``)
populate their per-bootstrap candidate pool via joint TruVaRImp
(``candidate_search="truvarimp_joint"``, see rid/truvarimp.py) instead of
exhaustively refitting every point of each family's discretized grid.
``rid_tree`` (single-family RID) is unaffected -- ``compute_rid``/
``RashomonImportanceDistribution`` has no ``candidate_search`` option, so it
stays on the grid path exactly as before.

Reuses everything else (DGP simulators, metrics, stepwise ranking, plotting,
output schema) directly from ``run_nonlinear_interaction_simulation`` rather
than duplicating it, so results from the two scripts stay directly comparable
column-for-column; only the cross-family RID call and the CLI knob that
controls its budget differ.

Usage:
    python run_nonlinear_interaction_simulation_truvarimp.py
    sbatch run_nonlinear_interaction_simulation_truvarimp.sl
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rid import CrossFamilyRashomonImportanceDistribution

import run_nonlinear_interaction_simulation as base

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

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        estimator.fit(X, y)

    ranked_features = [feature for feature, _ in estimator.rank_features(rid_metric)]
    return ranked_features


def run_simulation_study_truvarimp(
    simulators,
    *,
    beta_grid,
    sample_size,
    n_repetitions,
    n_bootstraps,
    truvarimp_total_budget,
    n_models_per_class,
    noise_std,
    epsilon,
    base_seed,
    stepwise_cv_splits,
    stepwise_logreg_C,
    stepwise_rf_n_estimators,
    random_state,
    n_jobs=1,
    rid_metric="sub_mr",
):
    rows = []
    snr_rows = []
    start = time.time()

    total_cells = len(simulators) * len(beta_grid) * n_repetitions
    completed = 0

    for dgp_index, (dgp_name, simulator) in enumerate(simulators.items()):
        for beta_index, beta in enumerate(beta_grid):
            for rep in range(n_repetitions):
                seed = base_seed + dgp_index * 1_000_000 + beta_index * 10_000 + rep
                dataset = simulator(
                    n=sample_size,
                    beta=beta,
                    noise_std=noise_std,
                    seed=seed,
                )

                feature_columns = base.feature_columns_from_dataset(dataset)
                X = dataset[feature_columns]
                y = dataset["Y"].astype(int)
                ground_truth = tuple(dataset.attrs["relevant_features"])
                equivalence_classes = dataset.attrs.get("equivalence_classes") or {}
                snr_empirical = float(dataset.attrs["snr_empirical"])

                snr_rows.append(
                    {
                        "dgp": dgp_name,
                        "beta": float(beta),
                        "rep": rep,
                        "snr_empirical": snr_empirical,
                    }
                )

                cv = base.StratifiedKFold(
                    n_splits=stepwise_cv_splits, shuffle=True, random_state=random_state
                )
                stepwise_scoring = "roc_auc"

                method_rankings = {
                    "cross_family_unweighted": run_cross_family_rid_truvarimp(
                        X,
                        y,
                        n_bootstraps=n_bootstraps,
                        total_budget=truvarimp_total_budget,
                        epsilon=epsilon,
                        family_balance_mode="unweighted",
                        n_jobs=n_jobs,
                        rid_metric=rid_metric,
                    ),
                    "cross_family_weighted": run_cross_family_rid_truvarimp(
                        X,
                        y,
                        n_bootstraps=n_bootstraps,
                        total_budget=truvarimp_total_budget,
                        epsilon=epsilon,
                        family_balance_mode="weighted",
                        n_jobs=n_jobs,
                        rid_metric=rid_metric,
                    ),
                    "stepwise_logreg": base.run_forward_stepwise_ranking(
                        X,
                        y,
                        base.LogisticRegression,
                        {"C": stepwise_logreg_C, "max_iter": 100000, "random_state": random_state},
                        cv,
                        stepwise_scoring,
                    ),
                    "stepwise_rf": base.run_forward_stepwise_ranking(
                        X,
                        y,
                        base.RandomForestClassifier,
                        {
                            "n_estimators": stepwise_rf_n_estimators,
                            "random_state": random_state,
                            "n_jobs": 1,
                        },
                        cv,
                        stepwise_scoring,
                    ),
                    "rid_tree": base.run_single_family_tree_rid(
                        X,
                        y,
                        epsilon=epsilon,
                        n_bootstraps=n_bootstraps,
                        n_models_pool=n_models_per_class,
                        n_jobs=n_jobs,
                        rid_metric=rid_metric,
                    ),
                }

                for method_name, ranked_features in method_rankings.items():
                    metric_row = base.compute_selection_metrics(ranked_features, ground_truth)

                    equivalence_recall = base.compute_equivalence_recall(
                        ranked_features, ground_truth, equivalence_classes
                    )
                    credit_split_ratios = base.compute_credit_split_ratios(
                        ranked_features, equivalence_classes
                    )
                    valid_ratios = [v for v in credit_split_ratios.values() if v is not None]
                    credit_split_ratio_mean = (
                        float(base.np.mean(valid_ratios)) if valid_ratios else None
                    )

                    rows.append(
                        {
                            "dgp": dgp_name,
                            "beta": float(beta),
                            "rep": rep,
                            "method": method_name,
                            "snr_empirical": snr_empirical,
                            "ground_truth": ", ".join(ground_truth),
                            "ranked_top_k": ", ".join(ranked_features[: len(ground_truth)]),
                            **metric_row,
                            "equivalence_recall_at_k": equivalence_recall,
                            "credit_split_ratio_mean": credit_split_ratio_mean,
                        }
                    )

                completed += 1
                if completed % max(1, n_repetitions) == 0:
                    elapsed = time.time() - start
                    print(
                        f"progress {completed}/{total_cells} "
                        f"({100.0 * completed / total_cells:.1f}%) "
                        f"elapsed={elapsed / 60.0:.1f}m"
                    )

    return base.pd.DataFrame(rows), base.pd.DataFrame(snr_rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ground-truth feature recovery across five methods (cross-family RID "
            "unweighted, cross-family RID weighted, stepwise logistic regression, stepwise "
            "random forest, and single-family tree RID) on simulated nonlinear-interaction "
            "datasets, with the two cross-family RID methods using joint TruVaRImp candidate "
            "search (candidate_search='truvarimp_joint') instead of exhaustive grid refitting."
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
            "Shared per-bootstrap real-fit budget across all families for the two cross-family "
            "RID methods (overrides rid.truvarimp.DEFAULT_TRUVARIMP_CONFIG's total_budget; "
            "default: 300). Replaces --models-per-class for those two methods -- "
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
            "RID variable-importance metric used by cross_family_unweighted, "
            "cross_family_weighted, and rid_tree (default: sub_mr; also available: "
            "loco, coef, iloco)."
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
            "addition, off by default (default: False)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n_jobs = base.resolve_num_workers(args.num_workers)

    simulators = dict(base.SIMULATORS)
    if args.include_redundant_dgps:
        simulators.update(base.REDUNDANT_SIMULATORS)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Running nonlinear interaction simulation study (TruVaRImp candidate search)")
    print(f"output_dir={args.output_dir}")
    print(f"sample_size={args.sample_size}")
    print(f"repetitions={args.repetitions}")
    print(f"bootstraps={args.bootstraps}")
    print(f"truvarimp_total_budget={args.truvarimp_total_budget}")
    print(f"models_per_class (rid_tree only)={args.models_per_class}")
    print(f"beta_grid={args.beta_grid}")
    print(f"noise_std={args.noise_std}")
    print(f"epsilon={args.epsilon}")
    print(f"rid_metric={args.rid_metric}")
    print(f"stepwise_cv_splits={args.stepwise_cv_splits}")
    print(f"stepwise_logreg_C={args.stepwise_logreg_C}")
    print(f"stepwise_rf_n_estimators={args.stepwise_rf_n_estimators}")
    print(f"methods_compared={list(base.METHODS)}")
    print(f"include_redundant_dgps={args.include_redundant_dgps}")
    print(f"simulations_included={list(simulators.keys())}")
    print(f"n_jobs={n_jobs}")
    print("=" * 70)

    started = time.time()

    study_results, snr_results = run_simulation_study_truvarimp(
        simulators,
        beta_grid=args.beta_grid,
        sample_size=args.sample_size,
        n_repetitions=args.repetitions,
        n_bootstraps=args.bootstraps,
        truvarimp_total_budget=args.truvarimp_total_budget,
        n_models_per_class=args.models_per_class,
        noise_std=args.noise_std,
        epsilon=args.epsilon,
        base_seed=args.base_seed,
        stepwise_cv_splits=args.stepwise_cv_splits,
        stepwise_logreg_C=args.stepwise_logreg_C,
        stepwise_rf_n_estimators=args.stepwise_rf_n_estimators,
        random_state=args.random_state,
        n_jobs=n_jobs,
        rid_metric=args.rid_metric,
    )

    metric_table_by_dgp_beta, metric_table_by_dgp = base.build_metric_tables(study_results)
    snr_plot_table = base.build_snr_plot_table(snr_results)

    study_settings = {
        "candidate_search": "truvarimp_joint",
        "sample_size": args.sample_size,
        "repetitions_per_beta": args.repetitions,
        "bootstraps": args.bootstraps,
        "truvarimp_total_budget": args.truvarimp_total_budget,
        "models_per_class_rid_tree_only": args.models_per_class,
        "beta_grid": args.beta_grid,
        "noise_std": args.noise_std,
        "epsilon": args.epsilon,
        "rid_metric": args.rid_metric,
        "stepwise_cv_splits": args.stepwise_cv_splits,
        "stepwise_logreg_C": args.stepwise_logreg_C,
        "stepwise_rf_n_estimators": args.stepwise_rf_n_estimators,
        "random_state": args.random_state,
        "num_workers": n_jobs,
        "include_redundant_dgps": args.include_redundant_dgps,
        "simulations_included": list(simulators.keys()),
        "methods_compared": list(base.METHODS),
    }

    study_results_path = os.path.join(args.output_dir, "study_results.csv")
    dgp_beta_path = os.path.join(args.output_dir, "metric_table_by_dgp_beta.csv")
    dgp_path = os.path.join(args.output_dir, "metric_table_by_dgp.csv")
    snr_table_path = os.path.join(args.output_dir, "snr_plot_table.csv")
    snr_plot_path = os.path.join(args.output_dir, "snr_vs_beta.png")
    settings_path = os.path.join(args.output_dir, "study_settings.json")

    study_results.to_csv(study_results_path, index=False)
    metric_table_by_dgp_beta.to_csv(dgp_beta_path, index=False)
    metric_table_by_dgp.to_csv(dgp_path, index=False)
    snr_plot_table.round(6).to_csv(snr_table_path, index=False)
    base.save_snr_plot(snr_plot_table, snr_plot_path)

    with open(settings_path, "w", encoding="utf-8") as handle:
        json.dump(study_settings, handle, indent=2)

    elapsed = time.time() - started
    print("\nSaved outputs:")
    print(f"  {study_results_path}")
    print(f"  {dgp_beta_path}")
    print(f"  {dgp_path}")
    print(f"  {snr_table_path}")
    print(f"  {snr_plot_path}")
    print(f"  {settings_path}")
    print(f"\nFinished in {elapsed:.1f}s ({elapsed / 60.0:.1f} min)")


if __name__ == "__main__":
    main()
