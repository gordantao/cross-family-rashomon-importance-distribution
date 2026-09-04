#!/usr/bin/env python3
"""Compare feature-selection methods for the Falcon-Cano bioavailability study.

Compares three top-k feature sets on the Falcon-Cano oral-bioavailability
classification task (%F >= 50 vs. < 50):

1) Forward stepwise selection scored via logistic-regression CV.
2) Forward stepwise selection scored via random-forest CV.
3) Single-family RID using a fully enumerated decision-tree Rashomon set
   (RashomonImportanceDistribution over FullyEnumeratedTreeClassifier).

Usage:
    python run_falcon_cano_top40_comparison.py
    sbatch run_falcon_cano_top40_comparison.sl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rid import FullyEnumeratedTreeClassifier, RashomonImportanceDistribution  # noqa: E402
from run_rashomon_falcon_cano import (  # noqa: E402
    remove_highly_correlated_features,
    resolve_num_workers,
    select_curated_descriptors,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare forward-stepwise feature selection (logistic regression and "
            "random forest) against single-family RID on a fully enumerated "
            "decision-tree Rashomon set, for the Falcon-Cano bioavailability task."
        )
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).resolve().parent / "falcon_cano_featured.csv"),
        help="Path to the input CSV file (default: <script dir>/falcon_cano_featured.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "top40_feature_comparison"),
        help="Directory for outputs (default: <script dir>/results/top40_feature_comparison)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Number of top features to select from each method (default: 40)",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.8,
        help="Remove highly correlated descriptors above this absolute threshold (default: 0.8)",
    )
    parser.add_argument(
        "--use-curated-descriptors",
        action="store_true",
        default=False,
        help="Drop redundant RDKit descriptors before correlation filtering",
    )
    parser.add_argument(
        "--stepwise-scoring",
        type=str,
        default="roc_auc",
        help="sklearn scoring for stepwise CV (default: roc_auc)",
    )
    parser.add_argument(
        "--stepwise-cv-splits",
        type=int,
        default=5,
        help="CV folds for stepwise selection and evaluation (default: 5)",
    )
    parser.add_argument(
        "--stepwise-n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for scoring candidate features during stepwise selection (default: 1)",
    )
    parser.add_argument(
        "--stepwise-logreg-C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for the stepwise logistic regression model (default: 1.0)",
    )
    parser.add_argument(
        "--stepwise-rf-n-estimators",
        type=int,
        default=300,
        help="Random forest trees for the stepwise RF model (default: 300)",
    )
    parser.add_argument(
        "--stepwise-rf-max-depth",
        type=int,
        default=None,
        help="Random forest max_depth for the stepwise RF model (default: None)",
    )
    parser.add_argument(
        "--stepwise-rf-min-samples-leaf",
        type=int,
        default=1,
        help="Random forest min_samples_leaf for the stepwise RF model (default: 1)",
    )
    parser.add_argument(
        "--rid-metric",
        type=str,
        default="sub_mr",
        help="RID variable-importance metric (default: sub_mr)",
    )
    parser.add_argument(
        "--rid-epsilon",
        type=float,
        default=0.05,
        help="Rashomon epsilon for the single-family tree RID (default: 0.05)",
    )
    parser.add_argument(
        "--rid-n-bootstraps",
        type=int,
        default=100,
        help="Bootstrap iterations for the single-family tree RID (default: 100)",
    )
    parser.add_argument(
        "--rid-n-models-pool",
        type=int,
        default=50,
        help="Candidate refits per grid point, per bootstrap, for the tree Rashomon set (default: 50)",
    )
    parser.add_argument(
        "--rid-n-jobs",
        type=int,
        default=None,
        help="Parallel workers for the tree RID bootstraps (default: SLURM_CPUS_PER_TASK or os.cpu_count())",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for CV folds and the stepwise search models (default: 42)",
    )
    return parser.parse_args()


def _prepare_dataset(
    data_path: Path,
    use_curated_descriptors: bool,
    correlation_threshold: float,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    df = pd.read_csv(data_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    n_features_raw = X.shape[1]
    if use_curated_descriptors:
        X = select_curated_descriptors(X)
    n_features_curated = X.shape[1]

    X = remove_highly_correlated_features(X, correlation_threshold=correlation_threshold)

    metadata = {
        "dataset_path": str(data_path),
        "target_column": "target",
        "n_rows": int(X.shape[0]),
        "n_features_raw": int(n_features_raw),
        "n_features_after_curation": int(n_features_curated),
        "n_features_after_corr": int(X.shape[1]),
        "class_counts": y.value_counts().to_dict(),
    }
    return X, y, metadata


def _score_candidate_feature(
    feature: str,
    selected_features: tuple[str, ...],
    X: pd.DataFrame,
    y: pd.Series,
    model_cls,
    model_kwargs: dict,
    cv,
    scoring: str,
) -> tuple[str, float, float]:
    feature_set = list(selected_features) + [feature]
    estimator = model_cls(**model_kwargs)
    scores = cross_val_score(
        estimator,
        X[feature_set],
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )
    return feature, float(np.mean(scores)), float(np.std(scores))


def run_forward_stepwise_selection(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int,
    model_cls,
    model_kwargs: dict,
    cv,
    scoring: str,
    n_jobs: int,
) -> tuple[list[str], pd.DataFrame]:
    selected_features: list[str] = []
    history_rows: list[dict] = []
    all_features = list(X.columns)
    steps = min(top_k, len(all_features))

    for step_idx in range(1, steps + 1):
        remaining = [feature for feature in all_features if feature not in selected_features]

        scores = Parallel(n_jobs=n_jobs)(
            delayed(_score_candidate_feature)(
                feature=feature,
                selected_features=tuple(selected_features),
                X=X,
                y=y,
                model_cls=model_cls,
                model_kwargs=model_kwargs,
                cv=cv,
                scoring=scoring,
            )
            for feature in remaining
        )

        best_feature, best_mean, best_std = max(scores, key=lambda item: item[1])
        selected_features.append(best_feature)
        history_rows.append(
            {
                "rank": step_idx,
                "feature": best_feature,
                "cv_score_mean": best_mean,
                "cv_score_std": best_std,
                "n_features_in_model": len(selected_features),
            }
        )

        print(
            f"[stepwise:{model_cls.__name__}] step {step_idx:02d}/{steps}: "
            f"+{best_feature}  score={best_mean:.6f} +- {best_std:.6f}"
        )

    return selected_features, pd.DataFrame(history_rows)


def _evaluate_feature_set(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    model_cls,
    model_kwargs: dict,
    cv,
    scoring: str,
) -> dict:
    estimator = model_cls(**model_kwargs)
    scores = cross_val_score(
        estimator,
        X[features],
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )
    return {
        "scoring": scoring,
        "cv_score_mean": float(np.mean(scores)),
        "cv_score_std": float(np.std(scores)),
        "n_features": len(features),
    }


def run_single_family_tree_rid(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int,
    rid_metric: str,
    epsilon: float,
    n_bootstraps: int,
    n_models_pool: int,
    n_jobs: int,
) -> tuple[list[str], pd.DataFrame, dict]:
    estimator = RashomonImportanceDistribution(
        epsilon=epsilon,
        n_bootstraps=n_bootstraps,
        n_models_pool=n_models_pool,
        model_class=FullyEnumeratedTreeClassifier,
        vi_metrics=(rid_metric,),
        performance_metrics=("accuracy", "auprc"),
        n_jobs=n_jobs,
    )
    estimator.fit(X, y)

    if estimator.metric_results_ is None:
        raise RuntimeError(
            "Single-family tree RID returned no metric results; "
            "no valid Rashomon bootstraps were found"
        )

    ranking = estimator.rank_features(rid_metric)
    summary = estimator.metric_summary(rid_metric)
    top_pairs = ranking[:top_k]
    top_features = [feature for feature, _ in top_pairs]

    rows = []
    for rank, (feature, expected_importance) in enumerate(top_pairs, start=1):
        rows.append(
            {
                "rank": rank,
                "feature": feature,
                "expected_importance": float(expected_importance),
                "prob_positive": float(summary[feature]["prob_positive"]),
            }
        )

    perf_stats = dict(estimator.perf_stats_ or {})
    perf_stats["n_valid_bootstraps"] = estimator.n_valid_bootstraps_
    return top_features, pd.DataFrame(rows), perf_stats


def _build_comparison_table(method_feature_lists: dict[str, list[str]]) -> pd.DataFrame:
    ranks = {
        name: {feature: rank for rank, feature in enumerate(features, start=1)}
        for name, features in method_feature_lists.items()
    }
    all_features = sorted(set().union(*(set(v) for v in method_feature_lists.values())))

    rows = []
    for feature in all_features:
        row: dict = {"feature": feature}
        in_flags = []
        for name, rank_map in ranks.items():
            feature_rank = rank_map.get(feature)
            row[f"{name}_rank"] = feature_rank
            row[f"in_{name}_top_k"] = feature_rank is not None
            in_flags.append(feature_rank is not None)
        row["n_methods_selected"] = int(sum(in_flags))
        row["in_all_methods"] = bool(all(in_flags))
        rows.append(row)

    return pd.DataFrame(rows)


def _pairwise_overlap_counts(comparison: pd.DataFrame, method_names: list[str]) -> dict:
    overlaps = {}
    for i, name_a in enumerate(method_names):
        for name_b in method_names[i + 1 :]:
            key = f"overlap_{name_a}_{name_b}"
            overlaps[key] = int(
                (comparison[f"in_{name_a}_top_k"] & comparison[f"in_{name_b}_top_k"]).sum()
            )
    overlaps["overlap_all_methods"] = int(comparison["in_all_methods"].sum())
    return overlaps


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rid_n_jobs = resolve_num_workers(args.rid_n_jobs)

    print("=" * 70)
    print("Falcon-Cano Top-K Feature Comparison")
    print(f"  data:                  {args.data}")
    print(f"  output_dir:            {output_dir}")
    print(f"  top_k:                 {args.top_k}")
    print(f"  correlation_threshold: {args.correlation_threshold}")
    print(f"  use_curated_descriptors: {args.use_curated_descriptors}")
    print(f"  rid_n_jobs:            {rid_n_jobs}")
    print("=" * 70)

    start_time = time.time()

    X, y, data_meta = _prepare_dataset(
        data_path=Path(args.data),
        use_curated_descriptors=args.use_curated_descriptors,
        correlation_threshold=args.correlation_threshold,
    )
    print(
        f"[data] rows={data_meta['n_rows']} "
        f"features={data_meta['n_features_after_corr']} "
        f"(raw={data_meta['n_features_raw']})"
    )

    cv = StratifiedKFold(
        n_splits=args.stepwise_cv_splits,
        shuffle=True,
        random_state=args.random_state,
    )

    # --- Method 1: forward stepwise selection via logistic regression ---
    logreg_kwargs = {
        "penalty": "l2",
        "C": args.stepwise_logreg_C,
        "solver": "lbfgs",
        "max_iter": 100000,
        "random_state": args.random_state,
    }
    stepwise_logreg_features, stepwise_logreg_history = run_forward_stepwise_selection(
        X=X,
        y=y,
        top_k=args.top_k,
        model_cls=LogisticRegression,
        model_kwargs=logreg_kwargs,
        cv=cv,
        scoring=args.stepwise_scoring,
        n_jobs=args.stepwise_n_jobs,
    )
    stepwise_logreg_eval = _evaluate_feature_set(
        X, y, stepwise_logreg_features, LogisticRegression, logreg_kwargs, cv, args.stepwise_scoring
    )
    stepwise_logreg_history.to_csv(output_dir / "stepwise_logreg_top_features.csv", index=False)

    # --- Method 2: forward stepwise selection via random forest ---
    rf_kwargs = {
        "n_estimators": args.stepwise_rf_n_estimators,
        "max_depth": args.stepwise_rf_max_depth,
        "min_samples_leaf": args.stepwise_rf_min_samples_leaf,
        "random_state": args.random_state,
        "n_jobs": 1,
    }
    stepwise_rf_features, stepwise_rf_history = run_forward_stepwise_selection(
        X=X,
        y=y,
        top_k=args.top_k,
        model_cls=RandomForestClassifier,
        model_kwargs=rf_kwargs,
        cv=cv,
        scoring=args.stepwise_scoring,
        n_jobs=args.stepwise_n_jobs,
    )
    stepwise_rf_eval = _evaluate_feature_set(
        X, y, stepwise_rf_features, RandomForestClassifier, rf_kwargs, cv, args.stepwise_scoring
    )
    stepwise_rf_history.to_csv(output_dir / "stepwise_rf_top_features.csv", index=False)

    # --- Method 3: single-family RID on a fully enumerated tree Rashomon set ---
    print(
        f"[rid] metric={args.rid_metric} epsilon={args.rid_epsilon} "
        f"n_bootstraps={args.rid_n_bootstraps} n_models_pool={args.rid_n_models_pool}"
    )
    rid_tree_features, rid_tree_table, rid_tree_perf = run_single_family_tree_rid(
        X=X,
        y=y,
        top_k=args.top_k,
        rid_metric=args.rid_metric,
        epsilon=args.rid_epsilon,
        n_bootstraps=args.rid_n_bootstraps,
        n_models_pool=args.rid_n_models_pool,
        n_jobs=rid_n_jobs,
    )
    rid_tree_table.to_csv(output_dir / "rid_tree_top_features.csv", index=False)

    # --- Comparison across all three top-k feature sets ---
    method_feature_lists = {
        "stepwise_logreg": stepwise_logreg_features,
        "stepwise_rf": stepwise_rf_features,
        "rid_tree": rid_tree_features,
    }
    comparison = _build_comparison_table(method_feature_lists)
    comparison.sort_values(
        ["n_methods_selected", "stepwise_logreg_rank"],
        ascending=[False, True],
    ).to_csv(output_dir / "top_feature_overlap.csv", index=False)
    overlap_counts = _pairwise_overlap_counts(comparison, list(method_feature_lists.keys()))

    settings = {
        "args": vars(args),
        "data_metadata": data_meta,
        "stepwise_logreg": {
            "scoring": args.stepwise_scoring,
            "model_kwargs": logreg_kwargs,
            "top_features": stepwise_logreg_features,
            "evaluation": stepwise_logreg_eval,
        },
        "stepwise_rf": {
            "scoring": args.stepwise_scoring,
            "model_kwargs": rf_kwargs,
            "top_features": stepwise_rf_features,
            "evaluation": stepwise_rf_eval,
        },
        "rid_tree": {
            "metric": args.rid_metric,
            "epsilon": args.rid_epsilon,
            "top_features": rid_tree_features,
            "rashomon_perf_stats": rid_tree_perf,
        },
        "overlap_counts": overlap_counts,
    }
    with (output_dir / "run_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)

    overall_row = {
        "task_name": "falcon_cano_bioavailability",
        "task_type": "classification",
        "target_column": "target",
        "dataset_path": data_meta["dataset_path"],
        "n_rows": data_meta["n_rows"],
        "n_features": data_meta["n_features_after_corr"],
        "stepwise_logreg_scoring": args.stepwise_scoring,
        "stepwise_logreg_cv_score_mean": stepwise_logreg_eval["cv_score_mean"],
        "stepwise_logreg_cv_score_std": stepwise_logreg_eval["cv_score_std"],
        "stepwise_rf_scoring": args.stepwise_scoring,
        "stepwise_rf_cv_score_mean": stepwise_rf_eval["cv_score_mean"],
        "stepwise_rf_cv_score_std": stepwise_rf_eval["cv_score_std"],
        "rid_tree_metric": args.rid_metric,
        "rid_tree_accuracy_mean": rid_tree_perf.get("accuracy_mean"),
        "rid_tree_accuracy_std": rid_tree_perf.get("accuracy_std"),
        "rid_tree_auprc_mean": rid_tree_perf.get("auprc_mean"),
        "rid_tree_auprc_std": rid_tree_perf.get("auprc_std"),
        "rid_tree_n_valid_bootstraps": rid_tree_perf.get("n_valid_bootstraps"),
        **overlap_counts,
    }
    overall_summary_path = output_dir / "overall_task_summary.csv"
    pd.DataFrame([overall_row]).to_csv(overall_summary_path, index=False)

    run_manifest = {
        "args": vars(args),
        "overall_summary_csv": str(overall_summary_path),
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2)

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"stepwise_logreg top-10: {stepwise_logreg_features[:10]}")
    print(f"stepwise_rf top-10:     {stepwise_rf_features[:10]}")
    print(f"rid_tree top-10:        {rid_tree_features[:10]}")
    print(f"overlap counts: {overlap_counts}")
    print(f"Finished in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
