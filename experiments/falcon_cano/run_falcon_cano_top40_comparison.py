#!/usr/bin/env python3
"""Compare feature-selection methods for the Falcon-Cano bioavailability study.

Compares five top-k feature sets on the Falcon-Cano oral-bioavailability
classification task (%F >= 50 vs. < 50):

1) Forward stepwise selection scored via logistic-regression CV.
2) Forward stepwise selection scored via random-forest CV.
3) Single-family RID using a fully enumerated decision-tree Rashomon set
   (RashomonImportanceDistribution over FullyEnumeratedTreeClassifier).
4) Cross-family RID with family_balance_mode='unweighted'.
5) Cross-family RID with family_balance_mode='weighted'.

Every method's selected top-k feature set is scored with the SAME held-out
cross-validated evaluator (logistic regression, proper k-fold CV) so the five
methods are directly comparable. RID's own internal Rashomon-set performance
stats (accuracy/AUPRC computed on the same bootstrap sample each model was
fit on -- in-sample, not held-out) are recorded separately in
run_settings.json for diagnostic purposes only; they are NOT comparable to
the held-out CV scores and are not used to rank methods.

Optional diagnostic (--inject-redundant-duplicate FEATURE): adds a
correlated engineered duplicate of an existing real feature and reports each
method's credit-split ratio for that pair (0 = winner-take-all, 1 = shared
evenly). Needs no ground truth, since the duplicate is deliberately
constructed to be substitutable -- unlike ground-truth precision/recall,
this credit-sharing test works on real data. Off by default.

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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rid import (  # noqa: E402
    CrossFamilyRashomonImportanceDistribution,
    ElasticNetClassifier,
    FullyEnumeratedTreeClassifier,
    LassoClassifier,
    RashomonImportanceDistribution,
    RidgeClassifier,
)
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
            "decision-tree Rashomon set and cross-family RID (both unweighted and "
            "weighted balance modes), for the Falcon-Cano bioavailability task. "
            "Every method's selected features are scored with a common held-out "
            "CV evaluator for a fair comparison."
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
    parser.add_argument(
        "--inject-redundant-duplicate",
        type=str,
        default=None,
        help=(
            "Name of an existing feature to add a correlated engineered duplicate of "
            "(e.g. SlogP_VSA2), to test whether each method shares importance credit "
            "across the redundant pair or arbitrarily commits to one. No ground truth "
            "needed -- the duplicate is deliberately constructed, so both members are "
            "known to be substitutable by design. Off by default (default: None)."
        ),
    )
    parser.add_argument(
        "--redundancy-corr",
        type=float,
        default=0.95,
        help="Correlation strength between the injected duplicate and its source feature (default: 0.95)",
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


def _build_cross_family_model_configs() -> dict:
    return {
        "RandomForest": (RandomForestClassifier, {}),
        "GradientBoosting": (GradientBoostingClassifier, {}),
        "SVM": (SVC, {}),
        "Lasso": (LassoClassifier, {}),
        "ElasticNet": (ElasticNetClassifier, {}),
        "Ridge": (RidgeClassifier, {}),
    }


def run_cross_family_rid(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int,
    rid_metric: str,
    epsilon: float,
    n_bootstraps: int,
    n_models_per_class: int,
    family_balance_mode: str,
    n_jobs: int,
) -> tuple[list[str], pd.DataFrame, dict]:
    estimator = CrossFamilyRashomonImportanceDistribution(
        model_configs=_build_cross_family_model_configs(),
        epsilon=epsilon,
        n_bootstraps=n_bootstraps,
        n_models_per_class=n_models_per_class,
        vi_metrics=(rid_metric,),
        performance_metrics=("accuracy", "auprc"),
        family_balance_mode=family_balance_mode,
        n_jobs=n_jobs,
    )
    estimator.fit(X, y)

    if estimator.metric_results_ is None:
        raise RuntimeError(
            f"Cross-family RID ({family_balance_mode}) returned no metric results; "
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

    perf_summary = {
        "family_counts": estimator.family_counts_,
        "family_perf_stats": estimator.family_perf_stats_,
        "n_valid_bootstraps": estimator.n_valid_bootstraps_,
    }
    return top_features, pd.DataFrame(rows), perf_summary


def _summarize_cross_family_perf(perf_summary: dict) -> dict:
    family_perf_stats = perf_summary.get("family_perf_stats") or {}
    accuracy_means = [stats["accuracy_mean"] for stats in family_perf_stats.values()]
    auprc_means = [stats["auprc_mean"] for stats in family_perf_stats.values()]
    return {
        "accuracy_mean": float(np.mean(accuracy_means)) if accuracy_means else None,
        "auprc_mean": float(np.mean(auprc_means)) if auprc_means else None,
    }


def run_common_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    method_feature_lists: dict[str, list[str]],
    model_cls,
    model_kwargs: dict,
    cv,
    scoring: str,
) -> dict[str, dict]:
    """Evaluate every method's selected top-k features with the SAME held-out
    CV evaluator, so methods are compared on equal footing regardless of which
    model (if any) they used internally to select features."""

    return {
        name: _evaluate_feature_set(X, y, features, model_cls, model_kwargs, cv, scoring)
        for name, features in method_feature_lists.items()
    }


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


def inject_redundant_duplicate(
    X: pd.DataFrame,
    source_feature: str,
    redundancy_corr: float,
    random_state: int,
) -> tuple[pd.DataFrame, str]:
    """Add a correlated engineered duplicate of an existing real feature (not
    a synthetic ground-truth driver -- source_feature is whatever real column
    the caller chose). Requires no ground truth: the duplicate is
    substitutable for source_feature by construction, so any credit-sharing
    (or lack of it) a method exhibits between the two is directly
    attributable to how it handles redundancy, not to unknown confounders."""

    if source_feature not in X.columns:
        raise ValueError(f"--inject-redundant-duplicate feature '{source_feature}' not found in X")

    dup_name = f"{source_feature}_dup"
    rng = np.random.default_rng(random_state)
    dup_noise = rng.normal(0.0, X[source_feature].std(), size=len(X))
    X = X.copy()
    X[dup_name] = redundancy_corr * X[source_feature] + np.sqrt(1.0 - redundancy_corr**2) * dup_noise
    return X, dup_name


def compute_credit_split_ratio_top_k(ranked_top_k: list[str], pair: tuple[str, str]) -> float | None:
    """Credit-split ratio using only each method's OWN top-k selection (the
    top40 comparison scripts only ever rank the top-k, not the full feature
    set, unlike the nonlinear simulation's full-ranking methods). Both pair
    members present -> ratio via inverse rank-within-top-k (near 1 = shared
    evenly). Exactly one present -> 0.0 (complete winner-take-all: the method
    had a full top-k budget and still excluded the redundant twin entirely).
    Neither present -> None (this pair wasn't relevant to this method's
    selection at all)."""

    positions = {feature: rank for rank, feature in enumerate(ranked_top_k, start=1)}
    present = [positions[member] for member in pair if member in positions]
    if len(present) == 0:
        return None
    if len(present) == 1:
        return 0.0
    importances = [1.0 / rank for rank in present]
    return min(importances) / max(importances)


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

    redundant_pair = None
    if args.inject_redundant_duplicate:
        X, dup_name = inject_redundant_duplicate(
            X, args.inject_redundant_duplicate, args.redundancy_corr, args.random_state
        )
        redundant_pair = (args.inject_redundant_duplicate, dup_name)
        print(
            f"[redundancy] injected '{dup_name}' as a correlated duplicate of "
            f"'{args.inject_redundant_duplicate}' (corr={args.redundancy_corr})"
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

    # --- Methods 4-5: cross-family RID (unweighted, weighted) ---
    cross_family_features: dict[str, list[str]] = {}
    cross_family_perf: dict[str, dict] = {}
    for balance_mode in ("unweighted", "weighted"):
        print(
            f"[rid] cross-family metric={args.rid_metric} epsilon={args.rid_epsilon} "
            f"n_bootstraps={args.rid_n_bootstraps} n_models_per_class={args.rid_n_models_pool} "
            f"family_balance_mode={balance_mode}"
        )
        features, table, perf = run_cross_family_rid(
            X=X,
            y=y,
            top_k=args.top_k,
            rid_metric=args.rid_metric,
            epsilon=args.rid_epsilon,
            n_bootstraps=args.rid_n_bootstraps,
            n_models_per_class=args.rid_n_models_pool,
            family_balance_mode=balance_mode,
            n_jobs=rid_n_jobs,
        )
        table.to_csv(output_dir / f"rid_cross_family_{balance_mode}_top_features.csv", index=False)
        cross_family_features[balance_mode] = features
        cross_family_perf[balance_mode] = perf

    # --- Comparison across all five top-k feature sets ---
    method_feature_lists = {
        "stepwise_logreg": stepwise_logreg_features,
        "stepwise_rf": stepwise_rf_features,
        "rid_tree": rid_tree_features,
        "cross_family_unweighted": cross_family_features["unweighted"],
        "cross_family_weighted": cross_family_features["weighted"],
    }
    comparison = _build_comparison_table(method_feature_lists)
    comparison.sort_values(
        ["n_methods_selected", "stepwise_logreg_rank"],
        ascending=[False, True],
    ).to_csv(output_dir / "top_feature_overlap.csv", index=False)
    overlap_counts = _pairwise_overlap_counts(comparison, list(method_feature_lists.keys()))

    # --- Common held-out CV evaluation: every method's features scored the same way ---
    common_model_kwargs = {
        "penalty": "l2",
        "C": args.stepwise_logreg_C,
        "solver": "lbfgs",
        "max_iter": 100000,
        "random_state": args.random_state,
    }
    common_eval = run_common_evaluation(
        X, y, method_feature_lists, LogisticRegression, common_model_kwargs, cv, args.stepwise_scoring
    )

    cross_family_unweighted_rashomon = _summarize_cross_family_perf(cross_family_perf["unweighted"])
    cross_family_weighted_rashomon = _summarize_cross_family_perf(cross_family_perf["weighted"])

    credit_split_ratios = None
    if redundant_pair is not None:
        credit_split_ratios = {
            name: compute_credit_split_ratio_top_k(features, redundant_pair)
            for name, features in method_feature_lists.items()
        }
        print(f"[redundancy] credit-split ratios for {redundant_pair}: {credit_split_ratios}")

    settings = {
        "args": vars(args),
        "data_metadata": data_meta,
        "redundancy_diagnostic": (
            None
            if redundant_pair is None
            else {"pair": redundant_pair, "redundancy_corr": args.redundancy_corr, "credit_split_ratios": credit_split_ratios}
        ),
        "common_evaluator": {
            "model": "LogisticRegression",
            "model_kwargs": common_model_kwargs,
            "scoring": args.stepwise_scoring,
            "note": (
                "Every method's selected top-k features scored with this SAME held-out "
                "CV evaluator for a fair, apples-to-apples comparison."
            ),
            "results": common_eval,
        },
        "stepwise_logreg": {
            "scoring": args.stepwise_scoring,
            "model_kwargs": logreg_kwargs,
            "top_features": stepwise_logreg_features,
            "search_model_evaluation": stepwise_logreg_eval,
        },
        "stepwise_rf": {
            "scoring": args.stepwise_scoring,
            "model_kwargs": rf_kwargs,
            "top_features": stepwise_rf_features,
            "search_model_evaluation": stepwise_rf_eval,
        },
        "rid_tree": {
            "metric": args.rid_metric,
            "epsilon": args.rid_epsilon,
            "top_features": rid_tree_features,
            "rashomon_train_perf_stats": rid_tree_perf,
        },
        "cross_family_unweighted": {
            "metric": args.rid_metric,
            "epsilon": args.rid_epsilon,
            "top_features": cross_family_features["unweighted"],
            "rashomon_train_perf_stats": cross_family_perf["unweighted"],
        },
        "cross_family_weighted": {
            "metric": args.rid_metric,
            "epsilon": args.rid_epsilon,
            "top_features": cross_family_features["weighted"],
            "rashomon_train_perf_stats": cross_family_perf["weighted"],
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
        "common_eval_scoring": args.stepwise_scoring,
    }
    for name in method_feature_lists:
        overall_row[f"{name}_held_out_cv_score_mean"] = common_eval[name]["cv_score_mean"]
        overall_row[f"{name}_held_out_cv_score_std"] = common_eval[name]["cv_score_std"]
    overall_row.update(
        {
            "rid_tree_rashomon_train_accuracy_mean": rid_tree_perf.get("accuracy_mean"),
            "rid_tree_rashomon_train_auprc_mean": rid_tree_perf.get("auprc_mean"),
            "cross_family_unweighted_rashomon_train_accuracy_mean": cross_family_unweighted_rashomon["accuracy_mean"],
            "cross_family_unweighted_rashomon_train_auprc_mean": cross_family_unweighted_rashomon["auprc_mean"],
            "cross_family_weighted_rashomon_train_accuracy_mean": cross_family_weighted_rashomon["accuracy_mean"],
            "cross_family_weighted_rashomon_train_auprc_mean": cross_family_weighted_rashomon["auprc_mean"],
        }
    )
    if credit_split_ratios is not None:
        for name, ratio in credit_split_ratios.items():
            overall_row[f"{name}_credit_split_ratio"] = ratio
    overall_row.update(overlap_counts)
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
    for name, features in method_feature_lists.items():
        print(f"{name} top-10: {features[:10]}")
        print(f"  held-out CV ({args.stepwise_scoring}): {common_eval[name]['cv_score_mean']:.4f} +- {common_eval[name]['cv_score_std']:.4f}")
    print(f"overlap counts: {overlap_counts}")
    print(f"Finished in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
