#!/usr/bin/env python3
"""Run Staellert direct target prediction tasks with feature selection.

This script is classification-only (regression-task support was removed --
see TASK_REGISTRY) and compares five top-k feature sets (METHODS):
1) stepwise_rf: forward stepwise selection scored via random forest.
2) stepwise_logreg: forward stepwise selection scored via logistic regression.
3) rid_tree: single-family RID on a fully enumerated decision-tree Rashomon set.
4) cross_family_unweighted: cross-family RID, exhaustive grid candidate search.
5) cross_family_unweighted_truvarimp: cross-family RID, joint TruVaRImp
   candidate search (candidate_search='truvarimp_joint', rid/truvarimp.py).
The weighted cross-family variant was dropped, as in the nonlinear simulation.

Every method's selected top-k feature set is scored with the SAME panel of
held-out cross-validated evaluators so all methods are directly comparable.
RID's own Rashomon-set performance stats (accuracy/AUPRC on each bootstrap's
out-of-bag rows) are recorded in run_settings.json for diagnostics only; they
come from a different protocol than the evaluator panel and are not used to
rank methods.

Each method is checkpointed separately (<task>/<method>_result.json plus
<method>_top_features.csv), so --methods can run any subset -- e.g. one method
per SLURM array task -- and reruns skip methods already checkpointed. Once
every method has a checkpoint, the combined outputs (top_feature_overlap.csv,
run_settings.json, overall_task_summary.csv) are (re)built.

Default task is the classification target directly annotated in the
paper-style dataset: annotated_phase.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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

from rid import (
    CrossFamilyRashomonImportanceDistribution,
    ElasticNetClassifier,
    FullyEnumeratedTreeClassifier,
    LassoClassifier,
    RashomonImportanceDistribution,
    RidgeClassifier,
    add_logging_arguments,
    configure_logging,
    silence_warnings,
)

logger = logging.getLogger(__name__)


TASK_REGISTRY = {
    "annotated_phase": {
        "file": "control_manifold_allfeatures.csv",
        "target_column": "annotated phase",
        "task": "classification",
    },
    "phase": {
        "file": "control_manifold_allfeatures.csv",
        "target_column": "phase",
        "task": "classification",
    },
}

KNOWN_TARGET_COLUMNS = {
    "phase",
    "age",
    "annotated phase",
    "annotated age",
}

METHODS = (
    "stepwise_rf",
    "stepwise_logreg",
    "rid_tree",
    "cross_family_unweighted",
    "cross_family_unweighted_truvarimp",
)

# Arguments that change a method's result. A checkpoint written under different
# values is refused rather than silently reused (n_jobs/logging/paths don't count).
_DATA_SETTING_KEYS = ("top_k", "correlation_threshold", "include_phate_features", "random_state",
                      "stepwise_cv_splits", "stepwise_scoring_classification")
_STEPWISE_SETTING_KEYS = ("stepwise_n_estimators", "stepwise_max_depth", "stepwise_min_samples_leaf",
                          "stepwise_class_weight", "stepwise_logreg_C")
_RID_SETTING_KEYS = ("rid_metric", "rid_epsilon", "rid_n_bootstraps")
METHOD_SETTING_KEYS = {
    "stepwise_rf": _DATA_SETTING_KEYS + _STEPWISE_SETTING_KEYS,
    "stepwise_logreg": _DATA_SETTING_KEYS + _STEPWISE_SETTING_KEYS,
    "rid_tree": _DATA_SETTING_KEYS + _RID_SETTING_KEYS + ("rid_n_models_pool",),
    "cross_family_unweighted": _DATA_SETTING_KEYS + _RID_SETTING_KEYS + ("rid_n_models_pool",),
    "cross_family_unweighted_truvarimp": _DATA_SETTING_KEYS + _RID_SETTING_KEYS + ("truvarimp_total_budget",),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run classification-only direct target prediction tasks on Staellert data, "
            "comparing random-forest and logistic-regression forward stepwise feature "
            "selection against single-family RID on a fully enumerated decision-tree "
            "Rashomon set and unweighted cross-family RID (grid and TruVaRImp candidate search)."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "staellert_et_al"),
        help="Directory containing Staellert CSV files (default: <script dir>/data/staellert_et_al)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="annotated_phase",
        help=(
            "Comma-separated task names. Use 'all' for all known tasks. "
            "Default: annotated_phase"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Number of top features to select from each method (default: 40)",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=_correlation_threshold_arg,
        default=0.8,
        help=(
            "Remove highly correlated descriptors above this absolute threshold. "
            "Set to a value <= 0 to disable, or 'auto' to pick the Kneedle elbow "
            "point dynamically from this run's data (default: 0.8)."
        ),
    )
    parser.add_argument(
        "--include-phate-features",
        action="store_true",
        default=False,
        help="Keep PHATE-derived columns as candidate features (default: False)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "staellert_direct_target_tasks"),
        help="Directory for outputs (default: <script dir>/results/staellert_direct_target_tasks)",
    )

    parser.add_argument(
        "--stepwise-scoring-classification",
        type=str,
        default=None,
        help=(
            "sklearn scoring for classification stepwise CV. Defaults to roc_auc for "
            "binary and f1_macro for multiclass."
        ),
    )
    parser.add_argument(
        "--stepwise-cv-splits",
        type=int,
        default=5,
        help="CV folds for stepwise selection (default: 5)",
    )
    parser.add_argument(
        "--stepwise-n-estimators",
        type=int,
        default=300,
        help="Random forest trees for stepwise selector (default: 300)",
    )
    parser.add_argument(
        "--stepwise-max-depth",
        type=int,
        default=None,
        help="Random forest max_depth for stepwise selector (default: None)",
    )
    parser.add_argument(
        "--stepwise-min-samples-leaf",
        type=int,
        default=1,
        help="Random forest min_samples_leaf for stepwise selector (default: 1)",
    )
    parser.add_argument(
        "--stepwise-class-weight",
        type=str,
        default="balanced",
        help=(
            "Class weight for classification stepwise models. Use 'none' to disable "
            "(default: balanced)."
        ),
    )
    parser.add_argument(
        "--stepwise-logreg-C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for the stepwise logistic regression model (default: 1.0)",
    )
    parser.add_argument(
        "--stepwise-n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for candidate evaluation at each step (default: -1)",
    )

    parser.add_argument(
        "--rid-metric",
        type=str,
        default="sub_mr",
        help="RID metric used to rank features (classification tasks only, default: sub_mr)",
    )
    parser.add_argument(
        "--rid-epsilon",
        type=float,
        default=0.05,
        help="RID epsilon (default: 0.05)",
    )
    parser.add_argument(
        "--rid-n-bootstraps",
        type=int,
        default=100,
        help="RID bootstrap iterations (default: 100)",
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
        default=1,
        help="RID parallel jobs (default: 1)",
    )
    parser.add_argument(
        "--truvarimp-total-budget",
        type=int,
        default=300,
        help=(
            "Shared per-bootstrap real-fit budget across all families for "
            "cross_family_unweighted_truvarimp (default: 300). That method ignores "
            "--rid-n-models-pool."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help=(
            "Methods to run this invocation (default: all). Methods already "
            "checkpointed in --output-dir are skipped."
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    add_logging_arguments(parser)
    return parser.parse_args()


def _parse_task_names(raw_tasks: str) -> list[str]:
    if raw_tasks.strip().lower() == "all":
        return list(TASK_REGISTRY.keys())

    task_names = [task.strip() for task in raw_tasks.split(",") if task.strip()]
    if not task_names:
        raise ValueError("No tasks were provided")

    unknown = [task for task in task_names if task not in TASK_REGISTRY]
    if unknown:
        raise ValueError(
            "Unknown tasks: "
            f"{unknown}. Supported tasks: {list(TASK_REGISTRY.keys())}"
        )
    return task_names


def _drop_index_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [col for col in df.columns if not col.lower().startswith("unnamed")]
    return df[keep_cols].copy()


def _sanitize_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    if X.isna().any().any():
        X = X.fillna(X.median())

    constant_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
    return X


def _remove_highly_correlated_features(
    X: pd.DataFrame,
    correlation_threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    if correlation_threshold <= 0:
        return X, []

    corr_matrix = X.corr().abs()
    removed_features: set[str] = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] <= correlation_threshold:
                continue

            feature_one = corr_matrix.columns[i]
            feature_two = corr_matrix.columns[j]
            if feature_one in removed_features or feature_two in removed_features:
                continue

            if X[feature_one].var() >= X[feature_two].var():
                removed_features.add(feature_two)
            else:
                removed_features.add(feature_one)

    removed_sorted = sorted(removed_features)
    if removed_sorted:
        X = X.drop(columns=removed_sorted)
    return X, removed_sorted


_AUTO_CORRELATION_THRESHOLD_SWEEP = np.round(np.arange(0.50, 0.991, 0.01), 3)


def _correlation_threshold_arg(value: str):
    """argparse type: accepts a float, or the literal 'auto' for elbow detection."""

    if value == "auto":
        return "auto"
    return float(value)


def sweep_correlation_thresholds(X: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    """Sweep correlation threshold vs. surviving feature count, reusing the
    actual _remove_highly_correlated_features algorithm at each threshold (so
    the sweep can never drift from what filtering really does)."""

    rows = []
    for t in thresholds:
        X_t, removed_t = _remove_highly_correlated_features(X, correlation_threshold=float(t))
        rows.append({"threshold": float(t), "n_features_remaining": X_t.shape[1], "n_removed": len(removed_t)})
    return pd.DataFrame(rows)


def find_correlation_elbow(sweep_df: pd.DataFrame) -> float:
    """Kneedle elbow point, falling back to max-distance-from-chord if `kneed`
    is unavailable."""

    x = sweep_df["threshold"].to_numpy()
    y = sweep_df["n_features_remaining"].to_numpy()
    try:
        from kneed import KneeLocator

        kl = KneeLocator(x, y, curve="convex", direction="increasing")
        if kl.knee is not None:
            return float(kl.knee)
    except ImportError:
        pass

    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_unit = (p2 - p1) / np.linalg.norm(p2 - p1)
    distances = []
    for xi, yi in zip(x_norm, y_norm):
        p = np.array([xi, yi]) - p1
        proj = np.dot(p, line_unit) * line_unit
        distances.append(np.linalg.norm(p - proj))
    return float(x[int(np.argmax(distances))])


def _resolve_correlation_threshold(X: pd.DataFrame, correlation_threshold) -> tuple[float, dict]:
    """Resolve a fixed threshold as-is, or pick the Kneedle elbow point from
    this run's own data when correlation_threshold == 'auto'."""

    if correlation_threshold != "auto":
        threshold = float(correlation_threshold)
        return threshold, {"mode": "fixed", "value": threshold}

    sweep_df = sweep_correlation_thresholds(X, _AUTO_CORRELATION_THRESHOLD_SWEEP)
    elbow = find_correlation_elbow(sweep_df)
    n_at_elbow = int(sweep_df.loc[sweep_df["threshold"].sub(elbow).abs().idxmin(), "n_features_remaining"])
    logger.info(f"[correlation-threshold=auto] elbow={elbow:.3f} -> {n_at_elbow} features (from {X.shape[1]} raw)")
    return elbow, {"mode": "auto", "value": elbow, "n_features_at_elbow": n_at_elbow, "n_features_raw": int(X.shape[1])}


def _prepare_task_dataset(
    csv_path: Path,
    target_column: str,
    correlation_threshold,
    include_phate_features: bool,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    df = _drop_index_like_columns(df)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {csv_path}")

    drop_cols = set(KNOWN_TARGET_COLUMNS)
    drop_cols.add(target_column)

    if not include_phate_features:
        drop_cols.update([col for col in df.columns if col.upper().startswith("PHATE")])

    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = _sanitize_features(df[feature_cols])

    y_raw = df[target_column]
    valid_rows = y_raw.notna()
    y = y_raw.loc[valid_rows].astype(str)

    X = X.loc[valid_rows].copy()

    X_before_corr = X.shape[1]
    resolved_threshold, threshold_info = _resolve_correlation_threshold(X, correlation_threshold)
    X, removed_corr = _remove_highly_correlated_features(X, resolved_threshold)

    metadata = {
        "dataset_path": str(csv_path),
        "target_column": target_column,
        "task": "classification",
        "n_rows": int(X.shape[0]),
        "n_features_before_corr": int(X_before_corr),
        "n_features_after_corr": int(X.shape[1]),
        "n_removed_corr": int(len(removed_corr)),
        "removed_corr_features": removed_corr,
        "correlation_threshold_info": threshold_info,
        "n_classes": int(y.nunique()),
        "class_counts": y.value_counts().to_dict(),
    }
    return X, y, metadata


def _make_stepwise_cv_and_scoring(args: argparse.Namespace, y: pd.Series):
    cv = StratifiedKFold(
        n_splits=args.stepwise_cv_splits,
        shuffle=True,
        random_state=args.random_state,
    )
    if args.stepwise_scoring_classification:
        scoring = args.stepwise_scoring_classification
    else:
        scoring = "roc_auc" if int(y.nunique()) == 2 else "f1_macro"

    return cv, scoring


def _make_stepwise_rf_model(args: argparse.Namespace):
    class_weight = None if str(args.stepwise_class_weight).lower() == "none" else args.stepwise_class_weight
    return RandomForestClassifier, {
        "n_estimators": args.stepwise_n_estimators,
        "random_state": args.random_state,
        "n_jobs": 1,
        "max_depth": args.stepwise_max_depth,
        "min_samples_leaf": args.stepwise_min_samples_leaf,
        "class_weight": class_weight,
    }


def _make_stepwise_logreg_model(args: argparse.Namespace):
    class_weight = None if str(args.stepwise_class_weight).lower() == "none" else args.stepwise_class_weight
    return LogisticRegression, {
        "C": args.stepwise_logreg_C,
        "max_iter": 100000,
        "random_state": args.random_state,
        "class_weight": class_weight,
    }


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

        logger.info(
            f"[stepwise] step {step_idx:02d}/{steps}: "
            f"+{best_feature}  score={best_mean:.6f} +- {best_std:.6f}"
        )

    return selected_features, pd.DataFrame(history_rows)


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
    n_jobs: int,
    candidate_search: str = "grid",
    truvarimp_total_budget: int | None = None,
) -> tuple[list[str], pd.DataFrame, dict]:
    estimator = CrossFamilyRashomonImportanceDistribution(
        model_configs=_build_cross_family_model_configs(),
        epsilon=epsilon,
        n_bootstraps=n_bootstraps,
        n_models_per_class=n_models_per_class,
        vi_metrics=(rid_metric,),
        performance_metrics=("accuracy", "auprc"),
        family_balance_mode="unweighted",
        n_jobs=n_jobs,
        candidate_search=candidate_search,
        truvarimp_config=(
            {"total_budget": truvarimp_total_budget} if candidate_search == "truvarimp_joint" else None
        ),
    )
    estimator.fit(X, y)

    if estimator.metric_results_ is None:
        raise RuntimeError(
            f"Cross-family RID ({candidate_search}) returned no metric results; "
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


def _summarize_cross_family_perf(perf_summary: dict) -> dict:
    # Families with no Rashomon members report NaN stats; average only the rest.
    family_perf_stats = perf_summary.get("family_perf_stats") or {}

    def _mean_over_families(key: str) -> float | None:
        values = [stats[key] for stats in family_perf_stats.values() if not np.isnan(stats[key])]
        return float(np.mean(values)) if values else None

    return {
        "accuracy_mean": _mean_over_families("accuracy_mean"),
        "auprc_mean": _mean_over_families("auprc_mean"),
    }


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


def build_common_evaluator_panel(random_state: int) -> list[tuple[str, type, dict]]:
    """Three fixed evaluators spanning distinct inductive biases, so a feature
    set's held-out score isn't an artifact of one model's blind spots (e.g. a
    linear-only evaluator can't reward features whose signal is nonlinear or
    interaction-only -- exactly the kind of signal cross-family RID exists to
    surface). logreg = additive/linear, random_forest = axis-aligned
    interactions via bagging, svm_rbf = smooth nonlinear margin via a kernel:
    three genuinely different mechanisms rather than three flavors of tree.
    """

    return [
        (
            "logreg",
            LogisticRegression,
            {"penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 100000, "random_state": random_state},
        ),
        (
            "random_forest",
            RandomForestClassifier,
            {"n_estimators": 300, "min_samples_leaf": 1, "random_state": random_state, "n_jobs": 1},
        ),
        (
            "svm_rbf",
            SVC,
            {"kernel": "rbf", "C": 1.0, "gamma": "scale", "random_state": random_state},
        ),
    ]


def run_common_evaluation_panel(
    X: pd.DataFrame,
    y: pd.Series,
    method_feature_lists: dict[str, list[str]],
    evaluators: list[tuple[str, type, dict]],
    cv,
    scoring: str,
) -> dict[str, dict[str, dict]]:
    """Evaluate every method's selected top-k features under every evaluator
    in the panel, so methods are compared on equal footing regardless of which
    model (if any) they used internally to select features, and regardless of
    any single evaluator's own inductive bias."""

    return {
        name: {
            evaluator_name: _evaluate_feature_set(X, y, features, model_cls, model_kwargs, cv, scoring)
            for evaluator_name, model_cls, model_kwargs in evaluators
        }
        for name, features in method_feature_lists.items()
    }


def _write_json_atomic(payload: dict, path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _method_settings(method: str, args: argparse.Namespace) -> dict:
    settings = {key: getattr(args, key) for key in METHOD_SETTING_KEYS[method]}
    if method.startswith(("rid_", "cross_family_")):
        settings["rashomon_evaluation"] = "out_of_bag"
    return settings


def _method_result_path(task_output_dir: Path, method: str) -> Path:
    return task_output_dir / f"{method}_result.json"


def _load_method_checkpoint(task_output_dir: Path, method: str, args: argparse.Namespace) -> dict | None:
    """Return a method's checkpoint, or None if absent. Refuse one written under different settings."""

    path = _method_result_path(task_output_dir, method)
    if not path.exists():
        return None
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(_method_settings(method, args)))
    if checkpoint.get("settings") != expected:
        raise SystemExit(
            f"{path} was written with settings {checkpoint.get('settings')}, but this run uses "
            f"{expected}. Use a different --output-dir, or delete the checkpoint to recompute."
        )
    return checkpoint


def _run_method(
    method: str,
    X: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
    cv,
    stepwise_scoring: str,
) -> tuple[list[str], pd.DataFrame, dict]:
    """Run one method; return (top_features, per-feature table, method-specific info)."""

    if method in ("stepwise_rf", "stepwise_logreg"):
        if method == "stepwise_rf":
            model_cls, model_kwargs = _make_stepwise_rf_model(args)
        else:
            model_cls, model_kwargs = _make_stepwise_logreg_model(args)
        logger.info(
            f"[{method}] scoring={stepwise_scoring} cv_splits={args.stepwise_cv_splits} "
            f"model={model_cls.__name__}"
        )
        features, history = run_forward_stepwise_selection(
            X=X,
            y=y,
            top_k=args.top_k,
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            cv=cv,
            scoring=stepwise_scoring,
            n_jobs=args.stepwise_n_jobs,
        )
        search_eval = _evaluate_feature_set(
            X=X, y=y, features=features,
            model_cls=model_cls, model_kwargs=model_kwargs, cv=cv, scoring=stepwise_scoring,
        )
        info = {
            "scoring": stepwise_scoring,
            "model_kwargs": model_kwargs,
            "search_model_evaluation": search_eval,
        }
        return features, history, info

    if method == "rid_tree":
        logger.info(
            f"[rid_tree] metric={args.rid_metric} epsilon={args.rid_epsilon} "
            f"n_bootstraps={args.rid_n_bootstraps} n_models_pool={args.rid_n_models_pool}"
        )
        features, table, perf = run_single_family_tree_rid(
            X=X,
            y=y,
            top_k=args.top_k,
            rid_metric=args.rid_metric,
            epsilon=args.rid_epsilon,
            n_bootstraps=args.rid_n_bootstraps,
            n_models_pool=args.rid_n_models_pool,
            n_jobs=args.rid_n_jobs,
        )
        return features, table, {"metric": args.rid_metric, "rashomon_oob_perf_stats": perf}

    candidate_search = "truvarimp_joint" if method == "cross_family_unweighted_truvarimp" else "grid"
    logger.info(
        f"[{method}] metric={args.rid_metric} epsilon={args.rid_epsilon} "
        f"n_bootstraps={args.rid_n_bootstraps} candidate_search={candidate_search} "
        + (
            f"total_budget={args.truvarimp_total_budget}"
            if candidate_search == "truvarimp_joint"
            else f"n_models_per_class={args.rid_n_models_pool}"
        )
    )
    features, table, perf = run_cross_family_rid(
        X=X,
        y=y,
        top_k=args.top_k,
        rid_metric=args.rid_metric,
        epsilon=args.rid_epsilon,
        n_bootstraps=args.rid_n_bootstraps,
        n_models_per_class=args.rid_n_models_pool,
        n_jobs=args.rid_n_jobs,
        candidate_search=candidate_search,
        truvarimp_total_budget=args.truvarimp_total_budget,
    )
    info = {
        "metric": args.rid_metric,
        "candidate_search": candidate_search,
        "rashomon_oob_perf_stats": perf,
    }
    return features, table, info


def _run_single_task(
    task_name: str,
    task_config: dict,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict | None:
    """Run this invocation's --methods for one task, checkpointing each, then
    assemble the combined outputs if every method in METHODS is checkpointed.
    Returns the task's summary row, or None while methods are still missing."""

    task_output_dir = output_dir / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.data_dir) / task_config["file"]
    target_column = task_config["target_column"]
    task = task_config["task"]

    logger.info("-" * 70)
    logger.info(f"[task] {task_name}")
    logger.info(f"[task] file={csv_path}")
    logger.info(f"[task] target={target_column} task_type={task}")

    # Refuse mismatched checkpoints before spending hours on anything else.
    checkpoints = {method: _load_method_checkpoint(task_output_dir, method, args) for method in METHODS}

    X, y, data_meta = _prepare_task_dataset(
        csv_path=csv_path,
        target_column=target_column,
        correlation_threshold=args.correlation_threshold,
        include_phate_features=args.include_phate_features,
    )
    logger.info(
        f"[data] rows={data_meta['n_rows']} "
        f"features={data_meta['n_features_after_corr']} "
        f"(removed_corr={data_meta['n_removed_corr']})"
    )

    cv, stepwise_scoring = _make_stepwise_cv_and_scoring(args, y)
    evaluator_panel = build_common_evaluator_panel(args.random_state)

    for method in args.methods:
        if checkpoints[method] is not None:
            logger.info(f"[{method}] checkpoint found in {task_output_dir}; skipping")
            continue

        started = time.time()
        features, table, info = _run_method(method, X, y, args, cv, stepwise_scoring)
        elapsed = time.time() - started

        # Common held-out CV evaluation: every method's features scored the same way,
        # under a panel of evaluators spanning different inductive biases.
        common_eval = run_common_evaluation_panel(
            X, y, {method: features}, evaluator_panel, cv, stepwise_scoring
        )[method]

        _write_csv_atomic(table, task_output_dir / f"{method}_top_features.csv")
        checkpoint = {
            "settings": json.loads(json.dumps(_method_settings(method, args))),
            "top_features": features,
            "elapsed_seconds": elapsed,
            "common_eval": common_eval,
            **info,
        }
        _write_json_atomic(checkpoint, _method_result_path(task_output_dir, method))
        checkpoints[method] = checkpoint

        logger.info(f"[{method}] finished in {elapsed / 60.0:.1f} min; top-10: {features[:10]}")
        for evaluator_name, _, _ in evaluator_panel:
            evaluator_result = common_eval[evaluator_name]
            logger.info(
                f"       held-out CV [{evaluator_name}] ({stepwise_scoring}): "
                f"{evaluator_result['cv_score_mean']:.4f} +- {evaluator_result['cv_score_std']:.4f}"
            )

    completed = {method: checkpoint for method, checkpoint in checkpoints.items() if checkpoint is not None}
    missing = [method for method in METHODS if method not in completed]
    if missing:
        logger.info(f"[task] {task_name}: methods not yet checkpointed {missing}; skipping combined outputs")
        return None

    method_feature_lists = {method: completed[method]["top_features"] for method in METHODS}
    comparison = _build_comparison_table(method_feature_lists)
    _write_csv_atomic(
        comparison.sort_values(["n_methods_selected", "stepwise_rf_rank"], ascending=[False, True]),
        task_output_dir / "top_feature_overlap.csv",
    )
    overlap_counts = _pairwise_overlap_counts(comparison, list(METHODS))

    settings = {
        "task_name": task_name,
        "task_config": task_config,
        "args": vars(args),
        "data_metadata": data_meta,
        "common_evaluators": {
            "panel": {
                evaluator_name: {"model": model_cls.__name__, "model_kwargs": model_kwargs}
                for evaluator_name, model_cls, model_kwargs in evaluator_panel
            },
            "scoring": stepwise_scoring,
            "note": (
                "Every method's selected top-k features scored under every evaluator in "
                "this panel (spanning linear, bagged-tree, and kernel-margin inductive "
                "biases), so no method's score is an artifact of one evaluator's blind spot."
            ),
            "results": {method: completed[method]["common_eval"] for method in METHODS},
        },
        **{
            method: {key: value for key, value in completed[method].items() if key != "common_eval"}
            for method in METHODS
        },
        "overlap_counts": overlap_counts,
    }
    _write_json_atomic(settings, task_output_dir / "run_settings.json")

    row = {
        "task_name": task_name,
        "task_type": task,
        "target_column": target_column,
        "dataset_path": str(csv_path),
        "n_rows": data_meta["n_rows"],
        "n_features": data_meta["n_features_after_corr"],
        "common_eval_scoring": stepwise_scoring,
    }
    for method in METHODS:
        per_evaluator_means = []
        for evaluator_name, _, _ in evaluator_panel:
            evaluator_result = completed[method]["common_eval"][evaluator_name]
            row[f"{method}_held_out_cv_score_mean__{evaluator_name}"] = evaluator_result["cv_score_mean"]
            row[f"{method}_held_out_cv_score_std__{evaluator_name}"] = evaluator_result["cv_score_std"]
            per_evaluator_means.append(evaluator_result["cv_score_mean"])
        row[f"{method}_held_out_cv_score_mean_avg"] = float(np.mean(per_evaluator_means))
        row[f"{method}_elapsed_seconds"] = completed[method]["elapsed_seconds"]

    rid_tree_perf = completed["rid_tree"]["rashomon_oob_perf_stats"]
    row["rid_tree_rashomon_oob_accuracy_mean"] = rid_tree_perf.get("accuracy_mean")
    row["rid_tree_rashomon_oob_auprc_mean"] = rid_tree_perf.get("auprc_mean")
    for method in ("cross_family_unweighted", "cross_family_unweighted_truvarimp"):
        cross_family_rashomon = _summarize_cross_family_perf(completed[method]["rashomon_oob_perf_stats"])
        row[f"{method}_rashomon_oob_accuracy_mean"] = cross_family_rashomon["accuracy_mean"]
        row[f"{method}_rashomon_oob_auprc_mean"] = cross_family_rashomon["auprc_mean"]
    row.update(overlap_counts)

    logger.info(f"[task] combined outputs saved in {task_output_dir}")
    return row


def main() -> None:
    args = _parse_args()
    silence_warnings()
    configure_logging(args.log_level, args.log_file)
    task_names = _parse_task_names(args.tasks)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Staellert Direct Target Prediction Tasks")
    logger.info(f"  data_dir:                 {args.data_dir}")
    logger.info(f"  tasks:                    {task_names}")
    logger.info(f"  methods:                  {args.methods}")
    logger.info(f"  top_k:                    {args.top_k}")
    logger.info(f"  correlation_threshold:    {args.correlation_threshold}")
    logger.info(f"  include_phate_features:   {args.include_phate_features}")
    logger.info(f"  truvarimp_total_budget:   {args.truvarimp_total_budget}")
    logger.info(f"  output_dir:               {output_dir}")
    logger.info("=" * 70)

    overall_rows = []
    for task_name in task_names:
        row = _run_single_task(
            task_name=task_name,
            task_config=TASK_REGISTRY[task_name],
            args=args,
            output_dir=output_dir,
        )
        if row is not None:
            overall_rows.append(row)

    if len(overall_rows) < len(task_names):
        logger.info("=" * 70)
        logger.info("Checkpoints saved; combined summary waits until every method has run for every task")
        logger.info("=" * 70)
        return

    overall_summary_path = output_dir / "overall_task_summary.csv"
    _write_csv_atomic(pd.DataFrame(overall_rows), overall_summary_path)

    run_manifest = {
        "args": vars(args),
        "tasks_run": task_names,
        "methods": list(METHODS),
        "overall_summary_csv": str(overall_summary_path),
    }
    manifest_path = output_dir / "run_manifest.json"
    _write_json_atomic(run_manifest, manifest_path)

    logger.info("=" * 70)
    logger.info("Completed")
    logger.info(f"  Overall summary:          {overall_summary_path}")
    logger.info(f"  Run manifest:             {manifest_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
