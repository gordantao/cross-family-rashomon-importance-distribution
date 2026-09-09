#!/usr/bin/env python3
"""Run Staellert direct target prediction tasks with feature selection.

This script is classification-only (regression-task support was removed --
see TASK_REGISTRY) and compares five top-k feature sets:
1) Forward stepwise selection scored via random forest.
2) Forward stepwise selection scored via logistic regression.
3) Single-family RID on a fully enumerated decision-tree Rashomon set.
4) Cross-family RID with family_balance_mode='unweighted'.
5) Cross-family RID with family_balance_mode='weighted'.

Every method's selected top-k feature set is scored with the SAME held-out
cross-validated evaluator (logistic regression, proper k-fold CV) so all five
methods are directly comparable. RID's own internal Rashomon-set performance
stats (accuracy/AUPRC computed on the same bootstrap sample each model was
fit on -- in-sample, not held-out) are recorded separately in
run_settings.json for diagnostic purposes only; they are NOT comparable to
the held-out CV scores and are not used to rank methods.

Default task is the classification target directly annotated in the
paper-style dataset: annotated_phase.
"""

from __future__ import annotations

import argparse
import json
import sys
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
)


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run classification-only direct target prediction tasks on Staellert data, "
            "comparing random-forest and logistic-regression forward stepwise feature "
            "selection against single-family RID on a fully enumerated decision-tree "
            "Rashomon set and cross-family RID (both unweighted and weighted balance modes)."
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
        type=float,
        default=0.8,
        help=(
            "Remove highly correlated descriptors above this absolute threshold. "
            "Set to a value <= 0 to disable (default: 0.8)."
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
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
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


def _prepare_task_dataset(
    csv_path: Path,
    target_column: str,
    correlation_threshold: float,
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
    X, removed_corr = _remove_highly_correlated_features(X, correlation_threshold)

    metadata = {
        "dataset_path": str(csv_path),
        "target_column": target_column,
        "task": "classification",
        "n_rows": int(X.shape[0]),
        "n_features_before_corr": int(X_before_corr),
        "n_features_after_corr": int(X.shape[1]),
        "n_removed_corr": int(len(removed_corr)),
        "removed_corr_features": removed_corr,
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

        print(
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
    family_perf_stats = perf_summary.get("family_perf_stats") or {}
    accuracy_means = [stats["accuracy_mean"] for stats in family_perf_stats.values()]
    auprc_means = [stats["auprc_mean"] for stats in family_perf_stats.values()]
    return {
        "accuracy_mean": float(np.mean(accuracy_means)) if accuracy_means else None,
        "auprc_mean": float(np.mean(auprc_means)) if auprc_means else None,
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


def _run_single_task(
    task_name: str,
    task_config: dict,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    task_output_dir = output_dir / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.data_dir) / task_config["file"]
    target_column = task_config["target_column"]
    task = task_config["task"]

    print("-" * 70)
    print(f"[task] {task_name}")
    print(f"[task] file={csv_path}")
    print(f"[task] target={target_column} task_type={task}")

    X, y, data_meta = _prepare_task_dataset(
        csv_path=csv_path,
        target_column=target_column,
        correlation_threshold=args.correlation_threshold,
        include_phate_features=args.include_phate_features,
    )
    print(
        f"[data] rows={data_meta['n_rows']} "
        f"features={data_meta['n_features_after_corr']} "
        f"(removed_corr={data_meta['n_removed_corr']})"
    )

    cv, stepwise_scoring = _make_stepwise_cv_and_scoring(args, y)

    # --- Method 1: forward stepwise selection via random forest ---
    rf_model_cls, rf_model_kwargs = _make_stepwise_rf_model(args)
    print(
        f"[stepwise:rf] scoring={stepwise_scoring} cv_splits={args.stepwise_cv_splits} "
        f"n_estimators={args.stepwise_n_estimators}"
    )
    stepwise_rf_features, stepwise_rf_history = run_forward_stepwise_selection(
        X=X,
        y=y,
        top_k=args.top_k,
        model_cls=rf_model_cls,
        model_kwargs=rf_model_kwargs,
        cv=cv,
        scoring=stepwise_scoring,
        n_jobs=args.stepwise_n_jobs,
    )
    stepwise_rf_eval = _evaluate_feature_set(
        X=X, y=y, features=stepwise_rf_features,
        model_cls=rf_model_cls, model_kwargs=rf_model_kwargs, cv=cv, scoring=stepwise_scoring,
    )
    stepwise_rf_history.to_csv(task_output_dir / "stepwise_rf_top_features.csv", index=False)

    # --- Method 2: forward stepwise selection via logistic regression ---
    logreg_model_cls, logreg_model_kwargs = _make_stepwise_logreg_model(args)
    print(
        f"[stepwise:logreg] scoring={stepwise_scoring} cv_splits={args.stepwise_cv_splits} "
        f"model={logreg_model_cls.__name__}"
    )
    stepwise_logreg_features, stepwise_logreg_history = run_forward_stepwise_selection(
        X=X,
        y=y,
        top_k=args.top_k,
        model_cls=logreg_model_cls,
        model_kwargs=logreg_model_kwargs,
        cv=cv,
        scoring=stepwise_scoring,
        n_jobs=args.stepwise_n_jobs,
    )
    stepwise_logreg_eval = _evaluate_feature_set(
        X=X, y=y, features=stepwise_logreg_features,
        model_cls=logreg_model_cls, model_kwargs=logreg_model_kwargs, cv=cv, scoring=stepwise_scoring,
    )
    stepwise_logreg_history.to_csv(task_output_dir / "stepwise_logreg_top_features.csv", index=False)

    method_feature_lists = {
        "stepwise_rf": stepwise_rf_features,
        "stepwise_logreg": stepwise_logreg_features,
    }

    # --- Method 3: single-family RID on a fully enumerated tree Rashomon set ---
    # --- Methods 4-5: cross-family RID (unweighted, weighted) ---
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
        n_jobs=args.rid_n_jobs,
    )
    rid_tree_table.to_csv(task_output_dir / "rid_tree_top_features.csv", index=False)
    method_feature_lists["rid_tree"] = rid_tree_features

    cross_family_features: dict[str, list[str]] = {"unweighted": [], "weighted": []}
    cross_family_perf: dict[str, dict] = {"unweighted": {}, "weighted": {}}
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
            n_jobs=args.rid_n_jobs,
        )
        table.to_csv(task_output_dir / f"rid_cross_family_{balance_mode}_top_features.csv", index=False)
        cross_family_features[balance_mode] = features
        cross_family_perf[balance_mode] = perf
        method_feature_lists[f"cross_family_{balance_mode}"] = features

    comparison = _build_comparison_table(method_feature_lists)
    comparison.sort_values(
        ["n_methods_selected", "stepwise_rf_rank"],
        ascending=[False, True],
    ).to_csv(task_output_dir / "top_feature_overlap.csv", index=False)
    overlap_counts = _pairwise_overlap_counts(comparison, list(method_feature_lists.keys()))

    # --- Common held-out CV evaluation: every method's features scored the same way,
    # under a panel of evaluators spanning different inductive biases ---
    evaluator_panel = build_common_evaluator_panel(args.random_state)
    common_eval = run_common_evaluation_panel(
        X, y, method_feature_lists, evaluator_panel, cv, stepwise_scoring
    )

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
            "results": common_eval,
        },
        "stepwise_rf": {
            "scoring": stepwise_scoring,
            "model_kwargs": rf_model_kwargs,
            "top_features": stepwise_rf_features,
            "search_model_evaluation": stepwise_rf_eval,
        },
        "stepwise_logreg": {
            "scoring": stepwise_scoring,
            "model_kwargs": logreg_model_kwargs,
            "top_features": stepwise_logreg_features,
            "search_model_evaluation": stepwise_logreg_eval,
        },
        "rid_tree": {
            "metric": args.rid_metric,
            "top_features": rid_tree_features,
            "rashomon_train_perf_stats": rid_tree_perf,
        },
        "cross_family_unweighted": {
            "metric": args.rid_metric,
            "top_features": cross_family_features["unweighted"],
            "rashomon_train_perf_stats": cross_family_perf["unweighted"],
        },
        "cross_family_weighted": {
            "metric": args.rid_metric,
            "top_features": cross_family_features["weighted"],
            "rashomon_train_perf_stats": cross_family_perf["weighted"],
        },
        "overlap_counts": overlap_counts,
    }

    settings_path = task_output_dir / "run_settings.json"
    with settings_path.open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)

    print(f"[task] outputs saved in {task_output_dir}")
    for name, features in method_feature_lists.items():
        print(f"[task] {name} top-10: {features[:10]}")
        for evaluator_name, _, _ in evaluator_panel:
            evaluator_result = common_eval[name][evaluator_name]
            print(
                f"       held-out CV [{evaluator_name}] ({stepwise_scoring}): "
                f"{evaluator_result['cv_score_mean']:.4f} +- {evaluator_result['cv_score_std']:.4f}"
            )

    cross_family_unweighted_rashomon = _summarize_cross_family_perf(cross_family_perf["unweighted"])
    cross_family_weighted_rashomon = _summarize_cross_family_perf(cross_family_perf["weighted"])

    row = {
        "task_name": task_name,
        "task_type": task,
        "target_column": target_column,
        "dataset_path": str(csv_path),
        "n_rows": data_meta["n_rows"],
        "n_features": data_meta["n_features_after_corr"],
        "common_eval_scoring": stepwise_scoring,
    }
    for name in method_feature_lists:
        per_evaluator_means = []
        for evaluator_name, _, _ in evaluator_panel:
            evaluator_result = common_eval[name][evaluator_name]
            row[f"{name}_held_out_cv_score_mean__{evaluator_name}"] = evaluator_result["cv_score_mean"]
            row[f"{name}_held_out_cv_score_std__{evaluator_name}"] = evaluator_result["cv_score_std"]
            per_evaluator_means.append(evaluator_result["cv_score_mean"])
        row[f"{name}_held_out_cv_score_mean_avg"] = float(np.mean(per_evaluator_means))
    row.update(
        {
            "rid_tree_rashomon_train_accuracy_mean": rid_tree_perf.get("accuracy_mean"),
            "rid_tree_rashomon_train_auprc_mean": rid_tree_perf.get("auprc_mean"),
            "cross_family_unweighted_rashomon_train_accuracy_mean": cross_family_unweighted_rashomon["accuracy_mean"],
            "cross_family_unweighted_rashomon_train_auprc_mean": cross_family_unweighted_rashomon["auprc_mean"],
            "cross_family_weighted_rashomon_train_accuracy_mean": cross_family_weighted_rashomon["accuracy_mean"],
            "cross_family_weighted_rashomon_train_auprc_mean": cross_family_weighted_rashomon["auprc_mean"],
        }
    )
    row.update(overlap_counts)
    return row


def main() -> None:
    args = _parse_args()
    task_names = _parse_task_names(args.tasks)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Staellert Direct Target Prediction Tasks")
    print(f"  data_dir:                 {args.data_dir}")
    print(f"  tasks:                    {task_names}")
    print(f"  top_k:                    {args.top_k}")
    print(f"  correlation_threshold:    {args.correlation_threshold}")
    print(f"  include_phate_features:   {args.include_phate_features}")
    print(f"  output_dir:               {output_dir}")
    print("=" * 70)

    overall_rows = []
    for task_name in task_names:
        row = _run_single_task(
            task_name=task_name,
            task_config=TASK_REGISTRY[task_name],
            args=args,
            output_dir=output_dir,
        )
        overall_rows.append(row)

    overall_summary_path = output_dir / "overall_task_summary.csv"
    pd.DataFrame(overall_rows).to_csv(overall_summary_path, index=False)

    run_manifest = {
        "args": vars(args),
        "tasks_run": task_names,
        "overall_summary_csv": str(overall_summary_path),
    }
    manifest_path = output_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2)

    print("=" * 70)
    print("Completed")
    print(f"  Overall summary:          {overall_summary_path}")
    print(f"  Run manifest:             {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
