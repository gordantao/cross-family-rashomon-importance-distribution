#!/usr/bin/env python3
"""Run Staellert direct target prediction tasks with feature selection.

This script runs direct target prediction tasks on the Staellert control manifold
and compares two top-k feature sets where possible:
1) Random-forest forward stepwise selection.
2) CrossFamilyRashomonImportanceDistribution with family_balance_mode='unweighted'
   (classification tasks only).

Default tasks are those directly annotated in the paper-style dataset:
- annotated_phase (classification)
- annotated_age (regression)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.svm import SVC

from rid import (
    CrossFamilyRashomonImportanceDistribution,
    ElasticNetClassifier,
    LassoClassifier,
    RidgeClassifier,
)


TASK_REGISTRY = {
    "annotated_phase": {
        "file": "control_manifold_allfeatures.csv",
        "target_column": "annotated phase",
        "task": "classification",
    },
    "annotated_age": {
        "file": "control_manifold_allfeatures.csv",
        "target_column": "annotated age",
        "task": "regression",
    },
    "phase": {
        "file": "control_manifold_allfeatures.csv",
        "target_column": "phase",
        "task": "classification",
    },
    "age": {
        "file": "control_manifold_allfeatures.csv",
        "target_column": "age",
        "task": "regression",
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
            "Run direct target prediction tasks on Staellert data with random-forest "
            "forward stepwise feature selection, and unweighted RID on classification tasks."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/staellert_et_al",
        help="Directory containing Staellert CSV files (default: data/staellert_et_al)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="annotated_phase,annotated_age",
        help=(
            "Comma-separated task names. Use 'all' for all known tasks. "
            "Default: annotated_phase,annotated_age"
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
        default="results/staellert_direct_target_tasks",
        help="Directory for outputs (default: results/staellert_direct_target_tasks)",
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
        "--stepwise-scoring-regression",
        type=str,
        default="neg_root_mean_squared_error",
        help="sklearn scoring for regression stepwise CV (default: neg_root_mean_squared_error)",
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
            "Class weight for classification stepwise model. Use 'none' to disable "
            "(default: balanced)."
        ),
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
        "--rid-n-models-per-class",
        type=int,
        default=50,
        help="RID candidate models per family and bootstrap (default: 50)",
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
    task: str,
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
    if task == "classification":
        valid_rows = y_raw.notna()
        y = y_raw.loc[valid_rows].astype(str)
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        valid_rows = y.notna()
        y = y.loc[valid_rows]

    X = X.loc[valid_rows].copy()

    X_before_corr = X.shape[1]
    X, removed_corr = _remove_highly_correlated_features(X, correlation_threshold)

    if task == "classification":
        target_meta = {
            "n_classes": int(y.nunique()),
            "class_counts": y.value_counts().to_dict(),
        }
    else:
        target_meta = {
            "target_min": float(y.min()),
            "target_max": float(y.max()),
            "target_mean": float(y.mean()),
            "target_std": float(y.std()),
        }

    metadata = {
        "dataset_path": str(csv_path),
        "target_column": target_column,
        "task": task,
        "n_rows": int(X.shape[0]),
        "n_features_before_corr": int(X_before_corr),
        "n_features_after_corr": int(X.shape[1]),
        "n_removed_corr": int(len(removed_corr)),
        "removed_corr_features": removed_corr,
        **target_meta,
    }
    return X, y, metadata


def _make_stepwise_model_spec(
    args: argparse.Namespace,
    task: str,
    y: pd.Series,
):
    if task == "classification":
        class_weight = None if str(args.stepwise_class_weight).lower() == "none" else args.stepwise_class_weight
        model_cls = RandomForestClassifier
        model_kwargs = {
            "n_estimators": args.stepwise_n_estimators,
            "random_state": args.random_state,
            "n_jobs": 1,
            "max_depth": args.stepwise_max_depth,
            "min_samples_leaf": args.stepwise_min_samples_leaf,
            "class_weight": class_weight,
        }
        cv = StratifiedKFold(
            n_splits=args.stepwise_cv_splits,
            shuffle=True,
            random_state=args.random_state,
        )
        if args.stepwise_scoring_classification:
            scoring = args.stepwise_scoring_classification
        else:
            scoring = "roc_auc" if int(y.nunique()) == 2 else "f1_macro"
    else:
        model_cls = RandomForestRegressor
        model_kwargs = {
            "n_estimators": args.stepwise_n_estimators,
            "random_state": args.random_state,
            "n_jobs": 1,
            "max_depth": args.stepwise_max_depth,
            "min_samples_leaf": args.stepwise_min_samples_leaf,
        }
        cv = KFold(
            n_splits=args.stepwise_cv_splits,
            shuffle=True,
            random_state=args.random_state,
        )
        scoring = args.stepwise_scoring_regression

    return model_cls, model_kwargs, cv, scoring


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


def _build_cross_family_model_configs() -> dict:
    return {
        "RandomForest": (RandomForestClassifier, {}),
        "GradientBoosting": (GradientBoostingClassifier, {}),
        "SVM": (SVC, {}),
        "Lasso": (LassoClassifier, {}),
        "ElasticNet": (ElasticNetClassifier, {}),
        "Ridge": (RidgeClassifier, {}),
    }


def run_unweighted_rid(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int,
    rid_metric: str,
    epsilon: float,
    n_bootstraps: int,
    n_models_per_class: int,
    n_jobs: int,
) -> tuple[list[str], pd.DataFrame]:
    estimator = CrossFamilyRashomonImportanceDistribution(
        model_configs=_build_cross_family_model_configs(),
        epsilon=epsilon,
        n_bootstraps=n_bootstraps,
        n_models_per_class=n_models_per_class,
        vi_metrics=(rid_metric,),
        performance_metrics=("accuracy", "auprc"),
        family_balance_mode="unweighted",
        n_jobs=n_jobs,
    )
    estimator.fit(X, y)

    if estimator.metric_results_ is None:
        raise RuntimeError("RID returned no metric results; no valid Rashomon bootstraps were found")

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
    return top_features, pd.DataFrame(rows)


def _build_comparison_table(
    stepwise_features: Iterable[str],
    rid_features: Iterable[str],
) -> pd.DataFrame:
    stepwise_rank = {feature: rank for rank, feature in enumerate(stepwise_features, start=1)}
    rid_rank = {feature: rank for rank, feature in enumerate(rid_features, start=1)}

    all_features = sorted(set(stepwise_rank) | set(rid_rank))
    rows = []
    for feature in all_features:
        s_rank = stepwise_rank.get(feature)
        r_rank = rid_rank.get(feature)
        rows.append(
            {
                "feature": feature,
                "stepwise_rank": s_rank,
                "rid_rank": r_rank,
                "in_stepwise_top_k": s_rank is not None,
                "in_rid_top_k": r_rank is not None,
                "in_both": s_rank is not None and r_rank is not None,
            }
        )
    return pd.DataFrame(rows)


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
        task=task,
        correlation_threshold=args.correlation_threshold,
        include_phate_features=args.include_phate_features,
    )
    print(
        f"[data] rows={data_meta['n_rows']} "
        f"features={data_meta['n_features_after_corr']} "
        f"(removed_corr={data_meta['n_removed_corr']})"
    )

    model_cls, model_kwargs, cv, stepwise_scoring = _make_stepwise_model_spec(args, task, y)
    print(
        f"[stepwise] scoring={stepwise_scoring} cv_splits={args.stepwise_cv_splits} "
        f"n_estimators={args.stepwise_n_estimators}"
    )

    stepwise_features, stepwise_history = run_forward_stepwise_selection(
        X=X,
        y=y,
        top_k=args.top_k,
        model_cls=model_cls,
        model_kwargs=model_kwargs,
        cv=cv,
        scoring=stepwise_scoring,
        n_jobs=args.stepwise_n_jobs,
    )
    stepwise_eval = _evaluate_feature_set(
        X=X,
        y=y,
        features=stepwise_features,
        model_cls=model_cls,
        model_kwargs=model_kwargs,
        cv=cv,
        scoring=stepwise_scoring,
    )

    stepwise_path = task_output_dir / "stepwise_top_features.csv"
    stepwise_history.to_csv(stepwise_path, index=False)

    rid_top_features: list[str] = []
    rid_eval: dict | None = None
    overlap_count: int | None = None

    if task == "classification":
        print(
            f"[rid] metric={args.rid_metric} epsilon={args.rid_epsilon} "
            f"n_bootstraps={args.rid_n_bootstraps} n_models_per_class={args.rid_n_models_per_class} "
            "family_balance_mode=unweighted"
        )
        rid_top_features, rid_table = run_unweighted_rid(
            X=X,
            y=y,
            top_k=args.top_k,
            rid_metric=args.rid_metric,
            epsilon=args.rid_epsilon,
            n_bootstraps=args.rid_n_bootstraps,
            n_models_per_class=args.rid_n_models_per_class,
            n_jobs=args.rid_n_jobs,
        )
        comparison = _build_comparison_table(stepwise_features, rid_top_features)
        overlap_count = int(comparison["in_both"].sum())

        rid_eval = _evaluate_feature_set(
            X=X,
            y=y,
            features=rid_top_features,
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            cv=cv,
            scoring=stepwise_scoring,
        )

        rid_path = task_output_dir / "rid_unweighted_top_features.csv"
        overlap_path = task_output_dir / "top_feature_overlap.csv"
        rid_table.to_csv(rid_path, index=False)
        comparison.sort_values(
            ["in_both", "stepwise_rank", "rid_rank"],
            ascending=[False, True, True],
        ).to_csv(overlap_path, index=False)
    else:
        print("[rid] skipped: RID currently supports classification-style tasks only")

    settings = {
        "task_name": task_name,
        "task_config": task_config,
        "args": vars(args),
        "data_metadata": data_meta,
        "stepwise": {
            "scoring": stepwise_scoring,
            "top_features": stepwise_features,
            "evaluation": stepwise_eval,
        },
        "rid_unweighted": {
            "metric": args.rid_metric,
            "top_features": rid_top_features,
            "evaluation": rid_eval,
            "overlap_count": overlap_count,
            "overlap_ratio": (
                None
                if overlap_count is None
                else float(overlap_count / max(1, min(len(stepwise_features), len(rid_top_features))))
            ),
        },
    }

    settings_path = task_output_dir / "run_settings.json"
    with settings_path.open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)

    print(f"[task] outputs saved in {task_output_dir}")
    print(f"[task] stepwise top-10: {stepwise_features[:10]}")
    if rid_top_features:
        print(f"[task] rid top-10:      {rid_top_features[:10]}")

    return {
        "task_name": task_name,
        "task_type": task,
        "target_column": target_column,
        "dataset_path": str(csv_path),
        "n_rows": data_meta["n_rows"],
        "n_features": data_meta["n_features_after_corr"],
        "stepwise_scoring": stepwise_scoring,
        "stepwise_cv_score_mean": stepwise_eval["cv_score_mean"],
        "stepwise_cv_score_std": stepwise_eval["cv_score_std"],
        "rid_cv_score_mean": None if rid_eval is None else rid_eval["cv_score_mean"],
        "rid_cv_score_std": None if rid_eval is None else rid_eval["cv_score_std"],
        "top_k_overlap": overlap_count,
    }


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
