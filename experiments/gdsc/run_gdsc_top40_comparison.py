#!/usr/bin/env python3
"""Compare top-k feature selection methods on GDSC drug-sensitivity tasks.

Each task (gdsc_tasks.TASKS) predicts whether a cancer cell line is sensitive
to one drug (GDSC2 Z_SCORE below a threshold, set in prepare_gdsc_data.py)
from one shared, drug-agnostic feature panel: driver mutations, amplifications
and deletions, expression of the most variable genes plus every task's
biomarker genes, tissue and MSI status. Run prepare_gdsc_data.py first.

Mirrors experiments/staellert/run_staellert_top40_comparison.py (same methods,
checkpointing and evaluator panel) and adds one thing the Staellert data can't
support: an answer key. Each task's literature biomarkers (gdsc_tasks.py) are
looked up in every method's ranking and written to biomarker_recovery.csv.

Compares five top-k feature sets (METHODS):
1) stepwise_rf: forward stepwise selection scored via random forest.
2) stepwise_logreg: forward stepwise selection scored via logistic regression.
3) rid_tree: single-family RID on a fully enumerated decision-tree Rashomon set.
4) cross_family_unweighted: cross-family RID, exhaustive grid candidate search.
5) cross_family_unweighted_truvarimp: cross-family RID, joint TruVaRImp
   candidate search (candidate_search='truvarimp_joint', rid/truvarimp.py).
The weighted cross-family variant was dropped, as in the nonlinear simulation.

Evaluation protocol (outer stratified holdout): each task's rows are split once
into train and test (--holdout-fraction, default 0.2; --holdout-seed). Feature
selection never sees the test rows: the constant-column and correlation filters,
standardization, stepwise selection (whose inner CV runs on the train rows only)
and every RID fit use the train rows alone. Every method's selected top-k
feature set is then scored with the SAME panel of evaluators, each fit on the
train rows restricted to those features and scored once on the held-out test
rows (holdout_score) -- the primary, comparable score for every method. The
panel's in-train CV score (train_cv_score_*, the same folds stepwise selection
maximized) is kept alongside only to show how much reusing selection folds for
evaluation inflates the stepwise methods. RID's own Rashomon-set performance
stats (accuracy/AUPRC on each bootstrap's out-of-bag train rows) are recorded in
run_settings.json for diagnostics only and are not used to rank methods.

Each method is checkpointed separately (<task>/<method>_result.json plus
<method>_ranking.csv -- the k stepwise steps, or every feature for RID
methods), so --methods can run any subset -- e.g. one (task, method) pair per
SLURM array task -- and reruns skip methods already checkpointed. Once every
method has a checkpoint, the task's combined outputs (top_feature_overlap.csv,
biomarker_recovery.csv, run_settings.json) are (re)built, and
overall_task_summary.csv once every requested task is complete.

Differences from the Staellert script: features are standardized once after
correlation filtering (so the logistic and RBF-SVM evaluators see scaled
inputs, which the Staellert SVM evaluator didn't); the correlation filter keeps
the earlier column of a pair rather than the higher-variance one; and there are
no PHATE columns to exclude. Filtering and standardization are fit on the train
rows only and then applied to the test rows.
"""

from __future__ import annotations

import argparse
import hashlib
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
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gdsc_tasks import TASKS  # noqa: E402
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


TASK_REGISTRY = {name: {"task": "classification", **task} for name, task in TASKS.items()}

FEATURES_FILE = "features.csv"
RESPONSES_FILE = "responses.csv"
LABEL_COLUMN = "sensitive"

METHODS = (
    "stepwise_rf",
    "stepwise_logreg",
    "rid_tree",
    "cross_family_unweighted",
    "cross_family_unweighted_truvarimp",
)

# Arguments that change a method's result. A checkpoint written under different
# values is refused rather than silently reused (n_jobs/logging/paths don't count).
_DATA_SETTING_KEYS = ("data_fingerprint", "top_k", "correlation_threshold", "random_state",
                      "stepwise_cv_splits", "stepwise_scoring_classification",
                      "holdout_fraction", "holdout_seed")
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
            "Run GDSC drug-sensitivity classification tasks, "
            "comparing random-forest and logistic-regression forward stepwise feature "
            "selection against single-family RID on a fully enumerated decision-tree "
            "Rashomon set and unweighted cross-family RID (grid and TruVaRImp candidate search)."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "processed"),
        help="Directory with prepare_gdsc_data.py's outputs (default: <script dir>/data/processed)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help=f"Comma-separated task names from {list(TASKS)}, or 'all' (default: all)",
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
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "top40_feature_comparison_holdout"),
        help="Directory for outputs (default: <script dir>/results/top40_feature_comparison_holdout)",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of each task's rows held out (stratified) as the test set the "
            "evaluator panel scores on; feature selection only sees the rest (default: 0.2)"
        ),
    )
    parser.add_argument(
        "--holdout-seed",
        type=int,
        default=42,
        help="Random seed for the outer stratified train/test split (default: 42)",
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
        help="CV folds for stepwise selection and the in-train CV diagnostic, on the train rows only (default: 5)",
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


def _remove_highly_correlated_features(
    X: pd.DataFrame,
    correlation_threshold: float,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Drop one feature of every pair correlated above the threshold.

    Unlike the Staellert copy (which keeps the higher-variance feature), this keeps
    the earlier column. Variance isn't comparable across this panel's units -- a
    binary feature's is at most 0.25, so expression would always win -- while
    column order is prepare_gdsc_data.py's block order (mutation, amplification,
    deletion, expression, tissue, MSI): a genomic alteration is kept over the
    expression it drives. Also returns {removed: the kept feature it lost to}.
    """

    if correlation_threshold <= 0:
        return X, [], {}

    corr_matrix = X.corr().abs()
    removed_features: set[str] = set()
    kept_in_favour_of: dict[str, str] = {}

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] <= correlation_threshold:
                continue

            feature_one = corr_matrix.columns[i]
            feature_two = corr_matrix.columns[j]
            if feature_one in removed_features or feature_two in removed_features:
                continue

            # feature_one is final here: it could only have been removed as an earlier pair's feature_two.
            removed_features.add(feature_two)
            kept_in_favour_of[feature_two] = feature_one

    removed_sorted = sorted(removed_features)
    if removed_sorted:
        X = X.drop(columns=removed_sorted)
    return X, removed_sorted, kept_in_favour_of


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
        X_t, removed_t, _ = _remove_highly_correlated_features(X, correlation_threshold=float(t))
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


def _make_holdout_split(y: pd.Series, holdout_fraction: float, holdout_seed: int) -> tuple[pd.Index, pd.Index, dict]:
    """One outer stratified train/test split of a task's rows (original row order
    kept within each part). Feature selection sees only the train rows; the
    evaluator panel scores on the test rows. Also returns the split's metadata,
    including a hash of the test rows so checkpoints from another split are refused."""

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"--holdout-fraction must be in (0, 1), got {holdout_fraction}")
    if not y.index.is_unique:
        raise ValueError("Row index must be unique to define a holdout split")
    train_pos, test_pos = train_test_split(
        np.arange(len(y)), test_size=holdout_fraction, stratify=y.to_numpy(), random_state=holdout_seed
    )
    train_index, test_index = y.index[np.sort(train_pos)], y.index[np.sort(test_pos)]
    test_ids = [str(i) for i in test_index]
    metadata = {
        "protocol": "outer_stratified_holdout",
        "holdout_fraction": holdout_fraction,
        "holdout_seed": holdout_seed,
        "n_train": int(len(train_index)),
        "n_test": int(len(test_index)),
        "train_class_counts": {str(k): int(v) for k, v in y.loc[train_index].value_counts().items()},
        "test_class_counts": {str(k): int(v) for k, v in y.loc[test_index].value_counts().items()},
        "test_rows_sha256": hashlib.sha256("\n".join(test_ids).encode("utf-8")).hexdigest()[:16],
        "test_rows": test_ids,
    }
    return train_index, test_index, metadata


def _prepare_task_dataset(
    data_dir: Path,
    task_name: str,
    correlation_threshold,
    holdout_fraction: float,
    holdout_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """Load one task, split it into train/test rows, and fit every column filter
    and the standardization on the train rows only. Returns
    (X_train, X_test, y_train, y_test, metadata)."""

    features_path = data_dir / FEATURES_FILE
    responses_path = data_dir / RESPONSES_FILE
    for path in (features_path, responses_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run experiments/gdsc/prepare_gdsc_data.py first")

    features = pd.read_csv(features_path, index_col="model_id")
    responses = pd.read_csv(responses_path)
    responses = responses[responses["task"] == task_name].set_index("model_id")
    if responses.empty:
        raise ValueError(f"No responses for task '{task_name}' in {responses_path}; rerun prepare_gdsc_data.py")

    model_ids = responses.index.intersection(features.index)
    X = features.loc[model_ids]
    # Integer 0/1, not strings: rid's AUPRC metric scores binary tasks with pos_label=1.
    y = responses.loc[model_ids, LABEL_COLUMN].astype(int)

    train_index, test_index, holdout_meta = _make_holdout_split(y, holdout_fraction, holdout_seed)

    # Binary panels can be all-zero within one task's (train) cell lines.
    constant_cols = [col for col in X.columns if X.loc[train_index, col].nunique() <= 1]
    X = X.drop(columns=constant_cols)

    X_before_corr = X.shape[1]
    resolved_threshold, threshold_info = _resolve_correlation_threshold(X.loc[train_index], correlation_threshold)
    X_train, removed_corr, corr_proxies = _remove_highly_correlated_features(X.loc[train_index], resolved_threshold)
    X = X[X_train.columns]
    # Standardize with train-row statistics so the logistic/SVM evaluators and stepwise
    # logreg see scaled inputs; RID rescales internally and trees are unaffected.
    train_mean, train_std = X.loc[train_index].mean(), X.loc[train_index].std(ddof=0)
    X = (X - train_mean) / train_std

    metadata = {
        "features_path": str(features_path),
        "responses_path": str(responses_path),
        "label_column": LABEL_COLUMN,
        "task": "classification",
        "n_rows": int(X.shape[0]),
        "n_features_before_corr": int(X_before_corr),
        "n_features_after_corr": int(X.shape[1]),
        "n_removed_constant": len(constant_cols),
        "n_removed_corr": int(len(removed_corr)),
        "removed_corr_features": removed_corr,
        "corr_proxies": corr_proxies,
        "correlation_threshold_info": threshold_info,
        "n_classes": int(y.nunique()),
        "class_counts": y.value_counts().to_dict(),
        "holdout": holdout_meta,
    }
    return X.loc[train_index], X.loc[test_index], y.loc[train_index], y.loc[test_index], metadata


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
                "train_cv_score_mean": best_mean,
                "train_cv_score_std": best_std,
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
    top_features = [feature for feature, _ in ranking[:top_k]]

    # Every feature, not just the top k, so biomarkers ranked below k still get a rank.
    rows = []
    for rank, (feature, expected_importance) in enumerate(ranking, start=1):
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
    top_features = [feature for feature, _ in ranking[:top_k]]

    # Every feature, not just the top k, so biomarkers ranked below k still get a rank.
    rows = []
    for rank, (feature, expected_importance) in enumerate(ranking, start=1):
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


def _evaluate_feature_set_holdout(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: list[str],
    model_cls,
    model_kwargs: dict,
    train_cv,
    scoring: str,
) -> dict:
    """Fit on the train rows restricted to `features`, score once on the held-out
    test rows (holdout_score, the primary score). Also report the same evaluator's
    CV score within the train rows (train_cv_score_*), on the folds stepwise
    selection searched over -- optimistic for the stepwise methods by construction."""

    estimator = model_cls(**model_kwargs).fit(X_train[features], y_train)
    holdout_score = float(get_scorer(scoring)(estimator, X_test[features], y_test))
    train_cv = _evaluate_feature_set(X_train, y_train, features, model_cls, model_kwargs, train_cv, scoring)
    return {
        "scoring": scoring,
        "holdout_score": holdout_score,
        "train_cv_score_mean": train_cv["cv_score_mean"],
        "train_cv_score_std": train_cv["cv_score_std"],
        "n_features": len(features),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
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
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    method_feature_lists: dict[str, list[str]],
    evaluators: list[tuple[str, type, dict]],
    train_cv,
    scoring: str,
) -> dict[str, dict[str, dict]]:
    """Evaluate every method's selected top-k features under every evaluator
    in the panel -- fit on the train rows, scored on the held-out test rows no
    method saw while selecting -- so methods are compared on equal footing
    regardless of which model (if any) they used internally to select features,
    and regardless of any single evaluator's own inductive bias."""

    return {
        name: {
            evaluator_name: _evaluate_feature_set_holdout(
                X_train, y_train, X_test, y_test, features, model_cls, model_kwargs, train_cv, scoring
            )
            for evaluator_name, model_cls, model_kwargs in evaluators
        }
        for name, features in method_feature_lists.items()
    }


def _biomarker_recovery(task_config: dict, method: str, ranking: list[str], top_k: int, data_meta: dict) -> list[dict]:
    """Where a method ranks each of the task's literature biomarkers.

    A biomarker is a group of interchangeable features (gdsc_tasks.py); the group's
    rank is its best-ranked member. Stepwise rankings stop at k, so a stepwise
    biomarker outside the top k has no rank; RID rankings cover every feature.
    A member dropped by the correlation filter can't be ranked, so the
    *_with_proxies columns also credit the kept feature it was dropped in favour of
    (|r| above the threshold, so it carries the same signal). The strict columns
    count group members only.
    """

    position = {feature: rank for rank, feature in enumerate(ranking, start=1)}
    proxies = data_meta["corr_proxies"]
    rows = []
    for label, members in task_config["biomarkers"].items():
        ranked = sorted((position[m], m) for m in members if m in position)
        best_rank, best_member = ranked[0] if ranked else (None, None)
        proxy_of = {proxies[m]: m for m in members if m in proxies}
        with_proxies = sorted(ranked + [(position[p], p) for p in proxy_of if p in position])
        proxy_rank, proxy_feature = with_proxies[0] if with_proxies else (None, None)
        rows.append({
            "method": method,
            "biomarker": label,
            "best_member": best_member,
            "best_rank": best_rank,
            "in_top_k": best_rank is not None and best_rank <= top_k,
            "best_feature_with_proxies": proxy_feature,
            "best_rank_with_proxies": proxy_rank,
            "in_top_k_with_proxies": proxy_rank is not None and proxy_rank <= top_k,
            "members": ";".join(members),
            "members_removed_by_corr_filter": ";".join(f"{m}->{proxies[m]}" for m in members if m in proxies),
        })
    return rows


def _data_fingerprint(data_dir: Path) -> str:
    """SHA-256 over the processed panel and labels, so a checkpoint built from a
    differently prepared panel (new --z-threshold, gene count, ...) is refused."""

    digest = hashlib.sha256()
    for name in (FEATURES_FILE, RESPONSES_FILE):
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run experiments/gdsc/prepare_gdsc_data.py first")
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _write_json_atomic(payload: dict, path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _method_settings(method: str, args: argparse.Namespace, holdout_meta: dict) -> dict:
    settings = {key: getattr(args, key) for key in METHOD_SETTING_KEYS[method]}
    if method.startswith(("rid_", "cross_family_")):
        settings["rashomon_evaluation"] = "out_of_bag"
    # Added 2026-11-11: checkpoints from the old protocol (selection and evaluation on
    # the same rows/folds) lack these and are refused, as are ones from another split.
    settings["evaluation_protocol"] = holdout_meta["protocol"]
    settings["holdout_test_rows_sha256"] = holdout_meta["test_rows_sha256"]
    return settings


def _run_key(args: argparse.Namespace) -> dict:
    """Every result-changing argument across methods. A task_summary_row.json written
    under a different key (or none, i.e. before this was recorded) is left out of
    overall_task_summary.csv, so it never mixes rows from different settings or splits
    (the test-row hash is determined by data_fingerprint, holdout_* and the task)."""

    keys = sorted(set().union(*METHOD_SETTING_KEYS.values()))
    return json.loads(json.dumps({key: getattr(args, key) for key in keys}))


def _method_result_path(task_output_dir: Path, method: str) -> Path:
    return task_output_dir / f"{method}_result.json"


def _load_method_checkpoint(
    task_output_dir: Path, method: str, args: argparse.Namespace, holdout_meta: dict
) -> dict | None:
    """Return a method's checkpoint, or None if absent. Refuse one written under different settings."""

    path = _method_result_path(task_output_dir, method)
    if not path.exists():
        return None
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(_method_settings(method, args, holdout_meta)))
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
    """Run one method on the train rows only (X, y are the holdout split's train
    part); return (top_features, per-feature table, method-specific info)."""

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
            "search_model_train_cv_evaluation": search_eval,
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

    data_dir = Path(args.data_dir)
    task = task_config["task"]

    logger.info("-" * 70)
    logger.info(f"[task] {task_name}: {task_config['drug_name']} (GDSC2 drug {task_config['drug_id']}, "
                f"target {task_config['target']})")

    X_train, X_test, y_train, y_test, data_meta = _prepare_task_dataset(
        data_dir=data_dir,
        task_name=task_name,
        correlation_threshold=args.correlation_threshold,
        holdout_fraction=args.holdout_fraction,
        holdout_seed=args.holdout_seed,
    )
    holdout_meta = data_meta["holdout"]
    if not X_train.index.intersection(X_test.index).empty:
        raise RuntimeError("Train and test rows overlap")
    logger.info(
        f"[data] rows={data_meta['n_rows']} "
        f"features={data_meta['n_features_after_corr']} "
        f"(removed_corr={data_meta['n_removed_corr']}) classes={data_meta['class_counts']}"
    )
    logger.info(
        f"[holdout] train={holdout_meta['n_train']} {holdout_meta['train_class_counts']} "
        f"test={holdout_meta['n_test']} {holdout_meta['test_class_counts']} "
        f"(test rows sha256 {holdout_meta['test_rows_sha256']})"
    )

    # Refuse mismatched checkpoints before spending hours on anything else (data prep
    # comes first only because the settings include the holdout split's hash).
    checkpoints = {
        method: _load_method_checkpoint(task_output_dir, method, args, holdout_meta) for method in METHODS
    }

    # cv only ever splits the train rows: stepwise selection's inner CV and the
    # in-train CV diagnostic. The test rows are touched only by the evaluator panel.
    cv, stepwise_scoring = _make_stepwise_cv_and_scoring(args, y_train)
    evaluator_panel = build_common_evaluator_panel(args.random_state)

    for method in args.methods:
        if checkpoints[method] is not None:
            logger.info(f"[{method}] checkpoint found in {task_output_dir}; skipping")
            continue

        started = time.time()
        features, table, info = _run_method(method, X_train, y_train, args, cv, stepwise_scoring)
        elapsed = time.time() - started

        # Common holdout evaluation: every method's features scored the same way (fit on
        # train, scored on the untouched test rows), under a panel of evaluators
        # spanning different inductive biases.
        common_eval = run_common_evaluation_panel(
            X_train, y_train, X_test, y_test, {method: features}, evaluator_panel, cv, stepwise_scoring
        )[method]

        # Stepwise: the k selection steps. RID: every feature, ranked.
        _write_csv_atomic(table, task_output_dir / f"{method}_ranking.csv")
        checkpoint = {
            "settings": json.loads(json.dumps(_method_settings(method, args, holdout_meta))),
            "top_features": features,
            "ranking": table["feature"].tolist(),
            "elapsed_seconds": elapsed,
            "common_eval": common_eval,
            **info,
        }
        _write_json_atomic(checkpoint, _method_result_path(task_output_dir, method))
        checkpoints[method] = checkpoint

        logger.info(f"[{method}] finished in {elapsed / 60.0:.1f} min; top-10: {features[:10]}")
        for row in _biomarker_recovery(task_config, method, checkpoint["ranking"], args.top_k, data_meta):
            logger.info(f"       biomarker {row['biomarker']!r}: rank {row['best_rank']} ({row['best_member']})")
        for evaluator_name, _, _ in evaluator_panel:
            evaluator_result = common_eval[evaluator_name]
            logger.info(
                f"       holdout [{evaluator_name}] ({stepwise_scoring}): {evaluator_result['holdout_score']:.4f}"
                f"  (in-train CV {evaluator_result['train_cv_score_mean']:.4f} "
                f"+- {evaluator_result['train_cv_score_std']:.4f})"
            )

    # Re-read: array tasks for this task's other methods run concurrently and may
    # have checkpointed since this invocation started.
    checkpoints = {
        method: _load_method_checkpoint(task_output_dir, method, args, holdout_meta) for method in METHODS
    }
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
            "protocol": holdout_meta["protocol"],
            "note": (
                "Every method's selected top-k features scored under every evaluator in "
                "this panel (spanning linear, bagged-tree, and kernel-margin inductive "
                "biases), so no method's score is an artifact of one evaluator's blind spot. "
                "holdout_score: fit on the train rows, scored on the held-out test rows no "
                "method selected on (primary). train_cv_score_*: CV within the train rows on "
                "the folds stepwise selection maximized (optimistic for stepwise; diagnostic only)."
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

    recovery = pd.DataFrame([
        row
        for method in METHODS
        for row in _biomarker_recovery(task_config, method, completed[method]["ranking"], args.top_k, data_meta)
    ])
    _write_csv_atomic(recovery, task_output_dir / "biomarker_recovery.csv")

    row = {
        "task_name": task_name,
        "task_type": task,
        "drug_id": task_config["drug_id"],
        "drug_name": task_config["drug_name"],
        "n_rows": data_meta["n_rows"],
        "n_features": data_meta["n_features_after_corr"],
        "n_train": holdout_meta["n_train"],
        "n_test": holdout_meta["n_test"],
        "holdout_test_rows_sha256": holdout_meta["test_rows_sha256"],
        "common_eval_scoring": stepwise_scoring,
    }
    for method in METHODS:
        holdout_scores, train_cv_means = [], []
        for evaluator_name, _, _ in evaluator_panel:
            evaluator_result = completed[method]["common_eval"][evaluator_name]
            row[f"{method}_holdout_score__{evaluator_name}"] = evaluator_result["holdout_score"]
            row[f"{method}_train_cv_score_mean__{evaluator_name}"] = evaluator_result["train_cv_score_mean"]
            row[f"{method}_train_cv_score_std__{evaluator_name}"] = evaluator_result["train_cv_score_std"]
            holdout_scores.append(evaluator_result["holdout_score"])
            train_cv_means.append(evaluator_result["train_cv_score_mean"])
        row[f"{method}_holdout_score_avg"] = float(np.mean(holdout_scores))
        row[f"{method}_train_cv_score_mean_avg"] = float(np.mean(train_cv_means))
        row[f"{method}_elapsed_seconds"] = completed[method]["elapsed_seconds"]

    rid_tree_perf = completed["rid_tree"]["rashomon_oob_perf_stats"]
    row["rid_tree_rashomon_oob_accuracy_mean"] = rid_tree_perf.get("accuracy_mean")
    row["rid_tree_rashomon_oob_auprc_mean"] = rid_tree_perf.get("auprc_mean")
    for method in ("cross_family_unweighted", "cross_family_unweighted_truvarimp"):
        cross_family_rashomon = _summarize_cross_family_perf(completed[method]["rashomon_oob_perf_stats"])
        row[f"{method}_rashomon_oob_accuracy_mean"] = cross_family_rashomon["accuracy_mean"]
        row[f"{method}_rashomon_oob_auprc_mean"] = cross_family_rashomon["auprc_mean"]
    for method in METHODS:
        method_recovery = recovery[recovery["method"] == method]
        row[f"{method}_biomarkers_in_top_k"] = int(method_recovery["in_top_k"].sum())
        row[f"{method}_biomarkers_in_top_k_with_proxies"] = int(method_recovery["in_top_k_with_proxies"].sum())
    row["n_biomarkers"] = len(task_config["biomarkers"])
    row.update(overlap_counts)

    # Saved so whichever invocation finishes last can assemble every task's row.
    _write_json_atomic({**row, "run_key": _run_key(args)}, task_output_dir / "task_summary_row.json")
    logger.info(f"[task] combined outputs saved in {task_output_dir}")
    return row


def main() -> None:
    args = _parse_args()
    args.data_fingerprint = _data_fingerprint(Path(args.data_dir))
    silence_warnings()
    configure_logging(args.log_level, args.log_file)
    task_names = _parse_task_names(args.tasks)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("GDSC Drug-Sensitivity Top-k Feature Comparison")
    logger.info(f"  data_dir:                 {args.data_dir}")
    logger.info(f"  tasks:                    {task_names}")
    logger.info(f"  methods:                  {args.methods}")
    logger.info(f"  top_k:                    {args.top_k}")
    logger.info(f"  correlation_threshold:    {args.correlation_threshold}")
    logger.info(f"  holdout:                  fraction={args.holdout_fraction} seed={args.holdout_seed}")
    logger.info(f"  truvarimp_total_budget:   {args.truvarimp_total_budget}")
    logger.info(f"  output_dir:               {output_dir}")
    logger.info("=" * 70)

    for task_name in task_names:
        _run_single_task(
            task_name=task_name,
            task_config=TASK_REGISTRY[task_name],
            args=args,
            output_dir=output_dir,
        )

    # An array task runs one (task, method) pair, so collect every task that any
    # invocation has finished, not only the ones this invocation ran.
    run_key = _run_key(args)
    finished, overall_rows = [], []
    for name in TASK_REGISTRY:
        row_path = output_dir / name / "task_summary_row.json"
        if not row_path.exists():
            continue
        row = json.loads(row_path.read_text(encoding="utf-8"))
        if row.pop("run_key", None) != run_key:
            logger.warning(f"[task] {name}: {row_path} was written under other settings; leaving it out")
            continue
        finished.append(name)
        overall_rows.append(row)
    if not finished:
        logger.info("=" * 70)
        logger.info("Checkpoints saved; no task has every method checkpointed yet")
        logger.info("=" * 70)
        return

    overall_summary_path = output_dir / "overall_task_summary.csv"
    _write_csv_atomic(pd.DataFrame(overall_rows), overall_summary_path)

    run_manifest = {
        "args": vars(args),
        "tasks_finished": finished,
        "tasks_pending": [name for name in TASK_REGISTRY if name not in finished],
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
