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

Evaluation protocol (outer stratified holdout): the rows are split once into
train and test (--holdout-fraction, default 0.2; --holdout-seed). Feature
selection never sees the test rows: the correlation filter (including
--correlation-threshold auto), stepwise selection (whose inner CV runs on the
train rows only), every RID fit, --top-k auto curves, mRMR reordering and the
redundant-duplicate's noise scale all use the train rows alone. Every method's
selected top-k feature set is then scored with the SAME panel of evaluators
(logreg, random forest, RBF SVM), each fit on the train rows restricted to those
features and scored once on the held-out test rows (holdout_score) -- the
primary, comparable score for every method. The panel's in-train CV score
(train_cv_score_*, the same folds stepwise selection maximized) is kept
alongside only to show how much reusing selection folds for evaluation inflates
the stepwise methods. RID's own internal Rashomon-set performance stats
(accuracy/AUPRC on each bootstrap's out-of-bag train rows) are recorded
separately in run_settings.json for diagnostic purposes only and are not used
to rank methods.

Optional diagnostic (--inject-redundant-duplicate FEATURE): adds a
correlated engineered duplicate of an existing real feature and reports each
method's credit-split ratio for that pair (0 = winner-take-all, 1 = shared
evenly). Needs no ground truth, since the duplicate is deliberately
constructed to be substitutable -- unlike ground-truth precision/recall,
this credit-sharing test works on real data. Off by default.

--rid-selection mrmr (default: marginal) greedily reorders RID's own
importance ranking to trade off importance against redundancy with
already-picked features (mrmr_selection.py), instead of RID's plain
importance sort. RID's raw ranking deliberately doesn't penalize redundancy
(two substitutable features can both look important across the Rashomon
set -- see --inject-redundant-duplicate above), which is correct for
characterizing importance but is a real handicap in a matched-k predictive
comparison against stepwise selection, which gets redundancy-avoidance for
free from its own greedy CV-scored search. Off (marginal) by default.

Usage:
    python run_falcon_cano_top40_comparison.py
    sbatch run_falcon_cano_top40_comparison.sl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
import logging
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from correlation_threshold_sweep import find_elbow, sweep_thresholds  # noqa: E402
from feature_count_selection import select_k_by_cv_gain, select_k_by_prob_positive  # noqa: E402
from mrmr_selection import reorder_by_mrmr  # noqa: E402
from rid.candidate_pool import DEFAULT_N_CANDIDATES_PER_FAMILY  # noqa: E402
from rid import (  # noqa: E402
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
from run_rashomon_falcon_cano import (  # noqa: E402
    remove_highly_correlated_features,
    resolve_num_workers,
    select_curated_descriptors,
)

logger = logging.getLogger(__name__)

_AUTO_CORRELATION_THRESHOLD_SWEEP = np.round(np.arange(0.50, 0.991, 0.01), 3)


def _correlation_threshold_arg(value: str):
    """argparse type: accepts a float, or the literal 'auto' for elbow detection."""

    if value == "auto":
        return "auto"
    return float(value)


def _top_k_arg(value: str):
    """argparse type: accepts an int, or the literal 'auto' for elbow-based selection."""

    if value == "auto":
        return "auto"
    return int(value)


def _resolve_correlation_threshold(X: pd.DataFrame, correlation_threshold) -> tuple[float, dict]:
    """Resolve a fixed threshold as-is, or pick the Kneedle elbow point from this
    run's own data when correlation_threshold == 'auto'."""

    if correlation_threshold != "auto":
        threshold = float(correlation_threshold)
        return threshold, {"mode": "fixed", "value": threshold}

    sweep_df = sweep_thresholds(X, _AUTO_CORRELATION_THRESHOLD_SWEEP)
    elbow = find_elbow(sweep_df)
    n_at_elbow = int(sweep_df.loc[sweep_df["threshold"].sub(elbow).abs().idxmin(), "n_features_remaining"])
    logger.info(f"[correlation-threshold=auto] elbow={elbow:.3f} -> {n_at_elbow} features (from {X.shape[1]} raw)")
    return elbow, {"mode": "auto", "value": elbow, "n_features_at_elbow": n_at_elbow, "n_features_raw": int(X.shape[1])}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare forward-stepwise feature selection (logistic regression and "
            "random forest) against single-family RID on a fully enumerated "
            "decision-tree Rashomon set and cross-family RID (both unweighted and "
            "weighted balance modes), for the Falcon-Cano bioavailability task. "
            "Every method selects on the train rows of an outer stratified holdout; its "
            "selected features are scored with a common evaluator panel fit on those "
            "train rows and scored on the held-out test rows, for a fair comparison."
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
        default=str(Path(__file__).resolve().parent / "results" / "top40_feature_comparison_holdout"),
        help="Directory for outputs (default: <script dir>/results/top40_feature_comparison_holdout)",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of rows held out (stratified) as the test set the evaluator panel "
            "scores on; feature selection only sees the rest (default: 0.2)"
        ),
    )
    parser.add_argument(
        "--holdout-seed",
        type=int,
        default=42,
        help="Random seed for the outer stratified train/test split (default: 42)",
    )
    parser.add_argument(
        "--top-k",
        type=_top_k_arg,
        default=40,
        help=(
            "Number of top features to select from each method, or 'auto' to pick a "
            "per-method count via elbow detection on that method's own native signal "
            "curve (see feature_count_selection.py) instead of a fixed number "
            "(default: 40)"
        ),
    )
    parser.add_argument(
        "--top-k-max-sweep",
        type=int,
        default=100,
        help=(
            "Ceiling on how many features to rank when --top-k=auto -- each method's "
            "signal curve (prob_positive for RID methods, in-train CV gain for stepwise "
            "methods) is computed over rank=1..this value, and the chosen k can't exceed "
            "it. Ignored when --top-k is a fixed integer (default: 100)"
        ),
    )
    parser.add_argument(
        "--correlation-threshold",
        type=_correlation_threshold_arg,
        default=0.8,
        help=(
            "Remove highly correlated descriptors above this absolute threshold, or "
            "'auto' to pick the Kneedle elbow point dynamically from this run's data "
            "(default: 0.8)"
        ),
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
        help=(
            "CV folds for stepwise selection, --top-k auto CV-gain curves and the in-train "
            "CV diagnostic, all on the train rows only (default: 5)"
        ),
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
        help=(
            "RID variable-importance metric (default: sub_mr -- permutation-based model "
            "reliance). iloco (ablation-based pairwise interaction importance) is available "
            "as an explicit opt-in, but is not a general-purpose replacement for sub_mr: its "
            "sum(|iLOCO|) reduction is structurally blind to additive main effects, which "
            "measurably hurt ground-truth recovery on 9/10 synthetic DGPs in an earlier "
            "internal analysis, and it timed out at 48h on the full-scale real dataset. Cost scales ~O(features^2) per "
            "model if you do opt in: ~6s/model at this pipeline's current "
            "0.8-correlation-threshold feature count (~138 features); do not pair with "
            "--correlation-threshold auto without re-checking cost, since auto keeps more "
            "features (~177) and roughly doubles this per-model cost."
        ),
    )
    parser.add_argument(
        "--rid-selection",
        type=str,
        choices=["marginal", "mrmr"],
        default="marginal",
        help=(
            "How RID-based methods (rid_tree, cross_family_unweighted, "
            "cross_family_weighted) order their ranked features. 'marginal' (default) "
            "uses RID's own expected_importance ranking as-is -- two substitutable "
            "features can both rank highly, since RID deliberately characterizes "
            "importance across the whole Rashomon set rather than penalizing "
            "redundancy. 'mrmr' greedily reorders that same ranking (mrmr_selection.py) "
            "to trade off importance against redundancy with already-picked features, "
            "for a fairer matched-k predictive comparison against stepwise selection, "
            "which gets this for free from its own greedy CV-scored search. Under "
            "--top-k=auto, 'mrmr' elbow-detects a held-out CV-gain curve instead of "
            "prob_positive, since mrmr's reordering breaks the plain "
            "sorted-by-importance assumption prob_positive-elbow relies on."
        ),
    )
    parser.add_argument(
        "--mrmr-lambda",
        type=float,
        default=1.0,
        help="Redundancy penalty weight for --rid-selection=mrmr (default: 1.0)",
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
        help=(
            "RID n_models_pool / n_models_per_class (default: 50). Only sets seed refits for "
            "seed-sensitive grids and custom trainers: the tree grid fits each config once, and "
            "cross-family RID draws a shared candidate pool of "
            "rid.candidate_pool.DEFAULT_N_CANDIDATES_PER_FAMILY per family instead"
        ),
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
        help="Random state for the inner CV folds, the models and the duplicate's noise (default: 42)",
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
    add_logging_arguments(parser)
    return parser.parse_args()


def _make_holdout_split(y: pd.Series, holdout_fraction: float, holdout_seed: int) -> tuple[pd.Index, pd.Index, dict]:
    """One outer stratified train/test split of the rows (original row order kept
    within each part). Feature selection sees only the train rows; the evaluator
    panel scores on the test rows. Also returns the split's metadata, including a
    hash of the test rows."""

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


def _load_filtered_dataset(
    data_path: Path,
    use_curated_descriptors: bool,
    correlation_threshold: float,
    holdout_fraction: float | None = None,
    holdout_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Index | None, pd.Index | None, dict]:
    """Shared loader: read the CSV and curate descriptors, then fit the
    correlation filter on the train rows of an outer holdout split, or on all
    rows when holdout_fraction is None. Returns (X, y, train_index, test_index,
    metadata) over all rows (indices None when unsplit); row ids are the CSV's
    row numbers."""

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    df = pd.read_csv(data_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    train_index = test_index = holdout_meta = None
    if holdout_fraction is not None:
        train_index, test_index, holdout_meta = _make_holdout_split(y, holdout_fraction, holdout_seed)

    n_features_raw = X.shape[1]
    if use_curated_descriptors:
        X = select_curated_descriptors(X)  # a fixed descriptor list; looks at no rows
    n_features_curated = X.shape[1]

    X_fit = X if train_index is None else X.loc[train_index]
    resolved_threshold, threshold_info = _resolve_correlation_threshold(X_fit, correlation_threshold)
    X_fit = remove_highly_correlated_features(X_fit, correlation_threshold=resolved_threshold)
    X = X_fit if train_index is None else X[X_fit.columns]

    metadata = {
        "dataset_path": str(data_path),
        "target_column": "target",
        "n_rows": int(X.shape[0]),
        "n_features_raw": int(n_features_raw),
        "n_features_after_curation": int(n_features_curated),
        "n_features_after_corr": int(X.shape[1]),
        "class_counts": y.value_counts().to_dict(),
        "correlation_threshold_info": threshold_info,
    }
    if holdout_meta is not None:
        metadata["holdout"] = holdout_meta
    return X, y, train_index, test_index, metadata


def _prepare_dataset_unsplit(
    data_path: Path,
    use_curated_descriptors: bool,
    correlation_threshold: float,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """All rows, correlation filter fit on all rows (the pre-holdout loader). For
    callers with no evaluation step, e.g. run_stability_analysis.py."""

    X, y, _, _, metadata = _load_filtered_dataset(data_path, use_curated_descriptors, correlation_threshold)
    return X, y, metadata


def _prepare_dataset(
    data_path: Path,
    use_curated_descriptors: bool,
    correlation_threshold: float,
    holdout_fraction: float,
    holdout_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """Load the data, split it into train/test rows, and fit the correlation
    filter on the train rows only. Returns (X_train, X_test, y_train, y_test,
    metadata); row ids are the CSV's row numbers."""

    X, y, train_index, test_index, metadata = _load_filtered_dataset(
        data_path, use_curated_descriptors, correlation_threshold, holdout_fraction, holdout_seed
    )
    return X.loc[train_index], X.loc[test_index], y.loc[train_index], y.loc[test_index], metadata


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


def _compute_incremental_cv_curve(
    X: pd.DataFrame,
    y: pd.Series,
    ranked_features: list[str],
    model_cls,
    model_kwargs: dict,
    cv,
    scoring: str,
) -> list[float]:
    """In-train CV score after adding each successive feature along an
    already-fixed ranking. Unlike run_forward_stepwise_selection's own
    history, this doesn't search over candidate features at each step -- the
    order was already decided elsewhere (e.g. by reorder_by_mrmr) -- so it's
    one CV fit per rank instead of one per (rank x remaining candidate).
    Used only to elbow-detect how many ranks are worth keeping when a
    method's ranking isn't a plain importance sort anymore."""

    scores = []
    for k in range(1, len(ranked_features) + 1):
        estimator = model_cls(**model_kwargs)
        cv_scores = cross_val_score(
            estimator,
            X[ranked_features[:k]],
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        scores.append(float(np.mean(cv_scores)))
    return scores


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
    # Families with no Rashomon members report NaN stats; average only the rest.
    family_perf_stats = perf_summary.get("family_perf_stats") or {}

    def _mean_over_families(key: str) -> float | None:
        values = [stats[key] for stats in family_perf_stats.values() if not np.isnan(stats[key])]
        return float(np.mean(values)) if values else None

    return {
        "accuracy_mean": _mean_over_families("accuracy_mean"),
        "auprc_mean": _mean_over_families("auprc_mean"),
    }


def scaled_logistic_regression(**kwargs):
    """LogisticRegression behind a StandardScaler, fit inside each CV fold.

    Falcon-Cano descriptor SDs span ~4 orders of magnitude (fr_azide ~0.03,
    BertzCT ~380) and Ipc reaches ~1e16. Unscaled, the L2 penalty makes
    small-scale descriptors expensive to use, so selection and scoring favor
    large-scale ones regardless of signal.
    """

    return make_pipeline(StandardScaler(), LogisticRegression(**kwargs))


def scaled_rbf_svc(**kwargs):
    """RBF SVC behind a StandardScaler: unscaled, kernel distances are dominated by
    the largest-scale descriptor (Ipc), so the SVM sees little else."""

    return make_pipeline(StandardScaler(), SVC(**kwargs))


def build_common_evaluator_panel(random_state: int) -> list[tuple[str, Callable, dict]]:
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
            scaled_logistic_regression,
            {"penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 100000, "random_state": random_state},
        ),
        (
            "random_forest",
            RandomForestClassifier,
            {"n_estimators": 300, "min_samples_leaf": 1, "random_state": random_state, "n_jobs": 1},
        ),
        (
            "svm_rbf",
            scaled_rbf_svc,
            {"kernel": "rbf", "C": 1.0, "gamma": "scale", "random_state": random_state},
        ),
    ]


def run_common_evaluation_panel(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    method_feature_lists: dict[str, list[str]],
    evaluators: list[tuple[str, Callable, dict]],
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
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    source_feature: str,
    redundancy_corr: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Add a correlated engineered duplicate of an existing real feature (not
    a synthetic ground-truth driver -- source_feature is whatever real column
    the caller chose). Requires no ground truth: the duplicate is
    substitutable for source_feature by construction, so any credit-sharing
    (or lack of it) a method exhibits between the two is directly
    attributable to how it handles redundancy, not to unknown confounders.
    The duplicate is added to both the train and test rows (the evaluator panel
    needs the column on both); its noise scale comes from the train rows only."""

    if source_feature not in X_train.columns:
        raise ValueError(f"--inject-redundant-duplicate feature '{source_feature}' not found in X")

    dup_name = f"{source_feature}_dup"
    rng = np.random.default_rng(random_state)
    noise_sd = X_train[source_feature].std()

    def _with_duplicate(X: pd.DataFrame) -> pd.DataFrame:
        dup_noise = rng.normal(0.0, noise_sd, size=len(X))
        X = X.copy()
        X[dup_name] = redundancy_corr * X[source_feature] + np.sqrt(1.0 - redundancy_corr**2) * dup_noise
        return X

    return _with_duplicate(X_train), _with_duplicate(X_test), dup_name


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
    silence_warnings()
    configure_logging(args.log_level, args.log_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rid_n_jobs = resolve_num_workers(args.rid_n_jobs)

    logger.info("=" * 70)
    logger.info("Falcon-Cano Top-K Feature Comparison")
    logger.info(f"  data:                  {args.data}")
    logger.info(f"  output_dir:            {output_dir}")
    logger.info(f"  top_k:                 {args.top_k}")
    if args.top_k == "auto":
        logger.info(f"  top_k_max_sweep:       {args.top_k_max_sweep}")
    logger.info(f"  rid_selection:         {args.rid_selection}")
    if args.rid_selection == "mrmr":
        logger.info(f"  mrmr_lambda:           {args.mrmr_lambda}")
    logger.info(f"  correlation_threshold: {args.correlation_threshold}")
    logger.info(f"  holdout:               fraction={args.holdout_fraction} seed={args.holdout_seed}")
    logger.info(f"  use_curated_descriptors: {args.use_curated_descriptors}")
    logger.info(f"  rid_n_jobs:            {rid_n_jobs}")
    logger.info("=" * 70)

    start_time = time.time()

    X_train, X_test, y_train, y_test, data_meta = _prepare_dataset(
        data_path=Path(args.data),
        use_curated_descriptors=args.use_curated_descriptors,
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
        f"(raw={data_meta['n_features_raw']})"
    )
    logger.info(
        f"[holdout] train={holdout_meta['n_train']} {holdout_meta['train_class_counts']} "
        f"test={holdout_meta['n_test']} {holdout_meta['test_class_counts']} "
        f"(test rows sha256 {holdout_meta['test_rows_sha256']})"
    )

    redundant_pair = None
    if args.inject_redundant_duplicate:
        X_train, X_test, dup_name = inject_redundant_duplicate(
            X_train, X_test, args.inject_redundant_duplicate, args.redundancy_corr, args.random_state
        )
        redundant_pair = (args.inject_redundant_duplicate, dup_name)
        logger.info(
            f"[redundancy] injected '{dup_name}' as a correlated duplicate of "
            f"'{args.inject_redundant_duplicate}' (corr={args.redundancy_corr})"
        )

    # Everything from here until the evaluator panel sees only the train rows; cv
    # only ever splits them (stepwise selection, CV-gain curves, in-train CV diagnostic).
    cv = StratifiedKFold(
        n_splits=args.stepwise_cv_splits,
        shuffle=True,
        random_state=args.random_state,
    )

    auto_k_selection = args.top_k == "auto"
    selection_top_k = args.top_k_max_sweep if auto_k_selection else args.top_k
    k_selection_results: dict[str, dict] = {}

    use_mrmr = args.rid_selection == "mrmr"
    corr_matrix = X_train.corr().abs() if use_mrmr else None

    def resolve_feature_count(
        method_name: str,
        ranked_features: list[str],
        signal_values: list[float],
        signal_kind: str,
    ) -> list[str]:
        """When --top-k=auto, elbow-detect this method's own native signal curve
        (prob_positive for RID methods, in-train CV gain for stepwise methods)
        over rank=1..top_k_max_sweep and truncate to the elbow rank. Otherwise a
        no-op (ranked_features is already exactly args.top_k long)."""

        if not auto_k_selection:
            return ranked_features
        if signal_kind == "prob_positive":
            selection = select_k_by_prob_positive(signal_values)
        elif signal_kind == "cv_gain":
            selection = select_k_by_cv_gain(signal_values)
        else:
            raise ValueError(f"Unknown signal_kind '{signal_kind}'")
        selection["signal_kind"] = signal_kind
        selection["curve"] = {
            "rank": list(range(1, len(signal_values) + 1)),
            "value": [float(v) for v in signal_values],
        }
        k_selection_results[method_name] = selection
        chosen_k = selection["k"]
        logger.info(
            f"[top-k=auto] {method_name}: chosen k={chosen_k} via {selection['criterion']}"
            f"{' (at ceiling)' if selection['at_ceiling'] else ''}"
        )
        return ranked_features[:chosen_k]

    def resolve_rid_feature_count(
        method_name: str, features: list[str], table: pd.DataFrame
    ) -> tuple[list[str], pd.DataFrame]:
        """Applies --rid-selection=mrmr reordering (if enabled) to a RID
        method's ranking, then resolves k the same way resolve_feature_count
        does, but using whichever native signal curve matches the CURRENT
        ordering: prob_positive for the untouched marginal ranking, in-train
        CV gain for the mrmr reordering (mrmr breaks the sorted-by-importance
        assumption prob_positive-elbow relies on)."""

        if use_mrmr:
            importance_map = dict(zip(table["feature"], table["expected_importance"]))
            prob_positive_map = dict(zip(table["feature"], table["prob_positive"]))
            reordered = reorder_by_mrmr(importance_map, corr_matrix, args.mrmr_lambda)
            table = pd.DataFrame(
                {
                    "rank": range(1, len(reordered) + 1),
                    "feature": reordered,
                    "expected_importance": [importance_map[f] for f in reordered],
                    "prob_positive": [prob_positive_map[f] for f in reordered],
                }
            )
            features = reordered
            signal_values = (
                _compute_incremental_cv_curve(
                    X_train, y_train, features, scaled_logistic_regression, logreg_kwargs, cv, args.stepwise_scoring
                )
                if auto_k_selection
                else None
            )
            signal_kind = "cv_gain"
        else:
            signal_values = table["prob_positive"].tolist()
            signal_kind = "prob_positive"

        features = resolve_feature_count(method_name, features, signal_values, signal_kind)
        return features, table.head(len(features))

    # --- Method 1: forward stepwise selection via logistic regression ---
    logreg_kwargs = {
        "penalty": "l2",
        "C": args.stepwise_logreg_C,
        "solver": "lbfgs",
        "max_iter": 100000,
        "random_state": args.random_state,
    }
    stepwise_logreg_features, stepwise_logreg_history = run_forward_stepwise_selection(
        X=X_train,
        y=y_train,
        top_k=selection_top_k,
        model_cls=scaled_logistic_regression,
        model_kwargs=logreg_kwargs,
        cv=cv,
        scoring=args.stepwise_scoring,
        n_jobs=args.stepwise_n_jobs,
    )
    stepwise_logreg_features = resolve_feature_count(
        "stepwise_logreg",
        stepwise_logreg_features,
        stepwise_logreg_history["train_cv_score_mean"].tolist(),
        "cv_gain",
    )
    stepwise_logreg_eval = _evaluate_feature_set(
        X_train, y_train, stepwise_logreg_features, scaled_logistic_regression, logreg_kwargs, cv, args.stepwise_scoring
    )
    stepwise_logreg_history.head(len(stepwise_logreg_features)).to_csv(
        output_dir / "stepwise_logreg_top_features.csv", index=False
    )

    # --- Method 2: forward stepwise selection via random forest ---
    rf_kwargs = {
        "n_estimators": args.stepwise_rf_n_estimators,
        "max_depth": args.stepwise_rf_max_depth,
        "min_samples_leaf": args.stepwise_rf_min_samples_leaf,
        "random_state": args.random_state,
        "n_jobs": 1,
    }
    stepwise_rf_features, stepwise_rf_history = run_forward_stepwise_selection(
        X=X_train,
        y=y_train,
        top_k=selection_top_k,
        model_cls=RandomForestClassifier,
        model_kwargs=rf_kwargs,
        cv=cv,
        scoring=args.stepwise_scoring,
        n_jobs=args.stepwise_n_jobs,
    )
    stepwise_rf_features = resolve_feature_count(
        "stepwise_rf",
        stepwise_rf_features,
        stepwise_rf_history["train_cv_score_mean"].tolist(),
        "cv_gain",
    )
    stepwise_rf_eval = _evaluate_feature_set(
        X_train, y_train, stepwise_rf_features, RandomForestClassifier, rf_kwargs, cv, args.stepwise_scoring
    )
    stepwise_rf_history.head(len(stepwise_rf_features)).to_csv(
        output_dir / "stepwise_rf_top_features.csv", index=False
    )

    # --- Method 3: single-family RID on a fully enumerated tree Rashomon set ---
    logger.info(
        f"[rid] metric={args.rid_metric} epsilon={args.rid_epsilon} "
        f"n_bootstraps={args.rid_n_bootstraps} n_models_pool={args.rid_n_models_pool}"
    )
    rid_tree_features, rid_tree_table, rid_tree_perf = run_single_family_tree_rid(
        X=X_train,
        y=y_train,
        top_k=selection_top_k,
        rid_metric=args.rid_metric,
        epsilon=args.rid_epsilon,
        n_bootstraps=args.rid_n_bootstraps,
        n_models_pool=args.rid_n_models_pool,
        n_jobs=rid_n_jobs,
    )
    rid_tree_features, rid_tree_table = resolve_rid_feature_count("rid_tree", rid_tree_features, rid_tree_table)
    rid_tree_table.to_csv(output_dir / "rid_tree_top_features.csv", index=False)

    # --- Methods 4-5: cross-family RID (unweighted, weighted) ---
    cross_family_features: dict[str, list[str]] = {}
    cross_family_perf: dict[str, dict] = {}
    for balance_mode in ("unweighted", "weighted"):
        logger.info(
            f"[rid] cross-family metric={args.rid_metric} epsilon={args.rid_epsilon} "
            f"n_bootstraps={args.rid_n_bootstraps} n_models_per_class={args.rid_n_models_pool} "
            f"family_balance_mode={balance_mode}"
        )
        features, table, perf = run_cross_family_rid(
            X=X_train,
            y=y_train,
            top_k=selection_top_k,
            rid_metric=args.rid_metric,
            epsilon=args.rid_epsilon,
            n_bootstraps=args.rid_n_bootstraps,
            n_models_per_class=args.rid_n_models_pool,
            family_balance_mode=balance_mode,
            n_jobs=rid_n_jobs,
        )
        features, table = resolve_rid_feature_count(f"cross_family_{balance_mode}", features, table)
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

    # --- Common holdout evaluation: every method's features scored the same way (fit on
    # train, scored on the untouched test rows), under a panel of evaluators spanning
    # different inductive biases ---
    evaluator_panel = build_common_evaluator_panel(args.random_state)
    common_eval = run_common_evaluation_panel(
        X_train, y_train, X_test, y_test, method_feature_lists, evaluator_panel, cv, args.stepwise_scoring
    )

    cross_family_unweighted_rashomon = _summarize_cross_family_perf(cross_family_perf["unweighted"])
    cross_family_weighted_rashomon = _summarize_cross_family_perf(cross_family_perf["weighted"])

    credit_split_ratios = None
    if redundant_pair is not None:
        credit_split_ratios = {
            name: compute_credit_split_ratio_top_k(features, redundant_pair)
            for name, features in method_feature_lists.items()
        }
        logger.info(f"[redundancy] credit-split ratios for {redundant_pair}: {credit_split_ratios}")

    settings = {
        "args": vars(args),
        "data_metadata": data_meta,
        "redundancy_diagnostic": (
            None
            if redundant_pair is None
            else {"pair": redundant_pair, "redundancy_corr": args.redundancy_corr, "credit_split_ratios": credit_split_ratios}
        ),
        "k_selection": (
            None
            if not auto_k_selection
            else {
                "top_k_max_sweep": args.top_k_max_sweep,
                "per_method": k_selection_results,
            }
        ),
        "common_evaluators": {
            "panel": {
                evaluator_name: {"model": model_cls.__name__, "model_kwargs": model_kwargs}
                for evaluator_name, model_cls, model_kwargs in evaluator_panel
            },
            "scoring": args.stepwise_scoring,
            "protocol": holdout_meta["protocol"],
            "note": (
                "Every method's selected top-k features scored under every evaluator in "
                "this panel (spanning linear, bagged-tree, and kernel-margin inductive "
                "biases), so no method's score is an artifact of one evaluator's blind spot. "
                "holdout_score: fit on the train rows, scored on the held-out test rows no "
                "method selected on (primary). train_cv_score_*: CV within the train rows on "
                "the folds stepwise selection maximized (optimistic for stepwise; diagnostic only)."
            ),
            "results": common_eval,
        },
        "stepwise_logreg": {
            "scoring": args.stepwise_scoring,
            "model_kwargs": logreg_kwargs,
            "top_features": stepwise_logreg_features,
            "search_model_train_cv_evaluation": stepwise_logreg_eval,
        },
        "stepwise_rf": {
            "scoring": args.stepwise_scoring,
            "model_kwargs": rf_kwargs,
            "top_features": stepwise_rf_features,
            "search_model_train_cv_evaluation": stepwise_rf_eval,
        },
        "rid_tree": {
            "metric": args.rid_metric,
            "epsilon": args.rid_epsilon,
            # Each grid config fit once per bootstrap; no criterion="log_loss" duplicates.
            "grid_refits": "seed_sensitive_only",
            "top_features": rid_tree_features,
            "rashomon_oob_perf_stats": rid_tree_perf,
        },
        "cross_family_unweighted": {
            "metric": args.rid_metric,
            "epsilon": args.rid_epsilon,
            # Grid search = every candidate of the shared pool fit once (rid.candidate_pool).
            "candidate_search": "grid",
            "n_candidates_per_family": DEFAULT_N_CANDIDATES_PER_FAMILY,
            "top_features": cross_family_features["unweighted"],
            "rashomon_oob_perf_stats": cross_family_perf["unweighted"],
        },
        "cross_family_weighted": {
            "metric": args.rid_metric,
            "epsilon": args.rid_epsilon,
            # Grid search = every candidate of the shared pool fit once (rid.candidate_pool).
            "candidate_search": "grid",
            "n_candidates_per_family": DEFAULT_N_CANDIDATES_PER_FAMILY,
            "top_features": cross_family_features["weighted"],
            "rashomon_oob_perf_stats": cross_family_perf["weighted"],
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
        "n_train": holdout_meta["n_train"],
        "n_test": holdout_meta["n_test"],
        "holdout_test_rows_sha256": holdout_meta["test_rows_sha256"],
        "common_eval_scoring": args.stepwise_scoring,
    }
    for name in method_feature_lists:
        holdout_scores, train_cv_means = [], []
        for evaluator_name, _, _ in evaluator_panel:
            evaluator_result = common_eval[name][evaluator_name]
            overall_row[f"{name}_holdout_score__{evaluator_name}"] = evaluator_result["holdout_score"]
            overall_row[f"{name}_train_cv_score_mean__{evaluator_name}"] = evaluator_result["train_cv_score_mean"]
            overall_row[f"{name}_train_cv_score_std__{evaluator_name}"] = evaluator_result["train_cv_score_std"]
            holdout_scores.append(evaluator_result["holdout_score"])
            train_cv_means.append(evaluator_result["train_cv_score_mean"])
        overall_row[f"{name}_holdout_score_avg"] = float(np.mean(holdout_scores))
        overall_row[f"{name}_train_cv_score_mean_avg"] = float(np.mean(train_cv_means))
        overall_row[f"{name}_n_features_selected"] = len(method_feature_lists[name])
    overall_row.update(
        {
            "rid_tree_rashomon_oob_accuracy_mean": rid_tree_perf.get("accuracy_mean"),
            "rid_tree_rashomon_oob_auprc_mean": rid_tree_perf.get("auprc_mean"),
            "cross_family_unweighted_rashomon_oob_accuracy_mean": cross_family_unweighted_rashomon["accuracy_mean"],
            "cross_family_unweighted_rashomon_oob_auprc_mean": cross_family_unweighted_rashomon["auprc_mean"],
            "cross_family_weighted_rashomon_oob_accuracy_mean": cross_family_weighted_rashomon["accuracy_mean"],
            "cross_family_weighted_rashomon_oob_auprc_mean": cross_family_weighted_rashomon["auprc_mean"],
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
    logger.info("=" * 70)
    for name, features in method_feature_lists.items():
        logger.info(f"{name} top-10: {features[:10]}")
        for evaluator_name, _, _ in evaluator_panel:
            evaluator_result = common_eval[name][evaluator_name]
            logger.info(
                f"  holdout [{evaluator_name}] ({args.stepwise_scoring}): {evaluator_result['holdout_score']:.4f}"
                f"  (in-train CV {evaluator_result['train_cv_score_mean']:.4f} "
                f"+- {evaluator_result['train_cv_score_std']:.4f})"
            )
    logger.info(f"overlap counts: {overlap_counts}")
    logger.info(f"Finished in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info(f"Results saved to: {output_dir}/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
