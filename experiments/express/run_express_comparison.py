#!/usr/bin/env python3
"""Compare top-k feature selection methods on the EXPRESS stand-in (VEN combinations in AML).

Predicts whether a venetoclax combination works better than expected on a
patient's AML cells, one row per (patient, combo), from the patient's expression
plus which partner drug the combo uses. Run prepare_express_data.py first.

Data (see prepare_express_data.py): a public stand-in for EXPRESS (Janizek et
al. 2023), NOT the original EXPRESS data -- combination-ratio outcomes from Eide
et al. 2023 Table S9 (manually downloaded; AACR terms) joined to BeatAML2
expression (CC-BY-4.0): 279 patients, one RNA sample each, 3,013 rows over 21
"VEN + X" combos. CR = AUC_combo / min(AUC_VEN, AUC_partner), AUCs floored at
1% of the maximum AUC; prepare keeps log CR unfitted.

Mirrors experiments/gdsc/run_gdsc_top40_comparison.py (same four methods,
checkpointing, settings-mismatch refusal, evaluator panel and answer-key
report). Compares four top-k feature sets (METHODS):
1) stepwise_rf: forward stepwise selection scored via random forest.
2) stepwise_logreg: forward stepwise selection scored via logistic regression.
3) rid_tree: single-family RID on a fully enumerated decision-tree Rashomon set.
4) cross_family_unweighted_truvarimp: cross-family RID, joint TruVaRImp
   candidate search (candidate_search='truvarimp_joint', rid/truvarimp.py).

What differs from GDSC, all because one patient contributes up to 21 rows:

Outer holdout by patient. The *patients* are split once (seeded, unstratified --
the label doesn't exist yet when the split is drawn; --holdout-fraction 0.2,
--holdout-seed 42), and every row of a test patient is a test row. The sha256 of
the sorted test patient IDs goes into each checkpoint's settings, so checkpoints
from another split are refused.

Label, fit on the train patients only (--label):
    zscore_mean (default)    within each combo, z-score log CR with the train
                             rows' mean and SD for that combo; label 1 when
                             -z > 0, i.e. log CR below that combo's train mean
                             (better-than-typical synergy for that combo). The
                             same per-combo thresholds label the test rows.
    cr_lt_075                label 1 when CR < 0.75; nothing to fit.
Class balance is reported per split.

Features, fit on the train patients only: the --n-top-variance-genes (500) genes
most variable across the train patients (one value per patient, not per row),
plus the answer-key genes (prepare_express_data.ANSWER_KEY_GENES) forced in;
named expr_<GENE> in alphabetical order. Then one drug_<partner> indicator per
combo. Each row repeats its patient's expression. Constant columns, the
correlation filter and standardization are fit on the train rows and applied to
the test rows, as in GDSC. As in GDSC, the answer-key genes are not exempt from
the correlation filter: a dropped one is credited through the feature it lost
to (the *_with_proxies columns of answer_key_recovery.csv). The filter keeps the
earlier column of a correlated pair, i.e. the alphabetically earlier gene.

Grouping everywhere: stepwise selection's inner CV and the in-train CV
diagnostic use StratifiedGroupKFold by patient; the evaluator panel fits on the
train rows and scores the test rows (disjoint patients); both RID methods fit
with groups=<train patient IDs>, so each bootstrap resamples whole patients and
its out-of-bag rows are the undrawn patients' rows (rid.core grouped bootstrap).

Outputs go to <--output-dir>/<label>/: per-method checkpoints
(<method>_result.json, <method>_ranking.csv -- the k stepwise steps, or every
feature for RID methods), and once all four are checkpointed,
top_feature_overlap.csv, answer_key_recovery.csv (each method's rank of each
answer-key gene), run_settings.json and summary_row.json;
<--output-dir>/overall_summary.csv collects every finished label.

--profile timers|cprofile (default off) profiles each RID bootstrap without
changing results: <label>/<method>_profile.csv (one row per bootstrap) and, for
cprofile, <label>/<method>_profile/bootstrap_<b>.prof. It is not a checkpoint
setting, so a profiling run resumes from (and is skipped by) existing
checkpoints; use a separate --output-dir to profile a method already done.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, train_test_split
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_express_data import ANSWER_KEY_GENES  # noqa: E402
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

logger = logging.getLogger(__name__)


ROWS_FILE = "rows.csv"
EXPRESSION_FILE = "expression.csv"
LABELS = ("zscore_mean", "cr_lt_075")
CR_THRESHOLD = 0.75

# The answer key as GDSC-style biomarker groups, one gene per group.
ANSWER_KEY = {"biomarkers": {gene: [f"expr_{gene}"] for gene in ANSWER_KEY_GENES}}

METHODS = (
    "stepwise_rf",
    "stepwise_logreg",
    "rid_tree",
    "cross_family_unweighted_truvarimp",
)

# Arguments that change a method's result. A checkpoint written under different
# values is refused rather than silently reused (n_jobs/logging/paths don't count).
_DATA_SETTING_KEYS = ("data_fingerprint", "label", "top_k", "n_top_variance_genes", "correlation_threshold",
                      "random_state", "stepwise_cv_splits", "stepwise_scoring_classification",
                      "holdout_fraction", "holdout_seed")
_STEPWISE_SETTING_KEYS = ("stepwise_n_estimators", "stepwise_max_depth", "stepwise_min_samples_leaf",
                          "stepwise_class_weight", "stepwise_logreg_C")
_RID_SETTING_KEYS = ("rid_metric", "rid_epsilon", "rid_n_bootstraps")
METHOD_SETTING_KEYS = {
    "stepwise_rf": _DATA_SETTING_KEYS + _STEPWISE_SETTING_KEYS,
    "stepwise_logreg": _DATA_SETTING_KEYS + _STEPWISE_SETTING_KEYS,
    "rid_tree": _DATA_SETTING_KEYS + _RID_SETTING_KEYS,
    "cross_family_unweighted_truvarimp": _DATA_SETTING_KEYS + _RID_SETTING_KEYS + ("truvarimp_total_budget",),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the EXPRESS stand-in (venetoclax-combination synergy in AML, patient-grouped), "
            "comparing random-forest and logistic-regression forward stepwise feature "
            "selection against single-family RID on a fully enumerated decision-tree "
            "Rashomon set and unweighted cross-family RID (TruVaRImp candidate search)."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "processed"),
        help="Directory with prepare_express_data.py's outputs (default: <script dir>/data/processed)",
    )
    parser.add_argument(
        "--label",
        choices=LABELS,
        default="zscore_mean",
        help=(
            "zscore_mean: log CR below its combo's train-patient mean (per-combo z-score < 0); "
            f"cr_lt_075: CR < {CR_THRESHOLD} (default: zscore_mean)"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Number of top features to select from each method (default: 40)",
    )
    parser.add_argument(
        "--n-top-variance-genes",
        type=int,
        default=500,
        help=(
            "Genes kept by variance across the train patients, before the answer-key "
            "genes are added (default: 500)"
        ),
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
        default=str(Path(__file__).resolve().parent / "results" / "express_comparison"),
        help="Directory for outputs; each label gets a subdirectory (default: <script dir>/results/express_comparison)",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of patients held out (all their rows) as the test set the "
            "evaluator panel scores on; feature selection only sees the rest (default: 0.2)"
        ),
    )
    parser.add_argument(
        "--holdout-seed",
        type=int,
        default=42,
        help="Random seed for the outer patient-grouped train/test split (default: 42)",
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
        help=(
            "Patient-grouped CV folds for stepwise selection and the in-train CV diagnostic, "
            "on the train rows only (default: 5)"
        ),
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
        help="RID bootstrap iterations, each resampling whole train patients (default: 100)",
    )
    parser.add_argument(
        "--rid-n-jobs",
        type=int,
        default=1,
        help="RID parallel jobs (default: 1)",
    )
    parser.add_argument(
        "--profile",
        choices=("off", "timers", "cprofile"),
        default="off",
        help=(
            "Per-bootstrap profiling of the RID methods (default: off). 'timers' writes "
            "<label>/<method>_profile.csv (one row per bootstrap: stage seconds, VI seconds per "
            "family and metric, Rashomon size per family, worker peak RSS) and logs each "
            "bootstrap's stage summary; 'cprofile' also writes <label>/<method>_profile/"
            "bootstrap_<b>.prof (cProfile, from inside each joblib worker). Doesn't change "
            "results, so it isn't a checkpoint setting."
        ),
    )
    parser.add_argument(
        "--truvarimp-total-budget",
        type=int,
        default=300,
        help=(
            "Shared per-bootstrap real-fit budget across all families for "
            "cross_family_unweighted_truvarimp (default: 300), spent on a shared pool of "
            "rid.candidate_pool.DEFAULT_N_CANDIDATES_PER_FAMILY candidates per family."
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


def _feature_name(prefix: str, label: str) -> str:
    return f"{prefix}_{re.sub(r'[^0-9A-Za-z]+', '_', str(label)).strip('_')}"


def _remove_highly_correlated_features(
    X: pd.DataFrame,
    correlation_threshold: float,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Drop one feature of every pair correlated above the threshold.

    Keeps the earlier column, as in the GDSC copy. Column order is alphabetical
    expression genes, then the drug indicators, so of two correlated genes the
    alphabetically earlier one is kept -- an arbitrary but label-blind rule.
    Also returns {removed: the kept feature it lost to}.
    """

    if correlation_threshold <= 0:
        return X, [], {}

    corr_matrix = X.corr().abs()
    corr_values = corr_matrix.to_numpy()
    columns = corr_matrix.columns
    removed_features: set[str] = set()
    kept_in_favour_of: dict[str, str] = {}

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if not corr_values[i, j] > correlation_threshold:
                continue

            feature_one = columns[i]
            feature_two = columns[j]
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


def _make_patient_holdout_split(
    patient_ids: pd.Series, holdout_fraction: float, holdout_seed: int
) -> tuple[pd.Index, pd.Index, dict]:
    """One outer train/test split of the *patients* (unstratified: drawn before any
    label exists). Returns (train_patients, test_patients, metadata) with a hash of
    the sorted test patient IDs, so checkpoints from another split are refused."""

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"--holdout-fraction must be in (0, 1), got {holdout_fraction}")
    patients = np.sort(patient_ids.unique())
    train_patients, test_patients = train_test_split(patients, test_size=holdout_fraction, random_state=holdout_seed)
    train_patients, test_patients = pd.Index(np.sort(train_patients)), pd.Index(np.sort(test_patients))
    test_ids = [str(p) for p in test_patients]
    metadata = {
        "protocol": "outer_patient_grouped_holdout",
        "holdout_fraction": holdout_fraction,
        "holdout_seed": holdout_seed,
        "n_train_patients": int(len(train_patients)),
        "n_test_patients": int(len(test_patients)),
        "test_patients_sha256": hashlib.sha256("\n".join(test_ids).encode("utf-8")).hexdigest()[:16],
        "test_patients": test_ids,
    }
    return train_patients, test_patients, metadata


def _fit_label_params(train_rows: pd.DataFrame, label: str) -> dict:
    """Per-combo log-CR mean and SD over the train rows (zscore_mean); nothing
    for cr_lt_075. Fit on train rows only, then applied to every row."""

    if label == "cr_lt_075":
        return {"label": label, "cr_threshold": CR_THRESHOLD}
    stats = train_rows.groupby("combo")["log_cr"].agg(["mean", "std", "count"])
    too_small = stats.index[(stats["count"] < 2) | ~(stats["std"] > 0)].tolist()
    if too_small:
        raise ValueError(f"combos with fewer than 2 distinct train rows can't be z-scored: {too_small}")
    return {
        "label": label,
        "combo_mean": stats["mean"].to_dict(),
        "combo_sd": stats["std"].to_dict(),
        "combo_n_train_rows": stats["count"].astype(int).to_dict(),
    }


def _apply_label(rows: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series | None]:
    """Binary label (int 0/1) and, for zscore_mean, the per-combo z-score."""

    if params["label"] == "cr_lt_075":
        return (rows["cr"] < params["cr_threshold"]).astype(int), None
    unseen = sorted(set(rows["combo"]) - set(params["combo_mean"]))
    if unseen:
        raise ValueError(f"combos without train rows, so no label threshold: {unseen}")
    z = (rows["log_cr"] - rows["combo"].map(params["combo_mean"])) / rows["combo"].map(params["combo_sd"])
    return (-z > 0).astype(int), z


def _select_variable_genes(train_expression: pd.DataFrame, n_genes: int) -> list[str]:
    """The n most variable genes across the train patients (one row per patient)."""

    return train_expression.var().sort_values(ascending=False, kind="stable").index[:n_genes].tolist()


def _fit_standardizer(X_train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return X_train.mean(), X_train.std(ddof=0)


def _class_counts(y: pd.Series) -> dict:
    return {str(k): int(v) for k, v in y.value_counts().sort_index().items()}


def _prepare_dataset(
    data_dir: Path,
    label: str,
    n_top_variance_genes: int,
    correlation_threshold,
    holdout_fraction: float,
    holdout_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, dict]:
    """Load the rows and expression, split the patients, then fit the label, gene
    selection, column filters and standardization on the train patients only.
    Returns (X_train, X_test, y_train, y_test, groups_train, groups_test, metadata);
    groups are patient IDs aligned with the rows."""

    rows_path = data_dir / ROWS_FILE
    expression_path = data_dir / EXPRESSION_FILE
    for path in (rows_path, expression_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run experiments/express/prepare_express_data.py first")

    rows = pd.read_csv(rows_path)
    expression = pd.read_csv(expression_path, index_col="patient_id")
    if rows.duplicated(["patient_id", "combo"]).any():
        raise ValueError(f"{rows_path} has more than one row per (patient, combo)")
    missing_patients = set(rows["patient_id"]) - set(expression.index)
    if missing_patients:
        raise ValueError(f"{len(missing_patients)} patients in {ROWS_FILE} have no expression")

    train_patients, test_patients, holdout_meta = _make_patient_holdout_split(
        rows["patient_id"], holdout_fraction, holdout_seed
    )
    is_train = rows["patient_id"].isin(train_patients)
    train_index, test_index = rows.index[is_train], rows.index[~is_train]

    label_params = _fit_label_params(rows.loc[train_index], label)
    # Integer 0/1: rid's AUPRC metric scores binary tasks with pos_label=1.
    y, z = _apply_label(rows, label_params)

    variable_genes = _select_variable_genes(expression.loc[train_patients], n_top_variance_genes)
    missing_answer_key = sorted(set(ANSWER_KEY_GENES) - set(expression.columns))
    if missing_answer_key:
        raise ValueError(f"answer-key genes missing from {EXPRESSION_FILE}: {missing_answer_key}")
    genes = sorted(set(variable_genes) | set(ANSWER_KEY_GENES))
    expression_features = expression.loc[rows["patient_id"], genes].set_axis(rows.index)
    expression_features.columns = [f"expr_{gene}" for gene in genes]

    combos = sorted(rows["combo"].unique())
    partner_of = rows.drop_duplicates("combo").set_index("combo")["partner"]
    drug_features = pd.DataFrame(
        {_feature_name("drug", partner_of[combo]): (rows["combo"] == combo).astype(int) for combo in combos},
        index=rows.index,
    )
    X = pd.concat([expression_features, drug_features], axis=1)
    if X.columns.duplicated().any():
        raise ValueError("duplicate feature names after building the panel")

    constant_cols = [col for col in X.columns if X.loc[train_index, col].nunique() <= 1]
    X = X.drop(columns=constant_cols)

    X_before_corr = X.shape[1]
    resolved_threshold, threshold_info = _resolve_correlation_threshold(X.loc[train_index], correlation_threshold)
    X_train, removed_corr, corr_proxies = _remove_highly_correlated_features(X.loc[train_index], resolved_threshold)
    X = X[X_train.columns]
    # Standardize with train-row statistics so the logistic/SVM evaluators and stepwise
    # logreg see scaled inputs; RID rescales internally and trees are unaffected.
    train_mean, train_std = _fit_standardizer(X.loc[train_index])
    X = (X - train_mean) / train_std

    groups = rows["patient_id"]
    holdout_meta.update({
        "n_train": int(len(train_index)),
        "n_test": int(len(test_index)),
        "train_class_counts": _class_counts(y.loc[train_index]),
        "test_class_counts": _class_counts(y.loc[test_index]),
        "train_positive_fraction_by_combo": y.loc[train_index].groupby(rows.loc[train_index, "combo"]).mean().to_dict(),
        "test_positive_fraction_by_combo": y.loc[test_index].groupby(rows.loc[test_index, "combo"]).mean().to_dict(),
        "test_rows_per_specimen_type": rows.loc[test_index, "specimen_type"].value_counts().to_dict(),
    })
    metadata = {
        "rows_path": str(rows_path),
        "expression_path": str(expression_path),
        "label": label,
        "label_params": label_params,
        "task": "classification",
        "n_rows": int(X.shape[0]),
        "n_patients": int(groups.nunique()),
        "n_combos": len(combos),
        "n_variable_genes": len(variable_genes),
        "n_answer_key_genes_added": len(set(ANSWER_KEY_GENES) - set(variable_genes)),
        "answer_key_genes_in_top_variance": sorted(set(ANSWER_KEY_GENES) & set(variable_genes)),
        "n_features_before_corr": int(X_before_corr),
        "n_features_after_corr": int(X.shape[1]),
        "n_removed_constant": len(constant_cols),
        "removed_constant_features": constant_cols,
        "n_removed_corr": int(len(removed_corr)),
        "removed_corr_features": removed_corr,
        "corr_proxies": corr_proxies,
        "correlation_threshold_info": threshold_info,
        "n_classes": int(y.nunique()),
        "class_counts": _class_counts(y),
        "holdout": holdout_meta,
    }
    if z is not None:
        metadata["test_z_mean"] = float(z.loc[test_index].mean())
    return (X.loc[train_index], X.loc[test_index], y.loc[train_index], y.loc[test_index],
            groups.loc[train_index], groups.loc[test_index], metadata)


def _make_stepwise_cv_and_scoring(args: argparse.Namespace, y: pd.Series):
    # Grouped by patient (pass groups= to every split/cross_val_score call), so no
    # patient's rows sit on both sides of an inner fold.
    cv = StratifiedGroupKFold(
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
    groups: pd.Series,
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
        groups=groups,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )
    return feature, float(np.mean(scores)), float(np.std(scores))


def run_forward_stepwise_selection(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
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
                groups=groups,
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


def _rid_ranking_table(estimator, rid_metric: str) -> tuple[list, pd.DataFrame]:
    ranking = estimator.rank_features(rid_metric)
    summary = estimator.metric_summary(rid_metric)
    # Every feature, not just the top k, so answer-key genes ranked below k still get a rank.
    rows = [
        {
            "rank": rank,
            "feature": feature,
            "expected_importance": float(expected_importance),
            "prob_positive": float(summary[feature]["prob_positive"]),
        }
        for rank, (feature, expected_importance) in enumerate(ranking, start=1)
    ]
    return ranking, pd.DataFrame(rows)


def _check_n_groups(estimator, groups: pd.Series) -> int:
    n_groups = estimator.n_groups_
    if n_groups != groups.nunique():
        raise RuntimeError(f"RID saw {n_groups} groups, expected {groups.nunique()} train patients")
    return int(n_groups)


def run_single_family_tree_rid(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    top_k: int,
    rid_metric: str,
    epsilon: float,
    n_bootstraps: int,
    n_jobs: int,
    profile: str | None = None,
    profile_dir: Path | None = None,
) -> tuple[list[str], pd.DataFrame, dict, list | None]:
    # n_models_pool is left at its default: it only sets seed refits for
    # refit_seeds=True grids, and the tree grid fits each config once.
    estimator = RashomonImportanceDistribution(
        epsilon=epsilon,
        n_bootstraps=n_bootstraps,
        model_class=FullyEnumeratedTreeClassifier,
        vi_metrics=(rid_metric,),
        performance_metrics=("accuracy", "auprc"),
        n_jobs=n_jobs,
        profile=profile,
        profile_dir=profile_dir,
    )
    estimator.fit(X, y, groups=groups.to_numpy())
    n_groups = _check_n_groups(estimator, groups)

    if estimator.metric_results_ is None:
        raise RuntimeError(
            "Single-family tree RID returned no metric results; "
            "no valid Rashomon bootstraps were found"
        )

    ranking, table = _rid_ranking_table(estimator, rid_metric)
    top_features = [feature for feature, _ in ranking[:top_k]]

    perf_stats = dict(estimator.perf_stats_ or {})
    perf_stats["n_valid_bootstraps"] = estimator.n_valid_bootstraps_
    perf_stats["n_groups"] = n_groups
    return top_features, table, perf_stats, estimator.profile_


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
    groups: pd.Series,
    top_k: int,
    rid_metric: str,
    epsilon: float,
    n_bootstraps: int,
    n_jobs: int,
    truvarimp_total_budget: int,
    profile: str | None = None,
    profile_dir: Path | None = None,
) -> tuple[list[str], pd.DataFrame, dict, list | None]:
    # Every family is in the shared candidate pool, so n_models_per_class is unused.
    estimator = CrossFamilyRashomonImportanceDistribution(
        model_configs=_build_cross_family_model_configs(),
        epsilon=epsilon,
        n_bootstraps=n_bootstraps,
        vi_metrics=(rid_metric,),
        performance_metrics=("accuracy", "auprc"),
        family_balance_mode="unweighted",
        n_jobs=n_jobs,
        candidate_search="truvarimp_joint",
        truvarimp_config={"total_budget": truvarimp_total_budget},
        profile=profile,
        profile_dir=profile_dir,
    )
    estimator.fit(X, y, groups=groups.to_numpy())
    n_groups = _check_n_groups(estimator, groups)

    if estimator.metric_results_ is None:
        raise RuntimeError(
            "Cross-family RID (truvarimp_joint) returned no metric results; "
            "no valid Rashomon bootstraps were found"
        )

    ranking, table = _rid_ranking_table(estimator, rid_metric)
    top_features = [feature for feature, _ in ranking[:top_k]]

    perf_summary = {
        "family_counts": estimator.family_counts_,
        "family_perf_stats": estimator.family_perf_stats_,
        "n_valid_bootstraps": estimator.n_valid_bootstraps_,
        "n_groups": n_groups,
    }
    return top_features, table, perf_summary, estimator.profile_


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
    groups: pd.Series,
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
        groups=groups,
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
    groups_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: list[str],
    model_cls,
    model_kwargs: dict,
    train_cv,
    scoring: str,
) -> dict:
    """Fit on the train rows restricted to `features`, score once on the held-out
    test patients' rows (holdout_score, the primary score). Also report the same
    evaluator's patient-grouped CV score within the train rows (train_cv_score_*),
    on the folds stepwise selection searched over -- optimistic for the stepwise
    methods by construction."""

    estimator = model_cls(**model_kwargs).fit(X_train[features], y_train)
    holdout_score = float(get_scorer(scoring)(estimator, X_test[features], y_test))
    train_cv = _evaluate_feature_set(
        X_train, y_train, groups_train, features, model_cls, model_kwargs, train_cv, scoring
    )
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
    groups_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    method_feature_lists: dict[str, list[str]],
    evaluators: list[tuple[str, type, dict]],
    train_cv,
    scoring: str,
) -> dict[str, dict[str, dict]]:
    """Evaluate every method's selected top-k features under every evaluator
    in the panel -- fit on the train rows, scored on the held-out test patients'
    rows no method saw while selecting -- so methods are compared on equal footing
    regardless of which model (if any) they used internally to select features,
    and regardless of any single evaluator's own inductive bias."""

    return {
        name: {
            evaluator_name: _evaluate_feature_set_holdout(
                X_train, y_train, groups_train, X_test, y_test, features, model_cls, model_kwargs,
                train_cv, scoring,
            )
            for evaluator_name, model_cls, model_kwargs in evaluators
        }
        for name, features in method_feature_lists.items()
    }


def _answer_key_recovery(method: str, ranking: list[str], top_k: int, data_meta: dict) -> list[dict]:
    """Where a method ranks each answer-key gene (GDSC's biomarker_recovery, one gene
    per group).

    Stepwise rankings stop at k, so a stepwise gene outside the top k has no rank;
    RID rankings cover every feature. A gene dropped by the correlation filter
    can't be ranked, so the *_with_proxies columns also credit the kept feature it
    was dropped in favour of (|r| above the threshold, so it carries the same
    signal). The strict columns count the gene's own feature only.
    """

    position = {feature: rank for rank, feature in enumerate(ranking, start=1)}
    proxies = data_meta["corr_proxies"]
    rows = []
    for gene, members in ANSWER_KEY["biomarkers"].items():
        ranked = sorted((position[m], m) for m in members if m in position)
        best_rank, best_member = ranked[0] if ranked else (None, None)
        proxy_of = {proxies[m]: m for m in members if m in proxies}
        with_proxies = sorted(ranked + [(position[p], p) for p in proxy_of if p in position])
        proxy_rank, proxy_feature = with_proxies[0] if with_proxies else (None, None)
        rows.append({
            "method": method,
            "gene": gene,
            "feature": members[0],
            "rank": best_rank,
            "in_top_k": best_rank is not None and best_rank <= top_k,
            "best_feature_with_proxies": proxy_feature,
            "best_rank_with_proxies": proxy_rank,
            "in_top_k_with_proxies": proxy_rank is not None and proxy_rank <= top_k,
            "n_ranked": len(ranking),
            "removed_by_corr_filter": ";".join(f"{m}->{proxies[m]}" for m in members if m in proxies),
        })
    return rows


def _data_fingerprint(data_dir: Path) -> str:
    """SHA-256 over the processed rows and expression, so a checkpoint built from
    differently prepared data (new AUC floor, ...) is refused."""

    digest = hashlib.sha256()
    for name in (ROWS_FILE, EXPRESSION_FILE):
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run experiments/express/prepare_express_data.py first")
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
        settings["rid_bootstrap"] = "grouped_by_patient"
    if method.startswith("cross_family_"):
        settings["n_candidates_per_family"] = DEFAULT_N_CANDIDATES_PER_FAMILY
    # The forced-in genes change the feature panel but not data_fingerprint.
    settings["answer_key_genes"] = list(ANSWER_KEY_GENES)
    settings["evaluation_protocol"] = holdout_meta["protocol"]
    settings["holdout_test_patients_sha256"] = holdout_meta["test_patients_sha256"]
    return settings


def _run_key(args: argparse.Namespace) -> dict:
    """Every result-changing argument across methods except the label. A
    summary_row.json written under a different key is left out of
    overall_summary.csv, so it never mixes rows from different settings or splits."""

    keys = sorted(set().union(*METHOD_SETTING_KEYS.values()) - {"label"})
    run_key = {key: getattr(args, key) for key in keys}
    run_key["n_candidates_per_family"] = DEFAULT_N_CANDIDATES_PER_FAMILY
    run_key["answer_key_genes"] = list(ANSWER_KEY_GENES)
    return json.loads(json.dumps(run_key))


def _method_result_path(label_output_dir: Path, method: str) -> Path:
    return label_output_dir / f"{method}_result.json"


def _load_method_checkpoint(
    label_output_dir: Path, method: str, args: argparse.Namespace, holdout_meta: dict
) -> dict | None:
    """Return a method's checkpoint, or None if absent. Refuse one written under different settings."""

    path = _method_result_path(label_output_dir, method)
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
    groups: pd.Series,
    args: argparse.Namespace,
    cv,
    stepwise_scoring: str,
    profile_dir: Path | None = None,
) -> tuple[list[str], pd.DataFrame, dict, list | None]:
    """Run one method on the train rows only (X, y, groups are the holdout split's
    train part); return (top_features, per-feature table, method-specific info,
    RID per-bootstrap profile records or None). --profile only affects the RID
    methods; profile_dir is where --profile cprofile puts their .prof files."""

    profile = None if args.profile == "off" else args.profile

    if method in ("stepwise_rf", "stepwise_logreg"):
        if method == "stepwise_rf":
            model_cls, model_kwargs = _make_stepwise_rf_model(args)
        else:
            model_cls, model_kwargs = _make_stepwise_logreg_model(args)
        logger.info(
            f"[{method}] scoring={stepwise_scoring} cv_splits={args.stepwise_cv_splits} (grouped by patient) "
            f"model={model_cls.__name__}"
        )
        features, history = run_forward_stepwise_selection(
            X=X,
            y=y,
            groups=groups,
            top_k=args.top_k,
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            cv=cv,
            scoring=stepwise_scoring,
            n_jobs=args.stepwise_n_jobs,
        )
        search_eval = _evaluate_feature_set(
            X=X, y=y, groups=groups, features=features,
            model_cls=model_cls, model_kwargs=model_kwargs, cv=cv, scoring=stepwise_scoring,
        )
        info = {
            "scoring": stepwise_scoring,
            "model_kwargs": model_kwargs,
            "search_model_train_cv_evaluation": search_eval,
        }
        return features, history, info, None

    if method == "rid_tree":
        logger.info(
            f"[rid_tree] metric={args.rid_metric} epsilon={args.rid_epsilon} "
            f"n_bootstraps={args.rid_n_bootstraps} (grouped by patient)"
        )
        features, table, perf, profile_records = run_single_family_tree_rid(
            X=X,
            y=y,
            groups=groups,
            top_k=args.top_k,
            rid_metric=args.rid_metric,
            epsilon=args.rid_epsilon,
            n_bootstraps=args.rid_n_bootstraps,
            n_jobs=args.rid_n_jobs,
            profile=profile,
            profile_dir=profile_dir,
        )
        info = {"metric": args.rid_metric, "rashomon_oob_perf_stats": perf}
        return features, table, info, profile_records

    candidate_search = "truvarimp_joint"
    logger.info(
        f"[{method}] metric={args.rid_metric} epsilon={args.rid_epsilon} "
        f"n_bootstraps={args.rid_n_bootstraps} (grouped by patient) candidate_search={candidate_search} "
        f"total_budget={args.truvarimp_total_budget}"
    )
    features, table, perf, profile_records = run_cross_family_rid(
        X=X,
        y=y,
        groups=groups,
        top_k=args.top_k,
        rid_metric=args.rid_metric,
        epsilon=args.rid_epsilon,
        n_bootstraps=args.rid_n_bootstraps,
        n_jobs=args.rid_n_jobs,
        truvarimp_total_budget=args.truvarimp_total_budget,
        profile=profile,
        profile_dir=profile_dir,
    )
    info = {
        "metric": args.rid_metric,
        "candidate_search": candidate_search,
        "rashomon_oob_perf_stats": perf,
    }
    return features, table, info, profile_records


def _run_label(args: argparse.Namespace, output_dir: Path) -> dict | None:
    """Run this invocation's --methods for --label, checkpointing each, then
    assemble the combined outputs if every method in METHODS is checkpointed.
    Returns the summary row, or None while methods are still missing."""

    label_output_dir = output_dir / args.label
    label_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("-" * 70)
    logger.info(f"[label] {args.label}")

    X_train, X_test, y_train, y_test, groups_train, groups_test, data_meta = _prepare_dataset(
        data_dir=Path(args.data_dir),
        label=args.label,
        n_top_variance_genes=args.n_top_variance_genes,
        correlation_threshold=args.correlation_threshold,
        holdout_fraction=args.holdout_fraction,
        holdout_seed=args.holdout_seed,
    )
    holdout_meta = data_meta["holdout"]
    if not X_train.index.intersection(X_test.index).empty:
        raise RuntimeError("Train and test rows overlap")
    if set(groups_train) & set(groups_test):
        raise RuntimeError("A patient has rows in both train and test")
    logger.info(
        f"[data] rows={data_meta['n_rows']} patients={data_meta['n_patients']} "
        f"features={data_meta['n_features_after_corr']} "
        f"(removed_corr={data_meta['n_removed_corr']}) classes={data_meta['class_counts']}"
    )
    logger.info(
        f"[holdout] train={holdout_meta['n_train']} rows / {holdout_meta['n_train_patients']} patients "
        f"{holdout_meta['train_class_counts']}  test={holdout_meta['n_test']} rows / "
        f"{holdout_meta['n_test_patients']} patients {holdout_meta['test_class_counts']} "
        f"(test patients sha256 {holdout_meta['test_patients_sha256']})"
    )

    # Refuse mismatched checkpoints before spending hours on anything else (data prep
    # comes first only because the settings include the holdout split's hash).
    checkpoints = {
        method: _load_method_checkpoint(label_output_dir, method, args, holdout_meta) for method in METHODS
    }

    # cv only ever splits the train rows: stepwise selection's inner CV and the
    # in-train CV diagnostic. The test rows are touched only by the evaluator panel.
    cv, stepwise_scoring = _make_stepwise_cv_and_scoring(args, y_train)
    evaluator_panel = build_common_evaluator_panel(args.random_state)

    for method in args.methods:
        if checkpoints[method] is not None:
            logger.info(f"[{method}] checkpoint found in {label_output_dir}; skipping")
            continue

        started = time.time()
        profile_dir = label_output_dir / f"{method}_profile" if args.profile == "cprofile" else None
        features, table, info, profile_records = _run_method(
            method, X_train, y_train, groups_train, args, cv, stepwise_scoring, profile_dir=profile_dir
        )
        elapsed = time.time() - started
        if profile_records is not None:
            # One row per bootstrap; written before the checkpoint so a profiling
            # run that dies in the evaluator panel still leaves its profile.
            profile_path = label_output_dir / f"{method}_profile.csv"
            _write_csv_atomic(pd.DataFrame(profile_records), profile_path)
            logger.info(f"[{method}] per-bootstrap profile: {profile_path}"
                        + (f" (cProfile dumps in {profile_dir})" if profile_dir is not None else ""))

        # Common holdout evaluation: every method's features scored the same way (fit on
        # train, scored on the untouched test patients), under a panel of evaluators
        # spanning different inductive biases.
        common_eval = run_common_evaluation_panel(
            X_train, y_train, groups_train, X_test, y_test, {method: features}, evaluator_panel, cv,
            stepwise_scoring,
        )[method]

        # Stepwise: the k selection steps. RID: every feature, ranked.
        _write_csv_atomic(table, label_output_dir / f"{method}_ranking.csv")
        checkpoint = {
            "settings": json.loads(json.dumps(_method_settings(method, args, holdout_meta))),
            "top_features": features,
            "ranking": table["feature"].tolist(),
            "elapsed_seconds": elapsed,
            # Not a setting: profiling doesn't change results (it does inflate elapsed_seconds).
            "profile": args.profile,
            "common_eval": common_eval,
            **info,
        }
        _write_json_atomic(checkpoint, _method_result_path(label_output_dir, method))
        checkpoints[method] = checkpoint

        logger.info(f"[{method}] finished in {elapsed / 60.0:.1f} min; top-10: {features[:10]}")
        for row in _answer_key_recovery(method, checkpoint["ranking"], args.top_k, data_meta):
            logger.info(f"       answer key {row['gene']}: rank {row['rank']} "
                        f"(with proxies {row['best_rank_with_proxies']})")
        for evaluator_name, _, _ in evaluator_panel:
            evaluator_result = common_eval[evaluator_name]
            logger.info(
                f"       holdout [{evaluator_name}] ({stepwise_scoring}): {evaluator_result['holdout_score']:.4f}"
                f"  (in-train CV {evaluator_result['train_cv_score_mean']:.4f} "
                f"+- {evaluator_result['train_cv_score_std']:.4f})"
            )

    # Re-read: array tasks for the other methods run concurrently and may have
    # checkpointed since this invocation started.
    checkpoints = {
        method: _load_method_checkpoint(label_output_dir, method, args, holdout_meta) for method in METHODS
    }
    completed = {method: checkpoint for method, checkpoint in checkpoints.items() if checkpoint is not None}
    missing = [method for method in METHODS if method not in completed]
    if missing:
        logger.info(f"[label] {args.label}: methods not yet checkpointed {missing}; skipping combined outputs")
        return None

    method_feature_lists = {method: completed[method]["top_features"] for method in METHODS}
    comparison = _build_comparison_table(method_feature_lists)
    _write_csv_atomic(
        comparison.sort_values(["n_methods_selected", "stepwise_rf_rank"], ascending=[False, True]),
        label_output_dir / "top_feature_overlap.csv",
    )
    overlap_counts = _pairwise_overlap_counts(comparison, list(METHODS))

    settings = {
        "label": args.label,
        "answer_key_genes": list(ANSWER_KEY_GENES),
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
                "holdout_score: fit on the train patients' rows, scored on the held-out test "
                "patients' rows no method selected on (primary). train_cv_score_*: "
                "patient-grouped CV within the train rows on the folds stepwise selection "
                "maximized (optimistic for stepwise; diagnostic only)."
            ),
            "results": {method: completed[method]["common_eval"] for method in METHODS},
        },
        **{
            method: {key: value for key, value in completed[method].items() if key != "common_eval"}
            for method in METHODS
        },
        "overlap_counts": overlap_counts,
    }
    _write_json_atomic(settings, label_output_dir / "run_settings.json")

    recovery = pd.DataFrame([
        row
        for method in METHODS
        for row in _answer_key_recovery(method, completed[method]["ranking"], args.top_k, data_meta)
    ])
    _write_csv_atomic(recovery, label_output_dir / "answer_key_recovery.csv")

    row = {
        "label": args.label,
        "n_rows": data_meta["n_rows"],
        "n_patients": data_meta["n_patients"],
        "n_features": data_meta["n_features_after_corr"],
        "n_train": holdout_meta["n_train"],
        "n_test": holdout_meta["n_test"],
        "n_train_patients": holdout_meta["n_train_patients"],
        "n_test_patients": holdout_meta["n_test_patients"],
        "train_positive_fraction": holdout_meta["train_class_counts"].get("1", 0) / holdout_meta["n_train"],
        "test_positive_fraction": holdout_meta["test_class_counts"].get("1", 0) / holdout_meta["n_test"],
        "holdout_test_patients_sha256": holdout_meta["test_patients_sha256"],
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
    cross_family_rashomon = _summarize_cross_family_perf(
        completed["cross_family_unweighted_truvarimp"]["rashomon_oob_perf_stats"]
    )
    row["cross_family_unweighted_truvarimp_rashomon_oob_accuracy_mean"] = cross_family_rashomon["accuracy_mean"]
    row["cross_family_unweighted_truvarimp_rashomon_oob_auprc_mean"] = cross_family_rashomon["auprc_mean"]
    for method in METHODS:
        method_recovery = recovery[recovery["method"] == method]
        row[f"{method}_answer_key_in_top_k"] = int(method_recovery["in_top_k"].sum())
        row[f"{method}_answer_key_in_top_k_with_proxies"] = int(method_recovery["in_top_k_with_proxies"].sum())
    row["n_answer_key_genes"] = len(ANSWER_KEY_GENES)
    row.update(overlap_counts)

    # Saved so whichever invocation finishes last can assemble every label's row.
    _write_json_atomic({**row, "run_key": _run_key(args)}, label_output_dir / "summary_row.json")
    logger.info(f"[label] combined outputs saved in {label_output_dir}")
    return row


def main() -> None:
    args = _parse_args()
    args.data_fingerprint = _data_fingerprint(Path(args.data_dir))
    silence_warnings()
    configure_logging(args.log_level, args.log_file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("EXPRESS stand-in (VEN combinations, AML) Top-k Feature Comparison")
    logger.info(f"  data_dir:                 {args.data_dir}")
    logger.info(f"  label:                    {args.label}")
    logger.info(f"  methods:                  {args.methods}")
    logger.info(f"  top_k:                    {args.top_k}")
    logger.info(f"  n_top_variance_genes:     {args.n_top_variance_genes}")
    logger.info(f"  correlation_threshold:    {args.correlation_threshold}")
    logger.info(f"  holdout:                  fraction={args.holdout_fraction} seed={args.holdout_seed} (by patient)")
    logger.info(f"  truvarimp_total_budget:   {args.truvarimp_total_budget}")
    logger.info(f"  profile:                  {args.profile}")
    logger.info(f"  output_dir:               {output_dir}")
    logger.info("=" * 70)

    _run_label(args, output_dir)

    # An array task runs one method, so collect every label that any invocation has
    # finished, not only this invocation's.
    run_key = _run_key(args)
    finished, overall_rows = [], []
    for label in LABELS:
        row_path = output_dir / label / "summary_row.json"
        if not row_path.exists():
            continue
        row = json.loads(row_path.read_text(encoding="utf-8"))
        if row.pop("run_key", None) != run_key:
            logger.warning(f"[label] {label}: {row_path} was written under other settings; leaving it out")
            continue
        finished.append(label)
        overall_rows.append(row)
    if not finished:
        logger.info("=" * 70)
        logger.info("Checkpoints saved; no label has every method checkpointed yet")
        logger.info("=" * 70)
        return

    overall_summary_path = output_dir / "overall_summary.csv"
    _write_csv_atomic(pd.DataFrame(overall_rows), overall_summary_path)

    run_manifest = {
        "args": vars(args),
        "labels_finished": finished,
        "labels_pending": [label for label in LABELS if label not in finished],
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
