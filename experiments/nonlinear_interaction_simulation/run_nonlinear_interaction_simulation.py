#!/usr/bin/env python3
"""
Cross-family RID simulation study: grid-vs-TruVaRImp validation + ground-truth recovery.

Real-data experiments run cross-family RID with TruVaRImp candidate search only.
This study is where that choice is validated: on synthetic DGPs with known
relevant features, it runs cross-family RID with BOTH search modes over the
identical shared candidate pool (rid.candidate_pool; same data, bootstrap
indices, family names, per-candidate seeds), so the exhaustive grid's
per-bootstrap Rashomon set is the exact ground truth for TruVaRImp's. Five
methods per simulated dataset:
1) cross_family_unweighted: cross-family RID, candidate_search='grid' (every
   one of --n-candidates-per-family pool candidates per family is fit).
2) cross_family_unweighted_truvarimp: cross-family RID,
   candidate_search='truvarimp_joint' (--truvarimp-total-budget fits per
   bootstrap, chosen adaptively from the same pool).
3) stepwise_logreg: forward stepwise selection scored via logistic regression.
4) stepwise_rf: forward stepwise selection scored via random forest.
5) rid_tree: single-family RID on a fully enumerated decision-tree Rashomon set.

Per (dgp, beta, rep) cell it records:
- ground-truth recovery (precision@k/recall@k/NDCG@k/exact match, k = number of
  relevant features) for all five methods -> ``study_results*.csv``;
- search coverage of TruVaRImp against the grid -> ``search_coverage*.csv``:
  Rashomon-membership recall |TV & G|/|G| and precision |TV & G|/|TV| per
  bootstrap, the min-loss gap TV_min - G_min and whether TruVaRImp evaluated
  the grid's best candidate, per-family recall and member counts, fits and
  wall-clock per bootstrap in each mode, and agreement of the two modes' RID
  outputs (Spearman of expected_importance, top-k overlap, max |delta
  prob_positive|). Since a pool candidate fits to the identical model in both
  modes, ``loss_mismatch_max`` (max |delta OOB loss| over candidates both modes
  evaluated) must be 0. Even identical Rashomon sets don't give bit-identical
  RID outputs: sub_mr's per-bootstrap permutation RNG is consumed in member
  order, which is candidate-index order for the grid but evaluation order for
  TruVaRImp, so the agreement metrics include that Monte Carlo noise.

Results are checkpointed to one CSV per (DGP, beta) and kind
(``per_dgp/study_results_<dgp>__beta<beta>.csv`` and
``per_dgp/search_coverage_<dgp>__beta<beta>.csv``) after every completed
(dgp, beta, rep) cell, and a rerun into the same --output-dir skips cells
already on disk, so a run that hits the SLURM time limit can simply be
resubmitted (optionally restricted via --dgps / --betas). One file per
(DGP, beta) lets concurrent array tasks each own different betas of a DGP.

Ground-truth recovery (precision@k/recall@k/NDCG@k/exact-match vs. the known
relevant features) only measures ranking overlap with ground truth, never
predictive performance. RID judges Rashomon membership, VI and its own
performance metrics on each bootstrap's out-of-bag rows (see rid/core.py).

Usage:
    python run_nonlinear_interaction_simulation.py
    sbatch run_nonlinear_interaction_simulation.sl
"""

import argparse
import functools
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rid.candidate_pool import DEFAULT_N_CANDIDATES_PER_FAMILY  # noqa: E402
from rid import (
    CrossFamilyRashomonImportanceDistribution,
    FullyEnumeratedTreeClassifier,
    LassoClassifier,
    RashomonImportanceDistribution,
    RidgeClassifier,
    add_logging_arguments,
    configure_logging,
    performance_accuracy,
    performance_auprc,
    performance_f1,
    silence_warnings,
)
from rid._logging import format_duration  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 400
DEFAULT_REPETITIONS = 12
DEFAULT_BOOTSTRAPS = 12
DEFAULT_TRUVARIMP_TOTAL_BUDGET = 300
DEFAULT_BETA_GRID = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
DEFAULT_NOISE_STD = 1.0
DEFAULT_BASE_SEED = 20260514
DEFAULT_EPSILON = 0.05
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "results" / "nonlinear_interaction_simulation")
DEFAULT_STEPWISE_CV_SPLITS = 5
DEFAULT_STEPWISE_LOGREG_C = 1.0
DEFAULT_STEPWISE_RF_N_ESTIMATORS = 100
DEFAULT_RANDOM_STATE = 42
METHODS = (
    "cross_family_unweighted",
    "cross_family_unweighted_truvarimp",
    "stepwise_logreg",
    "stepwise_rf",
    "rid_tree",
)
# The two cross-family methods differ ONLY in candidate_search: same
# RID_MODEL_CONFIGS (hence family names, pools and seeds), data and bootstraps.
CROSS_FAMILY_SEARCH_MODES = {
    "cross_family_unweighted": "grid",
    "cross_family_unweighted_truvarimp": "truvarimp_joint",
}
GRID_METHOD, TRUVARIMP_METHOD = CROSS_FAMILY_SEARCH_MODES
PER_DGP_SUBDIR = "per_dgp"
STUDY_RESULTS_PREFIX = "study_results_"
SEARCH_COVERAGE_PREFIX = "search_coverage_"

# No per-family search_grid overrides here: every family now inherits its
# grid straight from rid.models.TRAINER_REGISTRY, standardizing this
# experiment's search space on the same grids Falcon-Cano and Staellert use
# rather than its own, much smaller, bespoke ones.
RID_MODEL_CONFIGS = {
    "RF": {
        "model": RandomForestClassifier,
        "kwargs": {"min_samples_leaf": 2},
    },
    "GBM": {
        "model": GradientBoostingClassifier,
        "kwargs": {},
    },
    "SVM": {
        "model": SVC,
        "kwargs": {},
    },
    "Lasso": {
        "model": LassoClassifier,
        "kwargs": {},
    },
    "Ridge": {
        "model": RidgeClassifier,
        "kwargs": {},
    },
}

RID_PERFORMANCE_METRICS = (
    performance_accuracy,
    performance_f1,
    performance_auprc,
)

# Three evaluators spanning distinct inductive biases (additive/linear, bagged
# axis-aligned interactions, smooth kernel margin), so a method's top-k set
# isn't rewarded/penalized by one model's blind spots -- same panel used by
# the Falcon-Cano/Staellert top-k comparison scripts' build_common_evaluator_panel.
COMMON_EVALUATOR_NAMES = ("logreg", "random_forest", "svm_rbf")


def resolve_num_workers(cli_override=None):
    """Determine number of workers from Slurm env first, then CLI."""

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        n_workers = int(slurm_cpus)
        logger.info(f"[env] SLURM_CPUS_PER_TASK={n_workers} -> using {n_workers} workers")
        return n_workers

    if cli_override is not None:
        logger.info(f"[cli] --num-workers={cli_override}")
        return cli_override

    n_workers = os.cpu_count() or 1
    logger.info(f"[fallback] os.cpu_count()={n_workers}")
    return n_workers


def build_simulated_dataset(
    features,
    terms,
    y,
    epsilon,
    *,
    dgp_name,
    relevant_features,
    beta,
    snr_empirical,
    classification_cutoff,
    equivalence_classes=None,
):
    features = features.reset_index(drop=True).copy()
    terms = terms.reset_index(drop=True).copy()
    y = np.asarray(y)
    epsilon = np.asarray(epsilon, dtype=float)

    dataset = pd.concat([features, terms], axis=1)
    dataset["epsilon"] = epsilon
    dataset["Y"] = y.astype(int)

    dataset.attrs["dgp_name"] = dgp_name
    dataset.attrs["relevant_features"] = tuple(relevant_features)
    dataset.attrs["beta"] = float(beta)
    dataset.attrs["snr_empirical"] = float(snr_empirical)
    dataset.attrs["classification_cutoff"] = float(classification_cutoff)
    # Maps a "true driver" feature name to the tuple of features that are
    # substitutable for it (itself plus any correlated duplicate/proxy
    # features that aren't actually in the outcome equation). Empty for DGPs
    # with no deliberate redundancy.
    dataset.attrs["equivalence_classes"] = dict(equivalence_classes or {})
    return dataset


def apply_beta_signal(signal, beta, rng, noise_std=1.0, cutoff="median"):
    signal_arr = np.asarray(signal, dtype=float)
    epsilon = rng.normal(loc=0.0, scale=noise_std, size=signal_arr.shape[0])
    beta_signal = beta * signal_arr
    latent_score = beta_signal + epsilon

    if cutoff == "median":
        classification_cutoff = float(np.median(latent_score))
    elif cutoff == "mean":
        classification_cutoff = float(np.mean(latent_score))
    else:
        classification_cutoff = float(cutoff)

    y = (latent_score >= classification_cutoff).astype(int)

    epsilon_var = float(np.var(epsilon))
    if epsilon_var == 0.0:
        snr_empirical = np.inf
    else:
        snr_empirical = float(np.var(beta_signal) / epsilon_var)

    return y, epsilon, latent_score, classification_cutoff, snr_empirical


def simulate_custom_nonlinear(n=1000, beta=1.0, noise_std=1.0, seed=42):
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {
            "X1": rng.uniform(0.0, 1.0, size=n),
            "X2": rng.uniform(0.0, 1.0, size=n),
            "X3": rng.uniform(0.0, 1.0, size=n),
            "X4": rng.uniform(0.0, 1.0, size=n),
            "X5": rng.uniform(0.0, 1.0, size=n),
            "X6": rng.uniform(0.0, 1.0, size=n),
        }
    )

    nonlinear_x1_sq = features["X1"] ** 2
    step_x2 = (features["X2"] > 0.5).astype(float)
    interaction_x3_x4 = features["X3"] * features["X4"]
    signal = nonlinear_x1_sq + 2.0 * step_x2 + 2.0 * interaction_x3_x4

    y, epsilon, latent_score, cutoff_value, snr_empirical = apply_beta_signal(
        signal,
        beta,
        rng,
        noise_std=noise_std,
        cutoff="median",
    )

    terms = pd.DataFrame(
        {
            "nonlinear_x1_sq": nonlinear_x1_sq,
            "step_x2": step_x2,
            "interaction_x3_x4": interaction_x3_x4,
            "signal": signal,
            "latent_score": latent_score,
        }
    )

    return build_simulated_dataset(
        features,
        terms,
        y,
        epsilon,
        dgp_name="custom_nonlinear",
        relevant_features=("X1", "X2", "X3", "X4"),
        beta=beta,
        snr_empirical=snr_empirical,
        classification_cutoff=cutoff_value,
    )


def simulate_custom_sin_log(n=1000, beta=1.0, noise_std=1.0, seed=42):
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {
            "X1": rng.uniform(0.0, 1.0, size=n),
            "X2": rng.uniform(0.0, 1.0, size=n),
            "X3": rng.uniform(0.0, 1.0, size=n),
            "X4": rng.uniform(0.0, 1.0, size=n),
            "X5": rng.uniform(0.0, 1.0, size=n),
            "X6": rng.uniform(0.0, 1.0, size=n),
        }
    )

    sin_x1 = 1.8 * np.sin(np.pi * features["X1"])
    log_x3 = 1.4 * np.log1p(3.0 * features["X3"])
    interaction_x2_x4 = 1.6 * (features["X2"] * features["X4"])
    signal = sin_x1 + log_x3 + interaction_x2_x4

    y, epsilon, latent_score, cutoff_value, snr_empirical = apply_beta_signal(
        signal,
        beta,
        rng,
        noise_std=noise_std,
        cutoff="median",
    )

    terms = pd.DataFrame(
        {
            "sin_x1": sin_x1,
            "log_x3": log_x3,
            "interaction_x2_x4": interaction_x2_x4,
            "signal": signal,
            "latent_score": latent_score,
        }
    )

    return build_simulated_dataset(
        features,
        terms,
        y,
        epsilon,
        dgp_name="custom_sin_log",
        relevant_features=("X1", "X2", "X3", "X4"),
        beta=beta,
        snr_empirical=snr_empirical,
        classification_cutoff=cutoff_value,
    )


def simulate_chen(n=1000, beta=1.0, noise_std=1.0, seed=42):
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {f"X{i}": rng.normal(0.0, 1.0, size=n) for i in range(1, 11)}
    )

    minus_2_sin_x1 = -2.0 * np.sin(features["X1"])
    max_x2_0 = np.maximum(features["X2"], 0.0)
    linear_x3 = features["X3"]
    exp_minus_x4 = np.exp(-features["X4"])
    signal = minus_2_sin_x1 + max_x2_0 + linear_x3 + exp_minus_x4

    y, epsilon, latent_score, cutoff_value, snr_empirical = apply_beta_signal(
        signal,
        beta,
        rng,
        noise_std=noise_std,
        cutoff="median",
    )

    terms = pd.DataFrame(
        {
            "minus_2_sin_x1": minus_2_sin_x1,
            "max_x2_0": max_x2_0,
            "linear_x3": linear_x3,
            "exp_minus_x4": exp_minus_x4,
            "signal": signal,
            "latent_score": latent_score,
        }
    )

    return build_simulated_dataset(
        features,
        terms,
        y,
        epsilon,
        dgp_name="chen",
        relevant_features=("X1", "X2", "X3", "X4"),
        beta=beta,
        snr_empirical=snr_empirical,
        classification_cutoff=cutoff_value,
    )


def simulate_friedman(n=1000, beta=1.0, noise_std=1.0, seed=42):
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {f"X{i}": rng.uniform(0.0, 1.0, size=n) for i in range(1, 7)}
    )

    interaction_x1_x2 = 10.0 * np.sin(np.pi * features["X1"] * features["X2"])
    quadratic_x3 = 20.0 * (features["X3"] - 0.5) ** 2
    linear_x4 = 10.0 * features["X4"]
    linear_x5 = 5.0 * features["X5"]
    signal = interaction_x1_x2 + quadratic_x3 + linear_x4 + linear_x5

    y, epsilon, latent_score, cutoff_value, snr_empirical = apply_beta_signal(
        signal,
        beta,
        rng,
        noise_std=noise_std,
        cutoff="median",
    )

    terms = pd.DataFrame(
        {
            "interaction_x1_x2": interaction_x1_x2,
            "quadratic_x3": quadratic_x3,
            "linear_x4": linear_x4,
            "linear_x5": linear_x5,
            "signal": signal,
            "latent_score": latent_score,
        }
    )

    return build_simulated_dataset(
        features,
        terms,
        y,
        epsilon,
        dgp_name="friedman",
        relevant_features=("X1", "X2", "X3", "X4", "X5"),
        beta=beta,
        snr_empirical=snr_empirical,
        classification_cutoff=cutoff_value,
    )


def _add_redundant_duplicate(features, rng, source_col, dup_col, redundancy_corr):
    """Add a correlated duplicate of `source_col` that is NOT used in the
    outcome equation -- a proxy feature that's statistically substitutable
    for a true driver but isn't itself causally part of the DGP."""

    dup_noise = rng.normal(0.0, 1.0, size=len(features))
    features[dup_col] = (
        redundancy_corr * features[source_col] + np.sqrt(1.0 - redundancy_corr**2) * dup_noise
    )
    return features


def simulate_chen_redundant(n=1000, beta=1.0, noise_std=1.0, seed=42, redundancy_corr=0.95):
    """Same DGP as simulate_chen, plus a correlated duplicate of the true
    driver X1 (X1_dup) that the outcome equation never touches. Tests whether
    a method correctly shares importance/credit across a redundant pair
    instead of arbitrarily crediting only one."""

    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {f"X{i}": rng.normal(0.0, 1.0, size=n) for i in range(1, 11)}
    )
    features = _add_redundant_duplicate(features, rng, "X1", "X1_dup", redundancy_corr)

    minus_2_sin_x1 = -2.0 * np.sin(features["X1"])
    max_x2_0 = np.maximum(features["X2"], 0.0)
    linear_x3 = features["X3"]
    exp_minus_x4 = np.exp(-features["X4"])
    signal = minus_2_sin_x1 + max_x2_0 + linear_x3 + exp_minus_x4

    y, epsilon, latent_score, cutoff_value, snr_empirical = apply_beta_signal(
        signal,
        beta,
        rng,
        noise_std=noise_std,
        cutoff="median",
    )

    terms = pd.DataFrame(
        {
            "minus_2_sin_x1": minus_2_sin_x1,
            "max_x2_0": max_x2_0,
            "linear_x3": linear_x3,
            "exp_minus_x4": exp_minus_x4,
            "signal": signal,
            "latent_score": latent_score,
        }
    )

    return build_simulated_dataset(
        features,
        terms,
        y,
        epsilon,
        dgp_name=f"chen_redundant_r{round(redundancy_corr * 100):03d}",
        relevant_features=("X1", "X2", "X3", "X4"),
        beta=beta,
        snr_empirical=snr_empirical,
        classification_cutoff=cutoff_value,
        equivalence_classes={"X1": ("X1", "X1_dup")},
    )


def simulate_friedman_redundant(n=1000, beta=1.0, noise_std=1.0, seed=42, redundancy_corr=0.95):
    """Same DGP as simulate_friedman, plus a correlated duplicate of the true
    driver X1 (X1_dup, X1 being part of the X1*X2 interaction term)."""

    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {f"X{i}": rng.uniform(0.0, 1.0, size=n) for i in range(1, 7)}
    )
    features = _add_redundant_duplicate(features, rng, "X1", "X1_dup", redundancy_corr)

    interaction_x1_x2 = 10.0 * np.sin(np.pi * features["X1"] * features["X2"])
    quadratic_x3 = 20.0 * (features["X3"] - 0.5) ** 2
    linear_x4 = 10.0 * features["X4"]
    linear_x5 = 5.0 * features["X5"]
    signal = interaction_x1_x2 + quadratic_x3 + linear_x4 + linear_x5

    y, epsilon, latent_score, cutoff_value, snr_empirical = apply_beta_signal(
        signal,
        beta,
        rng,
        noise_std=noise_std,
        cutoff="median",
    )

    terms = pd.DataFrame(
        {
            "interaction_x1_x2": interaction_x1_x2,
            "quadratic_x3": quadratic_x3,
            "linear_x4": linear_x4,
            "linear_x5": linear_x5,
            "signal": signal,
            "latent_score": latent_score,
        }
    )

    return build_simulated_dataset(
        features,
        terms,
        y,
        epsilon,
        dgp_name=f"friedman_redundant_r{round(redundancy_corr * 100):03d}",
        relevant_features=("X1", "X2", "X3", "X4", "X5"),
        beta=beta,
        snr_empirical=snr_empirical,
        classification_cutoff=cutoff_value,
        equivalence_classes={"X1": ("X1", "X1_dup")},
    )


def sample_monk_features(n, rng):
    monk_domains = {
        "X1": np.array([1, 2, 3]),
        "X2": np.array([1, 2, 3]),
        "X3": np.array([1, 2]),
        "X4": np.array([1, 2, 3]),
        "X5": np.array([1, 2, 3, 4]),
        "X6": np.array([1, 2]),
    }
    return pd.DataFrame(
        {name: rng.choice(domain, size=n) for name, domain in monk_domains.items()}
    )


def simulate_monk1(n=1000, beta=1.0, noise_std=1.0, seed=42):
    rng = np.random.default_rng(seed)
    features = sample_monk_features(n, rng)

    x1_eq_x2 = (features["X1"] == features["X2"]).astype(float)
    x5_eq_1 = (features["X5"] == 1).astype(float)
    signal = np.maximum(x1_eq_x2, x5_eq_1)

    y, epsilon, latent_score, cutoff_value, snr_empirical = apply_beta_signal(
        signal,
        beta,
        rng,
        noise_std=noise_std,
        cutoff="median",
    )

    terms = pd.DataFrame(
        {
            "x1_eq_x2": x1_eq_x2,
            "x5_eq_1": x5_eq_1,
            "signal": signal,
            "latent_score": latent_score,
        }
    )

    return build_simulated_dataset(
        features,
        terms,
        y,
        epsilon,
        dgp_name="monk1",
        relevant_features=("X1", "X2", "X5"),
        beta=beta,
        snr_empirical=snr_empirical,
        classification_cutoff=cutoff_value,
    )


def simulate_monk3(n=1000, beta=1.0, noise_std=1.0, seed=42):
    rng = np.random.default_rng(seed)
    features = sample_monk_features(n, rng)

    x5_eq_3_and_x4_eq_1 = (
        (features["X5"] == 3) & (features["X4"] == 1)
    ).astype(float)
    x5_ne_4_and_x2_ne_3 = (
        (features["X5"] != 4) & (features["X2"] != 3)
    ).astype(float)
    signal = np.maximum(x5_eq_3_and_x4_eq_1, x5_ne_4_and_x2_ne_3)

    y, epsilon, latent_score, cutoff_value, snr_empirical = apply_beta_signal(
        signal,
        beta,
        rng,
        noise_std=noise_std,
        cutoff="median",
    )

    terms = pd.DataFrame(
        {
            "x5_eq_3_and_x4_eq_1": x5_eq_3_and_x4_eq_1,
            "x5_ne_4_and_x2_ne_3": x5_ne_4_and_x2_ne_3,
            "signal": signal,
            "latent_score": latent_score,
        }
    )

    return build_simulated_dataset(
        features,
        terms,
        y,
        epsilon,
        dgp_name="monk3",
        relevant_features=("X2", "X4", "X5"),
        beta=beta,
        snr_empirical=snr_empirical,
        classification_cutoff=cutoff_value,
    )


SIMULATORS = {
    "chen": simulate_chen,
    "friedman": simulate_friedman,
    "monk1": simulate_monk1,
    "monk3": simulate_monk3,
    "custom_nonlinear": simulate_custom_nonlinear,
    "custom_sin_log": simulate_custom_sin_log,
}

# Redundant-driver variants: not included in SIMULATORS by default (opt in via
# --include-redundant-dgps) since they're a diagnostic addition, not part of
# the standard ground-truth-recovery sweep. Each duplicates one true driver
# (X1) as a correlated proxy feature at two redundancy strengths, to test
# whether a method shares importance/credit across the redundant pair instead
# of arbitrarily crediting only one (see compute_equivalence_recall and
# compute_credit_split_ratios below).
REDUNDANT_SIMULATORS = {
    "chen_redundant_r095": functools.partial(simulate_chen_redundant, redundancy_corr=0.95),
    "chen_redundant_r070": functools.partial(simulate_chen_redundant, redundancy_corr=0.70),
    "friedman_redundant_r095": functools.partial(simulate_friedman_redundant, redundancy_corr=0.95),
    "friedman_redundant_r070": functools.partial(simulate_friedman_redundant, redundancy_corr=0.70),
}

# Canonical DGP order. A DGP's per-cell seed is derived from its index here
# (not its position in whatever subset --dgps selected), so running a DGP alone
# reproduces exactly the datasets it would get in a full run.
ALL_SIMULATORS = {**SIMULATORS, **REDUNDANT_SIMULATORS}


def feature_columns_from_dataset(dataset):
    def sort_key(name):
        match = re.match(r"X(\d+)", name)
        numeric_part = int(match.group(1)) if match else 0
        return (numeric_part, name)

    return sorted(
        [column for column in dataset.columns if column.startswith("X")],
        key=sort_key,
    )


def compute_selection_metrics(ranked_features, ground_truth):
    k = len(ground_truth)
    gt_set = set(ground_truth)
    top_k = ranked_features[:k]
    hits = len(gt_set.intersection(top_k))

    dcg = sum(1.0 / np.log2(i + 2) for i, feature in enumerate(top_k) if feature in gt_set)
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(k))
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    rank_lookup = {feature: idx + 1 for idx, feature in enumerate(ranked_features)}
    mean_gt_rank = float(np.mean([rank_lookup[feature] for feature in ground_truth]))

    return {
        "precision_at_k": hits / k,
        "recall_at_k": hits / len(ground_truth),
        "ndcg_at_k": ndcg,
        "exact_match": int(set(top_k) == gt_set),
        "mean_gt_rank": mean_gt_rank,
    }


def compute_equivalence_recall(ranked_features, ground_truth, equivalence_classes):
    """Recall@k that credits a hit if the top-k contains EITHER the literal
    ground-truth feature or any of its substitutable duplicates. Fair to
    methods that pick a statistically-interchangeable proxy instead of the
    "official" true-driver name. Returns None when there's no redundancy to
    account for (equivalence_classes empty)."""

    if not equivalence_classes:
        return None

    k = len(ground_truth)
    top_k = set(ranked_features[:k])
    hits = 0
    for true_feature in ground_truth:
        group = equivalence_classes.get(true_feature, (true_feature,))
        if any(member in top_k for member in group):
            hits += 1
    return hits / len(ground_truth)


def compute_credit_split_ratios(ranked_features, equivalence_classes):
    """For each redundant pair, how evenly is importance "credit" shared
    between its members? Uses inverse rank position (1/rank) as a
    method-agnostic importance proxy, since it's the one quantity every
    method here (RID or stepwise) produces via its full feature ranking.

    Ratio near 1.0 = credit shared evenly across the redundant pair (RID's
    expected behavior: several Rashomon-set models arbitrate the pair
    differently, so both retain some importance). Ratio near 0.0 =
    winner-take-all (a greedy/single-model method's expected behavior: pick
    one, then the other adds no further marginal value and drops far down
    the ranking)."""

    if not equivalence_classes:
        return {}

    rank_lookup = {feature: idx + 1 for idx, feature in enumerate(ranked_features)}
    ratios = {}
    for true_feature, group in equivalence_classes.items():
        importances = [1.0 / rank_lookup[member] for member in group if member in rank_lookup]
        if len(importances) < 2:
            ratios[true_feature] = None
            continue
        ratios[true_feature] = min(importances) / max(importances)
    return ratios


def run_cross_family_rid(
    X,
    y,
    *,
    n_bootstraps,
    epsilon,
    family_balance_mode,
    candidate_search,
    n_candidates_per_family=DEFAULT_N_CANDIDATES_PER_FAMILY,
    truvarimp_total_budget=DEFAULT_TRUVARIMP_TOTAL_BUDGET,
    n_jobs=1,
    rid_metric="sub_mr",
):
    """Fit cross-family RID with per-bootstrap search diagnostics; return the estimator.

    Every RID_MODEL_CONFIGS family is in the shared candidate pool, so
    ``n_models_per_class`` is never used and is left at its default.
    """

    estimator = CrossFamilyRashomonImportanceDistribution(
        model_configs=RID_MODEL_CONFIGS,
        epsilon=epsilon,
        n_bootstraps=n_bootstraps,
        vi_metrics=(rid_metric,),
        performance_metrics=RID_PERFORMANCE_METRICS,
        family_balance_mode=family_balance_mode,
        n_jobs=n_jobs,
        candidate_search=candidate_search,
        truvarimp_config={"total_budget": truvarimp_total_budget},
        n_candidates_per_family=n_candidates_per_family,
        record_search_diagnostics=True,
    )

    estimator.fit(X, y)
    return estimator


def _score_candidate_feature(feature, selected_features, X, y, model_cls, model_kwargs, cv, scoring):
    feature_set = list(selected_features) + [feature]
    estimator = model_cls(**model_kwargs)
    scores = cross_val_score(
        estimator, X[feature_set], y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise"
    )
    return feature, float(np.mean(scores))


def run_forward_stepwise_ranking(X, y, model_cls, model_kwargs, cv, scoring, n_jobs=1):
    """Rank every feature via greedy forward selection (a full ranking, not just top-k)."""

    selected_features = []
    all_features = list(X.columns)

    for _ in range(len(all_features)):
        remaining = [feature for feature in all_features if feature not in selected_features]
        scores = Parallel(n_jobs=n_jobs)(
            delayed(_score_candidate_feature)(
                feature, tuple(selected_features), X, y, model_cls, model_kwargs, cv, scoring
            )
            for feature in remaining
        )
        best_feature, _ = max(scores, key=lambda item: item[1])
        selected_features.append(best_feature)

    return selected_features


def run_single_family_tree_rid(
    X,
    y,
    *,
    epsilon,
    n_bootstraps,
    n_jobs=1,
    rid_metric="sub_mr",
):
    # n_models_pool is left at its default: the tree grid has refit_seeds=False,
    # so each config is fit exactly once whatever it is set to.
    estimator = RashomonImportanceDistribution(
        epsilon=epsilon,
        n_bootstraps=n_bootstraps,
        model_class=FullyEnumeratedTreeClassifier,
        vi_metrics=(rid_metric,),
        performance_metrics=RID_PERFORMANCE_METRICS,
        n_jobs=n_jobs,
    )

    estimator.fit(X, y)

    ranked_features = [feature for feature, _ in estimator.rank_features(rid_metric)]
    return ranked_features


def run_methods_with_logging(method_runners, cell_label):
    """Run each method's zero-arg runner in order, logging start/finish and wall time.

    Returns ``(method_rankings, method_seconds)``, both keyed by method name in
    the runners' order.
    """

    method_rankings = {}
    method_seconds = {}
    for method_name, runner in method_runners.items():
        logger.info(f"{cell_label}: starting {method_name}")
        method_start = time.time()
        method_rankings[method_name] = runner()
        method_seconds[method_name] = time.time() - method_start
        logger.info(f"{cell_label}: {method_name} done in {format_duration(method_seconds[method_name])}")
    return method_rankings, method_seconds


def log_cell_progress(completed, total_cells, cell_label, method_seconds, start):
    elapsed = time.time() - start
    eta = elapsed / completed * (total_cells - completed)
    timings = ", ".join(f"{name}={format_duration(sec)}" for name, sec in method_seconds.items())
    logger.info(
        f"progress {completed}/{total_cells} ({100.0 * completed / total_cells:.1f}%) "
        f"{cell_label} | {timings} | elapsed={format_duration(elapsed)} "
        f"ETA={format_duration(eta)}"
    )


def build_common_evaluator_panel(random_state):
    """Three fixed evaluators spanning distinct inductive biases, so a
    feature set's held-out score isn't an artifact of one model's blind spots
    -- the same three models as the Falcon-Cano/Staellert top-k comparison
    scripts' build_common_evaluator_panel, but NOT the same panel since
    2026-11-06: those scripts now standardize features inside each fold for
    logreg and svm_rbf (their descriptor/marker scales span 4-7 orders of
    magnitude), while this one doesn't. Every simulated feature in a DGP shares
    one distribution (uniform(0,1) or N(0,1)), so scaling matters little here,
    but held-out logreg/svm_rbf numbers aren't strictly comparable across them."""

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


def _evaluate_feature_set(X, y, features, model_cls, model_kwargs, cv, scoring):
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


def run_common_evaluation_panel(X, y, method_feature_lists, evaluators, cv, scoring):
    """Evaluate every method's selected top-k features under every evaluator
    in the panel, so methods are compared on equal footing regardless of which
    model (if any) they used internally to select features."""

    return {
        name: {
            evaluator_name: _evaluate_feature_set(X, y, features, model_cls, model_kwargs, cv, scoring)
            for evaluator_name, model_cls, model_kwargs in evaluators
        }
        for name, features in method_feature_lists.items()
    }


def build_held_out_scores(method_eval):
    """Flatten one method's ``run_common_evaluation_panel`` entry into the
    ``HELD_OUT_SCORE_COLUMNS`` fields of a study_results row."""

    held_out_scores = {
        f"held_out_cv_score_{stat}__{evaluator_name}": method_eval[evaluator_name][f"cv_score_{stat}"]
        for evaluator_name in COMMON_EVALUATOR_NAMES
        for stat in ("mean", "std")
    }
    held_out_scores["held_out_cv_score_mean_avg"] = float(
        np.mean([method_eval[name]["cv_score_mean"] for name in COMMON_EVALUATOR_NAMES])
    )
    return held_out_scores


def build_method_runners(
    X,
    y,
    cv,
    scoring,
    fitted_estimators,
    *,
    n_bootstraps,
    n_candidates_per_family,
    truvarimp_total_budget,
    epsilon,
    stepwise_logreg_C,
    stepwise_rf_n_estimators,
    random_state,
    n_jobs=1,
    rid_metric="sub_mr",
):
    """Zero-arg runner per method in ``METHODS`` order, each returning a full feature ranking.

    The cross-family runners also store their fitted estimator in
    ``fitted_estimators[method]``, for ``compute_search_coverage``.
    """

    def cross_family_runner(method_name):
        def run():
            estimator = run_cross_family_rid(
                X,
                y,
                n_bootstraps=n_bootstraps,
                epsilon=epsilon,
                family_balance_mode="unweighted",
                candidate_search=CROSS_FAMILY_SEARCH_MODES[method_name],
                n_candidates_per_family=n_candidates_per_family,
                truvarimp_total_budget=truvarimp_total_budget,
                n_jobs=n_jobs,
                rid_metric=rid_metric,
            )
            fitted_estimators[method_name] = estimator
            return [feature for feature, _ in estimator.rank_features(rid_metric)]

        return run

    return {
        GRID_METHOD: cross_family_runner(GRID_METHOD),
        TRUVARIMP_METHOD: cross_family_runner(TRUVARIMP_METHOD),
        "stepwise_logreg": lambda: run_forward_stepwise_ranking(
            X,
            y,
            LogisticRegression,
            {"C": stepwise_logreg_C, "max_iter": 100000, "random_state": random_state},
            cv,
            scoring,
        ),
        "stepwise_rf": lambda: run_forward_stepwise_ranking(
            X,
            y,
            RandomForestClassifier,
            {"n_estimators": stepwise_rf_n_estimators, "random_state": random_state, "n_jobs": 1},
            cv,
            scoring,
        ),
        "rid_tree": lambda: run_single_family_tree_rid(
            X,
            y,
            epsilon=epsilon,
            n_bootstraps=n_bootstraps,
            n_jobs=n_jobs,
            rid_metric=rid_metric,
        ),
    }


def _search_keys(evaluated_or_members):
    """``{family: indices}`` (or ``{family: (indices, losses)}``) -> set of (family, index) keys."""

    keys = set()
    for family, value in evaluated_or_members.items():
        indices = value[0] if isinstance(value, tuple) else value
        keys.update((family, int(index)) for index in indices)
    return keys


def _key_losses(evaluated):
    return {
        (family, int(index)): float(loss)
        for family, (indices, losses) in evaluated.items()
        for index, loss in zip(indices, losses)
    }


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def _nanmean(values):
    values = [value for value in values if value is not None and not np.isnan(value)]
    return float(np.mean(values)) if values else np.nan


def compute_search_coverage(grid_estimator, truvarimp_estimator, rid_metric, top_k):
    """TruVaRImp-vs-grid coverage for one cell, aggregated over bootstraps.

    Both estimators searched the identical per-bootstrap pool, keyed by
    (family, candidate_index), so the grid's Rashomon set G is the exact ground
    truth for TruVaRImp's set TV. Only bootstraps with a record in both modes
    count (both modes share the bootstrap split, so they're valid together).
    """

    families = list(grid_estimator.family_counts_)
    per_boot = []
    for grid_rec, tv_rec in zip(grid_estimator.search_diagnostics_, truvarimp_estimator.search_diagnostics_):
        if grid_rec is None or tv_rec is None:
            continue
        grid_members = _search_keys(grid_rec["rashomon"])
        tv_members = _search_keys(tv_rec["rashomon"])
        grid_losses = _key_losses(grid_rec["evaluated"])
        tv_losses = _key_losses(tv_rec["evaluated"])
        shared = grid_losses.keys() & tv_losses.keys()
        grid_best = {key for key, loss in grid_losses.items() if loss == grid_rec["min_loss"]}
        overlap = grid_members & tv_members
        row = {
            "n_overlap": len(overlap),
            "n_grid": len(grid_members),
            "n_tv": len(tv_members),
            "recall": _safe_ratio(len(overlap), len(grid_members)),
            "precision": _safe_ratio(len(overlap), len(tv_members)),
            "min_loss_gap": tv_rec["min_loss"] - grid_rec["min_loss"],
            "found_grid_best": float(bool(grid_best & tv_losses.keys())),
            "loss_mismatch": max((abs(grid_losses[k] - tv_losses[k]) for k in shared), default=0.0),
            "grid_fits": grid_rec["n_fits"],
            "tv_fits": tv_rec["n_fits"],
            "grid_seconds": grid_rec["seconds"],
            "tv_seconds": tv_rec["seconds"],
            "grid_search_seconds": grid_rec["search_seconds"],
            "tv_search_seconds": tv_rec["search_seconds"],
        }
        for family in families:
            g_f = {key for key in grid_members if key[0] == family}
            t_f = {key for key in tv_members if key[0] == family}
            row[f"recall__{family}"] = _safe_ratio(len(g_f & t_f), len(g_f))
            row[f"grid_members__{family}"] = len(g_f)
            row[f"truvarimp_members__{family}"] = len(t_f)
        per_boot.append(row)

    coverage = {"n_bootstraps_compared": len(per_boot)}
    if per_boot:
        frame = pd.DataFrame(per_boot)
        coverage.update(
            {
                "recall_mean": _nanmean(frame["recall"]),
                "recall_min": float(frame["recall"].min()),
                "recall_pooled": _safe_ratio(frame["n_overlap"].sum(), frame["n_grid"].sum()),
                "precision_mean": _nanmean(frame["precision"]),
                "precision_min": float(frame["precision"].min()),
                "precision_pooled": _safe_ratio(frame["n_overlap"].sum(), frame["n_tv"].sum()),
                "min_loss_gap_mean": float(frame["min_loss_gap"].mean()),
                "min_loss_gap_max": float(frame["min_loss_gap"].max()),
                "found_grid_best_rate": float(frame["found_grid_best"].mean()),
                "loss_mismatch_max": float(frame["loss_mismatch"].max()),
                "grid_rashomon_size_mean": float(frame["n_grid"].mean()),
                "truvarimp_rashomon_size_mean": float(frame["n_tv"].mean()),
                "grid_fits_mean": float(frame["grid_fits"].mean()),
                "truvarimp_fits_mean": float(frame["tv_fits"].mean()),
                "grid_seconds_mean": float(frame["grid_seconds"].mean()),
                "truvarimp_seconds_mean": float(frame["tv_seconds"].mean()),
                "grid_search_seconds_mean": float(frame["grid_search_seconds"].mean()),
                "truvarimp_search_seconds_mean": float(frame["tv_search_seconds"].mean()),
            }
        )
        coverage["fit_ratio"] = _safe_ratio(coverage["truvarimp_fits_mean"], coverage["grid_fits_mean"])
        coverage["time_ratio"] = _safe_ratio(coverage["truvarimp_seconds_mean"], coverage["grid_seconds_mean"])
        for family in families:
            coverage[f"recall__{family}"] = _nanmean(frame[f"recall__{family}"])
            coverage[f"grid_members__{family}"] = float(frame[f"grid_members__{family}"].mean())
            coverage[f"truvarimp_members__{family}"] = float(frame[f"truvarimp_members__{family}"].mean())

    # RID output agreement between the two modes.
    if grid_estimator.metric_results_ and truvarimp_estimator.metric_results_:
        grid_summary = grid_estimator.metric_summary(rid_metric)
        tv_summary = truvarimp_estimator.metric_summary(rid_metric)
        features = list(grid_summary)
        grid_imp = np.array([grid_summary[f]["expected_importance"] for f in features])
        tv_imp = np.array([tv_summary[f]["expected_importance"] for f in features])
        grid_rank = [f for f, _ in grid_estimator.rank_features(rid_metric)]
        tv_rank = [f for f, _ in truvarimp_estimator.rank_features(rid_metric)]
        coverage.update(
            {
                "importance_spearman": float(spearmanr(grid_imp, tv_imp)[0]),
                "top_k": top_k,
                "top_k_overlap": len(set(grid_rank[:top_k]) & set(tv_rank[:top_k])) / top_k,
                "max_abs_delta_prob_positive": float(
                    max(abs(grid_summary[f]["prob_positive"] - tv_summary[f]["prob_positive"]) for f in features)
                ),
                "max_abs_delta_expected_importance": float(np.max(np.abs(grid_imp - tv_imp))),
                "rankings_identical": int(grid_rank == tv_rank),
            }
        )
    return coverage


def _beta_tag(beta):
    return f"beta{float(beta):g}"


def per_dgp_results_path(output_dir, dgp_name, beta):
    return os.path.join(output_dir, PER_DGP_SUBDIR, f"{STUDY_RESULTS_PREFIX}{dgp_name}__{_beta_tag(beta)}.csv")


def per_dgp_coverage_path(output_dir, dgp_name, beta):
    return os.path.join(output_dir, PER_DGP_SUBDIR, f"{SEARCH_COVERAGE_PREFIX}{dgp_name}__{_beta_tag(beta)}.csv")


def write_csv_atomic(df, path):
    """Write via a temp file + rename so a job killed mid-write (or a concurrent
    array task) never leaves a truncated CSV behind."""

    tmp_path = f"{path}.{os.getpid()}.tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def load_completed_cells(results_path, methods):
    """Rows of every (beta, rep) cell already checkpointed in one (DGP, beta)
    file with all ``methods`` present, plus the set of those ``(beta, rep)`` keys."""

    if not os.path.exists(results_path):
        return [], set()

    existing = pd.read_csv(results_path)
    existing["beta"] = existing["beta"].astype(float)
    complete = existing.groupby(["beta", "rep"])["method"].apply(lambda m: set(methods) <= set(m))
    done_cells = {(float(beta), int(rep)) for (beta, rep), ok in complete.items() if ok}
    keep = [(float(b), int(r)) in done_cells for b, r in zip(existing["beta"], existing["rep"])]
    return existing[keep].to_dict("records"), done_cells


def load_coverage_rows(coverage_path, done_cells):
    """Coverage rows of cells whose study results are complete. Coverage is written
    before study results, so a cell interrupted between the two is rerun and its
    stale coverage row dropped here."""

    if not os.path.exists(coverage_path):
        return []
    existing = pd.read_csv(coverage_path)
    keep = [(float(b), int(r)) in done_cells for b, r in zip(existing["beta"], existing["rep"])]
    return existing[keep].to_dict("records")


def prepare_output_dir(output_dir, study_settings):
    """Create ``output_dir`` and write ``study_settings.json``, refusing to mix
    checkpointed per-DGP results produced under different settings.

    ``num_workers`` and the selected DGP/beta subsets are excluded from the
    check (they are not settings keys at all), since none changes any cell's result.
    """

    os.makedirs(os.path.join(output_dir, PER_DGP_SUBDIR), exist_ok=True)
    settings_path = os.path.join(output_dir, "study_settings.json")
    ignored_keys = {"num_workers"}

    has_checkpoints = any(
        name.endswith(".csv") for name in os.listdir(os.path.join(output_dir, PER_DGP_SUBDIR))
    )
    if has_checkpoints and os.path.exists(settings_path):
        with open(settings_path, encoding="utf-8") as handle:
            previous = json.load(handle)
        mismatched = sorted(
            key
            for key in (set(previous) | set(study_settings)) - ignored_keys
            if previous.get(key) != study_settings.get(key)
        )
        if mismatched:
            raise SystemExit(
                f"{output_dir} already holds per-DGP checkpoints produced with different "
                f"settings ({', '.join(mismatched)}). Use a new --output-dir, or delete "
                f"{os.path.join(output_dir, PER_DGP_SUBDIR)} to start over."
            )

    tmp_path = f"{settings_path}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(study_settings, handle, indent=2)
    os.replace(tmp_path, settings_path)


def run_simulation_study(
    simulators,
    *,
    beta_grid,
    betas=None,
    sample_size,
    n_repetitions,
    noise_std,
    base_seed,
    stepwise_cv_splits,
    random_state,
    make_method_runners,
    output_dir,
    rid_metric,
):
    """Run every (dgp, beta, rep) cell, checkpointing each (DGP, beta)'s rows to
    its own CSVs after every cell and skipping cells already checkpointed in
    ``output_dir``.

    ``betas`` (default: all of ``beta_grid``) restricts the run to a subset of
    ``beta_grid``; each cell's seed still comes from its beta's index in the full
    ``beta_grid``, so a subset reproduces the full run's datasets.

    ``make_method_runners(X, y, cv, scoring, fitted_estimators)`` returns the
    ``{method: runner}`` mapping for one cell (see ``build_method_runners``).
    """

    start = time.time()
    evaluator_panel = build_common_evaluator_panel(random_state)
    stepwise_scoring = "roc_auc"
    selected_betas = [beta for beta in beta_grid if betas is None or beta in betas]

    checkpoints = {
        (dgp_name, beta): load_completed_cells(per_dgp_results_path(output_dir, dgp_name, beta), METHODS)
        for dgp_name in simulators
        for beta in selected_betas
    }
    n_skipped = sum(len(done_cells) for _, done_cells in checkpoints.values())
    total_cells = len(simulators) * len(selected_betas) * n_repetitions - n_skipped
    if n_skipped:
        logger.info(f"Resuming: {n_skipped} cells already checkpointed, {total_cells} remaining")
    completed = 0

    for dgp_name, simulator in simulators.items():
        dgp_index = list(ALL_SIMULATORS).index(dgp_name)

        for beta_index, beta in enumerate(beta_grid):
            if beta not in selected_betas:
                continue
            results_path = per_dgp_results_path(output_dir, dgp_name, beta)
            coverage_path = per_dgp_coverage_path(output_dir, dgp_name, beta)
            rows, done_cells = checkpoints[(dgp_name, beta)]
            coverage_rows = load_coverage_rows(coverage_path, done_cells)

            for rep in range(n_repetitions):
                if (float(beta), rep) in done_cells:
                    continue

                seed = base_seed + dgp_index * 1_000_000 + beta_index * 10_000 + rep
                dataset = simulator(
                    n=sample_size,
                    beta=beta,
                    noise_std=noise_std,
                    seed=seed,
                )

                feature_columns = feature_columns_from_dataset(dataset)
                X = dataset[feature_columns]
                y = dataset["Y"].astype(int)
                ground_truth = tuple(dataset.attrs["relevant_features"])
                equivalence_classes = dataset.attrs.get("equivalence_classes") or {}
                snr_empirical = float(dataset.attrs["snr_empirical"])

                cv = StratifiedKFold(n_splits=stepwise_cv_splits, shuffle=True, random_state=random_state)

                cell_label = f"[{dgp_name} beta={beta} rep={rep}]"
                fitted_estimators = {}
                method_runners = make_method_runners(X, y, cv, stepwise_scoring, fitted_estimators)
                method_rankings, method_seconds = run_methods_with_logging(method_runners, cell_label)

                method_feature_lists = {
                    method_name: ranked_features[: len(ground_truth)]
                    for method_name, ranked_features in method_rankings.items()
                }
                common_eval = run_common_evaluation_panel(
                    X, y, method_feature_lists, evaluator_panel, cv, stepwise_scoring
                )

                for method_name, ranked_features in method_rankings.items():
                    metric_row = compute_selection_metrics(ranked_features, ground_truth)

                    equivalence_recall = compute_equivalence_recall(
                        ranked_features, ground_truth, equivalence_classes
                    )
                    credit_split_ratios = compute_credit_split_ratios(ranked_features, equivalence_classes)
                    valid_ratios = [v for v in credit_split_ratios.values() if v is not None]
                    credit_split_ratio_mean = float(np.mean(valid_ratios)) if valid_ratios else None

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
                            **build_held_out_scores(common_eval[method_name]),
                        }
                    )

                coverage = compute_search_coverage(
                    fitted_estimators[GRID_METHOD],
                    fitted_estimators[TRUVARIMP_METHOD],
                    rid_metric,
                    len(ground_truth),
                )
                coverage_rows.append(
                    {
                        "dgp": dgp_name,
                        "beta": float(beta),
                        "rep": rep,
                        "snr_empirical": snr_empirical,
                        "grid_method_seconds": method_seconds[GRID_METHOD],
                        "truvarimp_method_seconds": method_seconds[TRUVARIMP_METHOD],
                        **coverage,
                    }
                )
                logger.info(
                    f"{cell_label}: TruVaRImp vs grid: recall={coverage.get('recall_mean', np.nan):.3f} "
                    f"precision={coverage.get('precision_mean', np.nan):.3f} "
                    f"min_loss_gap={coverage.get('min_loss_gap_mean', np.nan):.4f} "
                    f"spearman={coverage.get('importance_spearman', np.nan):.3f} "
                    f"fits={coverage.get('truvarimp_fits_mean', np.nan):.0f}/{coverage.get('grid_fits_mean', np.nan):.0f}"
                )
                # Coverage first: study results decide which cells are done.
                write_csv_atomic(
                    pd.DataFrame(coverage_rows).sort_values(["beta", "rep"], kind="stable"), coverage_path
                )
                write_csv_atomic(pd.DataFrame(rows).sort_values(["beta", "rep"], kind="stable"), results_path)

                completed += 1
                log_cell_progress(completed, total_cells, cell_label, method_seconds, start)

            logger.info(f"[{dgp_name} beta={beta}] complete -> {results_path}")

        logger.info(f"[{dgp_name}] complete")
        write_combined_outputs(output_dir)


def write_combined_outputs(output_dir):
    """Rebuild the combined study_results/metric/SNR and search-coverage outputs from
    every per-(DGP, beta) checkpoint in ``output_dir`` (including those written by
    earlier or concurrent jobs)."""

    per_dgp_dir = os.path.join(output_dir, PER_DGP_SUBDIR)

    def checkpoint_paths(prefix):
        return sorted(
            os.path.join(per_dgp_dir, name)
            for name in os.listdir(per_dgp_dir)
            if name.startswith(prefix) and name.endswith(".csv")
        )

    per_dgp_paths = checkpoint_paths(STUDY_RESULTS_PREFIX)
    if not per_dgp_paths:
        logger.info("No per-DGP results yet; skipping combined outputs")
        return []

    study_results = pd.concat([pd.read_csv(path) for path in per_dgp_paths], ignore_index=True)
    snr_results = study_results.drop_duplicates(["dgp", "beta", "rep"])[["dgp", "beta", "rep", "snr_empirical"]]

    metric_table_by_dgp_beta, metric_table_by_dgp = build_metric_tables(study_results)
    snr_plot_table = build_snr_plot_table(snr_results)

    study_results_path = os.path.join(output_dir, "study_results.csv")
    dgp_beta_path = os.path.join(output_dir, "metric_table_by_dgp_beta.csv")
    dgp_path = os.path.join(output_dir, "metric_table_by_dgp.csv")
    snr_table_path = os.path.join(output_dir, "snr_plot_table.csv")
    snr_plot_path = os.path.join(output_dir, "snr_vs_beta.png")

    write_csv_atomic(study_results, study_results_path)
    write_csv_atomic(metric_table_by_dgp_beta, dgp_beta_path)
    write_csv_atomic(metric_table_by_dgp, dgp_path)
    write_csv_atomic(snr_plot_table.round(6), snr_table_path)
    save_snr_plot(snr_plot_table, snr_plot_path)
    output_paths = [study_results_path, dgp_beta_path, dgp_path, snr_table_path, snr_plot_path]

    coverage_paths = checkpoint_paths(SEARCH_COVERAGE_PREFIX)
    if coverage_paths:
        coverage = pd.concat([pd.read_csv(path) for path in coverage_paths], ignore_index=True)
        coverage = coverage.sort_values(["dgp", "beta", "rep"], kind="stable")
        by_dgp_beta, by_dgp, by_family = build_coverage_tables(coverage)
        coverage_outputs = {
            "search_coverage.csv": coverage,
            "search_coverage_by_dgp_beta.csv": by_dgp_beta,
            "search_coverage_by_dgp.csv": by_dgp,
            "search_coverage_by_family.csv": by_family,
        }
        for name, table in coverage_outputs.items():
            path = os.path.join(output_dir, name)
            write_csv_atomic(table, path)
            output_paths.append(path)
        output_paths.extend(save_coverage_plots(coverage, output_dir))

    return output_paths


# Per-cell search-coverage columns summarized in the aggregate tables/plots.
COVERAGE_SUMMARY_COLUMNS = [
    "recall_mean",
    "recall_min",
    "precision_mean",
    "precision_min",
    "min_loss_gap_mean",
    "min_loss_gap_max",
    "found_grid_best_rate",
    "loss_mismatch_max",
    "grid_rashomon_size_mean",
    "truvarimp_rashomon_size_mean",
    "grid_fits_mean",
    "truvarimp_fits_mean",
    "fit_ratio",
    "grid_seconds_mean",
    "truvarimp_seconds_mean",
    "time_ratio",
    "grid_method_seconds",
    "truvarimp_method_seconds",
    "importance_spearman",
    "top_k_overlap",
    "max_abs_delta_prob_positive",
    "rankings_identical",
]


def build_coverage_tables(coverage):
    """Mean/SEM of the coverage columns by (dgp, beta), means by dgp (plus an
    ``ALL`` row), and per-family recall/member counts in long form by dgp."""

    columns = [column for column in COVERAGE_SUMMARY_COLUMNS if column in coverage]
    by_dgp_beta = coverage.groupby(["dgp", "beta"])[columns].agg(["mean", "sem"]).reset_index()
    by_dgp_beta.columns = ["_".join(col).strip("_") for col in by_dgp_beta.columns.to_flat_index()]

    by_dgp = coverage.groupby("dgp")[columns].mean().reset_index()
    overall = coverage[columns].mean().to_frame().T
    overall.insert(0, "dgp", "ALL")
    by_dgp = pd.concat([by_dgp, overall], ignore_index=True)

    families = sorted(column.split("__", 1)[1] for column in coverage if column.startswith("recall__"))
    family_rows = []
    for dgp_name, group in [*coverage.groupby("dgp"), ("ALL", coverage)]:
        for family in families:
            family_rows.append(
                {
                    "dgp": dgp_name,
                    "family": family,
                    "recall_mean": group[f"recall__{family}"].mean(),
                    "grid_members_mean": group[f"grid_members__{family}"].mean(),
                    "truvarimp_members_mean": group[f"truvarimp_members__{family}"].mean(),
                }
            )
    by_family = pd.DataFrame(family_rows)

    for table in (by_dgp_beta, by_dgp, by_family):
        numeric_cols = table.select_dtypes(include=[np.number]).columns
        table[numeric_cols] = table[numeric_cols].round(4)
    return by_dgp_beta, by_dgp, by_family


def _plot_metric_panels(coverage, panels, title, output_path):
    """One panel per (column, label): mean +/- SEM across reps vs beta, one line per DGP."""

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.5), squeeze=False)
    for ax, (column, label) in zip(axes[0], panels):
        stats = coverage.groupby(["dgp", "beta"])[column].agg(["mean", "sem"]).reset_index()
        for dgp_name, group in stats.groupby("dgp"):
            group = group.sort_values("beta")
            ax.plot(group["beta"], group["mean"], marker="o", linewidth=2, label=dgp_name)
            ax.fill_between(
                group["beta"],
                group["mean"] - group["sem"].fillna(0.0),
                group["mean"] + group["sem"].fillna(0.0),
                alpha=0.15,
            )
        ax.set_xlabel("beta")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0][-1].legend(title="Simulation", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_coverage_plots(coverage, output_dir):
    paths = {
        "search_coverage_recall_precision.png": (
            [
                ("recall_mean", "Rashomon recall |TV & G| / |G|"),
                ("precision_mean", "Rashomon precision |TV & G| / |TV|"),
                ("found_grid_best_rate", "Share of bootstraps TruVaRImp fit the grid's best"),
            ],
            "TruVaRImp Rashomon-set coverage vs. exhaustive grid (same pool)",
        ),
        "search_rid_agreement.png": (
            [
                ("importance_spearman", "Spearman of expected_importance"),
                ("top_k_overlap", "Top-k overlap (k = # relevant features)"),
                ("max_abs_delta_prob_positive", "max |delta prob_positive|"),
            ],
            "RID output agreement: TruVaRImp vs. grid",
        ),
        "search_cost_ratio.png": (
            [
                ("fit_ratio", "Fits per bootstrap, TruVaRImp / grid"),
                ("time_ratio", "Seconds per bootstrap, TruVaRImp / grid"),
            ],
            "TruVaRImp cost relative to exhaustive grid",
        ),
    }
    written = []
    for name, (panels, title) in paths.items():
        panels = [(column, label) for column, label in panels if column in coverage]
        if not panels:
            continue
        path = os.path.join(output_dir, name)
        _plot_metric_panels(coverage, panels, title, path)
        written.append(path)
    return written


HELD_OUT_SCORE_COLUMNS = [
    f"held_out_cv_score_{stat}__{evaluator_name}"
    for evaluator_name in COMMON_EVALUATOR_NAMES
    for stat in ("mean", "std")
] + ["held_out_cv_score_mean_avg"]


def build_metric_tables(study_results):
    grouped = (
        study_results.groupby(["dgp", "beta", "method"])[
            [
                "snr_empirical",
                "precision_at_k",
                "recall_at_k",
                "ndcg_at_k",
                "exact_match",
                "mean_gt_rank",
                "equivalence_recall_at_k",
                "credit_split_ratio_mean",
                *HELD_OUT_SCORE_COLUMNS,
            ]
        ]
        .agg(["mean", "sem"])
        .reset_index()
    )
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.to_flat_index()]

    overall = (
        study_results.groupby(["dgp", "method"])
        [
            [
                "precision_at_k",
                "recall_at_k",
                "ndcg_at_k",
                "exact_match",
                "mean_gt_rank",
                "equivalence_recall_at_k",
                "credit_split_ratio_mean",
                *HELD_OUT_SCORE_COLUMNS,
            ]
        ]
        .mean()
        .reset_index()
    )

    for table in (grouped, overall):
        numeric_cols = table.select_dtypes(include=[np.number]).columns
        table[numeric_cols] = table[numeric_cols].round(3)

    return grouped, overall


def build_snr_plot_table(snr_results):
    return (
        snr_results.groupby(["dgp", "beta"])["snr_empirical"]
        .agg(["mean", "sem"])
        .reset_index()
        .sort_values(["dgp", "beta"])
    )


def save_snr_plot(snr_plot_table, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    for dgp_name, group in snr_plot_table.groupby("dgp"):
        group = group.sort_values("beta")
        ax.plot(group["beta"], group["mean"], marker="o", linewidth=2, label=dgp_name)
        ax.fill_between(
            group["beta"],
            group["mean"] - group["sem"],
            group["mean"] + group["sem"],
            alpha=0.15,
        )

    ax.set_xlabel("beta in latent score: beta * signal + epsilon")
    ax.set_ylabel("Empirical SNR = Var(beta * signal) / Var(epsilon)")
    ax.set_title("SNR vs beta across simulation families")
    ax.grid(alpha=0.3)
    ax.legend(title="Simulation", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate TruVaRImp candidate search against the exhaustive grid on simulated "
            "nonlinear-interaction datasets: cross-family RID runs in both modes over the "
            "identical shared candidate pool, so the grid's per-bootstrap Rashomon set is the "
            "exact ground truth for TruVaRImp's (membership recall/precision, min-loss gap, "
            "per-family recall, fits/time, RID output agreement). Also benchmarks ground-truth "
            "feature recovery across five methods (cross-family RID grid and TruVaRImp, stepwise "
            "logistic regression, stepwise random forest, single-family tree RID), and saves "
            "tables/plots."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for result tables and plot",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Sample size per simulation run",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help="Repetitions per beta per simulation",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=DEFAULT_BOOTSTRAPS,
        help="RID bootstraps per fit",
    )
    parser.add_argument(
        "--n-candidates-per-family",
        type=int,
        default=DEFAULT_N_CANDIDATES_PER_FAMILY,
        help=(
            "Shared candidate pool size per family per bootstrap (rid.candidate_pool), used by "
            "both cross-family methods: cross_family_unweighted fits all of it, "
            f"cross_family_unweighted_truvarimp searches it (default: {DEFAULT_N_CANDIDATES_PER_FAMILY})"
        ),
    )
    parser.add_argument(
        "--truvarimp-total-budget",
        type=int,
        default=DEFAULT_TRUVARIMP_TOTAL_BUDGET,
        help=(
            "Real fits per bootstrap, shared across all families, for "
            "cross_family_unweighted_truvarimp (default: "
            f"{DEFAULT_TRUVARIMP_TOTAL_BUDGET}). Even a budget above the pool size need not "
            "be spent: TruVaRImp stops once no unevaluated candidate is worth a fit"
        ),
    )
    parser.add_argument(
        "--beta-grid",
        type=float,
        nargs="+",
        default=DEFAULT_BETA_GRID,
        help=(
            "List of beta values used in latent score beta*signal+epsilon. Seeds derive from a "
            "beta's index in this list, so keep it fixed and use --betas to run a subset"
        ),
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Run only these betas (each must be in --beta-grid), e.g. one array task per "
            "(DGP, beta). Seeds still come from each beta's index in --beta-grid, so a subset "
            "reproduces the full run's datasets. Default: all of --beta-grid"
        ),
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=DEFAULT_NOISE_STD,
        help="Noise standard deviation for epsilon",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help="Base random seed",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="Rashomon epsilon",
    )
    parser.add_argument(
        "--rid-metric",
        type=str,
        default="sub_mr",
        help=(
            "RID variable-importance metric used by both cross-family methods and "
            "rid_tree (default: sub_mr; also available: loco, coef, iloco). DGP feature counts are tiny (6-11), so iloco's "
            "O(features^2) pairwise cost is negligible here, unlike on the real "
            "Falcon-Cano/Staellert datasets."
        ),
    )
    parser.add_argument(
        "--stepwise-cv-splits",
        type=int,
        default=DEFAULT_STEPWISE_CV_SPLITS,
        help="CV folds for the forward-stepwise comparison methods (default: 5)",
    )
    parser.add_argument(
        "--stepwise-logreg-C",
        type=float,
        default=DEFAULT_STEPWISE_LOGREG_C,
        help="Inverse regularization strength for the stepwise logistic regression method (default: 1.0)",
    )
    parser.add_argument(
        "--stepwise-rf-n-estimators",
        type=int,
        default=DEFAULT_STEPWISE_RF_N_ESTIMATORS,
        help="Random forest trees for the stepwise random-forest method (default: 100)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
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
    add_dgps_argument(parser)
    add_logging_arguments(parser)
    return parser.parse_args()


def add_dgps_argument(parser):
    parser.add_argument(
        "--dgps",
        type=str,
        nargs="+",
        choices=list(ALL_SIMULATORS),
        default=None,
        help=(
            "Run only these DGPs (any of the standard or redundant ones), e.g. to finish the "
            "DGPs a timed-out run didn't reach. Overrides --include-redundant-dgps. Cells "
            "already checkpointed in --output-dir are skipped either way."
        ),
    )


def select_simulators(dgps, include_redundant_dgps):
    if dgps:
        return {name: ALL_SIMULATORS[name] for name in dgps}
    simulators = dict(SIMULATORS)
    if include_redundant_dgps:
        simulators.update(REDUNDANT_SIMULATORS)
    return simulators


def main():
    args = parse_args()
    silence_warnings()
    configure_logging(args.log_level, args.log_file)
    n_jobs = resolve_num_workers(args.num_workers)

    simulators = select_simulators(args.dgps, args.include_redundant_dgps)
    unknown_betas = sorted(set(args.betas or []) - set(args.beta_grid))
    if unknown_betas:
        raise SystemExit(f"--betas {unknown_betas} not in --beta-grid {args.beta_grid}")

    study_settings = {
        "rashomon_evaluation": "out_of_bag",
        # Added 2026-11-12: both cross-family methods search the same shared pool
        # (rid.candidate_pool): grid fits all of it, TruVaRImp spends this budget on it.
        # rid_tree fits each grid config once.
        "candidate_search_modes": dict(CROSS_FAMILY_SEARCH_MODES),
        "n_candidates_per_family": args.n_candidates_per_family,
        "truvarimp_total_budget": args.truvarimp_total_budget,
        "grid_refits": "seed_sensitive_only",
        # Checkpoints are one file per (DGP, beta), with search-coverage rows alongside.
        "checkpoint_layout": "per_dgp_beta_with_search_coverage",
        "sample_size": args.sample_size,
        "repetitions_per_beta": args.repetitions,
        "bootstraps": args.bootstraps,
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
        "methods_compared": list(METHODS),
    }
    prepare_output_dir(args.output_dir, study_settings)

    logger.info("=" * 70)
    logger.info("Running nonlinear interaction simulation study (grid vs. TruVaRImp validation)")
    logger.info(f"output_dir={args.output_dir}")
    logger.info(f"sample_size={args.sample_size}")
    logger.info(f"repetitions={args.repetitions}")
    logger.info(f"bootstraps={args.bootstraps}")
    logger.info(f"n_candidates_per_family={args.n_candidates_per_family}")
    logger.info(f"truvarimp_total_budget={args.truvarimp_total_budget}")
    logger.info(f"beta_grid={args.beta_grid}")
    logger.info(f"betas={args.betas or args.beta_grid}")
    logger.info(f"noise_std={args.noise_std}")
    logger.info(f"epsilon={args.epsilon}")
    logger.info(f"rid_metric={args.rid_metric}")
    logger.info(f"stepwise_cv_splits={args.stepwise_cv_splits}")
    logger.info(f"stepwise_logreg_C={args.stepwise_logreg_C}")
    logger.info(f"stepwise_rf_n_estimators={args.stepwise_rf_n_estimators}")
    logger.info(f"methods_compared={list(METHODS)}")
    logger.info(f"simulations_included={list(simulators.keys())}")
    logger.info(f"n_jobs={n_jobs}")
    logger.info("=" * 70)

    started = time.time()

    run_simulation_study(
        simulators,
        beta_grid=args.beta_grid,
        betas=args.betas,
        sample_size=args.sample_size,
        n_repetitions=args.repetitions,
        noise_std=args.noise_std,
        base_seed=args.base_seed,
        stepwise_cv_splits=args.stepwise_cv_splits,
        random_state=args.random_state,
        make_method_runners=lambda X, y, cv, scoring, fitted_estimators: build_method_runners(
            X,
            y,
            cv,
            scoring,
            fitted_estimators,
            n_bootstraps=args.bootstraps,
            n_candidates_per_family=args.n_candidates_per_family,
            truvarimp_total_budget=args.truvarimp_total_budget,
            epsilon=args.epsilon,
            stepwise_logreg_C=args.stepwise_logreg_C,
            stepwise_rf_n_estimators=args.stepwise_rf_n_estimators,
            random_state=args.random_state,
            n_jobs=n_jobs,
            rid_metric=args.rid_metric,
        ),
        output_dir=args.output_dir,
        rid_metric=args.rid_metric,
    )
    output_paths = write_combined_outputs(args.output_dir)

    elapsed = time.time() - started
    logger.info("Saved outputs:")
    for path in [os.path.join(args.output_dir, PER_DGP_SUBDIR), *output_paths]:
        logger.info(f"  {path}")
    logger.info(f"Finished in {elapsed:.1f}s ({elapsed / 60.0:.1f} min)")


if __name__ == "__main__":
    main()
