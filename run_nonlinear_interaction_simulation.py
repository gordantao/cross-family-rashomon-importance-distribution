#!/usr/bin/env python3
"""
Cross-family RID simulation study runner (cluster-friendly).

This script reproduces the cleaned nonlinear interaction notebook as a
command-line workflow suitable for Slurm jobs.

Usage:
    python run_nonlinear_interaction_simulation.py
    sbatch run_nonlinear_interaction_simulation.sl
"""

import argparse
import contextlib
import io
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

from rid import (
    CrossFamilyRashomonImportanceDistribution,
    LassoClassifier,
    RidgeClassifier,
    performance_accuracy,
    performance_auprc,
    performance_f1,
    vi_sub_mr,
)

DEFAULT_SAMPLE_SIZE = 400
DEFAULT_REPETITIONS = 12
DEFAULT_BOOTSTRAPS = 12
DEFAULT_MODELS_PER_CLASS = 6
DEFAULT_BETA_GRID = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
DEFAULT_NOISE_STD = 1.0
DEFAULT_BASE_SEED = 20260514
DEFAULT_EPSILON = 0.05
DEFAULT_OUTPUT_DIR = "results/nonlinear_interaction_simulation"
DEFAULT_BALANCE_MODE = "unweighted"

RID_MODEL_CONFIGS = {
    "RF": {
        "model": RandomForestClassifier,
        "kwargs": {"min_samples_leaf": 2},
        "search_grid": {
            "n_estimators": [100, 200],
            "max_depth": [4, None],
            "max_features": ["sqrt"],
        },
    },
    "GBM": {
        "model": GradientBoostingClassifier,
        "kwargs": {},
        "search_grid": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        },
    },
    "SVM": {
        "model": SVC,
        "kwargs": {},
        "search_grid": {
            "C": [0.5, 2.0],
            "gamma": ["scale", "auto"],
        },
    },
    "Lasso": {
        "model": LassoClassifier,
        "kwargs": {},
        "search_grid": {"C": [0.1, 1.0, 10.0]},
    },
    "Ridge": {
        "model": RidgeClassifier,
        "kwargs": {},
        "search_grid": {"C": [0.1, 1.0, 10.0]},
    },
}

RID_PERFORMANCE_METRICS = (
    performance_accuracy,
    performance_f1,
    performance_auprc,
)


def resolve_num_workers(cli_override=None):
    """Determine number of workers from Slurm env first, then CLI."""

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        n_workers = int(slurm_cpus)
        print(f"[env] SLURM_CPUS_PER_TASK={n_workers} -> using {n_workers} workers")
        return n_workers

    if cli_override is not None:
        print(f"[cli] --num-workers={cli_override}")
        return cli_override

    n_workers = os.cpu_count() or 1
    print(f"[fallback] os.cpu_count()={n_workers}")
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


def feature_columns_from_dataset(dataset):
    return sorted(
        [column for column in dataset.columns if column.startswith("X")],
        key=lambda name: int(name[1:]),
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


def run_cross_family_rid(
    X,
    y,
    *,
    n_bootstraps,
    n_models_per_class,
    epsilon,
    family_balance_mode,
    n_jobs=1,
):
    estimator = CrossFamilyRashomonImportanceDistribution(
        model_configs=RID_MODEL_CONFIGS,
        epsilon=epsilon,
        n_bootstraps=n_bootstraps,
        n_models_per_class=n_models_per_class,
        vi_metrics=(vi_sub_mr,),
        performance_metrics=RID_PERFORMANCE_METRICS,
        family_balance_mode=family_balance_mode,
        n_jobs=n_jobs,
    )

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        estimator.fit(X, y)

    ranked_features = [feature for feature, _ in estimator.rank_features(vi_sub_mr)]
    return ranked_features


def run_simulation_study(
    simulators,
    *,
    beta_grid,
    sample_size,
    n_repetitions,
    n_bootstraps,
    n_models_per_class,
    noise_std,
    epsilon,
    base_seed,
    family_balance_mode,
    n_jobs=1,
):
    rows = []
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

                feature_columns = feature_columns_from_dataset(dataset)
                X = dataset[feature_columns]
                y = dataset["Y"].astype(int)
                ground_truth = tuple(dataset.attrs["relevant_features"])

                ranked_features = run_cross_family_rid(
                    X,
                    y,
                    n_bootstraps=n_bootstraps,
                    n_models_per_class=n_models_per_class,
                    epsilon=epsilon,
                    family_balance_mode=family_balance_mode,
                    n_jobs=n_jobs,
                )

                metric_row = compute_selection_metrics(ranked_features, ground_truth)
                rows.append(
                    {
                        "dgp": dgp_name,
                        "beta": float(beta),
                        "rep": rep,
                        "snr_empirical": float(dataset.attrs["snr_empirical"]),
                        "ground_truth": ", ".join(ground_truth),
                        "ranked_top_k": ", ".join(ranked_features[: len(ground_truth)]),
                        **metric_row,
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

    return pd.DataFrame(rows)


def build_metric_tables(study_results):
    grouped = (
        study_results.groupby(["dgp", "beta"])[
            [
                "snr_empirical",
                "precision_at_k",
                "recall_at_k",
                "ndcg_at_k",
                "exact_match",
                "mean_gt_rank",
            ]
        ]
        .agg(["mean", "sem"])
        .reset_index()
    )
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.to_flat_index()]

    overall = (
        study_results.groupby("dgp")
        [["precision_at_k", "recall_at_k", "ndcg_at_k", "exact_match", "mean_gt_rank"]]
        .mean()
        .reset_index()
    )

    for table in (grouped, overall):
        numeric_cols = table.select_dtypes(include=[np.number]).columns
        table[numeric_cols] = table[numeric_cols].round(3)

    return grouped, overall


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
        description="Run cross-family RID simulation benchmark and save tables/plots."
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
        "--models-per-class",
        type=int,
        default=DEFAULT_MODELS_PER_CLASS,
        help="RID models per class",
    )
    parser.add_argument(
        "--beta-grid",
        type=float,
        nargs="+",
        default=DEFAULT_BETA_GRID,
        help="List of beta values used in latent score beta*signal+epsilon",
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
        "--family-balance-mode",
        type=str,
        default=DEFAULT_BALANCE_MODE,
        help="Cross-family balance mode (for example: unweighted, weighted, count)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel workers (defaults to SLURM_CPUS_PER_TASK)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n_jobs = resolve_num_workers(args.num_workers)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Running nonlinear interaction simulation study")
    print(f"output_dir={args.output_dir}")
    print(f"sample_size={args.sample_size}")
    print(f"repetitions={args.repetitions}")
    print(f"bootstraps={args.bootstraps}")
    print(f"models_per_class={args.models_per_class}")
    print(f"beta_grid={args.beta_grid}")
    print(f"noise_std={args.noise_std}")
    print(f"epsilon={args.epsilon}")
    print(f"family_balance_mode={args.family_balance_mode}")
    print(f"n_jobs={n_jobs}")
    print("=" * 70)

    started = time.time()

    study_results = run_simulation_study(
        SIMULATORS,
        beta_grid=args.beta_grid,
        sample_size=args.sample_size,
        n_repetitions=args.repetitions,
        n_bootstraps=args.bootstraps,
        n_models_per_class=args.models_per_class,
        noise_std=args.noise_std,
        epsilon=args.epsilon,
        base_seed=args.base_seed,
        family_balance_mode=args.family_balance_mode,
        n_jobs=n_jobs,
    )

    metric_table_by_dgp_beta, metric_table_by_dgp = build_metric_tables(study_results)

    snr_plot_table = (
        study_results.groupby(["dgp", "beta"])["snr_empirical"]
        .agg(["mean", "sem"])
        .reset_index()
        .sort_values(["dgp", "beta"])
    )

    study_settings = {
        "sample_size": args.sample_size,
        "repetitions_per_beta": args.repetitions,
        "bootstraps": args.bootstraps,
        "models_per_class": args.models_per_class,
        "beta_grid": args.beta_grid,
        "noise_std": args.noise_std,
        "epsilon": args.epsilon,
        "family_balance_mode": args.family_balance_mode,
        "num_workers": n_jobs,
        "simulations_included": list(SIMULATORS.keys()),
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
    save_snr_plot(snr_plot_table, snr_plot_path)

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
