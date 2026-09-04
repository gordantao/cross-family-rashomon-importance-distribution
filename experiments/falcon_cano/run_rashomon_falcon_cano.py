#!/usr/bin/env python3
"""
Rashomon Importance Distribution (RID) analysis - SLURM-compatible script.

NUM_WORKERS is automatically synced with SLURM's --cpus-per-task via the
SLURM_CPUS_PER_TASK environment variable. Falls back to os.cpu_count() for
local runs.

Usage:
    python run_rashomon_falcon_cano.py
    sbatch run_rashomon_falcon_cano.sh
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rid import (
    CrossFamilyRashomonImportanceDistribution,
    ElasticNetClassifier,
    LassoClassifier,
    LinearClassifier,
    RashomonImportanceDistribution,
    RidgeClassifier,
    VALID_CROSS_FAMILY_BALANCE_MODES,
)

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Curated RDKit descriptor exclusion list
#
# Rationale for each group:
#   mol_weight_variants   – HeavyAtomMolWt and ExactMolWt are near-perfect
#                           surrogates for MolWt; keep MolWt only.
#   estate_abs            – MaxAbsEStateIndex / MinAbsEStateIndex are |max| /
#                           |min| of their signed counterparts; sign carries
#                           information, so keep the signed versions.
#   partial_charge_abs    – Same abs-vs-signed argument as EState extremes.
#   bcut_low_end          – Each BCUT2D pair (hi/lo) spans the same spectrum;
#                           the high-end eigenvalue is sufficient.
#   logp_bcut             – BCUT2D_LOGPHI and _LOGPLOW are eigenvalue-based
#                           logP surrogates; use Crippen MolLogP directly.
#   mr_bcut               – BCUT2D_MRHI/_MRLOW are eigenvalue-based MR
#                           surrogates; use Crippen MolMR directly.
#   chi_base_and_norm     – Chi0/1 (bond-based) and Chi*n (normalized) are
#                           nested in the valence-weighted Chi*v; keep Chi*v.
#   ring_count_sums       – NumAliphatic/Aromatic/SaturatedRings equal the sum
#                           of the corresponding carbocycle + heterocycle pair;
#                           keep the components for finer resolution.
#   fr_coo_sums           – fr_COO and fr_COO2 double-count what fr_Al_COO +
#                           fr_Ar_COO already capture separately.
#   fr_oh_subsets         – fr_Al_OH_noTert is a strict subset of fr_Al_OH;
#                           fr_phenol and fr_phenol_noOrthoHbond both overlap
#                           heavily with fr_Ar_OH.
#   fr_nitro_subsets      – fr_nitro_arom and fr_nitro_arom_nonortho are
#                           subsets of fr_nitro.
#   fr_co_subset          – fr_C_O_noCOO = fr_C_O minus carboxylic acids;
#                           keep the broader fr_C_O.
#   fr_ketone_variant     – fr_ketone_Topliss is an alternate ketone count
#                           that largely duplicates fr_ketone.
#   fr_ndealk_duplicate   – fr_Ndealkylation2 is highly correlated with
#                           fr_Ndealkylation1; keep the first.
# ---------------------------------------------------------------------------
EXCLUDED_RDKIT_DESCRIPTORS: dict[str, str] = {
    # mol_weight_variants
    "HeavyAtomMolWt": "mol_weight_variants",
    "ExactMolWt": "mol_weight_variants",
    # estate_abs
    "MaxAbsEStateIndex": "estate_abs",
    "MinAbsEStateIndex": "estate_abs",
    # partial_charge_abs
    "MaxAbsPartialCharge": "partial_charge_abs",
    "MinAbsPartialCharge": "partial_charge_abs",
    # bcut_low_end
    "BCUT2D_MWLOW": "bcut_low_end",
    "BCUT2D_CHGLO": "bcut_low_end",
    # logp_bcut
    "BCUT2D_LOGPHI": "logp_bcut",
    "BCUT2D_LOGPLOW": "logp_bcut",
    # mr_bcut
    "BCUT2D_MRHI": "mr_bcut",
    "BCUT2D_MRLOW": "mr_bcut",
    # chi_base_and_norm
    "Chi0": "chi_base_and_norm",
    "Chi0n": "chi_base_and_norm",
    "Chi1": "chi_base_and_norm",
    "Chi1n": "chi_base_and_norm",
    "Chi2n": "chi_base_and_norm",
    "Chi3n": "chi_base_and_norm",
    "Chi4n": "chi_base_and_norm",
    # ring_count_sums
    "NumAliphaticRings": "ring_count_sums",
    "NumAromaticRings": "ring_count_sums",
    "NumSaturatedRings": "ring_count_sums",
    # fr_coo_sums
    "fr_COO": "fr_coo_sums",
    "fr_COO2": "fr_coo_sums",
    # fr_oh_subsets
    "fr_Al_OH_noTert": "fr_oh_subsets",
    "fr_phenol": "fr_oh_subsets",
    "fr_phenol_noOrthoHbond": "fr_oh_subsets",
    # fr_nitro_subsets
    "fr_nitro_arom": "fr_nitro_subsets",
    "fr_nitro_arom_nonortho": "fr_nitro_subsets",
    # fr_co_subset
    "fr_C_O_noCOO": "fr_co_subset",
    # fr_ketone_variant
    "fr_ketone_Topliss": "fr_ketone_variant",
    # fr_ndealk_duplicate
    "fr_Ndealkylation2": "fr_ndealk_duplicate",
}


def select_curated_descriptors(df_descriptors):
    """Drop redundant RDKit descriptors according to EXCLUDED_RDKIT_DESCRIPTORS."""

    cols_to_drop = [c for c in EXCLUDED_RDKIT_DESCRIPTORS if c in df_descriptors.columns]
    df_curated = df_descriptors.drop(columns=cols_to_drop)

    by_reason: dict[str, list[str]] = {}
    for col in cols_to_drop:
        reason = EXCLUDED_RDKIT_DESCRIPTORS[col]
        by_reason.setdefault(reason, []).append(col)

    print(f"Curated descriptor set: dropped {len(cols_to_drop)} descriptors "
          f"({df_descriptors.shape[1]} → {df_curated.shape[1]})")
    for reason, cols in sorted(by_reason.items()):
        print(f"  [{reason}] {', '.join(cols)}")

    return df_curated


def resolve_num_workers(cli_override=None):
    """Determine the number of parallel workers."""

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        n_workers = int(slurm_cpus)
        print(f"[env] SLURM_CPUS_PER_TASK={n_workers} - using {n_workers} workers")
        return n_workers

    if cli_override is not None:
        print(f"[cli] --num-workers={cli_override}")
        return cli_override

    n_workers = os.cpu_count() or 4
    print(f"[fallback] os.cpu_count()={n_workers}")
    return n_workers


def remove_highly_correlated_features(df_descriptors, correlation_threshold=0.95):
    """Remove highly correlated features, keeping the one with higher variance."""

    print(f"Starting with {df_descriptors.shape[1]} features")
    corr_matrix = df_descriptors.corr().abs()
    removed_features = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] <= correlation_threshold:
                continue

            feature_one = corr_matrix.columns[i]
            feature_two = corr_matrix.columns[j]
            if df_descriptors[feature_one].var() >= df_descriptors[feature_two].var():
                removed_features.add(feature_two)
            else:
                removed_features.add(feature_one)

    df_filtered = df_descriptors.drop(columns=removed_features)
    print(f"Removed {len(removed_features)} features, remaining: {df_filtered.shape[1]}")
    return df_filtered


def print_metric_rankings(title, estimator, perf_stats=None):
    """Print top features for every available RID metric."""

    print(f"\n{'=' * 60}")
    print(title)

    if perf_stats:
        print(
            "  Rashomon-set performance: "
            f"acc={perf_stats['accuracy_mean']:.3f}\u00b1{perf_stats['accuracy_std']:.3f}  "
            f"AUPRC={perf_stats['auprc_mean']:.3f}\u00b1{perf_stats['auprc_std']:.3f}"
        )

    if not estimator.metric_results_:
        print("  No RID metrics available.")
        return

    for metric in estimator.available_metrics_:
        summary = estimator.metric_summary(metric)
        ranking = estimator.rank_features(metric)
        print(f"\nTop 10 features [{metric}]:")
        for feature_name, importance in ranking[:10]:
            print(
                f"  {feature_name}: E[phi]={importance:.6f}, "
                f"P(phi>0)={summary[feature_name]['prob_positive']:.3f}"
            )


def save_metric_plots(estimator, output_dir, filename_prefix, title_prefix):
    """Persist CDF and density plots for each RID metric."""

    if not estimator.metric_results_:
        return

    for metric, (rid_cdfs, cdf_grid, raw_importances) in estimator.metric_results_.items():
        top_features = [feature_name for feature_name, _ in estimator.rank_features(metric)[:5]]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for feature_name in top_features:
            axes[0].plot(cdf_grid, rid_cdfs[feature_name], label=feature_name, linewidth=2)
        axes[0].set_title(f"{title_prefix} (CDF) [{metric}]")
        axes[0].set_xlabel(f"Variable Importance ({metric})")
        axes[0].set_ylabel("CDF: P(importance <= k)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        for feature_name in top_features:
            if len(raw_importances[feature_name]) > 1:
                sns.kdeplot(
                    raw_importances[feature_name],
                    label=feature_name,
                    fill=True,
                    alpha=0.1,
                    ax=axes[1],
                )
        axes[1].set_title(f"{title_prefix} Density (PDF) [{metric}]")
        axes[1].set_xlabel(f"Variable Importance ({metric})")
        axes[1].set_ylabel("Density")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{filename_prefix}_{metric}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved: {filename_prefix}_{metric}.png")


def save_family_composition_plot(output_dir, family_counts, balance_mode):
    """Persist the cross-family Rashomon composition plot."""

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(family_counts.keys(), family_counts.values())
    ax.set_title(
        f"Model Family Representation in Cross-Family Rashomon Sets ({balance_mode})"
    )
    ax.set_ylabel("Total models across all bootstraps")
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"rid_family_composition_{balance_mode}.png"),
        dpi=150,
    )
    plt.close(fig)
    print(f"  Saved: rid_family_composition_{balance_mode}.png")


def build_model_configs():
    """Return the single-family and cross-family model configuration maps."""

    model_configs = {
        "RandomForest": (RandomForestClassifier, {}),
        "GradientBoosting": (GradientBoostingClassifier, {}),
        "SVM": (SVC, {}),
        "Lasso": (LassoClassifier, {}),
        "ElasticNet": (ElasticNetClassifier, {}),
        "Ridge": (RidgeClassifier, {}),
        "Linear": (LinearClassifier, {}),
    }

    # Example custom grid override:
    # model_configs["SVM"] = {
    #     "model": SVC,
    #     "kwargs": {},
    #     "search_grid": {
    #         "C": [0.1, 1, 10],
    #         "gamma": [0.001, 0.01, 0.1],
    #     },
    # }

    cross_family_model_configs = {
        name: config
        for name, config in model_configs.items()
        if name != "Linear"
    }
    return model_configs, cross_family_model_configs


def main():
    parser = argparse.ArgumentParser(
        description="Run Rashomon Importance Distribution (RID) analysis."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: SLURM_CPUS_PER_TASK or os.cpu_count())",
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
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory for output figures and results (default: <script dir>/results)",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=100,
        help="Number of bootstrap iterations (default: 100)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Rashomon epsilon parameter (default: 0.05)",
    )
    parser.add_argument(
        "--n-models-pool",
        type=int,
        default=50,
        help="Candidate models per bootstrap (default: 50)",
    )
    parser.add_argument(
        "--cross-family-only",
        action="store_true",
        default=False,
        help="Skip per-family RID and run only the cross-family RID (faster)",
    )
    parser.add_argument(
        "--cross-family-balance-modes",
        nargs="+",
        default=["unweighted", "weighted"],
        choices=list(VALID_CROSS_FAMILY_BALANCE_MODES),
        help="Cross-family balancing modes to run (default: unweighted weighted)",
    )
    parser.add_argument(
        "--use-curated-descriptors",
        action="store_true",
        default=False,
        help="Drop redundant RDKit descriptors before correlation filtering",
    )
    args = parser.parse_args()

    num_workers = resolve_num_workers(args.num_workers)
    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()
    print(f"{'=' * 60}")
    print("RID Analysis")
    print(f"  Workers:      {num_workers}")
    print(f"  Data:         {args.data}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Bootstraps:   {args.n_bootstraps}")
    print(f"  Epsilon:      {args.epsilon}")
    print(f"  Models/pool:  {args.n_models_pool}")
    print(f"  Cross-family only: {args.cross_family_only}")
    print(f"  Cross-family balance modes: {args.cross_family_balance_modes}")
    print(f"  Curated descriptors: {args.use_curated_descriptors}")
    print(f"{'=' * 60}\n")

    df = pd.read_csv(args.data)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    X = df.drop(columns=["target"])
    y = df["target"]
    if args.use_curated_descriptors:
        X = select_curated_descriptors(X)
    X_filtered = remove_highly_correlated_features(X, correlation_threshold=0.8)

    model_configs, cross_family_model_configs = build_model_configs()
    rid_estimators = {}

    if args.cross_family_only:
        print("[cross-family-only] Skipping per-family RID.\n")
    else:
        for name, model_config in model_configs.items():
            print(f"\n{'=' * 60}")
            estimator = RashomonImportanceDistribution(
                epsilon=args.epsilon,
                n_bootstraps=args.n_bootstraps,
                n_models_pool=args.n_models_pool,
                model_config=model_config,
                n_jobs=num_workers,
            )
            estimator.fit(X_filtered, y)
            rid_estimators[name] = estimator

    for name, estimator in rid_estimators.items():
        if estimator.metric_results_ is None:
            continue
        data_path = os.path.join(args.output_dir, f"rid_{name}_data.pkl")
        with open(data_path, "wb") as handle:
            pickle.dump(
                {
                    "metrics": estimator.metric_results_,
                    "perf": estimator.perf_stats_,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"  Saved: rid_{name}_data.pkl")

    for name, estimator in rid_estimators.items():
        if estimator.metric_results_ is None:
            continue
        print_metric_rankings(f"Model class: {name}", estimator, estimator.perf_stats_)

    for name, estimator in rid_estimators.items():
        if estimator.metric_results_ is None:
            continue
        save_metric_plots(estimator, args.output_dir, f"rid_{name}", f"RID - {name}")

    for balance_mode in args.cross_family_balance_modes:
        print(f"\n{'=' * 60}")
        print(f"Running cross-family RID... mode={balance_mode}")
        estimator = CrossFamilyRashomonImportanceDistribution(
            model_configs=cross_family_model_configs,
            epsilon=args.epsilon,
            n_bootstraps=args.n_bootstraps,
            n_models_per_class=args.n_models_pool,
            family_balance_mode=balance_mode,
            n_jobs=num_workers,
        )
        estimator.fit(X_filtered, y)

        if estimator.metric_results_ is None:
            continue

        suffix = f"cross_family_{balance_mode}"
        data_path = os.path.join(args.output_dir, f"rid_{suffix}_data.pkl")
        with open(data_path, "wb") as handle:
            pickle.dump(
                {
                    "metrics": estimator.metric_results_,
                    "family_counts": estimator.family_counts_,
                    "family_perf": estimator.family_perf_stats_,
                    "family_balance_mode": balance_mode,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"  Saved: rid_{suffix}_data.pkl")

        print(f"\n{'=' * 60}")
        print(f"Cross-family RID results ({balance_mode}):")
        print(f"  Family representation: {estimator.family_counts_}")
        if estimator.family_perf_stats_:
            print("  Accuracy and AUPRC by model family (Rashomon models):")
            for family_name, perf_stats in estimator.family_perf_stats_.items():
                print(
                    f"    {family_name}: acc={perf_stats['accuracy_mean']:.3f}\u00b1{perf_stats['accuracy_std']:.3f}  "
                    f"AUPRC={perf_stats['auprc_mean']:.3f}\u00b1{perf_stats['auprc_std']:.3f}"
                )
        print_metric_rankings(
            f"Cross-family RID ({balance_mode})",
            estimator,
        )
        save_metric_plots(
            estimator,
            args.output_dir,
            f"rid_{suffix}",
            f"Cross-Family RID (mode={balance_mode})",
        )
        save_family_composition_plot(args.output_dir, estimator.family_counts_, balance_mode)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Finished in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Results saved to: {args.output_dir}/")
    print("  Load data with: pickle.load(open('results/rid_<model>_data.pkl', 'rb'))")


if __name__ == "__main__":
    main()