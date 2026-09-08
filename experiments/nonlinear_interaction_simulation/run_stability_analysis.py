#!/usr/bin/env python3
"""Test whether RID's own reported uncertainty predicts genuine single-model
instability under resampling.

RID's stated purpose (see docs/cross_family_rashomon_importance_distribution.md)
is answering "how stable is importance across many near-optimal models" -- not
"which top-k subset predicts best" (that's what the *_top40_comparison.py
scripts and the ground-truth precision/recall metrics in
run_nonlinear_interaction_simulation.py already test). This script targets
the stability claim directly:

1) Repeatedly bootstrap-refit ONE single model (no Rashomon epsilon
   filtering) on a fixed dataset, and measure how often each feature's
   importance (sub_mr) comes out POSITIVE across those refits, deriving a
   flip_rate from that fraction (0 = always positive or always non-positive,
   1 = as unstable as a coin flip). This mirrors EXACTLY how RID derives its
   own ambiguity from P(phi > 0) -- both sides measure sign-of-contribution
   instability, just estimated two different ways. An earlier version of
   this script used top-k RANK membership instead, which measures a
   different thing (rank stability, not sign stability) and produced
   spurious disagreement for features with a modest-but-reliably-positive
   effect that were nonetheless rank-competitive with other features.
2) Fit RID (single-family, same model family) once on the same dataset, and
   derive an ambiguity score from its own P(phi > 0) output (0 = RID is
   fully confident either way, 1 = RID itself reports maximal uncertainty).
3) Correlate flip_rate against RID's ambiguity across features. A positive
   correlation validates that RID's own uncertainty signal is predictive of
   genuine single-model instability, without needing the expensive repeated
   refit procedure to discover it after the fact.

Run on both the standard DGPs and the redundant-driver DGP variants (see
REDUNDANT_SIMULATORS in run_nonlinear_interaction_simulation.py) -- a
redundant duplicate of a true driver is exactly the scenario expected to
produce high single-model instability (a single model arbitrarily picks one
of the interchangeable pair each refit), making the redundant DGPs the
sharpest test of whether RID's ambiguity score tracks real instability.

Usage:
    python run_stability_analysis.py
    sbatch run_stability_analysis.sl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rid import FullyEnumeratedTreeClassifier, LassoClassifier, RashomonImportanceDistribution, vi_sub_mr  # noqa: E402
from run_nonlinear_interaction_simulation import (  # noqa: E402
    REDUNDANT_SIMULATORS,
    SIMULATORS,
    feature_columns_from_dataset,
    resolve_num_workers,
)

DEFAULT_SAMPLE_SIZE = 400
DEFAULT_NOISE_STD = 1.0
DEFAULT_N_REPEATS = 30
DEFAULT_EPSILON = 0.05
DEFAULT_RID_N_BOOTSTRAPS = 100
DEFAULT_RID_N_MODELS_POOL = 50
DEFAULT_RID_METRIC = "sub_mr"
DEFAULT_RANDOM_STATE = 42
DEFAULT_BASE_SEED = 20260907
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "results" / "stability_analysis")
DEFAULT_DGPS = "chen,chen_redundant_r095"
DEFAULT_BETAS = [1.0, 2.0]

ALL_SIMULATORS = {**SIMULATORS, **REDUNDANT_SIMULATORS}

# family_name -> (model_class, fixed kwargs template; random_state filled in per fit)
FAMILY_CONFIGS = {
    "Lasso": (
        LassoClassifier,
        {"penalty": "l1", "solver": "saga", "C": 1.0, "max_iter": 100000},
    ),
    "FullyEnumeratedTree": (
        FullyEnumeratedTreeClassifier,
        {"max_depth": 4, "min_samples_leaf": 5, "criterion": "gini"},
    ),
}


def _spearman_corr(a, b):
    """Spearman rank correlation without a scipy dependency."""

    a = pd.Series(a)
    b = pd.Series(b)
    if a.nunique() <= 1 or b.nunique() <= 1:
        return None
    corr = np.corrcoef(a.rank(), b.rank())[0, 1]
    return float(corr) if np.isfinite(corr) else None


def repeated_single_model_sign_instability(
    X_scaled,
    y,
    model_cls,
    model_kwargs,
    feature_names,
    n_repeats,
    base_seed,
):
    """Bootstrap-refit a single model n_repeats times (no Rashomon filtering);
    return, per feature, the fraction of refits where its importance
    (sub_mr) came out positive, and a flip_rate derived from that fraction.

    This mirrors EXACTLY how RID derives its own ambiguity from P(phi>0) --
    both sides measure sign-of-contribution instability, estimated two
    different ways (naive repeated single-model refits vs. RID's internal
    Rashomon-filtered bootstrap ensemble). An earlier version of this
    function used top-k RANK membership instead, which measures a different
    thing (rank stability, not sign stability) and produced spurious
    disagreement for features with a modest-but-reliably-positive effect
    that were nonetheless rank-competitive with other features -- see the
    write-up for the chen_redundant_r095/beta=2.0 case that exposed this."""

    positive_counts = {name: 0 for name in feature_names}
    rng = np.random.default_rng(base_seed)

    for i in range(n_repeats):
        X_boot, y_boot = resample(X_scaled, y, random_state=base_seed + i)
        model = model_cls(random_state=base_seed, **model_kwargs)
        model.fit(X_boot, y_boot)
        importances = vi_sub_mr(model, X_boot, y_boot, rng=rng)
        for name, value in zip(feature_names, importances):
            if value > 0:
                positive_counts[name] += 1

    prob_positive_single = {name: count / n_repeats for name, count in positive_counts.items()}
    flip_rate = {name: 1.0 - abs(2.0 * rate - 1.0) for name, rate in prob_positive_single.items()}
    return prob_positive_single, flip_rate


def run_stability_cell(
    X_scaled,
    y,
    feature_names,
    model_cls,
    model_kwargs,
    n_repeats,
    epsilon,
    rid_n_bootstraps,
    rid_n_models_pool,
    rid_metric,
    n_jobs,
    random_state,
):
    prob_positive_single, flip_rate = repeated_single_model_sign_instability(
        X_scaled, y, model_cls, model_kwargs, feature_names, n_repeats, random_state
    )

    # NOTE: model_kwargs (the single model's fixed hyperparameters) is deliberately
    # NOT passed here -- RID searches its own full grid for this family (that's the
    # whole point of single-family RID), and passing fixed hyperparameters that
    # overlap with the grid's own would collide with the trainer's kwargs (e.g.
    # max_depth/C already come from TRAINER_REGISTRY's grid, not from us).
    estimator = RashomonImportanceDistribution(
        epsilon=epsilon,
        n_bootstraps=rid_n_bootstraps,
        n_models_pool=rid_n_models_pool,
        model_class=model_cls,
        vi_metrics=(rid_metric,),
        performance_metrics=("accuracy", "auprc"),
        n_jobs=n_jobs,
    )
    estimator.fit(pd.DataFrame(X_scaled, columns=feature_names), y)

    if estimator.metric_results_ is None:
        raise RuntimeError("RID returned no metric results; no valid Rashomon bootstraps were found")

    summary = estimator.metric_summary(rid_metric)
    prob_positive = {name: summary[name]["prob_positive"] for name in feature_names}
    ambiguity = {name: 1.0 - abs(2.0 * p - 1.0) for name, p in prob_positive.items()}

    flip_vals = [flip_rate[name] for name in feature_names]
    ambiguity_vals = [ambiguity[name] for name in feature_names]
    spearman_corr = _spearman_corr(flip_vals, ambiguity_vals)

    per_feature = pd.DataFrame(
        {
            "feature": feature_names,
            "single_model_prob_positive": [prob_positive_single[name] for name in feature_names],
            "single_model_flip_rate": flip_vals,
            "rid_prob_positive": [prob_positive[name] for name in feature_names],
            "rid_ambiguity": ambiguity_vals,
        }
    )
    return per_feature, spearman_corr


def _plot_flip_vs_ambiguity(per_feature, ground_truth, title, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    is_gt = per_feature["feature"].isin(ground_truth)
    ax.scatter(
        per_feature.loc[~is_gt, "single_model_flip_rate"],
        per_feature.loc[~is_gt, "rid_ambiguity"],
        label="noise feature",
        alpha=0.6,
    )
    ax.scatter(
        per_feature.loc[is_gt, "single_model_flip_rate"],
        per_feature.loc[is_gt, "rid_ambiguity"],
        label="ground-truth driver",
        marker="D",
        s=80,
        color="crimson",
    )
    for _, row in per_feature.iterrows():
        ax.annotate(row["feature"], (row["single_model_flip_rate"], row["rid_ambiguity"]), fontsize=7)
    ax.set_xlabel("Single-model sign-instability (flip rate of importance sign, repeated refits)")
    ax.set_ylabel("RID ambiguity (1 - |2*P(phi>0) - 1|)")
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test whether RID's own P(phi>0)-derived ambiguity score predicts genuine "
            "single-model instability under bootstrap resampling, on both standard and "
            "redundant-driver DGPs."
        )
    )
    parser.add_argument(
        "--dgps",
        type=str,
        default=DEFAULT_DGPS,
        help=f"Comma-separated DGP names from SIMULATORS/REDUNDANT_SIMULATORS (default: {DEFAULT_DGPS})",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=DEFAULT_BETAS,
        help=f"Beta values to test (default: {DEFAULT_BETAS})",
    )
    parser.add_argument(
        "--families",
        type=str,
        default="Lasso,FullyEnumeratedTree",
        help="Comma-separated model families to test (default: Lasso,FullyEnumeratedTree)",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=DEFAULT_N_REPEATS,
        help="Number of independent bootstrap refits of the single model (default: 30)",
    )
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--rid-n-bootstraps", type=int, default=DEFAULT_RID_N_BOOTSTRAPS)
    parser.add_argument("--rid-n-models-pool", type=int, default=DEFAULT_RID_N_MODELS_POOL)
    parser.add_argument("--rid-metric", type=str, default=DEFAULT_RID_METRIC)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    n_jobs = resolve_num_workers(args.num_workers)

    dgp_names = [name.strip() for name in args.dgps.split(",") if name.strip()]
    unknown = [name for name in dgp_names if name not in ALL_SIMULATORS]
    if unknown:
        raise ValueError(f"Unknown DGPs: {unknown}. Available: {list(ALL_SIMULATORS.keys())}")

    family_names = [name.strip() for name in args.families.split(",") if name.strip()]
    unknown_families = [name for name in family_names if name not in FAMILY_CONFIGS]
    if unknown_families:
        raise ValueError(f"Unknown families: {unknown_families}. Available: {list(FAMILY_CONFIGS.keys())}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("RID stability-signal validation")
    print(f"  dgps:          {dgp_names}")
    print(f"  betas:         {args.betas}")
    print(f"  families:      {family_names}")
    print(f"  n_repeats:     {args.n_repeats}")
    print(f"  output_dir:    {output_dir}")
    print(f"  n_jobs:        {n_jobs}")
    print("=" * 70)

    started = time.time()
    per_feature_rows = []
    summary_rows = []

    for dgp_name in dgp_names:
        simulator = ALL_SIMULATORS[dgp_name]
        for beta in args.betas:
            seed = args.base_seed + hash((dgp_name, beta)) % 1_000_000
            dataset = simulator(n=args.sample_size, beta=beta, noise_std=args.noise_std, seed=seed)

            feature_columns = feature_columns_from_dataset(dataset)
            X = dataset[feature_columns]
            y = dataset["Y"].astype(int).to_numpy()
            ground_truth = tuple(dataset.attrs["relevant_features"])
            equivalence_classes = dataset.attrs.get("equivalence_classes") or {}

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.to_numpy())

            for family_name in family_names:
                model_cls, model_kwargs = FAMILY_CONFIGS[family_name]
                print(f"[cell] dgp={dgp_name} beta={beta} family={family_name}")

                per_feature, spearman_corr = run_stability_cell(
                    X_scaled=X_scaled,
                    y=y,
                    feature_names=feature_columns,
                    model_cls=model_cls,
                    model_kwargs=model_kwargs,
                    n_repeats=args.n_repeats,
                    epsilon=args.epsilon,
                    rid_n_bootstraps=args.rid_n_bootstraps,
                    rid_n_models_pool=args.rid_n_models_pool,
                    rid_metric=args.rid_metric,
                    n_jobs=n_jobs,
                    random_state=args.random_state,
                )
                per_feature["dgp"] = dgp_name
                per_feature["beta"] = beta
                per_feature["family"] = family_name
                per_feature["is_ground_truth"] = per_feature["feature"].isin(ground_truth)
                per_feature_rows.append(per_feature)

                plot_path = plots_dir / f"{dgp_name}_beta{beta}_{family_name}.png"
                _plot_flip_vs_ambiguity(
                    per_feature,
                    ground_truth,
                    title=f"{dgp_name} (beta={beta}, family={family_name})\nSpearman r={spearman_corr}",
                    output_path=plot_path,
                )

                summary_rows.append(
                    {
                        "dgp": dgp_name,
                        "beta": beta,
                        "family": family_name,
                        "n_features": len(feature_columns),
                        "n_ground_truth": len(ground_truth),
                        "has_redundancy": bool(equivalence_classes),
                        "spearman_flip_vs_ambiguity": spearman_corr,
                    }
                )
                print(f"       spearman(flip_rate, rid_ambiguity) = {spearman_corr}")

    per_feature_df = pd.concat(per_feature_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    per_feature_path = output_dir / "stability_per_feature.csv"
    summary_path = output_dir / "stability_summary.csv"
    per_feature_df.to_csv(per_feature_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    settings = {
        "args": vars(args),
        "dgp_names": dgp_names,
        "family_names": family_names,
    }
    with (output_dir / "run_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)

    elapsed = time.time() - started
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {per_feature_path}")
    print(f"Saved: {summary_path}")
    print(f"Plots: {plots_dir}/")
    print(f"Finished in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
