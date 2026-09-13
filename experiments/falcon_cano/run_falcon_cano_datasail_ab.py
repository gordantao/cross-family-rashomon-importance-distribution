#!/usr/bin/env python3
"""A/B test: does leakage-aware (DataSAIL) splitting change the held-out CV
comparison between methods, versus the current StratifiedKFold evaluator
which is blind to compound structure?

Motivation (see docs/findings_summary.md \\S11): Falcon-Cano's raw compound
list contains substantial congeneric families (e.g. 26 "-azole" antifungals,
25 "-pril" ACE inhibitors). Plain StratifiedKFold can put near-identical
analogs on both sides of a fold, inflating held_out_cv_score. DataSAIL's
cluster-based split (technique "C1e") keeps structurally similar compounds
in the same partition, using ECFP/Tanimoto similarity.

This is a RE-EVALUATION study, not a re-selection study: it reuses each
method's ALREADY-SELECTED top-40 feature list from
results/top40_feature_comparison/ rather than re-running the expensive RID
bootstrapping or stepwise search -- only the held-out scoring changes.

Arm A ("leaky"): the existing StratifiedKFold evaluator (_evaluate_feature_set
from run_falcon_cano_top40_comparison.py), unchanged.
Arm B ("leakage-aware"): N independent DataSAIL C1e 80/20 splits (repeated
holdout, since DataSAIL produces one train/test partition per run rather than
sklearn-style k-fold indices), averaged.

Both arms score the SAME feature sets under the SAME 3-model evaluator panel
(build_common_evaluator_panel) for a direct, apples-to-apples comparison.

Usage:
    python run_falcon_cano_datasail_ab.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasail.sail import datasail  # noqa: E402

from run_falcon_cano_top40_comparison import (  # noqa: E402
    _evaluate_feature_set,
    build_common_evaluator_panel,
)


def load_aligned_smiles(train_csv_path: Path) -> pd.DataFrame:
    """Reconstruct the row alignment between falcon_cano_featured.csv and the
    raw train.csv. Verified empirically: train.csv has 1161 rows, 4 of which
    have a missing (not invalid) `smile`; dropping those leaves exactly 1157
    rows matching falcon_cano_featured.csv, and recomputed RDKit descriptors
    for the first 5 aligned rows match the featured CSV exactly. Keys molecules
    by synthetic row_i, not compound Name -- Name is NOT a unique identifier
    in this dataset (several compounds appear 2-3x under the same name)."""

    raw = pd.read_csv(train_csv_path)
    raw = raw.dropna(subset=["smile"]).reset_index(drop=True)
    return raw


def load_method_feature_lists(top40_results_dir: Path) -> dict[str, list[str]]:
    file_map = {
        "stepwise_logreg": "stepwise_logreg_top_features.csv",
        "stepwise_rf": "stepwise_rf_top_features.csv",
        "rid_tree": "rid_tree_top_features.csv",
        "cross_family_unweighted": "rid_cross_family_unweighted_top_features.csv",
        "cross_family_weighted": "rid_cross_family_weighted_top_features.csv",
    }
    method_feature_lists = {}
    for method_name, filename in file_map.items():
        path = top40_results_dir / filename
        if not path.exists():
            print(f"[warn] {path} not found, skipping {method_name}")
            continue
        method_feature_lists[method_name] = pd.read_csv(path)["feature"].tolist()
    return method_feature_lists


def run_datasail_splits(
    smiles_map: dict[str, str],
    strat_map: dict[str, int],
    e_clusters: int,
    max_sec: int,
    n_runs: int,
) -> list[dict[str, str]]:
    """Get n_runs independent splits from a SINGLE datasail() call.

    datasail_main() unconditionally does random.seed(42); np.random.seed(42)
    at the start of every call -- calling datasail() n_runs separate times
    (each with runs=1) produces n_runs IDENTICAL splits, not independent ones
    (confirmed empirically: two such calls returned byte-identical results).
    DataSAIL's own runs=N reshuffles the dataset between runs internally
    (e_dataset.shuffle() for run > 0), which is the only way to get genuine
    inter-run variation from this library as installed.
    """

    e_splits, _, _ = datasail(
        techniques=["C1e"],
        splits=[0.8, 0.2],
        names=["train", "test"],
        runs=n_runs,
        solver="SCIP",
        max_sec=max_sec,
        e_type="M",
        e_data=smiles_map,
        e_strat=strat_map,
        e_clusters=e_clusters,
    )
    if not e_splits or "C1e" not in e_splits:
        raise RuntimeError("DataSAIL produced no assignment")
    return e_splits["C1e"]


def evaluate_leakage_aware(
    X: pd.DataFrame,
    y: pd.Series,
    method_feature_lists: dict[str, list[str]],
    evaluator_panel: list[tuple[str, type, dict]],
    split_assignments: list[dict[str, str]],
) -> dict[str, dict[str, dict]]:
    per_run_scores = {
        name: {evaluator_name: [] for evaluator_name, _, _ in evaluator_panel}
        for name in method_feature_lists
    }

    for split_assign in split_assignments:
        train_idx = [int(k.split("_")[1]) for k, v in split_assign.items() if v == "train"]
        test_idx = [int(k.split("_")[1]) for k, v in split_assign.items() if v == "test"]

        for name, features in method_feature_lists.items():
            X_train, X_test = X.iloc[train_idx][features], X.iloc[test_idx][features]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            for evaluator_name, model_cls, model_kwargs in evaluator_panel:
                model = model_cls(**model_kwargs)
                model.fit(X_train, y_train)
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)[:, 1]
                else:
                    y_score = model.decision_function(X_test)
                auc = roc_auc_score(y_test, y_score)
                per_run_scores[name][evaluator_name].append(float(auc))

    summary = {}
    for name, per_evaluator in per_run_scores.items():
        summary[name] = {}
        for evaluator_name, scores in per_evaluator.items():
            summary[name][evaluator_name] = {
                "cv_score_mean": float(np.mean(scores)),
                "cv_score_std": float(np.std(scores)),
                "n_runs": len(scores),
                "raw_scores": scores,
            }
    return summary


def evaluate_leaky(
    X: pd.DataFrame,
    y: pd.Series,
    method_feature_lists: dict[str, list[str]],
    evaluator_panel: list[tuple[str, type, dict]],
    cv,
    scoring: str,
) -> dict[str, dict[str, dict]]:
    summary = {}
    for name, features in method_feature_lists.items():
        summary[name] = {}
        for evaluator_name, model_cls, model_kwargs in evaluator_panel:
            summary[name][evaluator_name] = _evaluate_feature_set(
                X, y, features, model_cls, model_kwargs, cv, scoring
            )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).resolve().parent / "falcon_cano_featured.csv"),
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "falcon_cano" / "train.csv"),
        help="Raw train.csv with the 'smile' column, used to build DataSAIL's molecule input.",
    )
    parser.add_argument(
        "--top40-results-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "top40_feature_comparison"),
        help="Directory containing the already-computed *_top_features.csv files to re-evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "datasail_ab"),
    )
    parser.add_argument("--datasail-runs", type=int, default=5, help="Number of independent DataSAIL splits to average.")
    parser.add_argument("--datasail-clusters", type=int, default=50)
    parser.add_argument("--datasail-max-sec", type=int, default=90)
    parser.add_argument("--stepwise-cv-splits", type=int, default=5)
    parser.add_argument("--scoring", type=str, default="roc_auc")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Falcon-Cano DataSAIL leakage A/B test")
    print(f"datasail_runs={args.datasail_runs} datasail_clusters={args.datasail_clusters}")
    print("=" * 70)

    df = pd.read_csv(args.data)
    y = df["target"].astype(int)
    X = df.drop(columns=["target"])

    raw = load_aligned_smiles(Path(args.train_csv))
    assert len(raw) == len(df), (
        f"Row-count mismatch: train.csv (non-null smile)={len(raw)} vs "
        f"{args.data}={len(df)} -- alignment assumption from the docstring no longer holds"
    )
    smiles_map = {f"row_{i}": s for i, s in enumerate(raw["smile"])}
    strat_map = {f"row_{i}": int(v) for i, v in enumerate(y)}

    method_feature_lists = load_method_feature_lists(Path(args.top40_results_dir))
    print(f"[data] methods loaded: {list(method_feature_lists.keys())}")

    evaluator_panel = build_common_evaluator_panel(args.random_state)

    # --- Arm A: leaky (current) evaluator ---
    print("[arm A] evaluating under plain StratifiedKFold (leaky)...")
    cv = StratifiedKFold(n_splits=args.stepwise_cv_splits, shuffle=True, random_state=args.random_state)
    leaky_summary = evaluate_leaky(X, y, method_feature_lists, evaluator_panel, cv, args.scoring)

    # --- Arm B: leakage-aware (DataSAIL) evaluator ---
    print(f"[arm B] running {args.datasail_runs} independent DataSAIL C1e splits...")
    t0 = time.time()
    split_assignments = run_datasail_splits(
        smiles_map, strat_map, args.datasail_clusters, args.datasail_max_sec, args.datasail_runs
    )
    elapsed = time.time() - t0
    print(f"[arm B] {len(split_assignments)} splits obtained in {elapsed:.1f}s total")
    for run, split_assign in enumerate(split_assignments):
        counts = pd.Series(list(split_assign.values())).value_counts().to_dict()
        print(f"[arm B] run {run + 1}/{args.datasail_runs}: {counts}")

    leakage_aware_summary = evaluate_leakage_aware(
        X, y, method_feature_lists, evaluator_panel, split_assignments
    )

    # --- Build comparison table ---
    rows = []
    for name in method_feature_lists:
        row = {"method": name}
        leaky_means = []
        leakage_aware_means = []
        for evaluator_name, _, _ in evaluator_panel:
            leaky_mean = leaky_summary[name][evaluator_name]["cv_score_mean"]
            leakage_aware_mean = leakage_aware_summary[name][evaluator_name]["cv_score_mean"]
            row[f"leaky_{evaluator_name}"] = leaky_mean
            row[f"leakage_aware_{evaluator_name}"] = leakage_aware_mean
            row[f"delta_{evaluator_name}"] = leakage_aware_mean - leaky_mean
            leaky_means.append(leaky_mean)
            leakage_aware_means.append(leakage_aware_mean)
        row["leaky_avg"] = float(np.mean(leaky_means))
        row["leakage_aware_avg"] = float(np.mean(leakage_aware_means))
        row["delta_avg"] = row["leakage_aware_avg"] - row["leaky_avg"]
        rows.append(row)

    comparison_table = pd.DataFrame(rows).sort_values("leaky_avg", ascending=False)
    comparison_table.to_csv(output_dir / "leaky_vs_leakage_aware.csv", index=False)

    settings = {
        "args": vars(args),
        "n_rows": len(df),
        "n_features_per_method": {name: len(feats) for name, feats in method_feature_lists.items()},
        "datasail_split_sizes": [
            pd.Series(list(s.values())).value_counts().to_dict() for s in split_assignments
        ],
        "leaky_summary": leaky_summary,
        "leakage_aware_summary": leakage_aware_summary,
    }
    with (output_dir / "run_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)

    print("=" * 70)
    print(comparison_table.to_string(index=False))
    print(f"\nResults saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
