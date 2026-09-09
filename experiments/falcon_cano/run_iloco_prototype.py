"""Prototype: pairwise ablation-based iLOCO interaction importance on Falcon-Cano.

Restricts to a top-K already-important feature subset (from an existing
rid_tree top-40 result if available, else univariate correlation with the
target) to keep the O(K^2) pairwise sweep tractable, and skips pairs whose
features are already highly correlated (their "interaction" is redundant).
Fits ONE model (no Rashomon-set bootstrap loop -- this is a metric prototype,
not a full RID run) and compares sub_mr / loco / iloco rankings on it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rid.core import (
    compute_iloco_importance,
    compute_loco_importance,
    compute_model_reliance,
    describe_iloco_eligible_pairs,
)


def _select_top_k_features(df: pd.DataFrame, target_col: str, top_k: int, existing_top40_csv: Path | None) -> list[str]:
    if existing_top40_csv is not None and existing_top40_csv.exists():
        ranked = pd.read_csv(existing_top40_csv)
        features = ranked.sort_values("rank")["feature"].tolist()
        available = [f for f in features if f in df.columns]
        if len(available) >= top_k:
            print(f"[select] using existing rid_tree top-{top_k} list from {existing_top40_csv}")
            return available[:top_k]

    print("[select] no usable existing top-k list found; falling back to univariate |corr| with target")
    feature_cols = [c for c in df.columns if c != target_col]
    corrs = df[feature_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
    return corrs.head(top_k).index.tolist()


def run(data_path: Path, top_k: int, correlation_threshold: float, existing_top40_csv: Path | None) -> None:
    df = pd.read_csv(data_path)
    target_col = "target"
    assert target_col in df.columns, f"expected a '{target_col}' column, got {df.columns.tolist()[:5]}..."

    top_features = _select_top_k_features(df, target_col, top_k, existing_top40_csv)
    print(f"[data] restricted to {len(top_features)} features: {top_features[:8]}...")

    X_raw = df[top_features].to_numpy(dtype=float)
    y = df[target_col].to_numpy()
    X = StandardScaler().fit_transform(X_raw)

    model = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=100000, random_state=42)
    model.fit(X, y)

    print(f"[model] fit LogisticRegression on {X.shape[0]} rows x {X.shape[1]} features")

    t0 = time.time()
    sub_mr = compute_model_reliance(model, X, y, metric="sub_mr", rng=np.random.default_rng(0))
    t1 = time.time()
    loco = compute_loco_importance(model, X, y)
    t2 = time.time()
    iloco = compute_iloco_importance(model, X, y, correlation_threshold=correlation_threshold)
    n_eligible, n_total = describe_iloco_eligible_pairs(X, correlation_threshold=correlation_threshold)
    t3 = time.time()

    n_skipped = n_total - n_eligible
    print(
        f"[timing] sub_mr={t1 - t0:.3f}s  loco={t2 - t1:.3f}s  "
        f"iloco={t3 - t2:.3f}s (over {n_eligible}/{n_total} eligible pairs, "
        f"{n_skipped} skipped as |corr| >= {correlation_threshold})"
    )

    result = pd.DataFrame(
        {
            "feature": top_features,
            "sub_mr": sub_mr,
            "loco": loco,
            "iloco_sum_abs": iloco,
        }
    )
    result["rank_sub_mr"] = result["sub_mr"].rank(ascending=False, method="min").astype(int)
    result["rank_loco"] = result["loco"].rank(ascending=False, method="min").astype(int)
    result["rank_iloco"] = result["iloco_sum_abs"].rank(ascending=False, method="min").astype(int)
    result["rank_shift_iloco_vs_sub_mr"] = result["rank_sub_mr"] - result["rank_iloco"]
    result = result.sort_values("rank_iloco")

    out_dir = Path(__file__).resolve().parent / "results" / "iloco_prototype"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "iloco_vs_sub_mr_loco.csv"
    result.to_csv(out_csv, index=False)

    print(f"\n[result] saved comparison table to {out_csv}")
    print(result.to_string(index=False))

    biggest_shift = result.reindex(result["rank_shift_iloco_vs_sub_mr"].abs().sort_values(ascending=False).index)
    print("\n[result] top 5 features whose iloco rank differs most from sub_mr rank:")
    print(
        biggest_shift[["feature", "rank_sub_mr", "rank_iloco", "rank_shift_iloco_vs_sub_mr"]]
        .head(5)
        .to_string(index=False)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=str(Path(__file__).resolve().parent / "falcon_cano_featured.csv"))
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--correlation-threshold", type=float, default=0.8)
    parser.add_argument(
        "--existing-top40-csv",
        type=str,
        default=str(
            Path(__file__).resolve().parent / "results" / "top40_feature_comparison" / "rid_tree_top_features.csv"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    existing = Path(args.existing_top40_csv)
    run(
        data_path=Path(args.data),
        top_k=args.top_k,
        correlation_threshold=args.correlation_threshold,
        existing_top40_csv=existing if existing.exists() else None,
    )
