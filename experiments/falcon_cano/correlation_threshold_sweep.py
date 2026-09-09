"""Exploratory analysis: correlation threshold vs. number of surviving features.

Standalone diagnostic -- does NOT modify run_falcon_cano_top40_comparison.py's
default --correlation-threshold. Replicates that script's own greedy pairwise
correlation-filtering algorithm (remove_highly_correlated_features in
run_rashomon_falcon_cano.py: no skip of already-removed features when
evaluating later pairs, higher-variance member of a pair survives) across a
fine sweep of thresholds, and finds the elbow/knee point via the Kneedle
algorithm (falls back to a max-distance-from-chord heuristic if `kneed` is
unavailable).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA_PATH = Path(__file__).resolve().parent / "falcon_cano_featured.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "correlation_threshold_sweep"


def load_raw_features() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df.drop(columns=["target"])


def sweep_thresholds(X: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    """Replicates remove_highly_correlated_features's greedy algorithm exactly,
    but computes the correlation matrix once and reuses it across thresholds
    instead of recomputing per-threshold (the correlation values themselves
    don't depend on the threshold, only which pairs get flagged does)."""

    corr_matrix = X.corr().abs()
    variances = X.var()
    cols = list(corr_matrix.columns)
    n = len(cols)

    rows = []
    for t in thresholds:
        removed: set[str] = set()
        for i in range(n):
            for j in range(i + 1, n):
                if corr_matrix.iloc[i, j] <= t:
                    continue
                f1, f2 = cols[i], cols[j]
                if variances[f1] >= variances[f2]:
                    removed.add(f2)
                else:
                    removed.add(f1)
        rows.append({"threshold": float(t), "n_features_remaining": n - len(removed), "n_removed": len(removed)})
    return pd.DataFrame(rows)


def find_elbow(sweep_df: pd.DataFrame) -> float:
    x = sweep_df["threshold"].to_numpy()
    y = sweep_df["n_features_remaining"].to_numpy()
    try:
        from kneed import KneeLocator

        kl = KneeLocator(x, y, curve="convex", direction="increasing")
        if kl.knee is not None:
            return float(kl.knee)
    except ImportError:
        pass

    # Fallback: max perpendicular distance from the chord connecting the
    # first and last point, on axes normalized to [0, 1].
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / line_len
    distances = []
    for xi, yi in zip(x_norm, y_norm):
        p = np.array([xi, yi]) - p1
        proj_len = np.dot(p, line_unit)
        proj = proj_len * line_unit
        distances.append(np.linalg.norm(p - proj))
    return float(x[int(np.argmax(distances))])


def plot_sweep(sweep_df: pd.DataFrame, elbow: float, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep_df["threshold"], sweep_df["n_features_remaining"], marker=".", linewidth=1)
    ax.axvline(elbow, color="red", linestyle="--", label=f"elbow = {elbow:.3f}")
    ax.axvline(0.8, color="gray", linestyle=":", label="current default = 0.80")
    ax.set_xlabel("Correlation threshold")
    ax.set_ylabel("Number of features remaining")
    ax.set_title("Falcon-Cano: correlation threshold vs. surviving features")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X = load_raw_features()
    print(f"[falcon_cano] raw features: {X.shape[1]}")

    thresholds = np.round(np.arange(0.50, 0.991, 0.01), 3)
    sweep_df = sweep_thresholds(X, thresholds)
    sweep_df.to_csv(OUTPUT_DIR / "threshold_vs_feature_count.csv", index=False)

    elbow = find_elbow(sweep_df)
    plot_sweep(sweep_df, elbow, OUTPUT_DIR / "threshold_vs_feature_count.png")

    n_at_elbow = int(sweep_df.loc[sweep_df["threshold"].sub(elbow).abs().idxmin(), "n_features_remaining"])
    n_at_default = int(sweep_df.loc[sweep_df["threshold"].sub(0.80).abs().idxmin(), "n_features_remaining"])
    print(f"[falcon_cano] elbow threshold: {elbow:.3f} ({n_at_elbow} features)")
    print(f"[falcon_cano] current default 0.80: {n_at_default} features")
    print(f"[falcon_cano] outputs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
