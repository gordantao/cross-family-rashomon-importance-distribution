"""Shared Kneedle-based elbow/knee detection for (x, y) sweep curves.

Used by both `correlation_threshold_sweep.py` (threshold vs. surviving-feature-count)
and `feature_count_selection.py` (rank vs. a method's own native signal curve), so the
Kneedle-plus-fallback logic lives in one place instead of being copied per sweep.
"""

from __future__ import annotations

import numpy as np


def find_elbow(x_values, y_values, curve: str = "convex", direction: str = "decreasing"):
    """Elbow/knee x-value of a (x_values, y_values) curve, or None if no interior
    knee is found (falls back to a max-distance-from-chord heuristic if `kneed`
    isn't installed; that heuristic always returns a point, so None only ever
    comes from `kneed` itself declining to report a knee)."""

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if len(x) < 3:
        return None

    try:
        from kneed import KneeLocator

        kl = KneeLocator(x, y, curve=curve, direction=direction)
        if kl.knee is not None:
            return float(kl.knee)
    except ImportError:
        pass

    if x.max() == x.min() or y.max() == y.min():
        return None

    # Fallback: max perpendicular distance from the chord connecting the
    # first and last point, on axes normalized to [0, 1].
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_unit = line_vec / np.linalg.norm(line_vec)
    distances = []
    for xi, yi in zip(x_norm, y_norm):
        p = np.array([xi, yi]) - p1
        proj = np.dot(p, line_unit) * line_unit
        distances.append(np.linalg.norm(p - proj))
    return float(x[int(np.argmax(distances))])
