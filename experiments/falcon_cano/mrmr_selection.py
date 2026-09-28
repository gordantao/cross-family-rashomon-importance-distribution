"""Redundancy-aware feature reordering (mRMR: Peng, Long & Ding 2005), applied
on top of an already-computed importance ranking rather than as a from-scratch
selection method.

RID's own VI metrics (sub_mr, loco, iloco) deliberately do NOT penalize
redundancy between features -- that's the point of characterizing importance
across a Rashomon set: two substitutable features can both look important
because different equally-good models rely on different ones (see the
--inject-redundant-duplicate / credit_split_ratio diagnostic in
run_falcon_cano_top40_comparison.py, built specifically to demonstrate this).
Folding a redundancy penalty into compute_rid itself would erase that signal,
which is the whole reason to use a Rashomon set in the first place.

But when the actual downstream goal is picking a k-sized feature SET that's
jointly useful for prediction (not characterizing the space of plausible
importances), marginal top-k ranking is a real handicap: two highly
correlated features can both occupy top slots while adding almost no
incremental predictive value together. Forward stepwise selection sidesteps
this for free (it only ever scores marginal CV gain given what's already
selected), which is part of why it tends to win a matched-k predictive
comparison against RID's raw ranking -- not because RID's importances are
wrong, but because "top-k by marginal importance" and "the best jointly
predictive k-feature subset" are different targets.

This module bridges that gap without touching RID's own computation: reorder
an existing importance ranking greedily, trading off each candidate's
importance against its redundancy with whatever's already been picked. No
extra model refits needed -- reuses whatever importance values (e.g. RID's
expected_importance) the caller already computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reorder_by_mrmr(
    feature_importance: dict[str, float],
    corr_matrix: pd.DataFrame,
    redundancy_lambda: float = 1.0,
) -> list[str]:
    """Greedily reorder features by importance minus redundancy_lambda times
    the max absolute correlation with already-selected features, instead of
    pure importance. Returns a full permutation of feature_importance's keys
    (not a top-k truncation -- callers truncate afterward).

    Importance is min-max normalized to [0, 1] first so redundancy_lambda has
    a comparable, interpretable scale against |correlation| (also in [0, 1]);
    otherwise raw importance magnitudes (e.g. small sub_mr deltas) could be
    dwarfed by, or dwarf, the correlation penalty depending on the metric.
    """

    features = list(feature_importance)
    if len(features) <= 1:
        return features

    raw = np.array([feature_importance[f] for f in features], dtype=float)
    lo, hi = raw.min(), raw.max()
    normalized = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
    relevance = dict(zip(features, normalized))

    remaining = set(features)
    selected: list[str] = []

    first = max(remaining, key=lambda f: relevance[f])
    selected.append(first)
    remaining.discard(first)

    while remaining:
        def combined_score(f: str) -> float:
            redundancy = max(corr_matrix.at[f, s] for s in selected)
            return relevance[f] - redundancy_lambda * redundancy

        best = max(remaining, key=combined_score)
        selected.append(best)
        remaining.discard(best)

    return selected
