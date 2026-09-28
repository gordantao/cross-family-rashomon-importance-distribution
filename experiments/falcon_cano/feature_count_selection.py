"""Choose how many top-ranked features to report via elbow detection on each
method's own already-computed signal curve, instead of an arbitrary fixed
top-k or a deviance-based BIC/AIC refit sweep (this file's earlier approach --
see `docs/aic_bic_feature_count_selection.md`, local-only, for that rationale
and results, now superseded by this one).

RID-based methods (rid_tree, cross_family_unweighted, cross_family_weighted)
already compute, as a side effect of fitting, each feature's `prob_positive`
(P(phi > 0) -- the fraction of bootstrap Rashomon-set models under which that
feature's importance was positive; see `summarize_metric_result` in
`rid/core.py`). A truly unimportant feature's sign is close to a coin flip
across bootstraps (prob_positive near 0.5); a genuinely important one
approaches 1.0. Sorted by rank, this is a scree-plot-shaped curve with a real
elbow, and unlike the old BIC/AIC sweep it costs nothing extra to compute --
RID already produces it at every rank up to whatever ceiling it was run to.

Stepwise methods (stepwise_logreg, stepwise_rf) have no Rashomon set and thus
no prob_positive curve, but forward stepwise selection already tracks
held-out CV score at each step as it grows the feature set -- a classic
diminishing-returns curve. The same elbow detector applies there too, just
with the curve's shape flipped (increasing/concave instead of
decreasing/convex), so every method's k comes from a signal curve native to
that method rather than an externally imposed criterion.
"""

from __future__ import annotations

from elbow_selection import find_elbow


def _select_k(values: list[float], curve: str, direction: str, criterion: str) -> dict:
    k_max = len(values)
    ranks = list(range(1, k_max + 1))
    elbow_rank = find_elbow(ranks, values, curve=curve, direction=direction)

    if elbow_rank is None:
        k = k_max
        at_ceiling = True
    else:
        k = max(1, min(int(round(elbow_rank)), k_max))
        at_ceiling = k == k_max

    return {"k": k, "at_ceiling": at_ceiling, "criterion": criterion, "k_max": k_max}


def select_k_by_prob_positive(prob_positive_by_rank: list[float]) -> dict:
    """prob_positive sorted by descending importance rank: starts near 1.0 for
    genuinely important features, drops to a ~0.5 noise floor. Convex/decreasing,
    like a scree plot's eigenvalue curve but bounded below at the noise floor
    instead of at zero."""

    return _select_k(prob_positive_by_rank, curve="convex", direction="decreasing", criterion="prob_positive_elbow")


def select_k_by_cv_gain(cv_score_by_rank: list[float]) -> dict:
    """Held-out CV score at each forward-stepwise step: rises while new features
    add real signal, flattens once they stop helping. Concave/increasing."""

    return _select_k(cv_score_by_rank, curve="concave", direction="increasing", criterion="cv_gain_elbow")
