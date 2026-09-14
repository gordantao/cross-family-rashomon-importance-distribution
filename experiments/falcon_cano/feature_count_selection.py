"""Choose how many top-ranked features to report via deviance-based AIC/BIC,
instead of an arbitrary fixed top-k.

Motivation: the RID paper (Donnelly, Katta, Rudin, Browne, NeurIPS 2023,
arXiv:2309.13775) doesn't actually use any statistical criterion to pick a
number of features -- its synthetic experiments evaluate every feature
(classifying extraneous ones by known ground truth), and its one real-data
case study pre-filters to a round number (top 100 by univariate AUC) purely
for computational tractability, then displays "the ten genes with the
highest P(RIV>0)" as a table convenience, not a validated cutoff. This
repo's flat --top-k default is no less arbitrary -- just arbitrary in a
different spot.

Classical AIC = 2k - 2ln(L_hat) and BIC = k*ln(n) - 2ln(L_hat) need a genuine
likelihood, which only a fitted LogisticRegression has among this project's
evaluator panel. -2ln(L_hat) is exactly 2 * n * log_loss(y, y_prob)
(deviance) for any predict_proba-capable model, so a deviance-based AIC/BIC
generalizes cleanly to RandomForest/SVM too -- an established practical
approximation for RELATIVE comparison across k on the SAME model (it loses
classical AIC's exact asymptotic justification, which assumes a correctly
specified parametric likelihood, but that's not what it's being used for
here).

Fit in-sample (that's the point of AIC/BIC -- no held-out split needed) at
each k along a method's own ranking, for every evaluator in the panel used
elsewhere in this pipeline (avoids picking k based on one evaluator's blind
spots, same reasoning as the held-out CV panel). Unlike the
correlation-threshold sweep (a monotonic curve needing Kneedle-style elbow
detection), a BIC-vs-k curve has a genuine interior minimum in the typical
case: adding real signal lowers it, adding noise no longer worth its
complexity penalty raises it -- so the selection rule here is just argmin,
not an elbow heuristic. If the curve never turns (still decreasing at
k_max), that means k_max was set too low; this is flagged via `at_ceiling`
rather than silently forcing a fake elbow onto a monotonic curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def compute_deviance_criteria(y_true, y_prob, k: int, n: int) -> tuple[float, float]:
    """(aic, bic) from deviance = 2 * n * log_loss(y_true, y_prob)."""

    deviance = 2.0 * n * log_loss(y_true, y_prob, labels=[0, 1])
    aic = deviance + 2.0 * k
    bic = deviance + k * np.log(n)
    return float(aic), float(bic)


def sweep_feature_counts(
    X: pd.DataFrame,
    y: pd.Series,
    ranked_features: list[str],
    evaluator_panel: list[tuple[str, type, dict]],
    k_max: int,
) -> pd.DataFrame:
    """In-sample AIC/BIC at k=1..k_max along ranked_features, per evaluator."""

    n = len(y)
    k_max = min(k_max, len(ranked_features))
    rows = []
    for k in range(1, k_max + 1):
        features_k = ranked_features[:k]
        for evaluator_name, model_cls, model_kwargs in evaluator_panel:
            model = model_cls(**model_kwargs)
            model.fit(X[features_k], y)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X[features_k])[:, 1]
            else:
                raw = model.decision_function(X[features_k])
                y_prob = 1.0 / (1.0 + np.exp(-raw))
            aic, bic = compute_deviance_criteria(y, y_prob, k, n)
            rows.append({"k": k, "evaluator": evaluator_name, "aic": aic, "bic": bic})
    return pd.DataFrame(rows)


def select_k(sweep_df: pd.DataFrame, criterion: str = "bic") -> dict:
    """Per-evaluator argmin k, plus a consensus k (max across evaluators --
    conservative toward keeping more features any one evaluator found
    informative, rather than only the features a linear evaluator likes)."""

    k_max = int(sweep_df["k"].max())
    per_evaluator = {}
    for evaluator_name, group in sweep_df.groupby("evaluator"):
        best_row = group.loc[group[criterion].idxmin()]
        per_evaluator[evaluator_name] = {
            "best_k": int(best_row["k"]),
            "best_value": float(best_row[criterion]),
            "at_ceiling": int(best_row["k"]) == k_max,
        }
    consensus_k = max(v["best_k"] for v in per_evaluator.values())
    return {"per_evaluator": per_evaluator, "consensus_k": consensus_k, "criterion": criterion, "k_max": k_max}
