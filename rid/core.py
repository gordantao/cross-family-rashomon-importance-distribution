from __future__ import annotations

import cProfile
import logging
import os
import sys
import time
import warnings
from collections import Counter
from functools import partial
from inspect import signature
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from ._logging import ProgressLogger, format_duration
from .candidate_pool import (
    DEFAULT_N_CANDIDATES_PER_FAMILY,
    build_candidate_pool,
    fit_candidate_pool,
    resolve_default_search_space,
)
from .models import resolve_candidate_trainer, resolve_model_config, train_candidate_models
from .truvarimp import resolve_truvarimp_config, run_joint_truvarimp


VALID_VI_METRICS = ("sub_mr", "loco", "coef", "iloco")
VALID_PERFORMANCE_METRICS = ("accuracy", "f1", "auprc")
VALID_CROSS_FAMILY_BALANCE_MODES = ("unweighted", "weighted")
VALID_CANDIDATE_SEARCH_MODES = ("grid", "truvarimp_joint")
VALID_PROFILE_MODES = ("timers", "cprofile")

logger = logging.getLogger(__name__)


def compute_model_reliance(model, X, y, metric="sub_mr", rng=None):
    """Compute permutation-based model reliance for all features."""

    if rng is None:
        rng = np.random.default_rng()

    n_features = X.shape[1]
    y_prob = model.predict_proba(X)
    e_orig = log_loss(y, y_prob, labels=model.classes_)

    importances = np.zeros(n_features)
    for j in range(n_features):
        X_permuted = X.copy()
        X_permuted[:, j] = rng.permutation(X_permuted[:, j])
        y_prob_perm = model.predict_proba(X_permuted)
        e_switch = log_loss(y, y_prob_perm, labels=model.classes_)

        if metric == "sub_mr":
            importances[j] = e_switch - e_orig
        else:
            importances[j] = e_switch / e_orig if e_orig > 0 else 1.0

    return importances


def compute_loco_importance(model, X, y):
    """Compute leave-one-covariate-out importance.

    Ablation-based: each column is set to 0 on the already-fitted model. RID
    standardizes each bootstrap on its in-bag rows, so 0 is the in-bag mean.
    """

    e_orig = log_loss(y, model.predict_proba(X), labels=model.classes_)
    importances = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        X_loco = X.copy()
        X_loco[:, j] = 0.0
        importances[j] = log_loss(y, model.predict_proba(X_loco), labels=model.classes_) - e_orig
    return importances


def _eligible_interaction_pairs(X, correlation_threshold=0.8):
    """Return (j, k) index pairs whose features are NOT highly correlated.

    Near-duplicate features have an uninformative "interaction" by
    construction (removing both is basically the same as removing either),
    so skipping them keeps the O(M^2) pairwise sweep from being wasted on
    pairs nobody cares about. Returns (eligible_pairs, n_total_pairs).
    """

    n_features = X.shape[1]
    n_total_pairs = n_features * (n_features - 1) // 2
    if n_features < 2:
        return [], n_total_pairs

    corr = np.corrcoef(X, rowvar=False)
    # A column constant over these rows has undefined correlation (NaN, so its
    # pairs are skipped). Mark it explicitly: np.corrcoef's mean of n equal
    # nonzero values can be off by rounding, giving a garbage finite value that
    # would make eligibility depend on the column's offset/scale.
    constant = np.ptp(X, axis=0) == 0
    corr[constant, :] = np.nan
    corr[:, constant] = np.nan
    eligible_pairs = [
        (j, k)
        for j in range(n_features)
        for k in range(j + 1, n_features)
        if not np.isnan(corr[j, k]) and abs(corr[j, k]) < correlation_threshold
    ]
    return eligible_pairs, n_total_pairs


def compute_iloco_pairwise_importance(model, X, y, pairs):
    """Compute iLOCO_{j,k} = delta_j + delta_k - delta_{j,k} for given pairs.

    Ablation-based (zero the column(s) on the ALREADY-FITTED model -- under
    RID's per-bootstrap scaling, 0 is the in-bag mean -- and re-measure
    log-loss), matching compute_loco_importance's no-refit style,
    rather than the original iLOCO paper's refit-per-subset approach --
    consistent with how this codebase already computes loco, and far cheaper.
    Single-feature deltas are computed once and reused across every pair
    instead of being recomputed per pair.
    """

    e_orig = log_loss(y, model.predict_proba(X), labels=model.classes_)

    single_delta = {}

    def _delta_for(indices):
        key = tuple(sorted(indices))
        if key in single_delta:
            return single_delta[key]
        X_ablated = X.copy()
        for idx in indices:
            X_ablated[:, idx] = 0.0
        delta = log_loss(y, model.predict_proba(X_ablated), labels=model.classes_) - e_orig
        single_delta[key] = delta
        return delta

    iloco_by_pair = {}
    for j, k in pairs:
        delta_j = _delta_for((j,))
        delta_k = _delta_for((k,))
        delta_jk = _delta_for((j, k))
        iloco_by_pair[(j, k)] = delta_j + delta_k - delta_jk

    return iloco_by_pair


def describe_iloco_eligible_pairs(X, correlation_threshold=0.8):
    """Diagnostic helper: (n_eligible_pairs, n_total_pairs) for a feature matrix.

    Meant to be called ONCE by a caller inspecting a dataset/metric config
    (e.g. before kicking off a full RID run), not from inside the per-model,
    per-bootstrap hot path that compute_iloco_importance runs in -- that path
    can execute this tens of thousands of times per run, so it stays silent.
    """

    eligible_pairs, n_total_pairs = _eligible_interaction_pairs(X, correlation_threshold)
    return len(eligible_pairs), n_total_pairs


def compute_iloco_importance(model, X, y, correlation_threshold=0.8, reduction="sum_abs"):
    """Per-feature interaction-importance summary derived from pairwise iLOCO.

    Reduces the pairwise iLOCO_{j,k} matrix to one value per feature (sum, by
    default, of |iLOCO_{j,k}| over every partner k the feature was eligible
    to pair with) so it slots into the same per-feature VI-metric shape as
    sub_mr/loco (_coerce_importance_values expects exactly n_features values)
    rather than requiring a separate pairwise aggregation path. This is a
    summary of "how much does this feature's contribution move as part of
    2-way interactions", not the interaction detail itself -- the full
    pairwise dict is available via compute_iloco_pairwise_importance for
    anyone who wants the per-pair detail, and eligible/total pair counts via
    describe_iloco_eligible_pairs.

    Returns just the per-feature importance array (n_features,), matching the
    convention used by compute_model_reliance/compute_loco_importance -- this
    function is called once per Rashomon-set model per bootstrap, so it does
    not print or return diagnostics on every call.
    """

    n_features = X.shape[1]
    eligible_pairs, _ = _eligible_interaction_pairs(X, correlation_threshold)
    iloco_by_pair = compute_iloco_pairwise_importance(model, X, y, eligible_pairs)

    importances = np.zeros(n_features)
    for (j, k), value in iloco_by_pair.items():
        magnitude = abs(value) if reduction == "sum_abs" else value
        importances[j] += magnitude
        importances[k] += magnitude

    return importances


def compute_coef_importance(model):
    """Return normalized absolute coefficients for linear models."""

    if not hasattr(model, "coef_"):
        return None
    coefs = np.abs(model.coef_).mean(axis=0)
    total = coefs.sum()
    return coefs / total if total > 0 else np.zeros(coefs.shape)


def vi_sub_mr(model, X, y, rng=None):
    """Return subtraction-based model reliance importance."""

    return compute_model_reliance(model, X, y, metric="sub_mr", rng=rng)


def vi_loco(model, X, y, rng=None):
    """Return leave-one-covariate-out importance."""

    return compute_loco_importance(model, X, y)


def vi_coef(model, X, y=None, rng=None):
    """Return normalized coefficient importance for linear models."""

    return compute_coef_importance(model)


def vi_iloco(model, X, y, rng=None):
    """Return per-feature interaction-importance summary (pairwise iLOCO)."""

    return compute_iloco_importance(model, X, y)


def performance_accuracy(model, X, y, y_pred=None, y_prob=None):
    """Return classification accuracy on the bootstrap sample."""

    if y_pred is None:
        y_pred = model.predict(X)
    return float(accuracy_score(y, y_pred))


def performance_f1(model, X, y, y_pred=None, y_prob=None):
    """Return binary or macro F1, matching the previous RID behavior."""

    if y_pred is None:
        y_pred = model.predict(X)

    n_classes = len(np.unique(y))
    if n_classes == 2:
        return float(f1_score(y, y_pred))
    return float(f1_score(y, y_pred, average="macro"))


def performance_auprc(model, X, y, y_pred=None, y_prob=None):
    """Return binary or macro average precision, matching the previous RID behavior."""

    if y_prob is None:
        y_prob = model.predict_proba(X)

    n_classes = len(np.unique(y))
    if n_classes == 2:
        y_score = y_prob[:, 1] if np.ndim(y_prob) == 2 else y_prob
        return float(average_precision_score(y, y_score))
    return float(average_precision_score(y, y_prob, average="macro"))


DEFAULT_VI_METRIC_FUNCTIONS = {
    "sub_mr": vi_sub_mr,
    "loco": vi_loco,
    "coef": vi_coef,
    "iloco": vi_iloco,
}
DEFAULT_PERFORMANCE_METRIC_FUNCTIONS = {
    "accuracy": performance_accuracy,
    "f1": performance_f1,
    "auprc": performance_auprc,
}

CALLABLE_METRIC_ALIASES = {
    compute_model_reliance: "sub_mr",
    compute_loco_importance: "loco",
    compute_coef_importance: "coef",
    compute_iloco_importance: "iloco",
    vi_sub_mr: "sub_mr",
    vi_loco: "loco",
    vi_coef: "coef",
    vi_iloco: "iloco",
    accuracy_score: "accuracy",
    f1_score: "f1",
    average_precision_score: "auprc",
    performance_accuracy: "accuracy",
    performance_f1: "f1",
    performance_auprc: "auprc",
}


def _metric_name(metric):
    explicit_name = getattr(metric, "metric_name", None)
    if explicit_name:
        return str(explicit_name)

    if isinstance(metric, partial):
        if metric.func is compute_model_reliance:
            partial_metric = (metric.keywords or {}).get("metric")
            if partial_metric:
                return str(partial_metric)
        if metric.func in CALLABLE_METRIC_ALIASES:
            return CALLABLE_METRIC_ALIASES[metric.func]
        return getattr(metric.func, "__name__", metric.func.__class__.__name__)

    if metric in CALLABLE_METRIC_ALIASES:
        return CALLABLE_METRIC_ALIASES[metric]

    return getattr(metric, "__name__", metric.__class__.__name__)


def _normalize_metric_specs(metrics, default_metrics, label):
    if metrics is None:
        entries = list(default_metrics.items())
    else:
        raw_metrics = [metrics] if isinstance(metrics, str) or callable(metrics) else list(metrics)
        entries = []
        for metric in raw_metrics:
            if isinstance(metric, str):
                if metric not in default_metrics:
                    raise ValueError(
                        f"Unknown {label} metric '{metric}'. Supported metrics: {tuple(default_metrics)}"
                    )
                entries.append((metric, default_metrics[metric]))
                continue
            if not callable(metric):
                raise TypeError(
                    f"Each {label} metric must be a string key or callable, got {type(metric).__name__}"
                )
            entries.append((_metric_name(metric), metric))

    deduped = []
    seen = set()
    for name, metric in entries:
        if name not in seen:
            seen.add(name)
            deduped.append((name, metric))

    if not deduped:
        raise ValueError(f"{label} metrics cannot be empty")

    return tuple(deduped)


def _call_metric(metric, *, model, X, y, y_pred=None, y_prob=None, rng=None):
    y_score = None
    if y_prob is not None:
        y_score = y_prob[:, 1] if np.ndim(y_prob) == 2 and y_prob.shape[1] == 2 else y_prob

    context = {
        "model": model,
        "X": X,
        "y": y,
        "y_true": y,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_score": y_score,
        "rng": rng,
    }

    try:
        parameters = signature(metric).parameters.values()
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Could not inspect metric callable '{_metric_name(metric)}'. "
            "Provide a Python callable with an inspectable signature."
        ) from exc

    kwargs = {}
    accepts_var_kwargs = False
    for parameter in parameters:
        if parameter.kind == parameter.VAR_KEYWORD:
            accepts_var_kwargs = True
            continue
        if parameter.kind == parameter.VAR_POSITIONAL:
            continue
        if parameter.name in context and context[parameter.name] is not None:
            kwargs[parameter.name] = context[parameter.name]

    if accepts_var_kwargs:
        kwargs = {name: value for name, value in context.items() if value is not None}

    return metric(**kwargs)


def _coerce_importance_values(metric_name, values, n_features):
    if values is None:
        return None

    importance_array = np.asarray(values, dtype=float).reshape(-1)
    if importance_array.shape[0] != n_features:
        raise ValueError(
            f"VI metric '{metric_name}' must return one value per feature; "
            f"expected {n_features}, got {importance_array.shape[0]}"
        )
    return importance_array


def _coerce_performance_value(metric_name, value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Performance metric '{metric_name}' must return a scalar float-compatible value"
        ) from exc


def normalize_vi_metrics(vi_metrics):
    """Normalize VI metrics into `(name, callable)` pairs."""

    return _normalize_metric_specs(vi_metrics, DEFAULT_VI_METRIC_FUNCTIONS, "VI")


def normalize_performance_metrics(performance_metrics):
    """Normalize performance metrics into `(name, callable)` pairs."""

    return _normalize_metric_specs(
        performance_metrics,
        DEFAULT_PERFORMANCE_METRIC_FUNCTIONS,
        "performance",
    )


def normalize_cross_family_balance_mode(family_balance_mode):
    """Normalize and validate cross-family aggregation mode."""

    if family_balance_mode is None:
        return "unweighted"
    if family_balance_mode not in VALID_CROSS_FAMILY_BALANCE_MODES:
        raise ValueError(
            f"Unknown family_balance_mode '{family_balance_mode}'. "
            f"Supported modes: {VALID_CROSS_FAMILY_BALANCE_MODES}"
        )
    return family_balance_mode


def _declared_search_grid(model_config):
    """Return the custom search grid a model config declares, if any.

    Mirrors the shapes ``resolve_model_config`` accepts, so callers can tell
    whether an override exists without re-resolving the whole config.
    """

    if isinstance(model_config, Mapping):
        return model_config.get("search_grid")
    if isinstance(model_config, (tuple, list)):
        return model_config[2] if len(model_config) >= 3 else None
    return None


def normalize_candidate_search(candidate_search):
    """Normalize and validate the cross-family candidate-search strategy."""

    if candidate_search is None:
        return "grid"
    if candidate_search not in VALID_CANDIDATE_SEARCH_MODES:
        raise ValueError(
            f"Unknown candidate_search '{candidate_search}'. "
            f"Supported modes: {VALID_CANDIDATE_SEARCH_MODES}"
        )
    return candidate_search


def normalize_profile(profile, profile_dir=None):
    """Validate the opt-in per-bootstrap profiler; return ``(profile, profile_dir)``.

    ``profile`` is ``None``/``"off"`` (no profiling; returned as ``None``),
    ``"timers"`` or ``"cprofile"``. ``"cprofile"`` needs ``profile_dir``, which
    is created here (in the parent, before any worker writes into it) and
    returned as an absolute path so loky workers resolve it the same way.
    """

    if profile is None or profile == "off":
        profile = None
    elif profile not in VALID_PROFILE_MODES:
        raise ValueError(f"Unknown profile={profile!r}. Choose None/'off', 'timers' or 'cprofile'.")
    if profile == "cprofile":
        if profile_dir is None:
            raise ValueError("profile='cprofile' needs profile_dir (where bootstrap_<b>.prof files go)")
        profile_dir = os.path.abspath(os.fspath(profile_dir))
        os.makedirs(profile_dir, exist_ok=True)
    elif profile_dir is not None:
        raise ValueError(f"profile_dir is only used with profile='cprofile', not profile={profile!r}")
    return profile, profile_dir


def summarize_metric_result(metric_result):
    """Compute per-feature expected importance and P(phi > 0).

    ``metric_result`` is one value of ``metric_results_``: the 5-tuple
    ``(rid_cdfs, cdf_grid, raw_importances, expected_importance,
    prob_positive)`` built by ``_aggregate_metric_results``.

    ``expected_importance`` is the mean of the RID distribution itself: the
    average, over valid bootstraps, of each bootstrap's mean importance across
    its Rashomon set, so every bootstrap counts equally (as in the averaged
    ECDFs) regardless of how many models it kept. Within a bootstrap the mean
    is weighted by the per-model weights when the metric result carries them
    (cross-family ``weighted`` balance mode) and plain otherwise
    (single-family, or cross-family ``unweighted``). ``raw_importances`` stays
    the pooled per-model values, so its plain mean is *not* this quantity.

    ``prob_positive`` is P(phi > 0) under the same distribution, computed
    exactly from the per-model values (not read off the discretized CDF, whose
    first grid point at or above 0 would drop the mass in (0, grid point]):
    the average, over valid bootstraps, of each bootstrap's (weighted, where
    per-model weights exist) fraction of Rashomon models with phi strictly
    greater than 0. Exact zeros (e.g. a feature a model does not use) do not
    count as positive. For ``iloco`` (``sum_abs`` reduction, always >= 0) it
    is the fraction of models where the feature has any nonzero interaction.
    """

    rid_cdfs, _cdf_grid, _raw_importances, expected_importance, prob_positive = metric_result

    summary = {}
    for feature_name in rid_cdfs:
        summary[feature_name] = {
            "expected_importance": expected_importance[feature_name],
            "prob_positive": float(prob_positive[feature_name]),
        }
    return summary


def _prepare_inputs(X, y):
    X_values = np.asarray(X)
    if X_values.ndim != 2:
        raise ValueError("X must be 2-dimensional")

    y_values = np.asarray(y).reshape(-1)
    if X_values.shape[0] != y_values.shape[0]:
        raise ValueError("X and y must contain the same number of rows")

    if hasattr(X, "columns"):
        feature_names = [str(column) for column in X.columns]
    else:
        feature_names = [f"x{i}" for i in range(X_values.shape[1])]

    return X_values, y_values, feature_names


def _prepare_groups(groups, n_rows):
    """Encode ``groups`` as integer codes for :func:`_bootstrap_split`.

    Returns ``(codes, n_groups)``, or ``(None, None)`` when ``groups`` is None.
    Codes index the unique labels in sorted order, so the grouped bootstrap's
    draw is deterministic for a given set of labels regardless of row order.
    """

    if groups is None:
        return None, None
    labels = np.asarray(groups)
    if labels.ndim != 1:
        # e.g. tuple labels, which np.asarray would expand into a 2-D array
        labels = np.empty(len(groups), dtype=object)
        labels[:] = list(groups)
    if labels.shape[0] != n_rows:
        raise ValueError(f"groups must have one label per row of X ({n_rows}), got {labels.shape[0]}")
    import pandas as pd  # only needed on the grouped path; every experiment env has it

    # np.unique would silently merge float NaN/NA/NaT rows into one fake group
    # (or give each object-dtype NaN its own group), so reject missing labels.
    missing = np.asarray(pd.isna(labels), dtype=bool)
    if missing.any():
        raise ValueError(f"groups has {int(missing.sum())} missing labels (NaN/None/NA); give every row a group")
    try:
        unique_labels, codes = np.unique(labels, return_inverse=True)
    except TypeError as exc:
        raise ValueError("groups labels must be mutually sortable (e.g. all str or all int)") from exc
    return codes.reshape(-1), len(unique_labels)


def _active_metrics(per_bootstrap_importances, feature_names, metric_names):
    return [
        metric
        for metric in metric_names
        if any(len(boot[metric][feature_names[0]]) > 0 for boot in per_bootstrap_importances)
    ]


def _aggregate_metric_results(per_bootstrap_importances, feature_names, metrics, n_cdf_points, per_bootstrap_weights=None):
    metric_results = {}

    for metric in metrics:
        raw_importances = {feature_name: [] for feature_name in feature_names}
        for boot_importances in per_bootstrap_importances:
            for feature_name in feature_names:
                raw_importances[feature_name].extend(boot_importances[metric][feature_name])

        all_values = [value for values in raw_importances.values() for value in values]
        cdf_grid = np.linspace(np.min(all_values), np.max(all_values), n_cdf_points)
        rid_cdfs = {feature_name: np.zeros(n_cdf_points) for feature_name in feature_names}
        # Per-bootstrap (weighted) means, summed and divided like the ECDFs below,
        # so expected_importance is the mean of the averaged-ECDF mixture: each
        # valid bootstrap counts once, regardless of its Rashomon-set size.
        boot_mean_sums = {feature_name: 0.0 for feature_name in feature_names}
        boot_mean_counts = {feature_name: 0 for feature_name in feature_names}
        # Same per-bootstrap averaging for P(phi > 0), computed exactly from the
        # values rather than read off the discretized CDF.
        boot_positive_sums = {feature_name: 0.0 for feature_name in feature_names}

        valid_count = 0
        for i, boot_importances in enumerate(per_bootstrap_importances):
            if not boot_importances[metric][feature_names[0]]:
                continue
            valid_count += 1
            boot_weights = per_bootstrap_weights[i][metric] if per_bootstrap_weights is not None else None
            for feature_name in feature_names:
                values = boot_importances[metric][feature_name]
                if len(values) == 0:
                    continue
                boot_mean_sums[feature_name] += float(np.average(values, weights=boot_weights))
                boot_mean_counts[feature_name] += 1
                boot_positive_sums[feature_name] += float(
                    np.average(np.asarray(values) > 0, weights=boot_weights)
                )
                if boot_weights is not None:
                    values_arr = np.array(values)
                    weights_arr = np.array(boot_weights)
                    sorted_idx = np.argsort(values_arr)
                    sorted_vals = values_arr[sorted_idx]
                    cum_weights = np.cumsum(weights_arr[sorted_idx])
                    insert_idx = np.searchsorted(sorted_vals, cdf_grid, side="right")
                    ecdf = np.where(insert_idx > 0, cum_weights[insert_idx - 1], 0.0)
                else:
                    values_arr = np.sort(np.array(values))
                    ecdf = np.searchsorted(values_arr, cdf_grid, side="right") / len(values_arr)
                rid_cdfs[feature_name] += ecdf

        for feature_name in feature_names:
            rid_cdfs[feature_name] /= valid_count

        expected_importance = {
            feature_name: (
                boot_mean_sums[feature_name] / boot_mean_counts[feature_name]
                if boot_mean_counts[feature_name]
                else 0.0
            )
            for feature_name in feature_names
        }
        prob_positive = {
            feature_name: (
                boot_positive_sums[feature_name] / boot_mean_counts[feature_name]
                if boot_mean_counts[feature_name]
                else 0.0
            )
            for feature_name in feature_names
        }

        metric_results[metric] = (rid_cdfs, cdf_grid, raw_importances, expected_importance, prob_positive)

    return metric_results


def _aggregate_perf_stats(per_bootstrap_perf, performance_metric_specs):
    aggregated = {}
    for metric_name, _ in performance_metric_specs:
        per_boot_values = [np.mean(perf[metric_name]) for perf in per_bootstrap_perf if perf[metric_name]]
        aggregated[f"{metric_name}_mean"] = (
            float(np.mean(per_boot_values)) if per_boot_values else float("nan")
        )
        aggregated[f"{metric_name}_std"] = (
            float(np.std(per_boot_values)) if per_boot_values else float("nan")
        )
    return aggregated


def _aggregate_family_perf_stats(per_bootstrap_family_perf, family_names, performance_metric_specs):
    per_family_metrics = {
        family_name: {metric_name: [] for metric_name, _ in performance_metric_specs}
        for family_name in family_names
    }

    for boot_family_perf in per_bootstrap_family_perf:
        for family_name in family_names:
            for metric_name, _ in performance_metric_specs:
                if boot_family_perf[family_name][metric_name]:
                    per_family_metrics[family_name][metric_name].append(
                        np.mean(boot_family_perf[family_name][metric_name])
                    )

    aggregated = {}
    for family_name in family_names:
        aggregated[family_name] = {}
        for metric_name, _ in performance_metric_specs:
            values = per_family_metrics[family_name][metric_name]
            aggregated[family_name][f"{metric_name}_mean"] = (
                float(np.mean(values)) if values else float("nan")
            )
            aggregated[family_name][f"{metric_name}_std"] = (
                float(np.std(values)) if values else float("nan")
            )
    return aggregated


def _format_perf_stats(perf_stats, performance_metric_specs):
    parts = []
    for metric_name, _ in performance_metric_specs:
        parts.append(
            f"{metric_name}={perf_stats[f'{metric_name}_mean']:.3f}\u00b1{perf_stats[f'{metric_name}_std']:.3f}"
        )
    return "  ".join(parts)


def _run_bootstrap_task(worker, b, suppress_warnings, *args, profile=None, profile_dir=None):
    """Run one bootstrap worker, returning ``(b, result, stats)``.

    Warnings are ignored here, inside the worker process, because joblib's loky
    workers don't share the parent's warning filters -- without this, every
    sklearn ConvergenceWarning/FutureWarning from every candidate fit lands in
    the job's stderr. ``catch_warnings`` restores the filters afterwards, which
    matters when ``n_jobs=1`` runs the worker in the caller's own process.

    ``profile`` (see :func:`normalize_profile`): ``"timers"`` passes the worker a
    ``timers`` dict to fill with stage wall-clocks and stores a flat record in
    ``stats["profile"]`` (:func:`_profile_record`); ``"cprofile"`` does the same
    and also runs the worker under :class:`cProfile.Profile`, dumping
    ``<profile_dir>/bootstrap_<b>.prof`` from here, i.e. inside the joblib
    worker where the time is actually spent. ``None`` calls the worker exactly
    as before.
    """

    start = time.perf_counter()
    kwargs = {}
    timers = None
    if profile is not None:
        timers = kwargs["timers"] = {}
    profiler = cProfile.Profile() if profile == "cprofile" else None
    if profiler is not None:
        profiler.enable()
    try:
        with warnings.catch_warnings():
            if suppress_warnings:
                warnings.simplefilter("ignore")
            result, stats = worker(b, *args, **kwargs)
        stats["seconds"] = time.perf_counter() - start
    finally:
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(os.path.join(profile_dir, f"bootstrap_{b}.prof"))
    if timers is not None:
        stats["profile"] = _profile_record(b, stats, timers)
    return b, result, stats


def _profile_task_kwargs(profile, profile_dir):
    """Keyword arguments for ``_run_bootstrap_task``: none at all when profiling is off."""

    return {"profile": profile, "profile_dir": profile_dir} if profile else {}


def _add_time(timers, key, start):
    """Add the wall-clock since ``start`` to ``timers[key]``; no-op when profiling is off."""

    if timers is not None:
        timers[key] = timers.get(key, 0.0) + (time.perf_counter() - start)


def _worker_max_rss_mb():
    """Peak resident memory of this (worker) process so far, in MB, or None if unavailable."""

    try:
        import resource
    except ImportError:  # Windows
        return None
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    return float(max_rss) / (1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0)


def _profile_record(b, stats, timers):
    """One bootstrap's flat profiler record (one entry of ``profile_``).

    Totals: ``seconds`` (whole bootstrap), ``split_seconds`` (bootstrap split
    and in-bag scaling), ``search_seconds`` (candidate pool and fits; for
    ``truvarimp_joint`` also the GP/acquisition, split into ``fit_seconds`` and
    ``gp_seconds``), ``vi_seconds`` and ``perf_seconds`` (VI / performance
    metrics over the Rashomon members) and ``other_seconds`` (the remainder:
    Rashomon filtering and bookkeeping). Breakdowns:
    ``vi_seconds/<family>/<metric>`` (single-family: ``vi_seconds/<metric>``),
    ``perf_seconds/<family>``, ``rashomon_size[/<family>]`` and
    ``n_fits[/<family>]``. ``worker_pid`` and ``worker_max_rss_mb`` (the worker
    process's peak RSS so far -- a loky worker runs many bootstraps, so this
    is a running maximum, not this bootstrap's own peak) help attribute memory.
    """

    vi_seconds = sum(v for k, v in timers.items() if k.startswith("vi_seconds/"))
    perf_seconds = sum(v for k, v in timers.items() if k == "perf_seconds" or k.startswith("perf_seconds/"))
    record = {
        "b": int(b),
        "valid": bool(stats.get("valid")),
        "seconds": float(stats["seconds"]),
        "split_seconds": timers.get("split_seconds", 0.0),
        "search_seconds": timers.get("search_seconds", 0.0),
    }
    for key in ("fit_seconds", "gp_seconds"):
        if key in timers:
            record[key] = timers[key]
    record["vi_seconds"] = vi_seconds
    record["perf_seconds"] = perf_seconds
    record["other_seconds"] = record["seconds"] - (
        record["split_seconds"] + record["search_seconds"] + vi_seconds + perf_seconds
    )
    record.update({k: v for k, v in timers.items() if k.startswith(("vi_seconds/", "perf_seconds/"))})
    for key in ("rashomon_size", "n_fits"):
        value = stats.get(key)
        if isinstance(value, Mapping):
            record.update({f"{key}/{name}": int(count) for name, count in value.items()})
        else:
            record[key] = int(value or 0)
    record["worker_pid"] = os.getpid()
    record["worker_max_rss_mb"] = _worker_max_rss_mb()
    return record


def _format_profile_record(record):
    """One-line stage summary of a profiler record, for the per-bootstrap INFO log."""

    search = f"search={record['search_seconds']:.1f}s"
    if "fit_seconds" in record:
        search += f" (fit={record['fit_seconds']:.1f}s gp={record['gp_seconds']:.1f}s)"
    vi_parts = [
        f"{key[len('vi_seconds/'):]}={value:.1f}s"
        for key, value in record.items()
        if key.startswith("vi_seconds/") and value > 0
    ]
    vi = f"vi={record['vi_seconds']:.1f}s" + (f" [{', '.join(vi_parts)}]" if vi_parts else "")
    sizes = {k[len("rashomon_size/"):]: v for k, v in record.items() if k.startswith("rashomon_size/")}
    parts = [
        f"total={record['seconds']:.1f}s",
        f"split={record['split_seconds']:.1f}s",
        search,
        vi,
        f"perf={record['perf_seconds']:.1f}s",
        f"other={record['other_seconds']:.1f}s",
        f"rashomon={sizes if sizes else record.get('rashomon_size', 0)}",
    ]
    if record.get("worker_max_rss_mb") is not None:
        parts.append(f"worker_max_rss={record['worker_max_rss_mb']:.0f}MB")
    return " ".join(parts)


def _make_parallel(n_jobs):
    # generator_unordered (joblib>=1.4) yields each bootstrap as soon as it
    # finishes, which is what makes live progress possible; older joblib only
    # has the in-order generator, which still works, just with laggier progress.
    try:
        return Parallel(n_jobs=n_jobs, return_as="generator_unordered")
    except ValueError:
        return Parallel(n_jobs=n_jobs, return_as="generator")


def _collect_bootstrap_results(tasks, n_jobs, n_bootstraps, label):
    """Consume bootstrap tasks as they complete, logging progress; return ``(results, stats)`` sorted by b.

    Sorting by bootstrap index keeps aggregation order -- and therefore every
    downstream number -- identical to the previous ordered ``Parallel`` call.
    """

    progress = ProgressLogger(label, n_bootstraps, log=logger)
    collected = []
    n_valid = 0
    for b, result, stats in _make_parallel(n_jobs)(tasks):
        collected.append((b, result, stats))
        n_valid += bool(stats.get("valid"))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("%s: bootstrap %d %s", label, b, _format_bootstrap_stats(stats))
        if "profile" in stats:
            logger.info("%s: bootstrap %d profile: %s", label, b, _format_profile_record(stats["profile"]))
        progress.update(extra=f"valid={n_valid}")
    collected.sort(key=lambda item: item[0])
    return [result for _, result, _ in collected], [stats for _, _, stats in collected]


def _format_bootstrap_stats(stats):
    parts = [f"{stats['seconds']:.1f}s", f"fits={stats['n_fits']}", f"rashomon={stats['rashomon_size']}"]
    if stats.get("truvarimp"):
        search_stats = {k: v for k, v in stats["truvarimp"].items() if k != "pinned"}
        parts.append(f"truvarimp={search_stats}")
    return " ".join(parts)


def _mean_by_key(dicts):
    keys = list(dicts[0]) if dicts else []
    return {key: float(np.mean([d.get(key, 0) for d in dicts])) for key in keys}


def _log_bootstrap_summary(label, bootstrap_stats):
    if not bootstrap_stats:
        return
    seconds = np.array([stats["seconds"] for stats in bootstrap_stats])
    logger.info(
        "%s: per-bootstrap time mean=%s median=%s max=%s",
        label,
        format_duration(seconds.mean()),
        format_duration(np.median(seconds)),
        format_duration(seconds.max()),
    )

    def fmt(means):
        return ", ".join(f"{key}={value:.1f}" for key, value in means.items())

    n_fits = [stats["n_fits"] for stats in bootstrap_stats]
    rashomon = [stats["rashomon_size"] for stats in bootstrap_stats]
    if isinstance(n_fits[0], dict):
        logger.info("%s: mean fits/bootstrap: %s", label, fmt(_mean_by_key(n_fits)))
        logger.info("%s: mean Rashomon size/bootstrap: %s", label, fmt(_mean_by_key(rashomon)))
    else:
        logger.info(
            "%s: mean fits/bootstrap=%.1f, mean Rashomon size/bootstrap=%.1f",
            label,
            float(np.mean(n_fits)),
            float(np.mean(rashomon)),
        )

    search_stats = [stats["truvarimp"] for stats in bootstrap_stats if stats.get("truvarimp")]
    if search_stats:
        numeric = [
            {k: v for k, v in s.items() if k not in ("stop_reason", "pinned")} for s in search_stats
        ]
        stop_reasons = Counter(s["stop_reason"] for s in search_stats)
        logger.info(
            "%s: TruVaRImp mean/bootstrap: %s; stop reasons: %s",
            label,
            fmt(_mean_by_key(numeric)),
            dict(stop_reasons),
        )
        # Pinned dimensions depend only on each family's model kwargs, so they're
        # the same every bootstrap; report them once here, from the parent process.
        pinned = {}
        for s in search_stats:
            for family_name, dims in s.get("pinned", {}).items():
                pinned.setdefault(family_name, dims)
        for family_name, dims in pinned.items():
            logger.info(
                "%s: family '%s' pins %s via its model kwargs; those dimensions are held fixed "
                "rather than searched.",
                label,
                family_name,
                dims,
            )


def _bootstrap_split(X, y, b, groups=None):
    """In-bag resample and out-of-bag rows for bootstrap ``b``.

    The in-bag rows are exactly what ``resample(X, y, random_state=b)``
    draws; the out-of-bag (OOB) rows -- about 1/e of the data -- are the ones it
    never drew. Every candidate is fit in-bag and judged out-of-bag: Rashomon
    membership, VI metrics and performance metrics all use the OOB rows, so a
    model that memorizes its bootstrap can't win membership on training loss.
    Rows come back as given; callers standardize them with :func:`_scale_in_bag`.

    Returns ``None`` (an invalid bootstrap) unless both the in-bag and the OOB
    rows contain every class in ``y``. A model fit on an in-bag sample missing a
    class has fewer ``classes_``/``predict_proba`` columns than its peers, and
    OOB rows missing a class break multiclass log-loss/AUPRC; rare classes hit
    this (e.g. Staellert's 13-cell M phase loses all its OOB rows at seeds 227
    and 416). Seeds are unchanged, so valid bootstraps are unaffected.

    ``groups`` (integer codes from :func:`_prepare_groups`, or None) switches to
    a cluster bootstrap: ``resample`` draws the sorted unique groups with
    replacement (same ``random_state=b``), the in-bag rows are every row of each
    drawn group repeated with the group's multiplicity, and the OOB rows are
    every row of the groups never drawn -- so no group is on both sides. With
    few or large groups more bootstraps fail the class check above; they count
    as invalid like any other. ``groups=None`` is the row bootstrap, unchanged.
    """

    if groups is None:
        in_bag = resample(np.arange(len(y)), random_state=b)
        oob_mask = np.ones(len(y), dtype=bool)
        oob_mask[in_bag] = False
    else:
        n_groups = int(groups.max()) + 1
        drawn = resample(np.arange(n_groups), random_state=b)
        order = np.argsort(groups, kind="stable")
        bounds = np.searchsorted(groups[order], np.arange(n_groups + 1))
        in_bag = np.concatenate([order[bounds[g]:bounds[g + 1]] for g in drawn])
        oob_mask = ~np.isin(groups, drawn)
    n_classes = np.unique(y).size
    if (
        n_classes < 2
        or np.unique(y[in_bag]).size < n_classes
        or np.unique(y[oob_mask]).size < n_classes
    ):
        return None
    return X[in_bag], y[in_bag], X[oob_mask], y[oob_mask]


def _scale_in_bag(X_boot, X_oob):
    """Standardize a bootstrap's rows with a scaler fit on its in-bag rows only.

    The OOB rows judge Rashomon membership, VI and performance, so they must not
    shape the mean/SD the in-bag models are fit on. A column constant in-bag
    gets scale 1 (StandardScaler's zero-variance rule), i.e. is only centered.
    """

    scaler = StandardScaler().fit(X_boot)
    return scaler.transform(X_boot), scaler.transform(X_oob)


def _run_single_bootstrap(
    b,
    X,
    y,
    model_trainer,
    model_kwargs,
    n_models_pool,
    epsilon,
    feature_names,
    vi_metric_specs,
    performance_metric_specs,
    groups=None,
    timers=None,
):
    # timers: None (profiling off) or a dict this fills with stage wall-clocks
    # (see _profile_record); never changes the result.
    if timers is not None:
        timers.update({"split_seconds": 0.0, "search_seconds": 0.0, "perf_seconds": 0.0})
        timers.update({f"vi_seconds/{metric_name}": 0.0 for metric_name, _ in vi_metric_specs})
    rng = np.random.default_rng(seed=b)
    stage_start = time.perf_counter()
    split = _bootstrap_split(X, y, b, groups)
    if split is None:
        _add_time(timers, "split_seconds", stage_start)
        return None, {"n_fits": 0, "rashomon_size": 0, "valid": False}
    X_boot, y_boot, X_oob, y_oob = split
    X_boot, X_oob = _scale_in_bag(X_boot, X_oob)
    _add_time(timers, "split_seconds", stage_start)

    stage_start = time.perf_counter()
    models, losses = train_candidate_models(
        model_trainer, X_boot, y_boot, X_oob, y_oob, n_models_pool, b, model_kwargs
    )
    _add_time(timers, "search_seconds", stage_start)

    min_loss = np.min(losses)
    rashomon_mask = losses <= min_loss + epsilon
    rashomon_models = [model for model, keep in zip(models, rashomon_mask) if keep]
    stats = {
        "n_fits": len(losses),
        "rashomon_size": len(rashomon_models),
        "valid": bool(rashomon_models),
    }
    if not rashomon_models:
        return None, stats

    boot_importances = {
        metric: {feature_name: [] for feature_name in feature_names}
        for metric, _ in vi_metric_specs
    }
    boot_perf = {metric_name: [] for metric_name, _ in performance_metric_specs}

    for model in rashomon_models:
        metric_values = {}
        for metric_name, metric_fn in vi_metric_specs:
            stage_start = time.perf_counter()
            metric_values[metric_name] = _coerce_importance_values(
                metric_name,
                _call_metric(metric_fn, model=model, X=X_oob, y=y_oob, rng=rng),
                len(feature_names),
            )
            _add_time(timers, f"vi_seconds/{metric_name}", stage_start)

        for index, feature_name in enumerate(feature_names):
            for metric_name, _ in vi_metric_specs:
                if metric_values[metric_name] is not None:
                    boot_importances[metric_name][feature_name].append(
                        float(metric_values[metric_name][index])
                    )

        stage_start = time.perf_counter()
        y_pred = model.predict(X_oob)
        y_prob = model.predict_proba(X_oob)
        for metric_name, metric_fn in performance_metric_specs:
            metric_value = _coerce_performance_value(
                metric_name,
                _call_metric(
                    metric_fn,
                    model=model,
                    X=X_oob,
                    y=y_oob,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    rng=rng,
                ),
            )
            if metric_value is not None:
                boot_perf[metric_name].append(metric_value)
        _add_time(timers, "perf_seconds", stage_start)

    return (boot_importances, boot_perf), stats


def _search_diagnostics(family_names, family_indices, family_losses, global_min_loss, epsilon, search_seconds):
    """One bootstrap's candidate-search record for ``search_diagnostics_``.

    ``evaluated`` maps each family to ``(candidate_indices, oob_losses)`` of every
    candidate fit (indices are the shared pool's ``candidate_index``, so the same
    key means the same model in either search mode); ``rashomon`` maps each family
    to the indices that passed ``loss <= min_loss + epsilon``.
    """

    evaluated = {}
    rashomon = {}
    for name in family_names:
        indices = np.asarray(family_indices[name], dtype=int)
        losses = np.asarray(family_losses[name], dtype=float)
        evaluated[name] = (indices, losses)
        rashomon[name] = indices[losses <= global_min_loss + epsilon]
    return {
        "evaluated": evaluated,
        "rashomon": rashomon,
        "min_loss": float(global_min_loss),
        "n_fits": int(sum(len(losses) for _, losses in evaluated.values())),
        "search_seconds": float(search_seconds),
    }


def _run_single_bootstrap_cross_family(
    b,
    X,
    y,
    model_configs,
    n_models_per_class,
    epsilon,
    feature_names,
    vi_metric_specs,
    performance_metric_specs,
    family_balance_mode,
    candidate_search="grid",
    search_spaces=None,
    truvarimp_config=None,
    n_candidates_per_family=DEFAULT_N_CANDIDATES_PER_FAMILY,
    record_diagnostics=False,
    groups=None,
    timers=None,
):
    # timers: None (profiling off) or a dict this fills with stage wall-clocks
    # (see _profile_record); never changes the result.
    if timers is not None:
        timers.update({"split_seconds": 0.0, "search_seconds": 0.0})
        if candidate_search == "truvarimp_joint":
            timers.update({"fit_seconds": 0.0, "gp_seconds": 0.0})
        for name in model_configs:
            timers.update({f"vi_seconds/{name}/{metric_name}": 0.0 for metric_name, _ in vi_metric_specs})
        timers.update({f"perf_seconds/{name}": 0.0 for name in model_configs})
    rng = np.random.default_rng(seed=b)
    stage_start = time.perf_counter()
    split = _bootstrap_split(X, y, b, groups)
    if split is None:
        _add_time(timers, "split_seconds", stage_start)
        stats = {
            "n_fits": {name: 0 for name in model_configs},
            "rashomon_size": {name: 0 for name in model_configs},
            "truvarimp": None,
            "valid": False,
        }
        if record_diagnostics:
            stats["diagnostics"] = None
        return (None, None, None, None), stats
    X_boot, y_boot, X_oob, y_oob = split
    X_boot, X_oob = _scale_in_bag(X_boot, X_oob)
    _add_time(timers, "split_seconds", stage_start)
    search_start = time.perf_counter()

    # Both search modes draw from this one pool, so for a given b they see the
    # identical candidates (and fit any one of them with the same random_state).
    candidate_pool = build_candidate_pool(
        model_configs, search_spaces or {}, b, n_candidates_per_family
    )
    search_stats = None
    if candidate_search == "truvarimp_joint":
        family_models, family_losses, search_stats = run_joint_truvarimp(
            model_configs,
            search_spaces or {},
            X_boot,
            y_boot,
            X_oob,
            y_oob,
            b,
            epsilon,
            truvarimp_config,
            return_stats=True,
            candidate_pool=candidate_pool,
        )
        family_indices = search_stats.pop("evaluated_candidate_indices")
        fit_seconds = search_stats.pop("fit_seconds")
    else:
        family_models = {}
        family_losses = {}
        for name, (trainer, kwargs) in model_configs.items():
            if candidate_pool.get(name) is not None:
                models, losses = fit_candidate_pool(candidate_pool[name], X_boot, y_boot, X_oob, y_oob)
            else:
                # Outside the pool (SingleCandidateTrainer, custom trainer, no
                # search space, or a custom search_grid): the family's own trainer.
                models, losses = train_candidate_models(
                    trainer, X_boot, y_boot, X_oob, y_oob, n_models_per_class, b, kwargs
                )
            family_models[name] = models
            family_losses[name] = losses
        # fit_candidate_pool fits in candidate_index order; a family outside the
        # pool is keyed by its own fit order.
        family_indices = {name: list(range(len(family_losses[name]))) for name in model_configs}
    search_seconds = time.perf_counter() - search_start
    if timers is not None:
        timers["search_seconds"] = search_seconds
        if candidate_search == "truvarimp_joint":
            # gp_seconds also covers pool construction and result collapsing.
            timers["fit_seconds"] = fit_seconds
            timers["gp_seconds"] = search_seconds - fit_seconds

    stats = {
        "n_fits": {name: len(family_losses[name]) for name in model_configs},
        "rashomon_size": {name: 0 for name in model_configs},
        "truvarimp": search_stats,
        "valid": False,
    }
    empty_result = (None, None, None, None)

    all_losses = np.concatenate(
        [losses for losses in family_losses.values() if len(losses) > 0]
    ) if any(len(losses) > 0 for losses in family_losses.values()) else np.array([])
    if all_losses.size == 0:
        return empty_result, stats
    global_min_loss = np.min(all_losses)

    if record_diagnostics:
        stats["diagnostics"] = _search_diagnostics(
            model_configs, family_indices, family_losses, global_min_loss, epsilon, search_seconds
        )

    family_rashomon = {name: [] for name in model_configs}
    for name in model_configs:
        models = family_models[name]
        losses = family_losses[name]
        if len(losses) == 0:
            continue

        # Rashomon-set membership is judged against the best loss achieved by
        # ANY family in this bootstrap, not each family's own best -- a family
        # that can't get within epsilon of the pooled optimum shouldn't be
        # propped up by comparing it only to itself. family_balance_mode only
        # changes how already-admitted models are aggregated afterward.
        family_rashomon_mask = losses <= global_min_loss + epsilon
        family_rashomon[name] = [model for model, keep in zip(models, family_rashomon_mask) if keep]

    stats["rashomon_size"] = {name: len(models) for name, models in family_rashomon.items()}
    non_empty_families = [name for name, models in family_rashomon.items() if models]
    if not non_empty_families:
        return empty_result, stats

    rashomon_models = []
    rashomon_labels = []
    for name in non_empty_families:
        for model in family_rashomon[name]:
            rashomon_models.append(model)
            rashomon_labels.append(name)

    if not rashomon_models:
        return empty_result, stats

    stats["valid"] = True
    boot_family_counts = {name: 0 for name in model_configs}
    for label in rashomon_labels:
        boot_family_counts[label] += 1

    boot_importances = {
        metric: {feature_name: [] for feature_name in feature_names}
        for metric, _ in vi_metric_specs
    }
    boot_family_perf = {
        name: {metric_name: [] for metric_name, _ in performance_metric_specs}
        for name in model_configs
    }
    # Family label of each model that produced values, per metric (a metric
    # may return None for some families, e.g. ``coef`` on trees), so weights
    # stay aligned with that metric's pooled values.
    metric_labels = {metric_name: [] for metric_name, _ in vi_metric_specs}

    for model, label in zip(rashomon_models, rashomon_labels):
        metric_values = {}
        for metric_name, metric_fn in vi_metric_specs:
            stage_start = time.perf_counter()
            metric_values[metric_name] = _coerce_importance_values(
                metric_name,
                _call_metric(metric_fn, model=model, X=X_oob, y=y_oob, rng=rng),
                len(feature_names),
            )
            _add_time(timers, f"vi_seconds/{label}/{metric_name}", stage_start)

        for index, feature_name in enumerate(feature_names):
            for metric_name, _ in vi_metric_specs:
                if metric_values[metric_name] is not None:
                    boot_importances[metric_name][feature_name].append(
                        float(metric_values[metric_name][index])
                    )
        for metric_name, _ in vi_metric_specs:
            if metric_values[metric_name] is not None:
                metric_labels[metric_name].append(label)

        stage_start = time.perf_counter()
        y_pred = model.predict(X_oob)
        y_prob = model.predict_proba(X_oob)
        for metric_name, metric_fn in performance_metric_specs:
            metric_value = _coerce_performance_value(
                metric_name,
                _call_metric(
                    metric_fn,
                    model=model,
                    X=X_oob,
                    y=y_oob,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    rng=rng,
                ),
            )
            if metric_value is not None:
                boot_family_perf[label][metric_name].append(metric_value)
        _add_time(timers, f"perf_seconds/{label}", stage_start)

    if family_balance_mode == "weighted":
        # Per metric, each family with >=1 scored Rashomon member gets total
        # mass 1/K (K = such families), split evenly over its s_f members:
        # per-model weight 1/(K * s_f). Weights sum to 1 within this bootstrap.
        model_weights = {}
        for metric_name, labels in metric_labels.items():
            family_sizes = {name: labels.count(name) for name in set(labels)}
            n_families = len(family_sizes)
            model_weights[metric_name] = [
                1.0 / (n_families * family_sizes[name]) for name in labels
            ]
    else:
        model_weights = None

    return (boot_importances, boot_family_counts, boot_family_perf, model_weights), stats


def compute_rid(
    X,
    y,
    epsilon=0.05,
    n_bootstraps=500,
    n_models_pool=50,
    model_config=None,
    model_class=None,
    model_kwargs=None,
    model_trainer=None,
    vi_metrics=None,
    performance_metrics=None,
    n_cdf_points=200,
    n_jobs=1,
    suppress_warnings=True,
    groups=None,
    profile=None,
    profile_dir=None,
):
    """Compute single-family RID and return metric results, performance, and valid bootstrap count.

    Candidates come from the family's trainer (for built-in families, its
    hand-made ``GridCandidateTrainer`` grid, each config fit once per
    bootstrap). ``n_models_pool`` only sets how many seed refits a
    ``refit_seeds=True`` trainer (RandomForest) or a custom trainer makes:
    ``max(1, n_models_pool // len(grid))`` per config.

    ``suppress_warnings`` (default True) ignores every Python warning raised
    while fitting and scoring candidates inside each bootstrap worker.

    ``groups`` (default None: the row bootstrap) is an optional length-n array
    of hashable, mutually sortable labels, e.g. patient IDs when a patient has
    many rows. Each bootstrap then resamples whole groups (a cluster
    bootstrap), so a group's rows are never both in-bag and out-of-bag; see
    :func:`_bootstrap_split`. Expect more invalid bootstraps with few groups.

    ``profile`` (default None: off) is an opt-in per-bootstrap profiler:
    ``"timers"`` records stage wall-clocks, ``"cprofile"`` also runs each
    bootstrap under :class:`cProfile.Profile` inside its joblib worker and
    dumps ``<profile_dir>/bootstrap_<b>.prof``. When on, a fourth return value
    is appended: one flat record per bootstrap, in bootstrap order (keys in
    :func:`_profile_record`), and each bootstrap's stage summary is logged at
    INFO as it finishes. Profiling never changes the other outputs; when off
    the return value is the usual 3-tuple.
    """

    profile, profile_dir = normalize_profile(profile, profile_dir)
    if model_config is not None:
        if model_class is not None or model_trainer is not None or model_kwargs is not None:
            raise ValueError(
                "Pass either model_config or the explicit model_class/model_trainer/model_kwargs "
                "arguments, not both"
            )
        model_class, model_trainer, model_kwargs = resolve_model_config(model_config)
    else:
        if model_class is None and model_trainer is None:
            raise ValueError("Either model_config, model_class, or model_trainer must be provided")
        if model_kwargs is None:
            model_kwargs = {}
        if model_trainer is None:
            model_trainer = resolve_candidate_trainer(model_class)

    vi_metric_specs = normalize_vi_metrics(vi_metrics)
    performance_metric_specs = normalize_performance_metrics(performance_metrics)
    vi_metric_names = tuple(metric_name for metric_name, _ in vi_metric_specs)
    performance_metric_names = tuple(metric_name for metric_name, _ in performance_metric_specs)

    # Unscaled: each bootstrap standardizes on its own in-bag rows (_scale_in_bag).
    X_values, y_values, feature_names = _prepare_inputs(X, y)
    group_codes, n_groups = _prepare_groups(groups, len(y_values))

    model_name = getattr(model_class, "__name__", type(model_trainer).__name__)
    logger.info(
        f"RID: {model_name}, {len(feature_names)} features, "
        f"B={n_bootstraps}, pool={n_models_pool}, epsilon={epsilon}, n_jobs={n_jobs}, "
        f"metrics={vi_metric_names}, performance_metrics={performance_metric_names}"
        + (f", grouped bootstrap over {n_groups} groups" if group_codes is not None else "")
    )

    progress_label = f"RID[{model_name}] bootstraps"
    tasks = (
        delayed(_run_bootstrap_task)(
            _run_single_bootstrap,
            b,
            suppress_warnings,
            X_values,
            y_values,
            model_trainer,
            model_kwargs,
            n_models_pool,
            epsilon,
            feature_names,
            vi_metric_specs,
            performance_metric_specs,
            group_codes,
            **_profile_task_kwargs(profile, profile_dir),
        )
        for b in range(n_bootstraps)
    )
    results, bootstrap_stats = _collect_bootstrap_results(
        tasks, n_jobs, n_bootstraps, progress_label
    )
    _log_bootstrap_summary(progress_label, bootstrap_stats)
    profile_records = [stats["profile"] for stats in bootstrap_stats] if profile else None
    profile_suffix = (profile_records,) if profile else ()

    per_bootstrap_importances = []
    per_bootstrap_perf = []
    for result in results:
        if result is None:
            continue
        boot_importances, boot_perf = result
        per_bootstrap_importances.append(boot_importances)
        per_bootstrap_perf.append(boot_perf)

    n_valid = len(per_bootstrap_importances)
    if n_valid == 0:
        logger.warning("No bootstraps produced a non-empty Rashomon set.")
        return (None, None, 0) + profile_suffix

    active_metrics = _active_metrics(per_bootstrap_importances, feature_names, vi_metric_names)
    if not active_metrics:
        logger.warning(f"No active VI metrics produced values for metrics={vi_metric_names}")

    metric_results = _aggregate_metric_results(
        per_bootstrap_importances,
        feature_names,
        active_metrics,
        n_cdf_points,
    )
    perf_stats = _aggregate_perf_stats(per_bootstrap_perf, performance_metric_specs)

    logger.info(
        f"Done. {n_valid}/{n_bootstraps} valid bootstraps. "
        f"Metrics: {list(metric_results.keys())}  "
        f"{_format_perf_stats(perf_stats, performance_metric_specs)}"
    )
    return (metric_results, perf_stats, n_valid) + profile_suffix


def compute_rid_cross_family(
    X,
    y,
    model_configs,
    epsilon=0.05,
    n_bootstraps=500,
    n_models_per_class=50,
    vi_metrics=None,
    performance_metrics=None,
    family_balance_mode="unweighted",
    n_cdf_points=200,
    n_jobs=1,
    candidate_search="grid",
    truvarimp_config=None,
    suppress_warnings=True,
    n_candidates_per_family=DEFAULT_N_CANDIDATES_PER_FAMILY,
    return_search_diagnostics=False,
    groups=None,
    profile=None,
    profile_dir=None,
):
    """Compute cross-family RID and return metric results, family counts, and family performance.

    ``return_search_diagnostics`` (default False) appends a fifth return value:
    one record per bootstrap (index ``b``), ``None`` for a bootstrap whose split
    was invalid or that fit nothing, otherwise a dict with ``evaluated``
    (``{family: (candidate_indices, oob_losses)}`` for every candidate fit),
    ``rashomon`` (``{family: candidate_indices}`` of the members), ``min_loss``,
    ``n_fits``, ``search_seconds`` (candidate fitting/search only) and
    ``seconds`` (the whole bootstrap, VI included). Pool candidates are keyed
    by their ``candidate_index``, so for the same data and ``b`` both search
    modes key the same model identically; this is what lets a ``"grid"`` run
    serve as the exact Rashomon-set ground truth for a ``"truvarimp_joint"``
    run. Recording it changes no other output.

    ``profile`` (default None: off) is an opt-in per-bootstrap profiler:
    ``"timers"`` records stage wall-clocks (split, candidate search -- for
    ``"truvarimp_joint"`` split into real fits vs. GP/acquisition -- VI per
    family and metric, performance metrics per family) plus per-family
    Rashomon sizes and fit counts; ``"cprofile"`` also runs each bootstrap
    under :class:`cProfile.Profile` inside its joblib worker and dumps
    ``<profile_dir>/bootstrap_<b>.prof``. When on, one more value is appended
    to the return tuple (after the search diagnostics, if requested): one flat
    record per bootstrap, in bootstrap order (keys in :func:`_profile_record`),
    and each bootstrap's stage summary is logged at INFO as it finishes.
    Profiling never changes the other outputs; when off the return value is
    exactly as without it.

    ``suppress_warnings`` (default True) ignores every Python warning raised
    while fitting and scoring candidates inside each bootstrap worker.

    ``groups`` (default None: the row bootstrap) is an optional length-n array
    of hashable, mutually sortable labels, e.g. patient IDs when a patient has
    many rows. Each bootstrap then resamples whole groups (a cluster
    bootstrap), so a group's rows are never both in-bag and out-of-bag; see
    :func:`_bootstrap_split`. Expect more invalid bootstraps with few groups.

    Each bootstrap first draws one shared candidate pool
    (:func:`rid.candidate_pool.build_candidate_pool`): ``n_candidates_per_family``
    random configurations per family from its continuous search space
    (``rid.candidate_pool.DEFAULT_CONTINUOUS_SEARCH_SPACES_BY_NAME``, or a
    ``"search_space"`` / ``"truvarimp_search_space"`` key in a family's config
    mapping), split evenly across its categorical sub-families and deduplicated
    after integer rounding. Each candidate's ``random_state`` is fixed by
    (bootstrap, family name, candidate index), so a candidate fits to the
    identical model in either mode. Any hyperparameter a family pins through its
    model kwargs is held fixed rather than searched. ``candidate_search`` then
    selects which pool candidates get fit:

    - ``"grid"`` (default): every candidate, once.
    - ``"truvarimp_joint"``: CASH-style joint active search (TruVaRImp,
      arXiv:2603.15321) over the pool, adaptively spending a shared evaluation
      budget on whichever family's candidate looks most likely to be near the
      (bootstrap-implicit) Rashomon threshold -- see ``rid.truvarimp``. Every
      evaluated candidate is still fit for real and scored by its exact
      out-of-bag log-loss; only the search policy differs. ``truvarimp_config``
      overrides ``rid.truvarimp.DEFAULT_TRUVARIMP_CONFIG`` (notably
      ``total_budget``, the shared per-bootstrap fit budget across all families).

    Families outside the pool -- a ``SingleCandidateTrainer`` (e.g.
    ``LinearClassifier``), a custom trainer, a class without a search space, or
    (in ``"grid"`` mode) a family declaring its own ``search_grid`` -- are trained
    by their own trainer instead; ``n_models_per_class`` is passed to it in
    ``"grid"`` mode (it sets seed refits for ``refit_seeds=True`` grids and
    custom trainers) and is otherwise unused. TruVaRImp ignores ``search_grid``.
    """

    vi_metric_specs = normalize_vi_metrics(vi_metrics)
    performance_metric_specs = normalize_performance_metrics(performance_metrics)
    vi_metric_names = tuple(metric_name for metric_name, _ in vi_metric_specs)
    performance_metric_names = tuple(metric_name for metric_name, _ in performance_metric_specs)
    family_balance_mode = normalize_cross_family_balance_mode(family_balance_mode)
    candidate_search = normalize_candidate_search(candidate_search)
    profile, profile_dir = normalize_profile(profile, profile_dir)

    # Unscaled: each bootstrap standardizes on its own in-bag rows (_scale_in_bag).
    X_values, y_values, feature_names = _prepare_inputs(X, y)
    group_codes, n_groups = _prepare_groups(groups, len(y_values))

    n_candidates_per_family = int(n_candidates_per_family)
    truvarimp_config = dict(truvarimp_config or {})
    if truvarimp_config.get("n_candidates_per_family", n_candidates_per_family) != n_candidates_per_family:
        raise ValueError(
            "truvarimp_config['n_candidates_per_family'] conflicts with n_candidates_per_family; "
            "the pool is shared by both search modes, so set it via n_candidates_per_family only"
        )
    truvarimp_config["n_candidates_per_family"] = n_candidates_per_family
    truvarimp_budget = resolve_truvarimp_config(truvarimp_config)["total_budget"]

    resolved_model_configs = {}
    search_spaces = {}
    for name, model_config in model_configs.items():
        model_class, trainer, model_kwargs = resolve_model_config(model_config)
        resolved_model_configs[name] = (trainer, model_kwargs)
        explicit_space = None
        if isinstance(model_config, Mapping):
            explicit_space = model_config.get("search_space", model_config.get("truvarimp_search_space"))
        search_spaces[name] = explicit_space or resolve_default_search_space(model_class)
        has_search_grid = _declared_search_grid(model_config) is not None
        if candidate_search == "grid" and has_search_grid:
            search_spaces[name] = None
            logger.info(
                f"Note: family '{name}' declares a custom search_grid; 'grid' candidate_search "
                "fits that grid via its own trainer instead of drawing from the shared pool."
            )
        elif search_spaces[name] is None:
            logger.info(
                f"Note: family '{name}' ({getattr(model_class, '__name__', model_class)}) has no "
                f"continuous search space; it is not in the shared candidate pool and is trained "
                "by its own trainer instead"
                + (" (a single fixed evaluation)." if candidate_search == "truvarimp_joint" else ".")
            )
        elif has_search_grid:
            logger.info(
                f"Note: family '{name}' declares a custom search_grid, which only the "
                "'grid' candidate_search uses; truvarimp_joint searches its continuous "
                "space instead. Pass 'search_space' in a mapping config to "
                "constrain the joint search."
            )

    family_names = list(resolved_model_configs.keys())
    pool_description = f"pool={n_candidates_per_family}/family" + (
        f", budget={truvarimp_budget} fits/bootstrap" if candidate_search == "truvarimp_joint" else ""
    )
    logger.info(
        f"Cross-family RID (per-family Rashomon): {family_names}, {len(feature_names)} features, "
        f"B={n_bootstraps}, candidate_search={candidate_search}, "
        f"{pool_description}, epsilon={epsilon}, n_jobs={n_jobs}, "
        f"metrics={vi_metric_names}, performance_metrics={performance_metric_names}, "
        f"family_balance_mode={family_balance_mode}"
        + (f", grouped bootstrap over {n_groups} groups" if group_codes is not None else "")
    )

    progress_label = f"cross-family RID[{family_balance_mode}, {candidate_search}] bootstraps"
    tasks = (
        delayed(_run_bootstrap_task)(
            _run_single_bootstrap_cross_family,
            b,
            suppress_warnings,
            X_values,
            y_values,
            resolved_model_configs,
            n_models_per_class,
            epsilon,
            feature_names,
            vi_metric_specs,
            performance_metric_specs,
            family_balance_mode,
            candidate_search,
            search_spaces,
            truvarimp_config,
            n_candidates_per_family,
            return_search_diagnostics,
            group_codes,
            **_profile_task_kwargs(profile, profile_dir),
        )
        for b in range(n_bootstraps)
    )
    results, bootstrap_stats = _collect_bootstrap_results(
        tasks, n_jobs, n_bootstraps, progress_label
    )
    _log_bootstrap_summary(progress_label, bootstrap_stats)

    search_diagnostics = None
    if return_search_diagnostics:
        search_diagnostics = []
        for stats in bootstrap_stats:
            record = stats.get("diagnostics")
            if record is not None:
                record = {**record, "seconds": stats["seconds"]}
            search_diagnostics.append(record)
    extra_returns = (search_diagnostics,) if return_search_diagnostics else ()
    if profile:
        extra_returns += ([stats["profile"] for stats in bootstrap_stats],)

    per_bootstrap_importances = []
    per_bootstrap_family_perf = []
    per_bootstrap_weights = []
    family_counts = {name: 0 for name in family_names}

    for boot_importances, boot_counts, boot_family_perf, boot_model_weights in results:
        if boot_importances is None:
            continue
        per_bootstrap_importances.append(boot_importances)
        per_bootstrap_family_perf.append(boot_family_perf)
        per_bootstrap_weights.append(boot_model_weights)
        for name in family_names:
            family_counts[name] += boot_counts[name]

    n_valid = len(per_bootstrap_importances)
    if n_valid == 0:
        logger.warning("No bootstraps produced a non-empty Rashomon set.")
        return (None, family_counts, None, 0) + extra_returns

    active_metrics = _active_metrics(per_bootstrap_importances, feature_names, vi_metric_names)
    if not active_metrics:
        logger.warning(f"No active VI metrics produced values for metrics={vi_metric_names}")

    weights_arg = per_bootstrap_weights if family_balance_mode == "weighted" else None
    metric_results = _aggregate_metric_results(
        per_bootstrap_importances,
        feature_names,
        active_metrics,
        n_cdf_points,
        per_bootstrap_weights=weights_arg,
    )
    family_perf_stats = _aggregate_family_perf_stats(
        per_bootstrap_family_perf,
        family_names,
        performance_metric_specs,
    )

    logger.info(f"Done. {n_valid}/{n_bootstraps} valid bootstraps. Metrics: {list(metric_results.keys())}")
    logger.info(f"Family representation in Rashomon sets: {family_counts}")
    logger.info("Performance by model family (Rashomon models, bootstrap mean):")
    for name, perf_stats in family_perf_stats.items():
        logger.info(f"  {name}: {_format_perf_stats(perf_stats, performance_metric_specs)}")

    return (metric_results, family_counts, family_perf_stats, n_valid) + extra_returns


class _RIDSummaryMixin:
    def _require_fit(self):
        if not hasattr(self, "metric_results_"):
            raise RuntimeError("Call fit(X, y) before accessing RID results")

    def metric_result(self, metric):
        self._require_fit()
        if callable(metric):
            metric = _metric_name(metric)
        if self.metric_results_ is None or metric not in self.metric_results_:
            raise KeyError(f"Metric '{metric}' is not available. Found: {self.available_metrics_}")
        return self.metric_results_[metric]

    def metric_summary(self, metric):
        return summarize_metric_result(self.metric_result(metric))

    def rank_features(self, metric, by="expected_importance", descending=True):
        summary = self.metric_summary(metric)
        return sorted(
            ((feature_name, values[by]) for feature_name, values in summary.items()),
            key=lambda item: item[1],
            reverse=descending,
        )


class RashomonImportanceDistribution(_RIDSummaryMixin, BaseEstimator):
    """Single-family RID estimator with a sklearn-style fit API.

    ``profile`` (None/"off", "timers" or "cprofile") and ``profile_dir`` turn on
    the opt-in per-bootstrap profiler (see :func:`compute_rid`); ``profile_``
    then holds one flat record per bootstrap, else ``None``. Profiling never
    changes any other fitted attribute.
    """

    def __init__(
        self,
        epsilon=0.05,
        n_bootstraps=500,
        n_models_pool=50,
        model_config=None,
        model_class=None,
        model_kwargs=None,
        model_trainer=None,
        vi_metrics=None,
        performance_metrics=None,
        n_cdf_points=200,
        n_jobs=1,
        suppress_warnings=True,
        profile=None,
        profile_dir=None,
    ):
        self.epsilon = epsilon
        self.n_bootstraps = n_bootstraps
        self.n_models_pool = n_models_pool
        self.model_config = model_config
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.model_trainer = model_trainer
        self.vi_metrics = vi_metrics
        self.performance_metrics = performance_metrics
        self.n_cdf_points = n_cdf_points
        self.n_jobs = n_jobs
        self.suppress_warnings = suppress_warnings
        self.profile = profile
        self.profile_dir = profile_dir

    def fit(self, X, y, groups=None):
        """Fit RID on ``X, y``.

        ``groups`` (optional, length n) switches to a cluster bootstrap over
        those labels (see :func:`compute_rid`); ``n_groups_`` records the number
        of distinct groups, or ``None`` for the default row bootstrap.
        """

        results = compute_rid(
            X,
            y,
            epsilon=self.epsilon,
            n_bootstraps=self.n_bootstraps,
            n_models_pool=self.n_models_pool,
            model_config=self.model_config,
            model_class=self.model_class,
            model_kwargs=self.model_kwargs,
            model_trainer=self.model_trainer,
            vi_metrics=self.vi_metrics,
            performance_metrics=self.performance_metrics,
            n_cdf_points=self.n_cdf_points,
            n_jobs=self.n_jobs,
            suppress_warnings=self.suppress_warnings,
            groups=groups,
            profile=self.profile,
            profile_dir=self.profile_dir,
        )
        self.metric_results_, self.perf_stats_, self.n_valid_bootstraps_ = results[:3]
        self.profile_ = results[3] if len(results) > 3 else None
        _, y_values, self.feature_names_ = _prepare_inputs(X, y)
        self.n_groups_ = _prepare_groups(groups, len(y_values))[1]
        self.available_metrics_ = tuple(self.metric_results_.keys()) if self.metric_results_ else tuple()
        return self


class CrossFamilyRashomonImportanceDistribution(_RIDSummaryMixin, BaseEstimator):
    """Cross-family Rashomon Importance Distribution estimator.

    This estimator builds a Rashomon set separately for each model family,
    merges feature-importance samples across families, and returns empirical
    importance distributions that can be summarized or ranked.

    Parameters
    ----------
    model_configs : mapping of str to model config spec
        Mapping from a family label (for example ``"Lasso"`` or ``"SVM"``)
        to a model configuration accepted by
        :func:`rid.models.resolve_model_config`.

        Each value can be one of:

        - Tuple forms:
          - ``(model_class_or_trainer, kwargs)``
          - ``(model_class_or_trainer, kwargs, search_grid)``
          - ``(model_class_or_trainer, kwargs, search_grid, random_state_multiplier)``
        - Mapping form with aliases:
          - ``{"model": ...}``, ``{"estimator": ...}``, or ``{"trainer": ...}``
            (all three keys are aliases for the model/trainer entry point)
          - Optional ``"kwargs"``
          - Optional ``"search_grid"``
          - Optional ``"random_state_multiplier"``
          - Optional ``"search_space"`` (alias ``"truvarimp_search_space"``): a
            :class:`rid.candidate_pool.ContinuousSearchSpace` replacing the
            family's default shared-pool search space

        Notes:
        - ``search_grid`` and ``random_state_multiplier`` overrides are supported
          when the config references a model class, not an already-created
          trainer instance.
        - Family labels are preserved in ``family_counts_`` and
          ``family_perf_stats_``.

    epsilon : float, default=0.05
        Rashomon tolerance. A candidate model is retained in the Rashomon set
        if its bootstrap loss is within ``(1 + epsilon) * best_loss``.

    n_bootstraps : int, default=500
        Number of bootstrap resamples used to estimate RID curves.

    n_models_per_class : int, default=50
        Passed to the trainer of each family outside the shared candidate pool
        (see ``compute_rid_cross_family``) in ``"grid"`` mode: seed refits for a
        ``refit_seeds=True`` grid or a custom trainer. Unused by the built-in
        pooled families.

    n_candidates_per_family : int, default=500
        Size of the shared candidate pool each pooled family draws per
        bootstrap, split evenly across its categorical sub-families and
        deduplicated after integer rounding. ``"grid"`` fits all of it;
        ``"truvarimp_joint"`` searches it within ``total_budget`` fits.

    candidate_search : {"grid", "truvarimp_joint"}, default="grid"
        How each bootstrap's pool is searched; see ``compute_rid_cross_family``.

    truvarimp_config : dict, optional
        Overrides for ``rid.truvarimp.DEFAULT_TRUVARIMP_CONFIG``.

    vi_metrics : {"sub_mr", "loco", "coef"}, callable, or sequence of these, default=None
        Feature-importance metrics to compute. If ``None``, uses all built-in
        VI metrics: ``("sub_mr", "loco", "coef")``.

        String aliases:
        - ``"sub_mr"``: subtraction-based model reliance.
        - ``"loco"``: leave-one-covariate-out importance.
        - ``"coef"``: normalized absolute coefficients for linear models.

        Callable aliases that normalize to the same canonical names:
        - ``compute_model_reliance`` and ``vi_sub_mr`` -> ``"sub_mr"``
        - ``compute_loco_importance`` and ``vi_loco`` -> ``"loco"``
        - ``compute_coef_importance`` and ``vi_coef`` -> ``"coef"``

        Custom callables are supported. The metric name is resolved by:
        - ``callable.metric_name`` if present,
        - else known callable aliases,
        - else the callable ``__name__``.

    performance_metrics : {"accuracy", "f1", "auprc"}, callable, or sequence of these, default=None
        Per-model performance metrics summarized per family over Rashomon
        members. If ``None``, uses all built-in performance metrics:
        ``("accuracy", "f1", "auprc")``.

        String aliases:
        - ``"accuracy"``: classification accuracy.
        - ``"f1"``: binary F1 for 2 classes, macro F1 otherwise.
        - ``"auprc"``: average precision (macro for multiclass).

        Callable aliases that normalize to canonical names:
        - ``sklearn.metrics.accuracy_score`` and ``performance_accuracy``
          -> ``"accuracy"``
        - ``sklearn.metrics.f1_score`` and ``performance_f1`` -> ``"f1"``
        - ``sklearn.metrics.average_precision_score`` and
          ``performance_auprc`` -> ``"auprc"``

        Custom callables follow the same naming rules as ``vi_metrics``.

    family_balance_mode : {"unweighted", "weighted"}, default="unweighted"
        Strategy for aggregating cross-family feature-importance samples.

        - ``"unweighted"``: each Rashomon model contributes equally.
        - ``"weighted"``: each family contributes equal total mass per
          bootstrap: each model gets weight ``1 / (K * s_f)``, where ``s_f``
          is its family's Rashomon count and ``K`` the number of families
          with at least one Rashomon member in that bootstrap.

        If ``None``, this is normalized to ``"unweighted"``.

    n_cdf_points : int, default=200
        Number of points in the empirical CDF grid used for each metric's RID
        curve.

    n_jobs : int, default=1
        Number of parallel jobs for bootstrap execution (passed to
        :class:`joblib.Parallel`).

    suppress_warnings : bool, default=True
        Ignore every Python warning (e.g. sklearn ``ConvergenceWarning``)
        raised while fitting and scoring candidates inside each bootstrap.

    record_search_diagnostics : bool, default=False
        Keep a per-bootstrap record of the candidate search in
        ``search_diagnostics_`` (see ``compute_rid_cross_family``'s
        ``return_search_diagnostics``). Changes no other output.

    profile : {None, "off", "timers", "cprofile"}, default=None
        Opt-in per-bootstrap profiler (see ``compute_rid_cross_family``'s
        ``profile``): ``"timers"`` records stage wall-clocks, VI time per
        family and metric, and Rashomon sizes per family in ``profile_``;
        ``"cprofile"`` additionally dumps ``<profile_dir>/bootstrap_<b>.prof``
        from inside each joblib worker. Changes no other output.

    profile_dir : str or path-like, optional
        Directory for the ``.prof`` files; required by (and only allowed with)
        ``profile="cprofile"``. Created if missing.

    Attributes
    ----------
    metric_results_ : dict or None
        Mapping ``metric_name -> (rid_cdfs, cdf_grid, raw_importances,
        expected_importance, prob_positive)``; the last two are per-feature
        dicts (see ``summarize_metric_result``). ``None`` if no bootstrap
        produced a non-empty Rashomon set.

    family_counts_ : dict of str to int
        Total number of Rashomon models selected for each family across valid
        bootstraps.

    family_perf_stats_ : dict or None
        Nested mapping of family-level performance summary statistics for the
        requested ``performance_metrics``. ``None`` when no valid bootstrap is
        available.

    n_valid_bootstraps_ : int
        Number of bootstraps that produced at least one Rashomon model.

    feature_names_ : list of str
        Feature names derived from ``X.columns`` when available, otherwise
        generated as ``x0, x1, ...``.

    available_metrics_ : tuple of str
        Canonical names of VI metrics available in ``metric_results_`` after
        fit-time filtering.

    search_diagnostics_ : list or None
        One record per bootstrap when ``record_search_diagnostics=True``
        (evaluated pool candidates with their OOB losses, Rashomon members,
        min loss, fit count, wall-clock), else ``None``.

    profile_ : list of dict or None
        One flat record per bootstrap (in bootstrap order) when ``profile`` is
        on -- keys such as ``seconds``, ``search_seconds``, ``fit_seconds``/
        ``gp_seconds`` (TruVaRImp only), ``vi_seconds``,
        ``vi_seconds/<family>/<metric>``, ``perf_seconds/<family>``,
        ``rashomon_size/<family>``, ``n_fits/<family>``,
        ``worker_max_rss_mb`` -- else ``None``. ``pandas.DataFrame(profile_)``
        gives one row per bootstrap.

    n_groups_ : int or None
        Number of distinct ``groups`` labels when ``fit`` was given ``groups``
        (cluster bootstrap), else ``None`` (row bootstrap).

    Notes
    -----
    Metric callables are invoked by signature introspection. Any subset of the
    following keyword arguments can be declared by custom metrics:
    ``model, X, y, y_true, y_pred, y_prob, y_score, rng``.
    """

    def __init__(
        self,
        model_configs,
        epsilon=0.05,
        n_bootstraps=500,
        n_models_per_class=50,
        vi_metrics=None,
        performance_metrics=None,
        family_balance_mode="unweighted",
        n_cdf_points=200,
        n_jobs=1,
        candidate_search="grid",
        truvarimp_config=None,
        suppress_warnings=True,
        n_candidates_per_family=DEFAULT_N_CANDIDATES_PER_FAMILY,
        record_search_diagnostics=False,
        profile=None,
        profile_dir=None,
    ):
        self.model_configs = model_configs
        self.epsilon = epsilon
        self.n_bootstraps = n_bootstraps
        self.n_models_per_class = n_models_per_class
        self.vi_metrics = vi_metrics
        self.performance_metrics = performance_metrics
        self.family_balance_mode = family_balance_mode
        self.n_cdf_points = n_cdf_points
        self.n_jobs = n_jobs
        self.candidate_search = candidate_search
        self.truvarimp_config = truvarimp_config
        self.suppress_warnings = suppress_warnings
        self.n_candidates_per_family = n_candidates_per_family
        self.record_search_diagnostics = record_search_diagnostics
        self.profile = profile
        self.profile_dir = profile_dir

    def fit(self, X, y, groups=None):
        """Fit cross-family RID on ``X, y``.

        ``groups`` (optional, length n) switches to a cluster bootstrap over
        those labels (see :func:`compute_rid_cross_family`); ``n_groups_``
        records the number of distinct groups, or ``None`` for the default row
        bootstrap.
        """

        results = compute_rid_cross_family(
            X,
            y,
            model_configs=self.model_configs,
            epsilon=self.epsilon,
            n_bootstraps=self.n_bootstraps,
            n_models_per_class=self.n_models_per_class,
            vi_metrics=self.vi_metrics,
            performance_metrics=self.performance_metrics,
            family_balance_mode=self.family_balance_mode,
            n_cdf_points=self.n_cdf_points,
            n_jobs=self.n_jobs,
            candidate_search=self.candidate_search,
            truvarimp_config=self.truvarimp_config,
            suppress_warnings=self.suppress_warnings,
            n_candidates_per_family=self.n_candidates_per_family,
            return_search_diagnostics=self.record_search_diagnostics,
            groups=groups,
            profile=self.profile,
            profile_dir=self.profile_dir,
        )
        (
            self.metric_results_,
            self.family_counts_,
            self.family_perf_stats_,
            self.n_valid_bootstraps_,
        ) = results[:4]
        self.search_diagnostics_ = results[4] if self.record_search_diagnostics else None
        # The profile records, when on, are always the last return value.
        n_base = 5 if self.record_search_diagnostics else 4
        self.profile_ = results[n_base] if len(results) > n_base else None
        _, y_values, self.feature_names_ = _prepare_inputs(X, y)
        self.n_groups_ = _prepare_groups(groups, len(y_values))[1]
        self.available_metrics_ = tuple(self.metric_results_.keys()) if self.metric_results_ else tuple()
        return self