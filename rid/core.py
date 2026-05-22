from __future__ import annotations

from functools import partial
from inspect import signature
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from .models import resolve_candidate_trainer, resolve_model_config, train_candidate_models


VALID_VI_METRICS = ("sub_mr", "loco", "coef")
VALID_PERFORMANCE_METRICS = ("accuracy", "f1", "auprc")
VALID_CROSS_FAMILY_BALANCE_MODES = ("unweighted", "weighted")


def compute_model_reliance(model, X, y, metric="sub_mr", rng=None):
    """Compute permutation-based model reliance for all features."""

    if rng is None:
        rng = np.random.default_rng()

    n_features = X.shape[1]
    y_prob = model.predict_proba(X)
    e_orig = log_loss(y, y_prob)

    importances = np.zeros(n_features)
    for j in range(n_features):
        X_permuted = X.copy()
        X_permuted[:, j] = rng.permutation(X_permuted[:, j])
        y_prob_perm = model.predict_proba(X_permuted)
        e_switch = log_loss(y, y_prob_perm)

        if metric == "sub_mr":
            importances[j] = e_switch - e_orig
        else:
            importances[j] = e_switch / e_orig if e_orig > 0 else 1.0

    return importances


def compute_loco_importance(model, X, y):
    """Compute leave-one-covariate-out importance."""

    e_orig = log_loss(y, model.predict_proba(X))
    importances = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        X_loco = X.copy()
        X_loco[:, j] = 0.0
        importances[j] = log_loss(y, model.predict_proba(X_loco)) - e_orig
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
    vi_sub_mr: "sub_mr",
    vi_loco: "loco",
    vi_coef: "coef",
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


def summarize_metric_result(metric_result):
    """Compute per-feature expected importance and P(phi > 0)."""

    rid_cdfs, cdf_grid, raw_importances = metric_result
    idx_zero = min(np.searchsorted(cdf_grid, 0), len(cdf_grid) - 1)

    summary = {}
    for feature_name in rid_cdfs:
        values = raw_importances[feature_name]
        summary[feature_name] = {
            "expected_importance": float(np.mean(values)) if values else 0.0,
            "prob_positive": float(1 - rid_cdfs[feature_name][idx_zero]),
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

        valid_count = 0
        for i, boot_importances in enumerate(per_bootstrap_importances):
            if not boot_importances[metric][feature_names[0]]:
                continue
            valid_count += 1
            boot_weights = per_bootstrap_weights[i] if per_bootstrap_weights is not None else None
            for feature_name in feature_names:
                values = boot_importances[metric][feature_name]
                if len(values) == 0:
                    continue
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

        metric_results[metric] = (rid_cdfs, cdf_grid, raw_importances)

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


def _run_single_bootstrap(
    b,
    X_scaled,
    y,
    model_trainer,
    model_kwargs,
    n_models_pool,
    epsilon,
    feature_names,
    vi_metric_specs,
    performance_metric_specs,
):
    rng = np.random.default_rng(seed=b)
    X_boot, y_boot = resample(X_scaled, y, random_state=b)

    models, losses = train_candidate_models(
        model_trainer, X_boot, y_boot, n_models_pool, b, model_kwargs
    )

    min_loss = np.min(losses)
    rashomon_mask = losses <= min_loss + epsilon
    rashomon_models = [model for model, keep in zip(models, rashomon_mask) if keep]
    if not rashomon_models:
        return None

    boot_importances = {
        metric: {feature_name: [] for feature_name in feature_names}
        for metric, _ in vi_metric_specs
    }
    boot_perf = {metric_name: [] for metric_name, _ in performance_metric_specs}

    for model in rashomon_models:
        metric_values = {}
        for metric_name, metric_fn in vi_metric_specs:
            metric_values[metric_name] = _coerce_importance_values(
                metric_name,
                _call_metric(metric_fn, model=model, X=X_boot, y=y_boot, rng=rng),
                len(feature_names),
            )

        for index, feature_name in enumerate(feature_names):
            for metric_name, _ in vi_metric_specs:
                if metric_values[metric_name] is not None:
                    boot_importances[metric_name][feature_name].append(
                        float(metric_values[metric_name][index])
                    )

        y_pred = model.predict(X_boot)
        y_prob = model.predict_proba(X_boot)
        for metric_name, metric_fn in performance_metric_specs:
            metric_value = _coerce_performance_value(
                metric_name,
                _call_metric(
                    metric_fn,
                    model=model,
                    X=X_boot,
                    y=y_boot,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    rng=rng,
                ),
            )
            if metric_value is not None:
                boot_perf[metric_name].append(metric_value)

    return boot_importances, boot_perf


def _run_single_bootstrap_cross_family(
    b,
    X_scaled,
    y,
    model_configs,
    n_models_per_class,
    epsilon,
    feature_names,
    vi_metric_specs,
    performance_metric_specs,
    family_balance_mode,
):
    rng = np.random.default_rng(seed=b)
    X_boot, y_boot = resample(X_scaled, y, random_state=b)

    family_models = {}
    family_losses = {}
    for name, (trainer, kwargs) in model_configs.items():
        models, losses = train_candidate_models(
            trainer, X_boot, y_boot, n_models_per_class, b, kwargs
        )
        family_models[name] = models
        family_losses[name] = losses

    family_rashomon = {name: [] for name in model_configs}
    for name in model_configs:
        models = family_models[name]
        losses = family_losses[name]
        if len(losses) == 0:
            continue

        family_min_loss = np.min(losses)
        family_rashomon_mask = losses <= family_min_loss + epsilon
        family_rashomon[name] = [model for model, keep in zip(models, family_rashomon_mask) if keep]

    non_empty_families = [name for name, models in family_rashomon.items() if models]
    if not non_empty_families:
        return None, None, None, None

    rashomon_models = []
    rashomon_labels = []
    for name in non_empty_families:
        for model in family_rashomon[name]:
            rashomon_models.append(model)
            rashomon_labels.append(name)

    if family_balance_mode == "weighted":
        family_sizes = {name: len(family_rashomon[name]) for name in non_empty_families}
        inv_prop_sum = sum(1.0 / s for s in family_sizes.values())
        model_weights = []
        for name in non_empty_families:
            size = family_sizes[name]
            per_model_w = 1.0 / (size * size * inv_prop_sum)
            for _ in family_rashomon[name]:
                model_weights.append(per_model_w)
    else:
        model_weights = None

    if not rashomon_models:
        return None, None, None, None

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

    for model, label in zip(rashomon_models, rashomon_labels):
        metric_values = {}
        for metric_name, metric_fn in vi_metric_specs:
            metric_values[metric_name] = _coerce_importance_values(
                metric_name,
                _call_metric(metric_fn, model=model, X=X_boot, y=y_boot, rng=rng),
                len(feature_names),
            )

        for index, feature_name in enumerate(feature_names):
            for metric_name, _ in vi_metric_specs:
                if metric_values[metric_name] is not None:
                    boot_importances[metric_name][feature_name].append(
                        float(metric_values[metric_name][index])
                    )

        y_pred = model.predict(X_boot)
        y_prob = model.predict_proba(X_boot)
        for metric_name, metric_fn in performance_metric_specs:
            metric_value = _coerce_performance_value(
                metric_name,
                _call_metric(
                    metric_fn,
                    model=model,
                    X=X_boot,
                    y=y_boot,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    rng=rng,
                ),
            )
            if metric_value is not None:
                boot_family_perf[label][metric_name].append(metric_value)

    return boot_importances, boot_family_counts, boot_family_perf, model_weights


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
):
    """Compute single-family RID and return metric results, performance, and valid bootstrap count."""

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

    X_values, y_values, feature_names = _prepare_inputs(X, y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_values)

    model_name = getattr(model_class, "__name__", type(model_trainer).__name__)
    print(
        f"RID: {model_name}, {len(feature_names)} features, "
        f"B={n_bootstraps}, pool={n_models_pool}, epsilon={epsilon}, n_jobs={n_jobs}, "
        f"metrics={vi_metric_names}, performance_metrics={performance_metric_names}"
    )

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_single_bootstrap)(
            b,
            X_scaled,
            y_values,
            model_trainer,
            model_kwargs,
            n_models_pool,
            epsilon,
            feature_names,
            vi_metric_specs,
            performance_metric_specs,
        )
        for b in range(n_bootstraps)
    )

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
        print("Warning: No bootstraps produced a non-empty Rashomon set.")
        return None, None, 0

    active_metrics = _active_metrics(per_bootstrap_importances, feature_names, vi_metric_names)
    if not active_metrics:
        print(f"Warning: No active VI metrics produced values for metrics={vi_metric_names}")

    metric_results = _aggregate_metric_results(
        per_bootstrap_importances,
        feature_names,
        active_metrics,
        n_cdf_points,
    )
    perf_stats = _aggregate_perf_stats(per_bootstrap_perf, performance_metric_specs)

    print(
        f"  Done. {n_valid}/{n_bootstraps} valid bootstraps. "
        f"Metrics: {list(metric_results.keys())}  "
        f"{_format_perf_stats(perf_stats, performance_metric_specs)}"
    )
    return metric_results, perf_stats, n_valid


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
):
    """Compute cross-family RID and return metric results, family counts, and family performance."""

    vi_metric_specs = normalize_vi_metrics(vi_metrics)
    performance_metric_specs = normalize_performance_metrics(performance_metrics)
    vi_metric_names = tuple(metric_name for metric_name, _ in vi_metric_specs)
    performance_metric_names = tuple(metric_name for metric_name, _ in performance_metric_specs)
    family_balance_mode = normalize_cross_family_balance_mode(family_balance_mode)

    X_values, y_values, feature_names = _prepare_inputs(X, y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_values)

    resolved_model_configs = {}
    for name, model_config in model_configs.items():
        _, trainer, model_kwargs = resolve_model_config(model_config)
        resolved_model_configs[name] = (trainer, model_kwargs)

    family_names = list(resolved_model_configs.keys())
    print(
        f"Cross-family RID (per-family Rashomon): {family_names}, {len(feature_names)} features, "
        f"B={n_bootstraps}, pool={n_models_per_class}/class, epsilon={epsilon}, n_jobs={n_jobs}, "
        f"metrics={vi_metric_names}, performance_metrics={performance_metric_names}, "
        f"family_balance_mode={family_balance_mode}"
    )

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_single_bootstrap_cross_family)(
            b,
            X_scaled,
            y_values,
            resolved_model_configs,
            n_models_per_class,
            epsilon,
            feature_names,
            vi_metric_specs,
            performance_metric_specs,
            family_balance_mode,
        )
        for b in range(n_bootstraps)
    )

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
        print("Warning: No bootstraps produced a non-empty Rashomon set.")
        return None, family_counts, None, 0

    active_metrics = _active_metrics(per_bootstrap_importances, feature_names, vi_metric_names)
    if not active_metrics:
        print(f"Warning: No active VI metrics produced values for metrics={vi_metric_names}")

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

    print(f"  Done. {n_valid}/{n_bootstraps} valid bootstraps. Metrics: {list(metric_results.keys())}")
    print(f"  Family representation in Rashomon sets: {family_counts}")
    print("  Performance by model family (Rashomon models, bootstrap mean):")
    for name, perf_stats in family_perf_stats.items():
        print(f"    {name}: {_format_perf_stats(perf_stats, performance_metric_specs)}")

    return metric_results, family_counts, family_perf_stats, n_valid


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
    """Single-family RID estimator with a sklearn-style fit API."""

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

    def fit(self, X, y):
        self.metric_results_, self.perf_stats_, self.n_valid_bootstraps_ = compute_rid(
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
        )
        _, _, self.feature_names_ = _prepare_inputs(X, y)
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
        Candidate pool size requested per family and bootstrap before Rashomon
        filtering.

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
          bootstrap (models are reweighted inversely by family Rashomon count).

        If ``None``, this is normalized to ``"unweighted"``.

    n_cdf_points : int, default=200
        Number of points in the empirical CDF grid used for each metric's RID
        curve.

    n_jobs : int, default=1
        Number of parallel jobs for bootstrap execution (passed to
        :class:`joblib.Parallel`).

    Attributes
    ----------
    metric_results_ : dict or None
        Mapping ``metric_name -> (rid_cdfs, cdf_grid, raw_importances)``.
        ``None`` if no bootstrap produced a non-empty Rashomon set.

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

    def fit(self, X, y):
        (
            self.metric_results_,
            self.family_counts_,
            self.family_perf_stats_,
            self.n_valid_bootstraps_,
        ) = compute_rid_cross_family(
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
        )
        _, _, self.feature_names_ = _prepare_inputs(X, y)
        self.available_metrics_ = tuple(self.metric_results_.keys()) if self.metric_results_ else tuple()
        return self