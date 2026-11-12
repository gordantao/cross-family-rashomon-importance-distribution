#!/usr/bin/env python3
"""Diagnose why some model families never enter the cross-family Rashomon set.

In the OOB TruVaRImp Staellert run, random forests had 0 Rashomon members across
all 100 bootstraps. compute_rid_cross_family discards every candidate's loss once
membership is decided, so this script fits the whole shared candidate pool
exhaustively (what candidate_search="grid" does; same data prep, scaler,
bootstrap splits, pool and per-candidate seeds as the
cross_family_unweighted_truvarimp method of run_staellert_top40_comparison.py,
whose TruVaRImp search fits only a budgeted subset of that pool, see
rid.candidate_pool) and records one row per candidate instead of a Rashomon set.

It mirrors the comparison's outer stratified holdout protocol: the labelled rows
are split once (--holdout-fraction, --holdout-seed; same defaults as the
comparison), imputation and column filters are fit on the train rows, and the
candidate search runs on the train rows only -- exactly the X, y that
run_cross_family_rid receives there. The held-out test rows are never touched.
Each candidate row records:

- OOB log-loss (exactly rid.models.candidate_loss, the membership loss),
  accuracy, balanced accuracy, Brier score, top-label ECE, per-class mean loss,
  and how concentrated the loss is in a few badly-predicted rows;
- the same log-loss after temperature scaling, cross-fitted on two halves of
  the OOB rows (the temperature for one half is fit on the other), to separate
  "ranks cells worse" from "reports less confident probabilities".

Each bootstrap also saves the OOB predicted probabilities of every family's
best candidate, for reliability diagrams. ``plot`` only reads collections of
shared-pool candidates (dict configs); it refuses older collections (with the
since-removed RF probe family or hand-grid tuple configs).

Usage:
    # cluster: collect (checkpoints one file pair per bootstrap; reruns skip them)
    python experiments/staellert/diagnose_rashomon_family_losses.py collect --n-jobs 50
    # local, after copying the results dir back: plots + summary table
    python experiments/staellert/diagnose_rashomon_family_losses.py plot
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))
sys.path.insert(0, str(SCRIPT_DIR))

from rid import add_logging_arguments, configure_logging, silence_warnings  # noqa: E402
from rid.core import _bootstrap_split, _prepare_inputs  # noqa: E402
from rid.candidate_pool import (  # noqa: E402
    DEFAULT_N_CANDIDATES_PER_FAMILY,
    build_candidate_pool,
    resolve_default_search_space,
)
from rid.models import GridCandidateTrainer, candidate_loss, resolve_model_config  # noqa: E402
from run_staellert_top40_comparison import (  # noqa: E402
    TASK_REGISTRY,
    _build_cross_family_model_configs,
    _prepare_task_dataset,
)

logger = logging.getLogger(__name__)

# Same clipping as sklearn's log_loss (dtype eps, then renormalize rows), so
# per-row losses agree with candidate_loss even where a model predicts p=0.
PROB_CLIP = np.finfo(np.float64).eps
ECE_BINS = 15
# Settings that change collected rows; a results dir written under others is refused.
SETTING_KEYS = ("task", "correlation_threshold", "holdout_fraction", "holdout_seed", "n_models_pool",
                "n_candidates_per_family", "families", "evaluation_protocol",
                "holdout_test_rows_sha256")


# --------------------------------------------------------------------------- collect


def _clip_probabilities(proba):
    clipped = np.clip(proba, PROB_CLIP, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


def _per_sample_true_loss(proba, y_idx):
    p_true = _clip_probabilities(proba)[np.arange(len(y_idx)), y_idx]
    return -np.log(p_true), p_true


def _top_label_ece(proba, y_idx):
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_idx
    bins = np.minimum((confidence * ECE_BINS).astype(int), ECE_BINS - 1)
    ece = 0.0
    for k in range(ECE_BINS):
        in_bin = bins == k
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def _fit_temperature(logits, y_idx):
    def loss(log_t):
        return -log_softmax(logits / np.exp(log_t), axis=1)[np.arange(len(y_idx)), y_idx].mean()

    return float(np.exp(minimize_scalar(loss, bounds=(-4.0, 4.0), method="bounded").x))


def _temperature_scaled_loss(proba, y_idx, folds):
    """Cross-fitted temperature-scaled log-loss over all OOB rows, plus the mean fitted T.

    T > 1 means the model was overconfident (scaling softens it); T < 1 means
    underconfident (scaling sharpens it)."""

    logits = np.log(_clip_probabilities(proba))
    row_loss = np.empty(len(y_idx))
    temperatures = []
    for fit_rows, eval_rows in folds:
        t = _fit_temperature(logits[fit_rows], y_idx[fit_rows])
        temperatures.append(t)
        scaled = log_softmax(logits[eval_rows] / t, axis=1)
        row_loss[eval_rows] = -scaled[np.arange(len(eval_rows)), y_idx[eval_rows]]
    return float(row_loss.mean()), float(np.mean(temperatures))


def _candidate_row(model, X_oob, y_oob, classes, folds):
    proba = model.predict_proba(X_oob)
    # Columns follow model.classes_; an in-bag sample missing a class would
    # leave that column out, so map OOB labels by name, not by position.
    col = {label: j for j, label in enumerate(model.classes_)}
    y_idx = np.array([col.get(label, -1) for label in y_oob])
    if (y_idx < 0).any():
        raise RuntimeError(f"OOB labels {set(y_oob[y_idx < 0])} missing from model.classes_")

    row_loss, p_true = _per_sample_true_loss(proba, y_idx)
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    sorted_loss = np.sort(row_loss)[::-1]
    n_top = max(1, int(np.ceil(0.01 * len(sorted_loss))))
    ts_loss, temperature = _temperature_scaled_loss(proba, y_idx, folds)

    row = {
        "log_loss": candidate_loss(model, X_oob, y_oob),
        "accuracy": float((proba.argmax(axis=1) == y_idx).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_idx, proba.argmax(axis=1))),
        "brier": float(((proba - onehot) ** 2).sum(axis=1).mean()),
        "ece": _top_label_ece(proba, y_idx),
        "mean_confidence": float(proba.max(axis=1).mean()),
        "min_p_true": float(p_true.min()),
        "frac_p_true_below_0.01": float((p_true < 0.01).mean()),
        "top1pct_loss_share": float(sorted_loss[:n_top].sum() / sorted_loss.sum()),
        "ts_log_loss": ts_loss,
        "temperature": temperature,
    }
    for label in classes:
        in_class = y_oob == label
        row[f"loss_class_{label}"] = float(row_loss[in_class].mean()) if in_class.any() else np.nan
    return row, proba


def _family_candidates(name, trainer, model_kwargs, family_pool, n_models_pool, b):
    """Yield (config, seed, estimator) exactly as compute_rid_cross_family's grid search
    builds them: every shared-pool candidate with its own seed, or, for a family outside
    the pool, GridCandidateTrainer.train's configs/seeds."""

    if family_pool is not None:
        for sub in family_pool.subfamilies:
            for config, seed in zip(sub.configs, sub.random_states):
                yield config, seed, family_pool.build_estimator(dict(config), seed, {})
        return
    if not isinstance(trainer, GridCandidateTrainer):
        raise TypeError(f"family {name} is not grid-trained; this diagnostic only handles grid families")
    per = max(1, n_models_pool // len(trainer.configs)) if trainer.refit_seeds else 1
    for config in trainer.configs:
        for i in range(per):
            seed = b * trainer.random_state_multiplier + i
            yield config, seed, trainer.build_estimator(config, seed, model_kwargs)


def _config_columns(config):
    if isinstance(config, dict):
        return {f"cfg_{k}": v for k, v in config.items()}
    if isinstance(config, tuple):
        return {f"cfg_{i}": v for i, v in enumerate(config)}
    return {"cfg_0": config}


def _collect_bootstrap(b, X_scaled, y, families, search_spaces, n_models_pool,
                       n_candidates_per_family, classes, out_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        split = _bootstrap_split(X_scaled, y, b)
        if split is None:
            return b, 0, 0.0
        X_boot, y_boot, X_oob, y_oob = split
        oob_rows = _oob_rows(len(y), b)
        if len(oob_rows) != len(y_oob) or not np.array_equal(y[oob_rows], y_oob):
            raise RuntimeError(f"bootstrap {b}: _oob_rows no longer matches rid.core._bootstrap_split")
        folds = list(KFold(n_splits=2, shuffle=True, random_state=b).split(X_oob))

        # A family's pool draws and seeds depend only on (b, family name), so a
        # --families subset sees the same candidates as the full comparison run.
        pool = build_candidate_pool(families, search_spaces, b, n_candidates_per_family)
        sources = [(name, "standard",
                    _family_candidates(name, trainer, kwargs, pool.get(name), n_models_pool, b))
                   for name, (trainer, kwargs) in families.items()]

        rows = []
        best = {}
        for name, pool, candidates in sources:
            for config, seed, model in candidates:
                fit_start = time.perf_counter()
                model.fit(X_boot, y_boot)
                fit_seconds = time.perf_counter() - fit_start
                row, proba = _candidate_row(model, X_oob, y_oob, classes, folds)
                rows.append({"bootstrap": b, "family": name, "pool": pool, "seed": seed,
                             "config": repr(config), **_config_columns(config),
                             "fit_seconds": fit_seconds, **row})
                if name not in best or row["log_loss"] < best[name][0]:
                    best[name] = (row["log_loss"], proba, list(model.classes_))
                del model

        arrays = {"oob_rows": oob_rows, "y_oob": y_oob.astype(str)}
        for name, (_, proba, model_classes) in best.items():
            arrays[f"proba__{name}"] = proba
            arrays[f"classes__{name}"] = np.array(model_classes, dtype=str)
        # npz first: the candidates CSV is what marks a bootstrap done (_bootstrap_done).
        _write_atomic(lambda p: np.savez_compressed(p, **arrays), out_dir / f"b{b:03d}_best_proba.npz")
        df = pd.DataFrame(rows)
        _write_atomic(lambda p: df.to_csv(p, index=False), out_dir / f"b{b:03d}_candidates.csv")
        return b, len(rows), time.perf_counter() - start


def _oob_rows(n, b):
    """Row indices _bootstrap_split treats as out-of-bag for bootstrap b, in the same order."""

    oob_mask = np.ones(n, dtype=bool)
    oob_mask[resample(np.arange(n), random_state=b)] = False
    return np.flatnonzero(oob_mask)


def _bootstrap_done(per_bootstrap_dir, b):
    return all((per_bootstrap_dir / f"b{b:03d}_{suffix}").exists()
               for suffix in ("candidates.csv", "best_proba.npz"))


def _write_atomic(write, path):
    tmp_path = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    write(tmp_path)
    os.replace(tmp_path, path)


def _check_settings(output_dir, settings):
    path = output_dir / "settings.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        # Collections from before the RF probe was removed record rf_probe (default on)
        # and hold probe rows; resuming one would mix in probe-less bootstraps.
        if existing.get("rf_probe") or (
            {k: existing.get(k) for k in SETTING_KEYS} != {k: settings[k] for k in SETTING_KEYS}
        ):
            raise SystemExit(
                f"{path} was written with {existing}, but this run uses {settings}. "
                "Use a different --output-dir, or delete it to recompute."
            )
    _write_atomic(lambda p: p.write_text(json.dumps(settings, indent=2), encoding="utf-8"), path)


def run_collect(args):
    task_config = TASK_REGISTRY[args.task]
    # Train rows only, as in the comparison; the test rows are discarded unseen.
    X, _, y, _, meta = _prepare_task_dataset(
        csv_path=Path(args.data_dir) / task_config["file"],
        target_column=task_config["target_column"],
        correlation_threshold=args.correlation_threshold,
        include_phate_features=False,
        holdout_fraction=args.holdout_fraction,
        holdout_seed=args.holdout_seed,
    )
    holdout_meta = meta["holdout"]
    # Same preprocessing as compute_rid_cross_family: scale on all (train) rows once.
    X_values, y_values, _ = _prepare_inputs(X, y)
    X_scaled = StandardScaler().fit_transform(X_values)
    classes = sorted(np.unique(y_values))

    all_families = _build_cross_family_model_configs()
    unknown = set(args.families or []) - set(all_families)
    if unknown:
        raise SystemExit(f"unknown --families {sorted(unknown)}; choose from {list(all_families)}")
    families = {}
    search_spaces = {}
    for name, model_config in all_families.items():
        if args.families and name not in args.families:
            continue
        model_class, trainer, kwargs = resolve_model_config(model_config)
        families[name] = (trainer, kwargs)
        search_spaces[name] = resolve_default_search_space(model_class)

    output_dir = Path(args.output_dir)
    per_bootstrap_dir = output_dir / "per_bootstrap"
    per_bootstrap_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "task": args.task,
        "correlation_threshold": args.correlation_threshold,
        "holdout_fraction": args.holdout_fraction,
        "holdout_seed": args.holdout_seed,
        "n_models_pool": args.n_models_pool,
        "n_candidates_per_family": DEFAULT_N_CANDIDATES_PER_FAMILY,
        "families": list(families),
        "evaluation_protocol": holdout_meta["protocol"],
        "holdout_test_rows_sha256": holdout_meta["test_rows_sha256"],
        "n_rows": meta["n_rows"],
        "n_train_rows": int(X.shape[0]),
        "n_features": meta["n_features_after_corr"],
        "class_counts": meta["class_counts"],
        "train_class_counts": holdout_meta["train_class_counts"],
        "rashomon_evaluation": "out_of_bag",
    }
    _check_settings(output_dir, settings)

    bootstraps = args.bootstraps if args.bootstraps else list(range(args.n_bootstraps))
    todo = [b for b in bootstraps if not _bootstrap_done(per_bootstrap_dir, b)]
    logger.info(f"{X.shape[0]} train rows (of {meta['n_rows']}; test rows sha256 "
                f"{holdout_meta['test_rows_sha256']}) x {X.shape[1]} features, "
                f"classes {holdout_meta['train_class_counts']}")
    logger.info(f"families: {list(families)}")
    logger.info(f"bootstraps: {len(bootstraps)} requested, {len(bootstraps) - len(todo)} already done, "
                f"{len(todo)} to run on {args.n_jobs} workers")

    start = time.perf_counter()
    done = 0
    results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
        delayed(_collect_bootstrap)(b, X_scaled, y_values, families, search_spaces, args.n_models_pool,
                                    DEFAULT_N_CANDIDATES_PER_FAMILY, classes,
                                    per_bootstrap_dir)
        for b in todo
    )
    for b, n_rows, seconds in results:
        done += 1
        logger.info(f"bootstrap {b}: {n_rows} candidates in {seconds / 60:.1f} min "
                    f"({done}/{len(todo)}, elapsed {(time.perf_counter() - start) / 3600:.2f} h)")

    combined = _load_candidates(per_bootstrap_dir)
    combined.to_csv(output_dir / "candidates.csv.gz", index=False)
    logger.info(f"wrote {len(combined)} candidate rows from {combined['bootstrap'].nunique()} bootstraps "
                f"(every finished one in {per_bootstrap_dir}) to {output_dir / 'candidates.csv.gz'}")


def _load_candidates(per_bootstrap_dir):
    paths = sorted(per_bootstrap_dir.glob("b*_candidates.csv"))
    if not paths:
        raise SystemExit(f"no finished bootstraps in {per_bootstrap_dir} (all OOB splits invalid, or not run yet)")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


# --------------------------------------------------------------------------- plot


FAMILY_ORDER = ["GradientBoosting", "SVM", "Lasso", "ElasticNet", "Ridge", "RandomForest"]
FAMILY_COLORS = {
    "GradientBoosting": "#4C72B0",
    "SVM": "#8172B3",
    "Lasso": "#55A868",
    "ElasticNet": "#64B5CD",
    "Ridge": "#937860",
    "RandomForest": "#C44E52",
}


def _with_gaps(df):
    """Add each candidate's gap to its bootstrap's best standard-pool loss (raw and temperature-scaled)."""

    standard = df[df["pool"] == "standard"]
    best = standard.groupby("bootstrap")[["log_loss", "ts_log_loss"]].min()
    df = df.join(best, on="bootstrap", rsuffix="_best")
    df["gap"] = df["log_loss"] - df["log_loss_best"]
    df["ts_gap"] = df["ts_log_loss"] - df["ts_log_loss_best"]
    return df


def _families_present(df):
    return [f for f in FAMILY_ORDER if f in set(df["family"])]


def _family_best(df):
    idx = df.groupby(["bootstrap", "family"])["log_loss"].idxmin()
    return df.loc[idx]


def plot_bootstrap_panels(df, epsilon, plt, path, n_panels=12):
    bootstraps = sorted(df["bootstrap"].unique())[:n_panels]
    families = _families_present(df)
    ncols = 4
    nrows = int(np.ceil(len(bootstraps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), sharey=False, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, b in zip(axes.flat, bootstraps):
        sub = df[df["bootstrap"] == b]
        best = sub.loc[sub["pool"] == "standard", "log_loss"].min()
        for i, family in enumerate(families):
            losses = sub.loc[sub["family"] == family, "log_loss"].to_numpy()
            ax.scatter(i + rng.uniform(-0.25, 0.25, len(losses)), losses, s=6, alpha=0.6,
                       color=FAMILY_COLORS[family])
        ax.axhline(best, color="black", lw=1, label="best")
        ax.axhline(best + epsilon, color="black", lw=1, ls="--", label=f"best + ε ({epsilon})")
        ax.set_yscale("log")
        ax.set_title(f"bootstrap {b}", fontsize=9)
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(families,
                           rotation=45, ha="right", fontsize=7)
    for ax in axes.flat[len(bootstraps):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7, loc="upper left")
    axes.flat[0].set_ylabel("OOB log-loss (log scale)")
    fig.suptitle("Every candidate's OOB log-loss, per bootstrap", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gap_by_family(df, epsilon, plt, path):
    families = _families_present(df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, column, title in [(axes[0], "gap", "raw probabilities"),
                              (axes[1], "ts_gap", "after temperature scaling (cross-fitted)")]:
        data = [df.loc[df["family"] == f, column].clip(lower=1e-4).to_numpy() for f in families]
        parts = ax.violinplot(data, showextrema=False, widths=0.8)
        for body, family in zip(parts["bodies"], families):
            body.set_facecolor(FAMILY_COLORS[family])
            body.set_alpha(0.55)
        medians = [np.median(d) for d in data]
        ax.scatter(range(1, len(families) + 1), medians, color="black", s=12, zorder=3, label="median")
        ax.axhline(epsilon, color="black", ls="--", lw=1, label=f"ε = {epsilon}")
        ax.set_yscale("log")
        ax.set_xticks(range(1, len(families) + 1))
        ax.set_xticklabels(families, rotation=30, ha="right")
        ax.set_title(title)
    axes[0].set_ylabel("loss − bootstrap's best loss (log scale; 0 clipped to 1e-4)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Gap to the best candidate, pooled over bootstraps (Rashomon members sit below ε)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_family_best_ecdf(df, epsilon, plt, path):
    best = _family_best(df)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for family in _families_present(df):
        gaps = np.sort(best.loc[best["family"] == family, "gap"].to_numpy())
        ax.step(gaps, np.arange(1, len(gaps) + 1) / len(gaps), where="post",
                color=FAMILY_COLORS[family], label=family, lw=1.8)
    ax.axvline(epsilon, color="black", ls="--", lw=1, label=f"ε = {epsilon}")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("family's best candidate: loss − bootstrap's best loss")
    ax.set_ylabel("fraction of bootstraps")
    ax.set_title("How close does each family's best model get? (left of ε = family has members)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_epsilon_sweep(df, epsilon, plt, path):
    standard = df[df["pool"] == "standard"]
    families = _families_present(standard)
    eps_grid = np.logspace(-3, 0, 60)
    share = {f: [] for f in families}
    any_member = {f: [] for f in families}
    for eps in eps_grid:
        members = standard[standard["gap"] <= eps]
        counts = members["family"].value_counts()
        total = counts.sum()
        per_boot = members.groupby("family")["bootstrap"].nunique()
        for f in families:
            share[f].append(counts.get(f, 0) / total if total else 0.0)
            any_member[f].append(per_boot.get(f, 0) / standard["bootstrap"].nunique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].stackplot(eps_grid, [share[f] for f in families], labels=families,
                      colors=[FAMILY_COLORS[f] for f in families], alpha=0.85)
    axes[0].set_ylabel("share of pooled Rashomon members")
    axes[0].set_ylim(0, 1)
    for f in families:
        axes[1].plot(eps_grid, any_member[f], color=FAMILY_COLORS[f], label=f, lw=1.8)
    axes[1].set_ylabel("fraction of bootstraps with ≥1 member")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("ε (additive log-loss gap)")
        ax.axvline(epsilon, color="black", ls="--", lw=1)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Rashomon set composition as ε varies (dashed: ε = {epsilon})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_accuracy_vs_logloss(df, epsilon, plt, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for family in _families_present(df):
        sub = df[df["family"] == family]
        for ax, x in [(axes[0], "accuracy"), (axes[1], "balanced_accuracy")]:
            ax.scatter(sub[x], sub["gap"].clip(lower=1e-4), s=4, alpha=0.3,
                       color=FAMILY_COLORS[family], label=family)
    for ax, x in [(axes[0], "accuracy"), (axes[1], "balanced_accuracy")]:
        ax.axhline(epsilon, color="black", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel(f"OOB {x.replace('_', ' ')}")
        ax.set_ylabel("log-loss gap to bootstrap's best")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=8, markerscale=3)
    fig.suptitle("Same accuracy but larger log-loss = a calibration problem, not a ranking one")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_temperature(df, plt, path):
    best = _family_best(df)
    families = _families_present(df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    data = [best.loc[best["family"] == f, "temperature"].to_numpy() for f in families]
    parts = axes[0].violinplot(data, showextrema=False, widths=0.8)
    for body, family in zip(parts["bodies"], families):
        body.set_facecolor(FAMILY_COLORS[family])
        body.set_alpha(0.55)
    axes[0].axhline(1.0, color="black", lw=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("fitted temperature T (family's best model)")
    axes[0].set_title("T > 1: overconfident · T < 1: underconfident")
    for ax in axes:
        ax.set_xticks(range(1, len(families) + 1))
        ax.set_xticklabels(families, rotation=30, ha="right")
    width = 0.38
    raw = [best.loc[best["family"] == f, "log_loss"].median() for f in families]
    scaled = [best.loc[best["family"] == f, "ts_log_loss"].median() for f in families]
    x = np.arange(1, len(families) + 1)
    axes[1].bar(x - width / 2, raw, width, color=[FAMILY_COLORS[f] for f in families], label="raw")
    axes[1].bar(x + width / 2, scaled, width, color=[FAMILY_COLORS[f] for f in families], alpha=0.45,
                hatch="//", label="temperature-scaled")
    axes[1].set_ylabel("median OOB log-loss of family's best model")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _rf_label(value):
    """Readable hyperparameter label: CSV round-trips turn ints into floats and None into NaN."""

    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "None":
        return "None"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _label_sort_key(label):
    """Numbers ascending, then named values (None = unbounded depth, sqrt/log2)."""

    try:
        return (0, float(label), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(label))


def plot_rf_heatmaps(df, epsilon, plt, path):
    panels = []
    rf = df[df["family"] == "RandomForest"]
    if not rf.empty and "cfg_n_estimators" in rf:
        # Shared-pool configs are continuous draws, so bin them for the heatmap.
        # Left-closed so exactly 100/200 trees land in "100-199"/">=200".
        rf = rf.assign(
            n_estimators=pd.cut(rf["cfg_n_estimators"], [0, 100, 200, np.inf], right=False,
                                labels=["<100", "100-199", ">=200"]).astype(str),
            max_depth=pd.cut(rf["cfg_max_depth"], [0, 3, 5, 8, 12, np.inf],
                             labels=["2-3", "4-5", "6-8", "9-12", "13-20"]).astype(str),
            max_features=rf["cfg_max_features"].map(_rf_label),
        )
        for n_est in ["<100", "100-199", ">=200"]:
            if (rf["n_estimators"] == n_est).any():
                panels.append((f"standard RF, n_estimators {n_est}", rf[rf["n_estimators"] == n_est],
                               "max_depth", "max_features"))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.6), squeeze=False)
    vmax = max(sub.groupby([r, c])["gap"].median().max() for _, sub, r, c in panels)
    for ax, (title, sub, row_key, col_key) in zip(axes.flat, panels):
        grid = sub.groupby([row_key, col_key])["gap"].median().unstack(col_key)
        grid = grid.loc[sorted(grid.index, key=_label_sort_key), sorted(grid.columns, key=_label_sort_key)]
        image = ax.imshow(grid.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(grid.shape[1]))
        ax.set_xticklabels([str(c) for c in grid.columns], fontsize=8)
        ax.set_yticks(range(grid.shape[0]))
        ax.set_yticklabels([str(r) for r in grid.index], fontsize=8)
        ax.set_xlabel(col_key)
        ax.set_ylabel(row_key)
        ax.set_title(title, fontsize=9)
        for (i, j), value in np.ndenumerate(grid.to_numpy(dtype=float)):
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if value < 0.6 * vmax else "black")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="median gap to best")
    fig.suptitle(f"Random forest: median log-loss gap by hyperparameter (members need ≤ {epsilon})", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(per_bootstrap_dir, families, plt, path, n_bins=10):
    confidence = {f: [] for f in families}
    correct = {f: [] for f in families}
    for npz_path in sorted(per_bootstrap_dir.glob("b*_best_proba.npz")):
        data = np.load(npz_path, allow_pickle=False)
        y_oob = data["y_oob"]
        for f in families:
            if f"proba__{f}" not in data:
                continue
            proba = data[f"proba__{f}"]
            model_classes = data[f"classes__{f}"]
            confidence[f].append(proba.max(axis=1))
            correct[f].append(model_classes[proba.argmax(axis=1)] == y_oob)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    edges = np.linspace(0, 1, n_bins + 1)
    for f in families:
        if not confidence[f]:
            continue
        conf = np.concatenate(confidence[f])
        corr = np.concatenate(correct[f])
        bins = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
        xs, ys = [], []
        for k in range(n_bins):
            mask = bins == k
            if mask.sum() >= 20:
                xs.append(conf[mask].mean())
                ys.append(corr[mask].mean())
        axes[0].plot(xs, ys, marker="o", ms=3, color=FAMILY_COLORS[f], label=f)
        axes[1].hist(conf, bins=edges, histtype="step", lw=1.6, color=FAMILY_COLORS[f], density=True)
    axes[0].plot([0, 1], [0, 1], color="gray", ls=":")
    axes[0].set_xlabel("predicted probability of the top class")
    axes[0].set_ylabel("observed accuracy")
    axes[0].set_title("Reliability (family's best model per bootstrap)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("predicted probability of the top class")
    axes[1].set_ylabel("density")
    axes[1].set_title("Confidence distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class_loss(df, plt, path):
    best = _family_best(df)
    class_cols = [c for c in df.columns if c.startswith("loss_class_")]
    families = _families_present(df)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.8 / len(class_cols)
    for k, col in enumerate(class_cols):
        medians = [best.loc[best["family"] == f, col].median() for f in families]
        ax.bar(np.arange(len(families)) + k * width, medians, width, label=col.replace("loss_class_", "class "))
    ax.set_xticks(np.arange(len(families)) + 0.4 - width / 2)
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_ylabel("median per-row log-loss within class")
    ax.set_title("Which class drives each family's loss? (family's best model per bootstrap)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary_table(df, epsilon):
    best = _family_best(df)
    n_boot = df["bootstrap"].nunique()
    best_acc = df[df["pool"] == "standard"].groupby("bootstrap")["accuracy"].max()
    best = best.join(best_acc, on="bootstrap", rsuffix="_best")
    rows = []
    for family in _families_present(df):
        sub = df[df["family"] == family]
        fb = best[best["family"] == family]
        rows.append({
            "family": family,
            "pool": sub["pool"].iloc[0],
            "candidates": len(sub),
            "rashomon_members": int((sub["gap"] <= epsilon).sum()),
            "bootstraps_with_member": f"{fb['gap'].le(epsilon).sum()}/{n_boot}",
            "best_gap_median": fb["gap"].median(),
            "best_gap_min": fb["gap"].min(),
            "best_ts_gap_median": fb["ts_gap"].median(),
            "bootstraps_with_member_after_ts": f"{fb['ts_gap'].le(epsilon).sum()}/{n_boot}",
            "best_accuracy_median": fb["accuracy"].median(),
            "accuracy_gap_median": (fb["accuracy_best"] - fb["accuracy"]).median(),
            "best_temperature_median": fb["temperature"].median(),
            "best_ece_median": fb["ece"].median(),
            "best_top1pct_loss_share_median": fb["top1pct_loss_share"].median(),
            "fit_seconds_median": sub["fit_seconds"].median(),
        })
    return pd.DataFrame(rows)


def run_plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(args.output_dir)
    per_bootstrap_dir = output_dir / "per_bootstrap"
    df = _load_candidates(per_bootstrap_dir)
    # Only shared-pool collections (standard families, dict configs -> named cfg_*
    # columns) are supported; older ones (an extra probe family, or hand-grid tuple
    # configs -> positional cfg_0/cfg_1/...) would give wrong plots, so refuse them.
    unknown = sorted(set(df["family"]) - set(FAMILY_ORDER))
    if unknown or (df["pool"] != "standard").any() or "cfg_0" in df.columns:
        raise SystemExit(
            f"{per_bootstrap_dir} is an older collection (families outside {FAMILY_ORDER}: {unknown}, "
            f"or non-shared-pool candidates/tuple configs); re-run collect into a fresh --output-dir"
        )
    df = _with_gaps(df)
    logger.info(f"{len(df)} candidates from {df['bootstrap'].nunique()} bootstraps")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    eps = args.epsilon

    plot_bootstrap_panels(df, eps, plt, plots_dir / "01_bootstrap_panels.png")
    plot_gap_by_family(df, eps, plt, plots_dir / "02_gap_by_family.png")
    plot_family_best_ecdf(df, eps, plt, plots_dir / "03_family_best_gap_ecdf.png")
    plot_epsilon_sweep(df, eps, plt, plots_dir / "04_epsilon_sweep.png")
    plot_accuracy_vs_logloss(df, eps, plt, plots_dir / "05_accuracy_vs_logloss.png")
    plot_temperature(df, plt, plots_dir / "06_temperature_scaling.png")
    plot_rf_heatmaps(df, eps, plt, plots_dir / "07_rf_hyperparameter_heatmaps.png")
    plot_reliability(per_bootstrap_dir, _families_present(df), plt, plots_dir / "08_reliability.png")
    plot_per_class_loss(df, plt, plots_dir / "09_per_class_loss.png")

    table = summary_table(df, eps)
    table.to_csv(output_dir / "family_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.precision", 4):
        print(table.to_string(index=False))
    logger.info(f"plots in {plots_dir}, summary in {output_dir / 'family_summary.csv'}")


# --------------------------------------------------------------------------- main


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["collect", "plot"])
    parser.add_argument("--data-dir", default=str(SCRIPT_DIR / "data" / "staellert_et_al"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "results" / "rashomon_family_diagnostics_holdout"))
    parser.add_argument("--task", default="annotated_phase", choices=list(TASK_REGISTRY))
    parser.add_argument("--correlation-threshold", type=float, default=0.8)
    parser.add_argument("--holdout-fraction", type=float, default=0.2,
                        help="Outer stratified test fraction, as in the comparison script (default: 0.2)")
    parser.add_argument("--holdout-seed", type=int, default=42,
                        help="Outer train/test split seed, as in the comparison script (default: 42)")
    parser.add_argument("--n-bootstraps", type=int, default=100)
    parser.add_argument("--bootstraps", type=int, nargs="+", default=None,
                        help="Run only these bootstrap indices (e.g. for a smoke test)")
    parser.add_argument("--n-models-pool", type=int, default=100,
                        help="Seed refits per config for a grid-trained family outside the shared "
                             "candidate pool; none of the default families is (default: 100)")
    parser.add_argument("--families", nargs="+", default=None,
                        help="Restrict the standard pool to these families (smoke tests only: "
                             "the bootstrap's best loss then ignores the others)")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Rashomon epsilon for plots (default: 0.05)")
    parser.add_argument("--n-jobs", type=int, default=1)
    add_logging_arguments(parser)
    return parser.parse_args()


def main():
    args = _parse_args()
    silence_warnings()
    configure_logging(args.log_level, args.log_file)
    if args.command == "collect":
        run_collect(args)
    else:
        run_plot(args)


if __name__ == "__main__":
    main()
