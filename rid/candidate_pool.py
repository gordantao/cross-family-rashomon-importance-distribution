"""Shared per-bootstrap candidate pool for cross-family RID.

Both cross-family candidate-search modes start from the same pool, drawn here:

- ``candidate_search="grid"`` fits every candidate in it once
  (:func:`fit_candidate_pool`);
- ``candidate_search="truvarimp_joint"`` (``rid.truvarimp``) spends a fit budget
  on an adaptively chosen subset of it.

Per bootstrap ``b``, each family with a continuous search space (see
``DEFAULT_CONTINUOUS_SEARCH_SPACES_BY_NAME``) and a ``GridCandidateTrainer``-style
``build_estimator`` gets ``n_candidates_per_family`` random draws, split evenly
across its categorical sub-families (one per combination of categorical
choices, e.g. RandomForest ``max_features`` {sqrt, log2} -> 250 + 250 of 500).
Configs that collide after integer rounding (``max_depth``, ``n_estimators``,
...) are dropped, keeping the first occurrence, so a family can end up with
slightly fewer than its quota.

Every candidate carries its own ``random_state``, a deterministic function of
(bootstrap, family position, candidate index) -- see :func:`pool_random_state`
-- so the same candidate yields the identical fitted model and loss in either
search mode. The draws themselves are seeded per (bootstrap, family position),
so both modes see the identical pool for a given ``b``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .models import GridCandidateTrainer, candidate_loss

DEFAULT_N_CANDIDATES_PER_FAMILY = 500

# ---------------------------------------------------------------------------
# Search-space specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Continuous:
    """A continuous hyperparameter dimension, optionally searched on a log scale."""

    low: float
    high: float
    log: bool = False

    def sample(self, rng, n):
        if self.log:
            return np.exp(rng.uniform(np.log(self.low), np.log(self.high), size=n))
        return rng.uniform(self.low, self.high, size=n)


@dataclass(frozen=True)
class Categorical:
    """A categorical hyperparameter dimension -- split into independent sub-families."""

    choices: Sequence[Any]


@dataclass(frozen=True)
class ContinuousSearchSpace:
    """A family's hyperparameter search space for the shared candidate pool.

    ``dimensions`` maps a hyperparameter name (matching the keyword the family's
    ``build_estimator`` accepts in its dict/Mapping config form) to a
    :class:`Continuous` or :class:`Categorical` spec.
    """

    dimensions: Mapping[str, Any]

    def categorical_dimensions(self):
        return {name: dim for name, dim in self.dimensions.items() if isinstance(dim, Categorical)}

    def continuous_dimensions(self):
        return {name: dim for name, dim in self.dimensions.items() if isinstance(dim, Continuous)}


def split_categorical_subspaces(search_space):
    """Expand a search space's categorical axes into (fixed_assignment, continuous_names) pairs.

    Returns a list of ``(fixed_values, continuous_names)`` tuples, one per
    combination of categorical choices, where ``fixed_values`` is a dict of
    categorical hyperparameter -> chosen value and ``continuous_names`` is the
    ordered list of continuous hyperparameter names to sample/encode for the GP.
    """

    categorical = search_space.categorical_dimensions()
    continuous_names = list(search_space.continuous_dimensions().keys())

    if not categorical:
        return [({}, continuous_names)]

    names = list(categorical.keys())
    choice_lists = [categorical[name].choices for name in names]
    combos = []
    for combo in itertools.product(*choice_lists):
        combos.append((dict(zip(names, combo)), continuous_names))
    return combos


# ---------------------------------------------------------------------------
# Default continuous search spaces for the model classes already registered in
# rid/models.py's TRAINER_REGISTRY. These mirror each family's discrete
# single-family grid's *range*, just sampled continuously instead of at a
# handful of fixed points. Used by both cross-family candidate_search modes.
# ---------------------------------------------------------------------------


def _logistic_search_space(penalty_choices):
    dims = {"C": Continuous(1e-4, 1e3, log=True)}
    if len(penalty_choices) > 1:
        dims["penalty"] = Categorical(penalty_choices)
    return ContinuousSearchSpace(dims)


DEFAULT_CONTINUOUS_SEARCH_SPACES_BY_NAME = {
    "LogisticRegression": _logistic_search_space(["l1", "l2", "elasticnet"]),
    "LassoClassifier": _logistic_search_space(["l1"]),
    "RidgeClassifier": _logistic_search_space(["l2"]),
    "ElasticNetClassifier": ContinuousSearchSpace(
        {
            "C": Continuous(1e-3, 1e2, log=True),
            "l1_ratio": Continuous(0.05, 0.95, log=False),
        }
    ),
    "RandomForestClassifier": ContinuousSearchSpace(
        {
            "n_estimators": Continuous(50, 300, log=True),
            "max_depth": Continuous(2, 20, log=True),
            "max_features": Categorical(["sqrt", "log2"]),
        }
    ),
    "GradientBoostingClassifier": ContinuousSearchSpace(
        {
            "n_estimators": Continuous(50, 300, log=True),
            "learning_rate": Continuous(1e-2, 0.5, log=True),
            "max_depth": Continuous(2, 5, log=True),
        }
    ),
    "SVC": ContinuousSearchSpace(
        {
            "C": Continuous(1e-3, 1e4, log=True),
            "gamma": Continuous(1e-3, 1e1, log=True),
        }
    ),
    "FullyEnumeratedTreeClassifier": ContinuousSearchSpace(
        {
            "max_depth": Continuous(2, 6, log=False),
            "min_samples_leaf": Continuous(1, 20, log=True),
            "criterion": Categorical(["gini", "entropy"]),
        }
    ),
}


def resolve_default_search_space(model_class):
    """Look up the built-in continuous search space for a model class, if any."""

    return DEFAULT_CONTINUOUS_SEARCH_SPACES_BY_NAME.get(getattr(model_class, "__name__", None))


def _round_integer_like(name, value):
    """Round hyperparameters that must be integers (sampled continuously above)."""

    if name in ("n_estimators", "max_depth", "min_samples_leaf", "minbucket", "minsplit"):
        return int(round(value))
    return value


def drop_pinned_dimensions(search_space, model_kwargs):
    """Remove search dimensions the caller already pinned via ``model_kwargs``.

    A family declared as ``(RandomForestClassifier, {"max_depth": 5})`` is
    asking for that value, so searching ``max_depth`` anyway would silently
    override it (the trainer path lets ``model_kwargs`` win, or raises on the
    duplicate keyword -- it never discards the caller's value). Dropping the
    dimension honors the pin AND keeps the GP from carrying a constant,
    uninformative input column. Returns ``(search_space, pinned_names)``.
    """

    if not model_kwargs:
        return search_space, ()

    pinned = tuple(name for name in search_space.dimensions if name in model_kwargs)
    if not pinned:
        return search_space, ()

    remaining = {
        name: dim for name, dim in search_space.dimensions.items() if name not in model_kwargs
    }
    return ContinuousSearchSpace(remaining), pinned


# ---------------------------------------------------------------------------
# Per-candidate random_state
# ---------------------------------------------------------------------------

# random_state = (b * _POOL_MAX_FAMILIES + family_index) * _POOL_MAX_CANDIDATES + candidate_index.
# Each bound is enforced, so distinct (b, family, candidate) triples never share a
# seed; the product keeps every seed below 2**32 (sklearn's limit) for b < 13422.
_POOL_MAX_FAMILIES = 32
_POOL_MAX_CANDIDATES = 10_000
_MAX_RANDOM_STATE = 2**32 - 1


def pool_random_state(b, family_index, candidate_index):
    """The ``random_state`` a pool candidate is fit with, in either search mode."""

    if not 0 <= family_index < _POOL_MAX_FAMILIES:
        raise ValueError(f"The shared candidate pool supports at most {_POOL_MAX_FAMILIES} families")
    if not 0 <= candidate_index < _POOL_MAX_CANDIDATES:
        raise ValueError(
            f"The shared candidate pool supports at most {_POOL_MAX_CANDIDATES} candidates per family"
        )
    seed = (b * _POOL_MAX_FAMILIES + family_index) * _POOL_MAX_CANDIDATES + candidate_index
    if not 0 <= seed <= _MAX_RANDOM_STATE:
        raise ValueError(f"Bootstrap index {b} is too large for a valid candidate random_state")
    return int(seed)


# ---------------------------------------------------------------------------
# Pool construction and exhaustive fitting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubfamilyPool:
    """Candidates of one categorical sub-family (or a whole family without categoricals)."""

    name: str
    fixed_values: Mapping[str, Any]
    continuous_names: Sequence[str]
    log_flags: Mapping[str, bool]
    configs: Sequence[dict]  # full estimator kwargs per candidate (model kwargs included)
    random_states: Sequence[int]


@dataclass(frozen=True)
class FamilyPool:
    """One family's share of the bootstrap's candidate pool."""

    family: str
    build_estimator: Callable[..., Any]
    subfamilies: Sequence[SubfamilyPool]
    pinned: Sequence[str]

    @property
    def n_candidates(self):
        return sum(len(sub.configs) for sub in self.subfamilies)


def _split_quota(n_total, n_parts):
    """Split ``n_total`` evenly over ``n_parts``; the first ``n_total % n_parts`` get one more."""

    base, remainder = divmod(n_total, n_parts)
    return [base + (1 if i < remainder else 0) for i in range(n_parts)]


def _config_key(config):
    return tuple((name, repr(value)) for name, value in sorted(config.items()))


def build_candidate_pool(resolved_model_configs, search_spaces, b, n_candidates_per_family):
    """Draw bootstrap ``b``'s shared candidate pool.

    Parameters
    ----------
    resolved_model_configs : dict[str, (trainer, model_kwargs)]
        Families in their declared order; a family's position in this mapping
        (counting every family, pooled or not) is its ``family_index`` for
        seeding, so the same configs always give the same pool.
    search_spaces : dict[str, ContinuousSearchSpace or None]
        Search space per family name.
    b : int
        Bootstrap index.
    n_candidates_per_family : int
        Draws per family, split evenly across its categorical sub-families.

    Returns
    -------
    dict[str, FamilyPool or None]
        ``None`` for a family that is not pooled -- no search space, or a
        trainer that is not a ``GridCandidateTrainer`` (e.g.
        ``SingleCandidateTrainer``, a custom trainer). Callers train those
        through the family's own ``trainer.train``.
    """

    n_candidates_per_family = int(n_candidates_per_family)
    if n_candidates_per_family < 1:
        raise ValueError("n_candidates_per_family must be at least 1")

    pools = {}
    for family_index, (family_name, (trainer, model_kwargs)) in enumerate(
        resolved_model_configs.items()
    ):
        search_space = search_spaces.get(family_name)
        # Only a GridCandidateTrainer's build_estimator takes the
        # (config, random_state, model_kwargs) signature used here.
        # SingleCandidateTrainer's takes model_kwargs alone, and a custom
        # trainer may expose an attribute of that name with any signature.
        if search_space is None or not isinstance(trainer, GridCandidateTrainer):
            pools[family_name] = None
            continue

        search_space, pinned = drop_pinned_dimensions(search_space, model_kwargs)
        log_flags = {name: dim.log for name, dim in search_space.continuous_dimensions().items()}
        subspaces = split_categorical_subspaces(search_space)
        rng = np.random.default_rng([b, family_index])

        subfamilies = []
        candidate_index = 0
        for (fixed_values, continuous_names), quota in zip(
            subspaces, _split_quota(n_candidates_per_family, len(subspaces))
        ):
            samples = {
                name: search_space.dimensions[name].sample(rng, quota) for name in continuous_names
            }
            configs = []
            random_states = []
            seen = set()
            for i in range(quota):
                point = dict(model_kwargs)
                point.update(fixed_values)
                for name in continuous_names:
                    point[name] = _round_integer_like(name, float(samples[name][i]))
                key = _config_key(point)
                if key in seen:
                    continue
                seen.add(key)
                configs.append(point)
                random_states.append(pool_random_state(b, family_index, candidate_index))
                candidate_index += 1
            if not configs:
                continue
            sub_name = family_name if not fixed_values else f"{family_name}::{fixed_values}"
            subfamilies.append(
                SubfamilyPool(
                    name=sub_name,
                    fixed_values=fixed_values,
                    continuous_names=continuous_names,
                    log_flags=log_flags,
                    configs=configs,
                    random_states=random_states,
                )
            )

        pools[family_name] = FamilyPool(
            family=family_name,
            build_estimator=trainer.build_estimator,
            subfamilies=subfamilies,
            pinned=pinned,
        )
    return pools


def fit_pool_candidate(build_estimator, config, random_state, X_boot, y_boot, X_eval, y_eval):
    """Fit one pool candidate in-bag and score it out-of-bag; shared by both search modes."""

    model = build_estimator(dict(config), random_state, {})
    model.fit(X_boot, y_boot)
    return model, candidate_loss(model, X_eval, y_eval)


def fit_candidate_pool(family_pool, X_boot, y_boot, X_eval, y_eval):
    """Exhaustively fit every candidate in one family's pool (``candidate_search="grid"``)."""

    models = []
    losses = []
    for sub in family_pool.subfamilies:
        for config, random_state in zip(sub.configs, sub.random_states):
            model, loss = fit_pool_candidate(
                family_pool.build_estimator, config, random_state, X_boot, y_boot, X_eval, y_eval
            )
            models.append(model)
            losses.append(loss)
    return models, np.array(losses)
