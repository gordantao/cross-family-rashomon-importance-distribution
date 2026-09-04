from __future__ import annotations

from dataclasses import replace
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class LassoClassifier(LogisticRegression):
    """LogisticRegression with L1 penalty."""


class ElasticNetClassifier(LogisticRegression):
    """LogisticRegression with elastic-net penalty."""


class RidgeClassifier(LogisticRegression):
    """LogisticRegression with L2 penalty."""


class LinearClassifier(LogisticRegression):
    """Unregularized LogisticRegression."""


class FullyEnumeratedTreeClassifier(DecisionTreeClassifier):
    """DecisionTreeClassifier whose Rashomon set is built by exhaustively
    enumerating every point of a discretized hyperparameter grid, rather than
    sampling it. Follows the single-family decision-tree treatment in the RID
    literature (Donnelly, Katta, Rudin & Wu), approximating a full Rashomon
    set of trees via grid enumeration instead of a specialized
    branch-and-bound tree enumerator (e.g. TreeFARMS/GOSDT)."""


@dataclass(frozen=True)
class GridCandidateTrainer:
    """Trainer for model families explored through a hyperparameter grid."""

    configs: Sequence[Any]
    random_state_multiplier: int
    build_estimator: Callable[..., Any]

    def train(self, X_boot, y_boot, n_models_pool, b, model_kwargs):
        models = []
        losses = []
        per = max(1, n_models_pool // len(self.configs))

        for config in self.configs:
            for i in range(per):
                random_state = b * self.random_state_multiplier + i
                clf = self.build_estimator(config, random_state, model_kwargs)
                clf.fit(X_boot, y_boot)
                models.append(clf)
                losses.append(log_loss(y_boot, clf.predict_proba(X_boot)))

        return models, np.array(losses)


@dataclass(frozen=True)
class SingleCandidateTrainer:
    """Trainer for single-candidate model families."""

    build_estimator: Callable[..., Any]

    def train(self, X_boot, y_boot, n_models_pool, b, model_kwargs):
        del n_models_pool, b
        clf = self.build_estimator(model_kwargs)
        clf.fit(X_boot, y_boot)
        return [clf], np.array([log_loss(y_boot, clf.predict_proba(X_boot))])


def _normalize_search_grid(search_grid):
    """Normalize custom search-grid input into trainer configs."""

    if search_grid is None:
        return None
    if isinstance(search_grid, Mapping):
        return list(ParameterGrid(search_grid))
    return list(search_grid)


def _build_logistic_estimator(config, random_state, model_kwargs):
    if isinstance(config, Mapping):
        return LogisticRegression(
            random_state=random_state,
            max_iter=100000,
            **config,
            **model_kwargs,
        )

    C, penalty = config
    kwargs = {
        "penalty": penalty,
        "solver": "saga",
        "C": C,
        "max_iter": 100000,
        **model_kwargs,
    }
    if penalty == "elasticnet":
        kwargs["l1_ratio"] = 0.5
    return LogisticRegression(random_state=random_state, **kwargs)


def _build_random_forest_estimator(config, random_state, model_kwargs):
    if isinstance(config, Mapping):
        return RandomForestClassifier(
            random_state=random_state,
            **config,
            **model_kwargs,
        )

    n_estimators, max_depth, max_features = config
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state,
        **model_kwargs,
    )


def _build_gradient_boosting_estimator(config, random_state, model_kwargs):
    if isinstance(config, Mapping):
        return GradientBoostingClassifier(
            random_state=random_state,
            **config,
            **model_kwargs,
        )

    n_estimators, learning_rate, max_depth = config
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        **model_kwargs,
    )


def _build_svc_estimator(config, random_state, model_kwargs):
    if isinstance(config, Mapping):
        return SVC(
            probability=True,
            max_iter=100000,
            random_state=random_state,
            **config,
            **model_kwargs,
        )

    C, gamma = config
    return SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        probability=True,
        max_iter=100000,
        random_state=random_state,
        **model_kwargs,
    )


def _build_lasso_estimator(config, random_state, model_kwargs):
    if isinstance(config, Mapping):
        return LassoClassifier(
            solver="saga",
            penalty="l1",
            max_iter=100000,
            random_state=random_state,
            **config,
            **model_kwargs,
        )

    return LassoClassifier(
        penalty="l1",
        solver="saga",
        C=config,
        max_iter=100000,
        random_state=random_state,
        **model_kwargs,
    )


def _build_elastic_net_estimator(config, random_state, model_kwargs):
    if isinstance(config, Mapping):
        return ElasticNetClassifier(
            solver="saga",
            penalty="elasticnet",
            max_iter=100000,
            random_state=random_state,
            **config,
            **model_kwargs,
        )

    C, l1_ratio = config
    return ElasticNetClassifier(
        penalty="elasticnet",
        solver="saga",
        C=C,
        l1_ratio=l1_ratio,
        max_iter=100000,
        random_state=random_state,
        **model_kwargs,
    )


def _build_ridge_estimator(config, random_state, model_kwargs):
    if isinstance(config, Mapping):
        return RidgeClassifier(
            solver="lbfgs",
            penalty="l2",
            max_iter=100000,
            random_state=random_state,
            **config,
            **model_kwargs,
        )

    return RidgeClassifier(
        penalty="l2",
        solver="lbfgs",
        C=config,
        max_iter=100000,
        random_state=random_state,
        **model_kwargs,
    )


def _build_linear_estimator(model_kwargs):
    return LinearClassifier(
        penalty=None,
        solver="lbfgs",
        max_iter=100000,
        **model_kwargs,
    )


def _build_decision_tree_estimator(config, random_state, model_kwargs):
    if isinstance(config, Mapping):
        return FullyEnumeratedTreeClassifier(
            random_state=random_state,
            **config,
            **model_kwargs,
        )

    max_depth, min_samples_leaf, criterion = config
    return FullyEnumeratedTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=random_state,
        **model_kwargs,
    )


TRAINER_REGISTRY = {
    LogisticRegression: GridCandidateTrainer(
        configs=[
            (C, penalty)
            for C in np.logspace(-3, 3, 7)
            for penalty in ["l1", "l2", "elasticnet"]
        ],
        random_state_multiplier=1000,
        build_estimator=_build_logistic_estimator,
    ),
    RandomForestClassifier: GridCandidateTrainer(
        configs=[
            (n_estimators, max_depth, max_features)
            for n_estimators in [100, 200]
            for max_depth in [5, 10, None]
            for max_features in ["sqrt", "log2"]
        ],
        random_state_multiplier=3000,
        build_estimator=_build_random_forest_estimator,
    ),
    GradientBoostingClassifier: GridCandidateTrainer(
        configs=[
            (n_estimators, learning_rate, max_depth)
            for n_estimators in [100, 200]
            for learning_rate in [0.05, 0.1, 0.2]
            for max_depth in [2, 3, 4]
        ],
        random_state_multiplier=4000,
        build_estimator=_build_gradient_boosting_estimator,
    ),
    SVC: GridCandidateTrainer(
        configs=[
            (C, gamma)
            for C in [0.1, 1, 10, 100]
            for gamma in [0.001, 0.01, 0.1, 1, 10]
        ],
        random_state_multiplier=5000,
        build_estimator=_build_svc_estimator,
    ),
    LassoClassifier: GridCandidateTrainer(
        configs=list(np.logspace(-4, 2, 9)),
        random_state_multiplier=6000,
        build_estimator=_build_lasso_estimator,
    ),
    ElasticNetClassifier: GridCandidateTrainer(
        configs=[
            (C, l1_ratio)
            for C in np.logspace(-3, 2, 5)
            for l1_ratio in [0.1, 0.5, 0.9]
        ],
        random_state_multiplier=7000,
        build_estimator=_build_elastic_net_estimator,
    ),
    RidgeClassifier: GridCandidateTrainer(
        configs=list(np.logspace(-4, 3, 8)),
        random_state_multiplier=8000,
        build_estimator=_build_ridge_estimator,
    ),
    LinearClassifier: SingleCandidateTrainer(
        build_estimator=_build_linear_estimator,
    ),
    FullyEnumeratedTreeClassifier: GridCandidateTrainer(
        configs=[
            (max_depth, min_samples_leaf, criterion)
            for max_depth in [2, 3, 4, 5, 6]
            for min_samples_leaf in [1, 5, 10, 20]
            for criterion in ["gini", "entropy"]
        ],
        random_state_multiplier=9000,
        build_estimator=_build_decision_tree_estimator,
    ),
}


def register_candidate_trainer(model_class, trainer):
    """Register a custom trainer so RID can sample candidates for a model family."""

    TRAINER_REGISTRY[model_class] = trainer


def resolve_candidate_trainer(model_class, search_grid=None, random_state_multiplier=None):
    """Resolve the trainer responsible for a model family."""

    if model_class not in TRAINER_REGISTRY:
        raise ValueError(f"No candidate trainer registered for {model_class}")

    trainer = TRAINER_REGISTRY[model_class]
    if search_grid is None and random_state_multiplier is None:
        return trainer

    if not isinstance(trainer, GridCandidateTrainer):
        raise ValueError(
            f"Model class {model_class} does not support grid overrides because its trainer "
            "does not use a parameter grid"
        )

    configs = _normalize_search_grid(search_grid)
    if not configs:
        raise ValueError("search_grid must contain at least one candidate configuration")

    return replace(
        trainer,
        configs=configs,
        random_state_multiplier=(
            trainer.random_state_multiplier
            if random_state_multiplier is None
            else random_state_multiplier
        ),
    )


def resolve_model_config(model_config):
    """Resolve a model-config spec into a model class, trainer, and kwargs.

    Supported forms:
    - (model_class_or_trainer, model_kwargs)
    - (model_class_or_trainer, model_kwargs, search_grid)
    - (model_class_or_trainer, model_kwargs, search_grid, random_state_multiplier)
    - {
          "model" | "estimator" | "trainer": ...,
          "kwargs": {...},
          "search_grid": ...,
          "random_state_multiplier": int,
      }
    """

    if isinstance(model_config, Mapping):
        model_or_trainer = None
        for key in ("model", "estimator", "trainer"):
            if key in model_config:
                model_or_trainer = model_config[key]
                break
        if model_or_trainer is None:
            raise ValueError("model_config mapping must define one of: model, estimator, trainer")
        model_kwargs = dict(model_config.get("kwargs", {}))
        search_grid = model_config.get("search_grid")
        random_state_multiplier = model_config.get("random_state_multiplier")
    else:
        if len(model_config) == 2:
            model_or_trainer, model_kwargs = model_config
            search_grid = None
            random_state_multiplier = None
        elif len(model_config) == 3:
            model_or_trainer, model_kwargs, search_grid = model_config
            random_state_multiplier = None
        elif len(model_config) == 4:
            model_or_trainer, model_kwargs, search_grid, random_state_multiplier = model_config
        else:
            raise ValueError(
                "model_config tuples must have 2, 3, or 4 entries: "
                "(model_or_trainer, kwargs[, search_grid[, random_state_multiplier]])"
            )
        model_kwargs = dict(model_kwargs or {})

    if hasattr(model_or_trainer, "train"):
        if search_grid is not None or random_state_multiplier is not None:
            raise ValueError(
                "Custom search_grid and random_state_multiplier overrides are only supported "
                "when model_config references a model class, not an already-instantiated trainer"
            )
        return None, model_or_trainer, model_kwargs

    trainer = resolve_candidate_trainer(
        model_or_trainer,
        search_grid=search_grid,
        random_state_multiplier=random_state_multiplier,
    )
    return model_or_trainer, trainer, model_kwargs


def train_candidate_models(model_trainer, X_boot, y_boot, n_models_pool, b, model_kwargs):
    """Train the candidate model pool for a single bootstrap."""

    return model_trainer.train(X_boot, y_boot, n_models_pool, b, model_kwargs)