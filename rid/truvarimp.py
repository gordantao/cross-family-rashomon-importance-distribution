"""Joint TruVaRImp: CASH-style active-learning Rashomon-set search across model families.

Implements the TruVaRImp algorithm (Ewald, Binder, Bischl, Feurer & Casalicchio,
"CASHomon Sets: Efficient Rashomon Sets Across Multiple Model Classes and their
Hyperparameters", arXiv:2603.15321), generalized across families the way the paper's
own CASH formulation does (Sec. 3.1/4.1): one independent GP surrogate per family
("sub-family" once categorical hyperparameters are split out, see below), joined only
through a single shared implicit threshold h and a single shared evaluation budget
that competes every family's best next candidate against each other.

Design choice -- this module only decides WHICH hyperparameter configuration to
spend a real model fit on next. Every evaluated candidate is actually fit on
X_boot/y_boot and scored with its real log-loss on the bootstrap's out-of-bag
rows (X_eval/y_eval), exactly like the exhaustive search -- so the GP surrogate
and the implicit threshold track the same held-out loss the exact filter uses.
Rashomon-set membership downstream in rid/core.py is still decided from that
exact observed loss (`loss <= global_min_loss + epsilon`), never
from a GP posterior -- TruVaRImp here is an adaptive *search policy* over a much
larger/continuous hyperparameter space, not a replacement for the exact filter.

The candidates it chooses among are the bootstrap's shared pool from
rid.candidate_pool -- the same pool candidate_search="grid" fits exhaustively,
with the same per-candidate random_state -- so any candidate TruVaRImp fits is
bit-for-bit the model the exhaustive search would have fit.

Math note on why "joint" reduces to independent per-family GPs: the paper's CASH
kernel is block-diagonal, k(lambda_m, lambda_m') = delta_mm' k_m(lambda_m, lambda_m'),
i.e. zero cross-family covariance. Evaluating a point in family A does not change
the posterior for family B at all. So the one-step-lookahead variance-reduction
score for a candidate in family A only needs that family's own GP and its own
U/M sets; only the shared scalar threshold (h_opt, h_pes) and the shared epoch
schedule (eta_(i)) couple the families together. We exploit this: each family's
best next candidate is computed independently and cheaply, and only the final
"whose turn is it" competition is global.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

# The search-space specs live in rid.candidate_pool (shared with the exhaustive
# "grid" mode) and are re-exported here so ``rid.truvarimp.<name>`` keeps working.
from .candidate_pool import (
    DEFAULT_CONTINUOUS_SEARCH_SPACES_BY_NAME,
    DEFAULT_N_CANDIDATES_PER_FAMILY,
    Categorical,
    Continuous,
    ContinuousSearchSpace,
    build_candidate_pool,
    drop_pinned_dimensions,
    fit_pool_candidate,
    resolve_default_search_space,
    split_categorical_subspaces,
)

__all__ = [
    "Categorical",
    "Continuous",
    "ContinuousSearchSpace",
    "DEFAULT_CONTINUOUS_SEARCH_SPACES_BY_NAME",
    "DEFAULT_TRUVARIMP_CONFIG",
    "drop_pinned_dimensions",
    "resolve_default_search_space",
    "resolve_truvarimp_config",
    "run_joint_truvarimp",
    "split_categorical_subspaces",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-(sub)family active-search state
# ---------------------------------------------------------------------------


# Minimum number of observations before a converged kernel is reused as the
# starting point for the next MLE (see _refit_gp).
_WARM_START_MIN_OBSERVATIONS = 8


class _FamilySearchState:
    """TruVaRImp state (GP + L/H/U/M sets) for one family or categorical sub-family."""

    def __init__(
        self,
        name,
        build_estimator,
        continuous_names,
        log_flags,
        categorical_fixed,
        candidates,
        random_states,
    ):
        self.name = name
        self.build_estimator = build_estimator
        self.continuous_names = continuous_names
        self.categorical_fixed = categorical_fixed
        self.candidate_configs = candidates  # list[dict] -- full kwargs dict per candidate
        # Fixed per candidate by the shared pool (rid.candidate_pool.pool_random_state),
        # so a candidate fit here is identical to the same candidate fit by "grid".
        self.candidate_random_states = random_states
        self.n_candidates = len(candidates)

        if continuous_names:
            X_raw = np.array([[cfg[name] for name in continuous_names] for cfg in candidates])
            log_mask = np.array([log_flags[name] for name in continuous_names])
            self._X_gp = np.where(log_mask, np.log(np.maximum(X_raw, 1e-300)), X_raw)
        else:
            self._X_gp = np.zeros((self.n_candidates, 1))

        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
            length_scale=np.ones(self._X_gp.shape[1]), nu=2.5
        ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
        self.gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=0, random_state=0
        )

        self.evaluated_idx = []
        self._evaluated_set = set()  # O(1) membership for the acquisition filter
        self.evaluated_y = []
        self.models = []
        self.losses = []

        self.mu = np.zeros(self.n_candidates)
        # Uninformed prior: huge variance until the GP has at least one observation.
        self.sigma2 = np.full(self.n_candidates, 1e6)
        self._k_star_star_all = None
        self._K_post_full = None

        self.U = set(range(self.n_candidates))
        self.M = set(range(self.n_candidates))
        self.L = set()
        self.H = set()
        self._all_idx = np.arange(self.n_candidates)

    def _refit_gp(self):
        X_obs = self._X_gp[self.evaluated_idx]
        y_obs = np.array(self.evaluated_y)
        if len(set(map(tuple, X_obs.tolist()))) < 2:
            # GP hyperparameter MLE is ill-posed with <2 distinct points; predict
            # the prior mean with high uncertainty everywhere else instead of
            # erroring out on the degenerate single-point fit.
            self.mu = np.full(self.n_candidates, float(np.mean(y_obs)))
            self.sigma2 = np.full(self.n_candidates, 1e6)
            self.sigma2[self.evaluated_idx] = 1e-6
            self._k_star_star_all = None
            return
        self.gp.fit(X_obs, y_obs)
        # Warm-start the NEXT refit's hyperparameter search from this one's
        # converged kernel instead of the original prior every time -- the
        # optimum barely moves when one point is added to a few dozen, so
        # this saves most of the L-BFGS iterations on every subsequent fit.
        # Only once there are enough observations for the MLE to be meaningful:
        # with n_restarts_optimizer=0 every later fit starts from wherever the
        # previous one landed, and a 2-or-3-point MLE routinely pins
        # length_scale/noise_level to a bound, which would then persist for the
        # rest of the bootstrap.
        if len(y_obs) >= _WARM_START_MIN_OBSERVATIONS:
            self.gp.kernel = self.gp.kernel_
        self.mu, sigma = self.gp.predict(self._X_gp, return_std=True)
        self.sigma2 = sigma**2
        self._precompute_lookahead_tables(X_obs)

    def _precompute_lookahead_tables(self, X_obs):
        """Precompute, ONCE per GP refit, everything every candidate's
        one-step-lookahead variance reduction needs -- avoids re-factorizing
        the (small but O(n_obs^3)) observation covariance matrix separately
        for every candidate considered during acquisition (Bogunovic et al.
        2016, Appendix B: the reduction a hypothetical new observation at
        lambda gives at every other point lambda' can be written purely in
        terms of the current kernel and the already-observed points).

        Two scale details matter here, because both tables are consumed
        alongside ``self.sigma2`` in :meth:`best_candidate_and_score`:

        1. The tables must hold *posterior* covariance, not prior covariance.
           The look-ahead reduction at lambda' from observing lambda is
           ``k_post(lambda', lambda)^2 / k_post(lambda, lambda)``; using the
           prior ``k(lambda', lambda)`` instead would claim a reduction even
           where the current observations have already removed the covariance,
           and would not respect ``reduction <= sigma2(lambda')``.
        2. ``self.gp`` is fit with ``normalize_y=True``, so the posterior
           variance sklearn returns from ``predict`` is rescaled by
           ``_y_train_std ** 2`` while ``kernel_`` still lives in normalized-y
           units. Both tables are therefore rescaled by the same factor, so
           that ``sigma2 - K_post**2 / k_star_star`` subtracts like-for-like.
           (Without this, with log-loss spreads well below 1, the kernel-scale
           ratio exceeds the loss-scale variance by orders of magnitude, every
           look-ahead variance clamps to zero, the acquisition score becomes
           constant across candidates, and argmax degenerates to "first
           candidate".)
        """

        kernel = self.gp.kernel_
        K_obs = kernel(X_obs) + np.eye(X_obs.shape[0]) * 1e-10
        try:
            K_inv = np.linalg.inv(K_obs)
        except np.linalg.LinAlgError:
            K_inv = np.linalg.pinv(K_obs)

        y_scale = float(np.asarray(getattr(self.gp, "_y_train_std", 1.0)).ravel()[0])
        y_var_scale = y_scale**2

        K_cross_all = kernel(X_obs, self._X_gp)  # (n_obs, n_candidates)
        K_inv_cross = K_inv @ K_cross_all  # (n_obs, n_candidates)
        quad = np.sum(K_cross_all * K_inv_cross, axis=0)  # (n_candidates,)
        diag_all = kernel.diag(self._X_gp)
        # k_star_star_all[i] = posterior predictive variance AT i given only the
        # current observations -- i.e. the variance of the hypothetical new
        # observation if i were evaluated next.
        self._k_star_star_all = np.maximum(diag_all - quad, 1e-12) * y_var_scale
        # Posterior covariance between every pair of candidates, in the same
        # y units as self.sigma2 (see the docstring's point 2).
        self._K_post_full = (kernel(self._X_gp) - K_cross_all.T @ K_inv_cross) * y_var_scale

    def evaluate(self, idx, X_boot, y_boot, X_eval, y_eval, refit=True):
        model, loss = fit_pool_candidate(
            self.build_estimator,
            self.candidate_configs[idx],
            self.candidate_random_states[idx],
            X_boot,
            y_boot,
            X_eval,
            y_eval,
        )

        self.evaluated_idx.append(idx)
        self._evaluated_set.add(idx)
        self.evaluated_y.append(loss)
        self.models.append(model)
        self.losses.append(loss)
        self.U.discard(idx)
        # ``refit=False`` lets a caller batch several evaluations (e.g. the
        # random warm-up draws, none of which steer the next pick) and refit
        # the GP once at the end instead of once per fit -- each refit costs an
        # MLE plus an O(n_candidates^2) kernel build.
        if refit:
            self._refit_gp()
        return loss

    def confidence_interval(self, beta_sqrt):
        half_width = beta_sqrt * np.sqrt(self.sigma2)
        return self.mu - half_width, self.mu + half_width

    def update_partition(self, h_opt, h_pes, c_min_pes, beta_sqrt):
        """Algorithm 1, lines 10-24, restricted to this family's own candidates.

        Per the paper's own Appendix E.1 implementation note: since the GP
        kernel is re-fit by MLE every iteration, sigma_t is non-monotonic in
        t, so L/H/U/M are recomputed from the FULL candidate set each call
        (allowing points to re-enter U or M) rather than only ever shrinking
        from the previous iteration's sets.
        """

        l_t, u_t = self.confidence_interval(beta_sqrt)

        self.L = set(self._all_idx[u_t <= h_opt].tolist())
        self.H = set(self._all_idx[l_t > h_pes].tolist())
        self.U = set(self._all_idx.tolist()) - self.L - self.H
        self.M = set(self._all_idx[l_t <= c_min_pes].tolist())

    def epoch_satisfied(self, beta_sqrt, eta, eps_rel, delta_bar):
        if not self.U and not self.M:
            return True
        sigma_scaled = beta_sqrt * np.sqrt(self.sigma2)
        u_ok = all(sigma_scaled[idx] <= (1 + delta_bar) * eta for idx in self.U)
        m_ok = all(
            sigma_scaled[idx] <= (1 + delta_bar) / (1 + eps_rel) * eta for idx in self.M
        )
        return u_ok and m_ok

    def best_candidate_and_score(self, beta_sqrt, eta, eps_rel):
        """Local (within-family) argmax of the Algorithm 1 acquisition function.

        Because the joint kernel is block-diagonal, a candidate's score only
        depends on how much evaluating it would reduce variance for OTHER
        candidates *in this same family* -- points in other families are
        provably unaffected, so this local optimum is also this family's
        contribution to the global joint argmax (see module docstring).

        Fully vectorized over every unclassified candidate at once using the
        lookahead tables :meth:`_precompute_lookahead_tables` built at the
        last GP refit, instead of looping candidate-by-candidate -- the whole
        (n_candidates x (|U|+|M|)) score surface is a handful of array ops.
        """

        unclassified = sorted((self.U | self.M) - self._evaluated_set)
        if not unclassified:
            return None, -np.inf
        if self._k_star_star_all is None:
            return unclassified[0], 0.0

        cand_arr = np.array(unclassified, dtype=int)
        U_arr = np.array(sorted(self.U), dtype=int)
        M_arr = np.array(sorted(self.M), dtype=int)
        k_ss = self._k_star_star_all[cand_arr]  # (n_cand,)

        def delta_before(idx_arr, power):
            if idx_arr.size == 0:
                return 0.0
            excess = (power**2) * (beta_sqrt**2) * self.sigma2[idx_arr] - eta**2
            return float(np.sum(np.maximum(excess, 0.0)))

        def delta_after(idx_arr, power):
            if idx_arr.size == 0:
                return np.zeros(len(cand_arr))
            # Posterior cross-covariance, already in self.sigma2's units -- see
            # _precompute_lookahead_tables. The clamp at 0 is numerical safety
            # only: Cauchy-Schwarz guarantees K_post**2 / k_star_star <=
            # sigma2, so a correctly scaled look-ahead variance is >= 0.
            K_cross = self._K_post_full[np.ix_(cand_arr, idx_arr)]  # (n_cand, n_idx)
            sigma2_lookahead = np.maximum(
                self.sigma2[idx_arr][None, :] - (K_cross**2) / k_ss[:, None], 0.0
            )
            excess = (power**2) * (beta_sqrt**2) * sigma2_lookahead - eta**2
            return np.sum(np.maximum(excess, 0.0), axis=1)

        scores = (
            (delta_before(U_arr, 1.0) - delta_after(U_arr, 1.0))
            + (delta_before(M_arr, 1 + eps_rel) - delta_after(M_arr, 1 + eps_rel))
        )

        best_pos = int(np.argmax(scores))
        return int(cand_arr[best_pos]), float(scores[best_pos])


# ---------------------------------------------------------------------------
# Joint orchestrator
# ---------------------------------------------------------------------------


DEFAULT_TRUVARIMP_CONFIG = {
    "total_budget": 300,
    # Size of the shared candidate pool each family draws per bootstrap (split
    # evenly across its categorical sub-families; see rid.candidate_pool).
    # compute_rid_cross_family sets this from its own n_candidates_per_family.
    "n_candidates_per_family": DEFAULT_N_CANDIDATES_PER_FAMILY,
    "beta_sqrt": 3.0,
    "eta_init": 1.0,
    "r": 0.1,
    "delta_bar": 0.0,
    "n_warmup_per_subfamily": 10,
    # Share of total_budget held back from the active loop and spent at the end
    # fitting the configurations the search concluded are *inside* the Rashomon
    # threshold (set L), best-predicted first. Without this the returned pool
    # would contain only configurations the policy was uncertain about -- the
    # acquisition deliberately avoids spending fits where it is already
    # confident, which is exactly where the Rashomon members are.
    "harvest_fraction": 0.2,
}

# Renamed keys: (old name, message). Passing one is an error, not a silent no-op.
_RENAMED_TRUVARIMP_KEYS = {
    "n_candidates_per_subfamily": (
        "'n_candidates_per_subfamily' was replaced by 'n_candidates_per_family' (a per-family "
        "pool size split evenly across categorical sub-families); pass n_candidates_per_family "
        "to compute_rid_cross_family / CrossFamilyRashomonImportanceDistribution instead."
    ),
}


def resolve_truvarimp_config(config=None):
    """Merge ``config`` over ``DEFAULT_TRUVARIMP_CONFIG``, rejecting renamed/unknown keys."""

    config = dict(config or {})
    for key, message in _RENAMED_TRUVARIMP_KEYS.items():
        if key in config:
            raise ValueError(message)
    unknown = sorted(set(config) - set(DEFAULT_TRUVARIMP_CONFIG))
    if unknown:
        raise ValueError(
            f"Unknown truvarimp_config keys {unknown}; supported: {sorted(DEFAULT_TRUVARIMP_CONFIG)}"
        )
    return {**DEFAULT_TRUVARIMP_CONFIG, **config}


def run_joint_truvarimp(
    resolved_model_configs,
    search_spaces,
    X_boot,
    y_boot,
    X_eval,
    y_eval,
    b,
    epsilon,
    config=None,
    return_stats=False,
    candidate_pool=None,
):
    """Run joint (CASH-style) TruVaRImp over all families for one bootstrap.

    Parameters
    ----------
    resolved_model_configs : dict[str, (trainer, model_kwargs)]
        Same shape ``compute_rid_cross_family`` already builds from its
        ``model_configs`` argument.
    search_spaces : dict[str, ContinuousSearchSpace]
        Continuous search space per family name. Families missing an entry
        (or whose trainer is not a ``GridCandidateTrainer``, e.g. a
        ``SingleCandidateTrainer`` or a hand-written custom trainer) fall back
        to a single fixed evaluation using that family's own ``train`` method,
        so every family still contributes at least one real observation to the
        shared threshold. Dimensions the family already pins through its model
        kwargs are dropped from its space rather than searched over.
    X_boot, y_boot : ndarray
        The bootstrap resample to fit candidates on.
    X_eval, y_eval : ndarray
        The bootstrap's out-of-bag rows, which every candidate's loss is scored on.
    b : int
        Bootstrap index, used to seed the pool draws and the warm-up picks.
    epsilon : float
        Rashomon tolerance, interpreted exactly as elsewhere in rid/core.py:
        an absolute additive tolerance on log-loss (``eps_rel=0, eps_abs=epsilon``
        in the paper's ``h = c_min*(1+eps_rel) + eps_abs`` notation).
    config : dict, optional
        Overrides for ``DEFAULT_TRUVARIMP_CONFIG``. ``total_budget`` bounds the
        TOTAL number of real model fits across all families (warm-up, active
        search, and harvest alike); ``harvest_fraction`` is the share of it
        reserved for the closing harvest phase; ``n_candidates_per_family``
        sizes the candidate pool when ``candidate_pool`` is not given.
    return_stats : bool, default=False
        Also return a dict of search diagnostics: real fits spent in each phase
        (``n_warmup``, ``n_active``, ``n_harvest``, ``n_fallback``), how many
        times the acquisition tolerance eta was tightened (``n_eta_shrinks``),
        and why the active loop ended (``stop_reason``: ``"budget"``,
        ``"exhausted"``, ``"no_threshold"`` or ``"no_candidate"``), plus
        ``pinned``: the search-space dimensions each family fixes through its
        model kwargs (``{family_name: [dimension, ...]}``), and
        ``evaluated_candidate_indices``: each family's pool ``candidate_index``
        per evaluated candidate, aligned with ``family_models``/``family_losses``.
    candidate_pool : dict, optional
        This bootstrap's shared pool from
        :func:`rid.candidate_pool.build_candidate_pool` (what
        ``compute_rid_cross_family`` passes, so "grid" and TruVaRImp search the
        identical candidates). Built here from ``config`` when omitted.

    Returns
    -------
    family_models : dict[str, list[estimator]]
    family_losses : dict[str, np.ndarray]
        Every candidate actually evaluated (fit + scored) during the search,
        keyed back by the ORIGINAL family name (categorical sub-families are
        merged back together) -- same shape ``_run_single_bootstrap_cross_family``
        already expects from the grid-based path.
    stats : dict
        Only when ``return_stats=True``; see above.
    """

    cfg = resolve_truvarimp_config(config)
    if candidate_pool is None:
        candidate_pool = build_candidate_pool(
            resolved_model_configs, search_spaces, b, cfg["n_candidates_per_family"]
        )

    # Only picks warm-up candidates: the pool itself is drawn from its own
    # per-family streams, so it doesn't depend on this generator.
    rng = np.random.default_rng(seed=b)
    eps_rel = 0.0  # matches this codebase's existing absolute-epsilon convention
    eta = cfg["eta_init"]
    beta_sqrt = cfg["beta_sqrt"]
    delta_bar = cfg["delta_bar"]

    states = []
    family_of_state = {}
    pinned_dims = {}
    for family_name, (trainer, model_kwargs) in resolved_model_configs.items():
        family_pool = candidate_pool.get(family_name)
        if family_pool is not None:
            if family_pool.pinned:
                pinned_dims[family_name] = list(family_pool.pinned)
            index_offset = 0
            for sub in family_pool.subfamilies:
                state = _FamilySearchState(
                    name=sub.name,
                    build_estimator=family_pool.build_estimator,
                    continuous_names=sub.continuous_names,
                    log_flags=sub.log_flags,
                    categorical_fixed=sub.fixed_values,
                    candidates=sub.configs,
                    random_states=sub.random_states,
                )
                # Pool candidate_index runs across the family's sub-families in
                # order, so a sub-family's local index plus this offset recovers it.
                state.index_offset = index_offset
                index_offset += len(sub.configs)
                states.append(state)
                family_of_state[sub.name] = family_name
        else:
            # Not in the pool (e.g. a SingleCandidateTrainer, or a custom
            # trainer with no known continuous space): fall back to its own
            # .train() once, and fold the result straight into the output
            # rather than active-searching it.
            models, losses = trainer.train(X_boot, y_boot, X_eval, y_eval, 1, b, model_kwargs)
            state = _FallbackFamilyState(family_name, models, losses)
            states.append(state)
            family_of_state[family_name] = family_name

    searchable = [s for s in states if not isinstance(s, _FallbackFamilyState)]
    budget = max(int(cfg["total_budget"]), 0)
    harvest_budget = min(int(round(budget * float(cfg["harvest_fraction"]))), budget)
    search_budget = budget - harvest_budget
    total_evaluations = 0
    n_eta_shrinks = 0
    stop_reason = "budget"

    # --- warm-start: a handful of random evaluations per searchable sub-family ---
    # Clamped against the remaining budget: without the clamp, enough families
    # (or categorical splits) would push the real fit count past total_budget
    # before the active loop ever ran.
    for state in searchable:
        remaining = search_budget - total_evaluations
        if remaining <= 0:
            break
        n_warmup = min(cfg["n_warmup_per_subfamily"], state.n_candidates, remaining)
        warm_idx = rng.choice(state.n_candidates, size=n_warmup, replace=False)
        for position, idx in enumerate(warm_idx):
            # Random draws don't steer each other, so refit the GP once at the
            # end of the batch instead of after every fit.
            state.evaluate(
                int(idx),
                X_boot,
                y_boot,
                X_eval,
                y_eval,
                refit=(position == n_warmup - 1),
            )
        total_evaluations += n_warmup
    n_warmup_total = total_evaluations

    # --- active-learning loop (Algorithm 1, generalized across sub-families) ---
    while total_evaluations < search_budget:
        if not any(s.U or s.M for s in searchable):
            stop_reason = "exhausted"
            break
        thresholds = _shared_thresholds(searchable, beta_sqrt, eps_rel, epsilon)
        if thresholds is None:
            stop_reason = "no_threshold"
            break
        c_min_pes, h_pes, h_opt = thresholds

        for state in searchable:
            state.update_partition(h_opt, h_pes, c_min_pes, beta_sqrt)

        if all(state.epoch_satisfied(beta_sqrt, eta, eps_rel, delta_bar) for state in searchable):
            eta *= cfg["r"]
            n_eta_shrinks += 1
            continue

        best_state, best_idx, best_score = None, None, -np.inf
        for state in searchable:
            idx, score = state.best_candidate_and_score(beta_sqrt, eta, eps_rel)
            if idx is not None and score > best_score:
                best_state, best_idx, best_score = state, idx, score

        if best_state is None:
            stop_reason = "no_candidate"
            break

        best_state.evaluate(best_idx, X_boot, y_boot, X_eval, y_eval)
        total_evaluations += 1

    # --- harvest: actually fit the configurations the search believes are IN
    # the Rashomon set. The acquisition above spends its budget where the GP is
    # uncertain, i.e. near the threshold, so without this step set L is pure
    # bookkeeping and the returned pool is a boundary-biased sample that omits
    # the confidently-good configurations the Rashomon filter downstream exists
    # to collect.
    n_harvest = _harvest_predicted_rashomon(
        searchable, X_boot, y_boot, X_eval, y_eval, beta_sqrt, eps_rel, epsilon, harvest_budget
    )

    # --- collapse sub-family states back into the caller's family keys ---
    family_models = {name: [] for name in resolved_model_configs}
    family_losses = {name: [] for name in resolved_model_configs}
    family_indices = {name: [] for name in resolved_model_configs}
    for state in states:
        parent = family_of_state[state.name] if state.name in family_of_state else state.name
        family_models[parent].extend(state.models)
        family_losses[parent].extend(state.losses)
        offset = getattr(state, "index_offset", 0)
        family_indices[parent].extend(offset + int(idx) for idx in state.evaluated_idx)

    family_losses = {name: np.array(values) for name, values in family_losses.items()}
    if not return_stats:
        for family_name, dims in pinned_dims.items():
            logger.debug(
                f"Note: family '{family_name}' pins {dims} via its model kwargs; "
                "those dimensions are held fixed rather than searched."
            )
        return family_models, family_losses

    stats = {
        "n_warmup": n_warmup_total,
        "n_active": total_evaluations - n_warmup_total,
        "n_harvest": n_harvest,
        "n_fallback": sum(
            len(state.losses) for state in states if isinstance(state, _FallbackFamilyState)
        ),
        "n_eta_shrinks": n_eta_shrinks,
        "stop_reason": stop_reason,
        # Reported back rather than logged here: under n_jobs > 1 this runs in a
        # joblib worker with no logging handler, so a log call would be dropped.
        "pinned": pinned_dims,
        # Pool candidate_index of every evaluated candidate, aligned with
        # family_models/family_losses (for a fallback family: its own fit order).
        "evaluated_candidate_indices": family_indices,
    }
    return family_models, family_losses, stats


def _shared_thresholds(searchable, beta_sqrt, eps_rel, epsilon):
    """Compute the thresholds that couple the families: (c_min_pes, h_pes, h_opt).

    Returns ``None`` when no family has a non-empty potential-minimizer set M,
    i.e. there is nothing left to anchor the shared threshold to.
    """

    with_m = [state for state in searchable if state.M]
    if not with_m:
        return None

    c_min_pes = min(
        min(state.mu[i] + beta_sqrt * np.sqrt(state.sigma2[i]) for i in state.M)
        for state in with_m
    )
    c_min_opt = min(
        min(state.mu[i] - beta_sqrt * np.sqrt(state.sigma2[i]) for i in state.M)
        for state in with_m
    )
    h_pes = c_min_pes * (1 + eps_rel) + epsilon
    h_opt = c_min_opt * (1 + eps_rel) + epsilon
    return c_min_pes, h_pes, h_opt


def _harvest_predicted_rashomon(
    searchable, X_boot, y_boot, X_eval, y_eval, beta_sqrt, eps_rel, epsilon, harvest_budget
):
    """Spend the reserved budget fitting the best *predicted* configurations.

    Prefers candidates the final partition placed in L (confidently within the
    Rashomon threshold); when L is empty everywhere -- the common case, since a
    pooled threshold set by the strongest family classifies most candidates
    into H -- falls back to the globally lowest predicted loss, which is the
    same thing the exact filter in ``rid/core.py`` is looking for. These fits
    are terminal, so the GP is not refit afterwards.
    """

    if harvest_budget <= 0 or not searchable:
        return 0

    thresholds = _shared_thresholds(searchable, beta_sqrt, eps_rel, epsilon)
    if thresholds is not None:
        c_min_pes, h_pes, h_opt = thresholds
        for state in searchable:
            state.update_partition(h_opt, h_pes, c_min_pes, beta_sqrt)

    preferred = []
    fallback = []
    for state in searchable:
        unevaluated = set(range(state.n_candidates)) - state._evaluated_set
        for idx in sorted(unevaluated):
            entry = (float(state.mu[idx]), state.name, idx, state)
            if idx in state.L:
                preferred.append(entry)
            else:
                fallback.append(entry)

    queue = preferred if preferred else fallback
    queue.sort(key=lambda entry: (entry[0], entry[1], entry[2]))

    evaluated = 0
    for _, _, idx, state in queue[:harvest_budget]:
        state.evaluate(idx, X_boot, y_boot, X_eval, y_eval, refit=False)
        evaluated += 1
    return evaluated


class _FallbackFamilyState:
    """Stand-in for a family with no continuous search space: one fixed evaluation."""

    def __init__(self, name, models, losses):
        self.name = name
        self.models = list(models)
        self.losses = list(losses)
        self.evaluated_idx = list(range(len(models)))
        self.U = set()
        self.M = set()
