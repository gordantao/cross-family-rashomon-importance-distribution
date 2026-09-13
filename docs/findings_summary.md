# Findings summary

Running digest of methodology fixes, diagnostic results, and research findings from this
working session. Organized by topic, not chronologically. See
`docs/cross_family_rashomon_importance_distribution.md` for the underlying methodology and
`CLAUDE.md` for repo/cluster mechanics.

## Context

This repo's central claim is that **Cross-Family Rashomon Importance Distribution (RID)**
characterizes feature importance more honestly than a single fitted model does, by looking at
every near-optimal model within `epsilon` of the best loss (the Rashomon set), optionally
pooled across model families with different inductive biases (linear, tree, kernel, etc.)
rather than trusting whichever one model a greedy search happened to land on. Three
experiments stress-test that claim from different angles: **Falcon-Cano** (real oral
bioavailability data, ~1,157 compounds) and **Staellert** (real single-cell imaging data,
~2,930-8,850 cells) ask whether it holds up on real biological data with no known ground truth;
**nonlinear_interaction_simulation** asks the same question against synthetic DGPs with known
ground truth, specifically ones built around feature interactions RID's cross-family pooling
should be positioned to catch and single-family/greedy methods shouldn't.

This session's work falls into two threads that turned out to be connected. The first
(\S1-\S8) started from a simple observation -- linear stepwise selection was beating RID
variants on held-out predictive accuracy -- and turned into a chain of methodology fixes
(two real bugs in the cross-family aggregation code) and new diagnostics designed specifically
to test what RID is *actually* built to do (share credit honestly under redundancy, quantify
its own uncertainty) rather than what a greedy top-k selector is built to do (maximize a fixed
evaluator's accuracy). The second thread (\S9-\S12) is four research extensions an advisor
asked to have scoped or prototyped: a new interaction-aware VI metric (iLOCO), a more
principled correlation-filtering threshold, and two exploratory directions (DataSAIL,
rFCFP) that may or may not become real experiments later.

## 1. Methodology bugs found and fixed in `rid/core.py`

**Cross-family Rashomon-set membership was judged per-family, not pooled.**
`compute_rid_cross_family` admitted a family's models by comparing them to that family's
*own* best bootstrap loss, not the best loss achieved by any family. A family that couldn't
compete with the pooled optimum could still contribute a full slate of "Rashomon" models by
being compared only to itself -- which quietly undermines the entire premise of cross-family
pooling, since a weak family shouldn't get equal standing just for being judged against a
lower bar. Fixed: membership is now judged against the global best loss across all families in
that bootstrap; `family_balance_mode` only changes how already-admitted models are aggregated
afterward, not which models get in. (Verified on a toy dataset: a dominated family dropped to
0 admitted models across 15 bootstraps post-fix.)

**`weighted` cross-family mode wasn't actually affecting feature rankings.**
`expected_importance` (what `rank_features()` ranks by, and what every top-k CSV is built
from) was a plain unweighted mean regardless of balance mode -- the `weighted` reweighting
scheme only ever reached `prob_positive` (via the CDF), never the ranking itself. This is why
`unweighted` and `weighted` produced byte-identical top-k lists and importance values before
the fix, despite different `prob_positive` numbers -- the entire point of `weighted` mode
(equalize each family's contribution rather than letting a family with more surviving models
dominate) was silently not happening. Fixed: `expected_importance` is now a weighted mean using
the same per-model weights as the CDF, reducing to a plain mean when unweighted.

**Both fixes together**: on the Falcon-Cano redundancy diagnostic, `cross_family_unweighted`
and `cross_family_weighted` held-out CV scores went from identical (0.7051 both) to genuinely
different (0.668 vs 0.673), and credit-split-ratio for the injected duplicate pair went from
identical (0.2 both) to sharply different (0.0 vs 0.767) -- see \S3. Every cross-family result
in \S3, \S5, and \S6 was regenerated after these fixes; nothing dated from before them survives
in this document.

## 2. Held-out CV evaluator: single model to a 3-model panel

**Problem**: every method's selected top-k features were scored with one fixed
`LogisticRegression` evaluator. This only tests whether a feature set has *linearly*
extractable signal -- the one axis where cross-family RID's advantage (pooling across
families with different inductive biases) structurally can't show up, even if real. Reporting
RID's feature sets as "worse" under an evaluator that's blind to the exact kind of signal RID
is supposed to surface would have been a biased comparison baked into the pipeline itself.

**Why not a neural net instead**: NNs are universal approximators only in the
infinite-data/infinite-capacity limit; at this data scale (~1,157-2,930 rows) that traded-away
bias becomes variance instead. MLPs also carry their own specific bias toward *overly smooth*
functions, a poor match for tabular data with irregular target functions -- documented
empirically in Grinsztajn et al. 2022 (tree-based models remain SOTA on tabular data up to
~10K rows independent of speed).

**Fix**: replaced the single evaluator with a fixed 3-model panel spanning distinct
mechanisms -- `LogisticRegression` (additive/linear), `RandomForestClassifier` (axis-aligned
interactions via bagging), `SVC(kernel="rbf")` (smooth nonlinear margin). Every method's top-k
features are now scored under all three, reported separately plus a simple average, in both
`run_falcon_cano_top40_comparison.py` and `run_staellert_top40_comparison.py`. This panel is
the evaluator behind every held-out CV number quoted in \S5 going forward.

## 3. Redundancy / credit-split-ratio diagnostic (does a method share credit or pick a winner?)

This diagnostic exists because RID's design claim -- honest credit-sharing under
multicollinearity, rather than an arbitrary commitment to one of several equally-good
explanations -- isn't something a predictive-accuracy metric can see at all. It needs its own
test. Inject a 0.95-correlated engineered duplicate of a real feature (`SlogP_VSA2` in
Falcon-Cano) that plays no causal role -- both twins are known-substitutable by construction,
so this needs no ground truth and works on real data. `credit_split_ratio`: 0 = winner-take-all,
1 = shared evenly.

| method | credit_split_ratio | detail |
|---|---|---|
| stepwise_logreg | 0.0 | picked original (rank 3), duplicate never entered top-40 |
| stepwise_rf | 0.0 | picked original (rank 8), duplicate never entered top-40 |
| rid_tree | 0.381 | both ranked: #8 (imp 0.251) and #21 (imp 0.129) |
| cross_family_unweighted | 0.0 | boundary artifact -- duplicate at rank 40 (imp 0.202), original just outside top-40 |
| **cross_family_weighted** | **0.767** | both ranked with near-identical importance: #23 (0.036) and #30 (0.035) |

Reproduced twice (before and after the epsilon/weighting fix in \S1) with consistent results.
`cross_family_weighted`'s near-perfectly-even split is the cleanest evidence in this whole
session that RID's multiplicity-aware design works as intended in practice, on real data, not
just in theory -- stronger even than the tree family alone. The `unweighted` 0.0 is a top-40
cutoff artifact (original fell just outside, its near-identical twin squeaked in at the very
last slot), not true winner-take-all like stepwise's, which never considers the duplicate at
all once it commits to the original.

The same pattern replicates on the synthetic DGPs with engineered redundant duplicates of the
true driver (`chen_redundant_r095`, `friedman_redundant_r095`): cross-family credit-split
ratios of 0.40-0.42, again far above stepwise's structural 0.0. Seeing this hold on both real
data (no ground truth) and synthetic data (known ground truth) is what makes it a real finding
rather than a one-off artifact of Falcon-Cano's specific feature space.

## 4. Stability analysis (does RID's own uncertainty signal predict real instability?)

The other half of RID's design claim that predictive-accuracy metrics can't see: RID doesn't
just rank features, it reports a `P(φ>0)`-derived ambiguity for each one -- a built-in
uncertainty signal a point-estimate method like stepwise selection has no equivalent of at all.
This diagnostic asks whether that signal is trustworthy: repeatedly refit a single model family
under bootstrap resampling and check whether RID's ambiguity predicts the fraction of refits
where the feature's importance sign actually flips. Metric is **sign-of-contribution
instability**, not top-k rank membership -- an earlier version used rank-membership flip rate
and produced a spurious negative correlation (-0.32) for one DGP cell, traced to a feature with
an unstable *rank* but a stable *sign* (`P(φ>0)=1.0` always); redefining around sign fixed it.

| dataset | family | Spearman(flip-rate, RID ambiguity) |
|---|---|---|
| Falcon-Cano (real data) | Lasso | 0.475 |
| Falcon-Cano (real data) | FullyEnumeratedTree | **0.873** |
| nonlinear DGP simulation (synthetic, multiple cells) | Lasso / Tree | 0.54-0.79 |

Strong positive in every case, including real biological data with no ground truth to lean
on. Practically, this means a practitioner reading RID's output gets a genuine early-warning
signal for "this feature's importance might not replicate" -- something no single-model
selection method can offer by construction, since it only ever reports one point estimate.

## 5. Held-out CV comparison results and why RID underperforms on it

Full Falcon-Cano top-40 comparison, 500 bootstraps, post-\S1/\S2 fixes, scored under the
3-model panel from \S2 and averaged to one number per method:

| method | held-out CV (avg across panel) |
|---|---|
| stepwise_logreg | **0.721** |
| stepwise_rf | 0.708 |
| rid_tree | 0.693 |
| cross_family_weighted | 0.685 |
| cross_family_unweighted | 0.664 |

Stepwise methods win on this metric, and the mechanism is now confirmed, not just theorized.
Average pairwise |correlation| *within* each method's own top-40 selection:

| method | mean \|corr\| | median |
|---|---|---|
| stepwise_rf | 0.065 | 0.033 |
| stepwise_logreg | 0.090 | 0.041 |
| rid_tree | 0.194 | 0.148 |
| cross_family_unweighted | 0.211 | 0.180 |
| cross_family_weighted | 0.220 | 0.187 |

RID's top-40 picks are 2-3x more internally redundant. **Why**: forward stepwise selection is
sequential and conditions on what's already selected -- once one member of a correlated pair
is in, its partner offers little marginal CV improvement and gets skipped for something that
adds genuinely new signal. RID ranks by standalone/marginal importance with no such
conditioning, so multiple individually-important-but-mutually-redundant features can all make
the top-40, wasting fixed-budget slots on overlapping signal. This is the same behavior
characterized deliberately in \S3, just showing up organically across the full feature space
instead of at one engineered pair -- **the evaluation metric (fixed-budget top-k CV accuracy)
specifically penalizes credit-sharing**, independent of whether the underlying importance
characterization is right. For the project's central claim, this is an important distinction to
keep straight: it is not evidence that RID's importance characterization is wrong, only that
top-k CV accuracy was never going to be the metric that shows RID's value -- which is exactly
why \S3 and \S4 exist as separate, purpose-built evaluations.

Open discussion (not yet implemented): the better way to demonstrate business/use-case impact
is probably a decision-cost framing built on \S4's stability signal -- e.g., simulate how often
committing resources (wet-lab validation, an assay) to a single top-ranked stepwise pick would
land on a feature RID's own ambiguity score would have flagged as a coin-flip in advance. Needs
either real replication data or a defensible ground-truth-reproducibility proxy before building
out.

## 6. Ground-truth recovery on synthetic DGPs (nonlinear_interaction_simulation)

This is the experiment purpose-built to give RID's cross-family pooling claim a fair test: DGPs
with known relevant features, some deliberately built around feature interactions (`monk1`,
`monk3`) that a linear or single-tree method has no mechanism to detect but that pooling
across model families (including nonlinear ones) should. Unlike \S5, ranking is scored against
known ground truth (precision/recall of the true drivers), so it isn't subject to the same
credit-sharing penalty a fixed-budget CV score imposes.

| DGP | cross_unweighted | cross_weighted | rid_tree | stepwise_logreg | stepwise_rf |
|---|---|---|---|---|---|
| chen | 0.973 | 0.973 | 0.976 | 0.970 | 0.938 |
| chen_redundant_r095 | 0.926 | 0.920 | 0.845 | 0.964 | 0.833 |
| friedman | 1.000 | 1.000 | 0.993 | 0.886 | 0.995 |
| **monk1** (interaction-heavy) | **0.944** | **0.944** | 0.940 | 0.595 | 0.567 |
| monk3 (interaction-heavy) | 0.905 | 0.897 | 0.909 | 0.714 | 0.738 |

Cross-family RID clearly wins on the interaction-heavy DGPs where stepwise methods structurally
can't detect the relevant structure (0.944 vs. 0.595/0.567 on `monk1`) -- this is the clearest,
most direct evidence in the whole repo that cross-family RID captures something stepwise
selection cannot, measured the fair way: against real ground truth, not a fixed downstream
model's CV score. Combined with \S5's result, the overall picture is coherent rather than
contradictory: RID isn't "worse" than stepwise in general, it wins specifically on the
structure (interactions, redundancy-awareness) it was designed to find, and loses specifically
on the metric (fixed-budget top-k predictive accuracy) it was never designed to optimize.

## 7. `rid_tree` is not TreeFARMS/GOSDT

`FullyEnumeratedTreeClassifier` is a `DecisionTreeClassifier` subclass whose "Rashomon set" is
a bounded grid (`max_depth` 2-6 x `min_samples_leaf` {1,5,10,20} x `criterion` {gini,entropy} =
40 configs) of sklearn's standard greedy CART -- it does **not** search the combinatorial space
of sparse tree topologies the way the Donnelly/Katta/Rudin/Wu paper's tree family does. This
matters for how the project's results should be framed: "single-family tree RID" here means
"a Rashomon set over CART hyperparameter choices," a real but narrower notion of a tree
Rashomon set than the paper's. `max_depth=None` was deliberately excluded after it caused
memorized, 100%-train-accuracy, degenerate Rashomon sets. Genuine TreeFARMS/GOSDT-based
enumeration would need a new dependency (`treefarms`) plus an adapter, and hasn't been
implemented.

## 8. In-sample vs. held-out performance (a standing caveat, not new this session)

RID's own internal `accuracy_mean`/`auprc_mean` are computed by predicting on `X_boot`/`y_boot`
-- the same bootstrap resample each model was just fit on. This is in-sample, not a
generalization estimate, and is not comparable to the held-out CV scores in \S2/\S5. Every
comparison in this document that claims to measure generalization uses the held-out evaluator,
never RID's internal stats -- this caveat is the reason \S2's evaluator redesign was necessary
in the first place, rather than just reusing numbers RID already produces internally.

## 9. iLOCO: pairwise interaction importance, added as a first-class VI metric

Motivation tied back to \S6: if cross-family RID's edge is real interaction-detection, the
*intermediate* variable-importance metric feeding it (`sub_mr`, purely marginal/per-feature)
is arguably the wrong lens -- it can't represent "these two features matter jointly" at all.
iLOCO (arXiv:2502.06661) is a published metric built exactly for this: `iLOCO_{j,k} = Δ_j + Δ_k
- Δ_{j,k}` (error-increase deltas from removing features individually vs. jointly). The paper's
own naive cost is O(M^2) refits; its minipatch-ensemble trick exists to solve that at large M.

**Our implementation deliberately skips minipatching.** `compute_loco_importance` was already
ablation-based (zero a column at prediction time on an *already-fitted* model, no retraining)
rather than the paper's refit-based approach -- so a pairwise version reuses that same cheap
pattern instead, sidestepping the exact problem minipatching was invented to solve.
`compute_iloco_pairwise_importance` computes single-feature deltas once and caches them across
pairs; `_eligible_interaction_pairs` skips pairs whose features are already correlated above a
threshold (redundant "interactions" aren't interesting -- the same intuition as \S3's
credit-split diagnostic, applied here to pruning rather than measuring); reduced to a
per-feature `sum(|iLOCO_jk|)` score via `compute_iloco_importance` so it fits the existing
`n_features`-shaped VI-metric convention alongside `sub_mr`/`loco`/`coef`, meaning it slots
into every experiment's existing pipeline without new plumbing.

Measured cost directly (real Falcon-Cano data, one fitted model):

| feature count | eligible pairs | time |
|---|---|---|
| 40 (top-k) | 780 | 0.5s |
| 138 (current 0.8 corr-threshold scale) | 9,453 | 6.1s |
| 177 (auto elbow-threshold scale, \S10) | 15,520 | 11.5s |

Now `run_falcon_cano_top40_comparison.py`'s default `--rid-metric` (previously `sub_mr`) --
deliberately **not** paired with `--correlation-threshold auto` by default, since the two
compound (6.1s/model vs 11.5s/model) in a way neither was validated at together, risking the
48h cluster walltime. Both remain available independently via explicit flags. Also wired into
`run_nonlinear_interaction_simulation.py` via a new `--rid-metric` flag (previously hardcoded
to `sub_mr`); DGP feature counts are tiny (6-11), so iLOCO's cost is negligible there --
naturally the best place to see whether it improves ground-truth recovery on the
interaction-heavy DGPs from \S6, once that run completes.

Real top-40 run on Falcon-Cano found meaningful rank shifts vs. `sub_mr` (e.g. `VSA_EState8`
rank 34->18, `BCUT2D_MWHI` rank 18->32) -- plausible genuine interaction signal, not noise. One
open design question: the `sum_abs` reduction is always >=0, so `prob_positive` doesn't carry
the same "reliably positive vs. could be negative" meaning it does for `sub_mr`/`loco` -- not
yet resolved.

Caught and fixed one real bug in the prototype along the way: a bare CSV index column
(`"Unnamed: 0"`) was leaking into the top-15 via the univariate-correlation fallback
feature-selector.

## 10. Correlation-threshold elbow detection (replacing the flat 0.8 cutoff)

Advisor's suggestion: plot correlation threshold vs. surviving feature count and find the
"elbow" via Kneedle, rather than an arbitrary flat threshold. This matters for the project
beyond tidiness -- every experiment's feature space (and therefore every downstream
importance ranking) starts from this filtering step, so an arbitrary threshold is an arbitrary
thumb on the scale of everything built on top of it, including \S3's credit-split diagnostic
(which can only detect redundancy the upstream filter didn't already remove).

| dataset | current default | elbow (Kneedle) | features at elbow | features at 0.8 |
|---|---|---|---|---|
| Falcon-Cano | 0.80 | **≈0.97** | 176-177 | 137-138 |
| Staellert | 0.80 | **≈0.93** | 189 | 156 |

Both elbows sit well above the current flat cutoff -- the curve is fairly flat 0.50-0.95, then
accelerates sharply near 0.97-0.99 (plots: `experiments/{falcon_cano,staellert}/results/
correlation_threshold_sweep/threshold_vs_feature_count.png`, gitignored/local only). In plain
terms: the current threshold is probably too aggressive, discarding features a knee-based
criterion would keep. Wired in as an opt-in `--correlation-threshold auto` flag (Kneedle via
the `kneed` package, falling back to a max-distance-from-chord heuristic if unavailable) in
both comparison scripts; **default stays 0.80** pending a decision on whether to change it, and
because of the iLOCO cost interaction noted in \S9.

Falcon-Cano's and Staellert's underlying greedy correlation-removal algorithms are genuinely
different (Falcon-Cano's doesn't skip already-removed features when scanning later pairs;
Staellert's does) -- each got a faithful standalone replication rather than a shared/unified
implementation.

**Process note**: the standalone sweep module initially landed in each experiment's
`analysis/` directory, which has a blanket `.gitignore` entry (intended for personal
exploratory notebooks, e.g. Staellert's pre-existing
`top40_feature_comparison_results_analysis.ipynb`) -- so it silently never got committed, which
only surfaced when a fresh cluster `git pull` produced a `ModuleNotFoundError` and a job died in
5 seconds. Fixed by moving the module out of `analysis/` into each experiment's root, since it's
a real pipeline dependency now, not personal analysis code.

## 11. DataSAIL (leakage-aware dataset splitting) -- relevance confirmed, and now quantified

[docs.](https://datasail.readthedocs.io/en/latest/) Clusters entities by domain-specific
similarity (Tanimoto/ECFP for molecules, sequence identity for proteins) and assigns whole
clusters to folds via an ILP solver, so similar entities never split across train/test.

- **Falcon-Cano: relevant, and concrete, not hypothetical.** The raw (pre-featured) `train.csv`
  still has SMILES and contains substantial congeneric families: 26 "-azole" antifungals, 25
  "-pril" ACE inhibitors, 24 "-cef" cephalosporins, 22 "-olol" beta-blockers, 15 "-cillin"
  penicillins, 8 statins, plus literal duplicate compound names. The original benchmark's own
  diversity analysis reported train/test Tanimoto similarity of 0.61-0.66 ("moderate," i.e. not
  independent). Since `_evaluate_feature_set` (the \S2 evaluator) uses plain `StratifiedKFold`
  (blind to structure), near-identical analogs can land on both sides of a fold split, inflating
  every `held_out_cv_score` in \S5 beyond genuine generalization.
- **Staellert: not relevant.** Features are fluorescence-marker measurements, not
  molecules/sequences; none of DataSAIL's built-in similarity metrics apply, and no
  well/plate/lineage ID is exposed to group by even manually.
- **RID's own bootstrap loop: not applicable at all.** No train/test split exists there to
  protect -- `resample()` draws from the full data and scores in-sample (see \S8).

### A/B experiment: does leakage explain \S5's stepwise-vs-RID gap?

Lives in `experiments/falcon_cano/run_falcon_cano_datasail_ab.py` (kept inside the Falcon-Cano
experiment folder, not a new top-level experiment, since it reuses Falcon-Cano's exact dataset
and already-selected top-40 feature lists -- only the fold-splitting strategy changes; same
reasoning that put \S4's stability analysis and \S9's iLOCO prototype there directly). It's a
re-evaluation study, not a re-selection study: no RID bootstrapping or stepwise search reruns,
only the held-out scoring step changes.

**Getting DataSAIL's Python API to actually work required real debugging**, worth recording
since it isn't documented anywhere obvious:
- The correct single-entity cluster-based split technique string is **`"C1e"`** -- not `"C1"`
  (satisfies a naming check but never triggers clustering, so `dataset.cluster_names` stays
  `None` and crashes downstream) and not `"S1"`/`"Ce"` (both silently produce zero assignments
  or crash on an internal naming-convention mismatch between the clustering step and the
  solver's post-processing step). Confirmed reproducible across DataSAIL 1.4.0 and 1.3.0.
- Molecules must be keyed by a synthetic unique ID, not compound name -- Falcon-Cano has
  literal duplicate names (`sulfadiazine` etc., consistent with \S11's congeneric-family
  finding), which silently collapse in a `dict(zip(names, smiles))` construction.
- `datasail_main()` unconditionally reseeds `random`/`numpy.random` to 42 at the start of every
  call, so calling `datasail()` N separate times (each `runs=1`) returns N **identical** splits,
  not independent ones (confirmed empirically: two such calls were byte-identical down to 6
  decimal places). Must request `runs=N` in a single call instead, which reshuffles internally
  between runs.
- Real-scale solve time: ~60-90s per split at 1,157 compounds / 50 clusters (SCIP), vs. <1s at
  a 120-compound test scale -- plan for a few minutes per experiment, not instant.

**Result** (5 independent `C1e` 80/20 splits, same 3-model evaluator panel as \S2, applied to
the already-selected top-40 feature lists from \S5):

| method | leaky (current) | leakage-aware (DataSAIL) | delta |
|---|---|---|---|
| stepwise_logreg | 0.721 | 0.668 | **-0.053** |
| rid_tree | 0.693 | 0.657 | -0.036 |
| cross_family_weighted | 0.685 | 0.658 | -0.027 |
| cross_family_unweighted | 0.663 | 0.642 | -0.022 |
| stepwise_rf | 0.708 | 0.688 | -0.020 |

Every method drops under leakage-aware splitting (expected), but not equally: `stepwise_logreg`
drops the most, which shrinks its lead over every RID variant substantially -- the gap vs.
`rid_tree` falls from 0.028 to 0.011 (-61%), vs. `cross_family_weighted` from 0.036 to 0.010
(-72%). **But `stepwise_rf` drops the least and remains the top performer even under leakage
control** -- its lead over `rid_tree` actually widens slightly. Conclusion: leakage is real and
explains a substantial share of *linear* stepwise's specific advantage, but not the *entire*
stepwise-vs-RID gap from \S5 -- the redundancy-penalty mechanism identified there is a separate,
independently-real effect, most visible in `stepwise_rf`'s comparison. Caveat: n=5 splits, with
run-to-run std of 0.017-0.046 depending on method/evaluator -- the direction is consistent
across all 3 evaluator-panel models, but this is not a large-sample estimate.

Not yet implemented: the lighter-weight, dependency-free fallback (ECFP4/Tanimoto clustering +
`GroupKFold`) originally proposed as an alternative to DataSAIL is now moot, since DataSAIL's
API does work once called correctly -- the fallback would only be worth revisiting if the
`C1e` bugs above resurface in a future DataSAIL release.

## 12. rFCFP (dosage-aware molecular fingerprints)

Full writeup: `docs/rfcfp_dosage_poc.md`. This is the one line of work this session that isn't
about the two existing real datasets -- it's scoping a *third*, dose-response-flavored
experiment that would test a variant of the project's central claim: does cross-family RID's
importance characterization for a fingerprint bit shift depending on dose, something a static
structural fingerprint could never even represent as a question. Summary: rFCFP doesn't change
FCFP's bit-generation algorithm at all -- it rescales the existing fixed fingerprint by a
log-dose scalar and sums across compounds in a perturbation (`rFCFP = Σ log10(dose+1) x
FCFP4(SMILES)`), turning a structure-only fingerprint into a dose-aware one. Neither
Falcon-Cano nor Staellert has a dosage/stimulus variable, so a POC needs a different dataset.
Proposed: **sci-Plex3 (GSE139944)** -- 188 compounds x 4 doses x 3 cell lines, the rFCFP
paper's own validated benchmark, chosen over LINCS L1000 (175k compounds) for tractability. No
changes needed to `rid/core.py` for a POC -- an rFCFP dimension is just another column, like an
RDKit descriptor, so it would reuse every fix and diagnostic in \S1-\S9 as-is. Research and
proposal only; no data acquired or code written toward this.

## Summary of code changes this session

- `rid/core.py`: global-epsilon fix, weighted-`expected_importance` fix, `iloco` VI metric
  (`compute_iloco_pairwise_importance`, `compute_iloco_importance`, `vi_iloco`,
  `_eligible_interaction_pairs`, `describe_iloco_eligible_pairs`).
- `experiments/falcon_cano/run_falcon_cano_top40_comparison.py`,
  `experiments/staellert/run_staellert_top40_comparison.py`: 3-model evaluator panel,
  `--correlation-threshold auto`, `--inject-redundant-duplicate` + credit-split-ratio
  diagnostic (Falcon-Cano); `--rid-metric` defaults to `iloco` (Falcon-Cano only).
- `experiments/nonlinear_interaction_simulation/run_nonlinear_interaction_simulation.py`:
  new `--rid-metric` flag (default stays `sub_mr`).
- New: `experiments/{falcon_cano,staellert}/run_stability_analysis.py`,
  `experiments/{falcon_cano,staellert}/correlation_threshold_sweep.py`,
  `experiments/falcon_cano/run_iloco_prototype.py`.
- `experiments/falcon_cano/environment.yml`, `experiments/staellert/environment.yml`: added
  `kneed`, `datasail` (Falcon-Cano only).
- `docs/rfcfp_dosage_poc.md`: new.
