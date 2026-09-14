# Extended results: DataSAIL leakage A/B, DGP simulation metrics, iLOCO

This is a data appendix, not a narrative summary -- for the synthesized "why does this
matter" version of everything here, see `docs/findings_summary.md` (\S6 ground-truth
recovery, \S9 iLOCO, \S11 DataSAIL). This document exists so a future reader can see the
actual underlying numbers without re-running anything, and includes enough methodology
context to interpret them without also re-reading the whole session's history.

## Context for future readers

This repo implements **Cross-Family Rashomon Importance Distribution (RID)**: instead of
trusting one fitted model's feature importances, characterize importance across every
near-optimal model within `epsilon` of the best loss (the Rashomon set), optionally pooled
across model families with different inductive biases. Three experiments test this claim:
**Falcon-Cano** (real oral-bioavailability data, ~1,157 compounds, no ground truth),
**Staellert** (real single-cell imaging data, not used in this document), and
**nonlinear_interaction_simulation** ("the DGP experiment": synthetic data-generating
processes with *known* relevant features, some deliberately interaction-heavy).

Everything below comes from two separate lines of work:
1. Does Falcon-Cano's held-out CV comparison (stepwise selection vs. RID) get distorted by
   *data leakage* -- near-identical compounds landing on both sides of a fold split? (DataSAIL)
2. What does the fuller metric picture look like for the DGP experiment, and does iLOCO (a
   pairwise interaction-importance metric) actually help RID's ground-truth recovery there,
   where it should be the cheapest and most natural place to test it?

## 1. Falcon-Cano: DataSAIL leakage-aware A/B (full breakdown)

**Setup**: `experiments/falcon_cano/run_falcon_cano_datasail_ab.py`. Falcon-Cano's raw
compound list contains substantial congeneric families (26 "-azole" antifungals, 25 "-pril"
ACE inhibitors, 24 "-cef" cephalosporins, 22 "-olol" beta-blockers, 15 "-cillin" penicillins,
8 statins, plus literal duplicate compound names) -- the kind of structure plain
`StratifiedKFold` (blind to molecular structure) can leak across train/test folds, inflating
held-out CV scores. [DataSAIL](https://datasail.readthedocs.io/en/latest/) clusters compounds
by ECFP/Tanimoto similarity and assigns whole clusters to a split so this can't happen.

This is a **re-evaluation** study: it reuses each method's already-selected top-40 feature
list from the standard comparison (`results/top40_feature_comparison/`) and only changes how
the held-out score is computed -- Arm A is the existing `StratifiedKFold` evaluator
("leaky"), Arm B is 5 independent DataSAIL `C1e` 80/20 splits ("leakage-aware", repeated
holdout since DataSAIL produces one train/test partition per run rather than sklearn-style
fold indices). Both arms score under the same 3-model evaluator panel (`LogisticRegression`,
`RandomForestClassifier`, `SVC(kernel="rbf")`) used throughout this project.

**Getting DataSAIL's Python API to actually work required real debugging** (reproducible
across DataSAIL 1.4.0 and 1.3.0, not a one-off environment issue):
- The correct technique string for a single-entity cluster-based split is **`"C1e"`**.
  `"C1"` satisfies an internal naming check but never triggers clustering (`dataset.cluster_names`
  stays `None`, crashes downstream); `"S1"` and `"Ce"` either return zero assignments or crash
  on a genuine mismatch between DataSAIL's clustering step and its solver's post-processing step.
- Molecules must be keyed by a synthetic unique ID, not compound name -- several Falcon-Cano
  compounds share a name (e.g. `sulfadiazine`), which silently collapses in a naive
  `dict(zip(names, smiles))`.
- `datasail_main()` unconditionally reseeds `random`/`numpy.random` to 42 on every call, so
  calling `datasail()` N separate times (each `runs=1`) returns N **identical** splits, not
  independent ones (confirmed empirically -- two such calls were byte-identical to 6 decimal
  places). Must request `runs=N` in a single call, which reshuffles internally between runs.
- Real-scale solve time: ~60-90s per split at 1,157 compounds / 50 clusters (SCIP solver),
  vs. under a second at a 120-compound toy scale.

**Full results** (5 independent `C1e` splits; per-run std across those 5 splits ranges
0.017-0.046 depending on method/evaluator -- a real but modest sample):

| method | leaky (logreg) | leakage-aware (logreg) | leaky (RF) | leakage-aware (RF) | leaky (SVM-RBF) | leakage-aware (SVM-RBF) | leaky (avg) | leakage-aware (avg) | Δ (avg) |
|---|---|---|---|---|---|---|---|---|---|
| stepwise_logreg | 0.745 | 0.696 | 0.732 | 0.655 | 0.686 | 0.652 | 0.721 | 0.668 | **-0.053** |
| stepwise_rf | 0.654 | 0.659 | 0.790 | 0.727 | 0.680 | 0.678 | 0.708 | 0.688 | -0.020 |
| rid_tree | 0.679 | 0.658 | 0.748 | 0.682 | 0.652 | 0.631 | 0.693 | 0.657 | -0.036 |
| cross_family_weighted | 0.672 | 0.660 | 0.742 | 0.690 | 0.641 | 0.625 | 0.685 | 0.658 | -0.027 |
| cross_family_unweighted | 0.673 | 0.652 | 0.721 | 0.684 | 0.597 | 0.588 | 0.663 | 0.642 | -0.022 |

**Reading it**: every method drops under leakage-aware splitting (expected -- leaky CV always
looks rosier), but not equally. `stepwise_logreg` drops the most under every single evaluator
in the panel, which shrinks its lead over the RID variants substantially:

| comparison | gap under leaky | gap under leakage-aware | reduction |
|---|---|---|---|
| stepwise_logreg vs. rid_tree | 0.028 | 0.011 | -61% |
| stepwise_logreg vs. cross_family_weighted | 0.036 | 0.010 | -72% |
| stepwise_logreg vs. cross_family_unweighted | 0.058 | 0.026 | -55% |

But `stepwise_rf` drops the *least* of all five methods and remains the top performer even
under leakage control -- its lead over `rid_tree` (0.708 vs. 0.693 leaky; 0.688 vs. 0.657
leakage-aware) actually widens slightly. **Conclusion**: leakage is real and explains a
substantial share of *linear* stepwise's specific advantage over RID, but it is not the whole
story behind the stepwise-vs-RID gap -- the redundancy-penalty mechanism (RID's top-40 picks
being 2-3x more internally correlated than stepwise's, see `findings_summary.md` \S5) is a
separate, independently real effect, and it's the one still standing once leakage is controlled.

Raw data: `experiments/falcon_cano/results/datasail_ab/leaky_vs_leakage_aware.csv` and
`run_settings.json` (per-run raw scores, full split compositions).

## 2. Nonlinear-interaction-simulation (DGP): full metric set, `sub_mr` baseline

**Config** (identical for both the `sub_mr` baseline and the `iloco` run below, confirmed via
`study_settings.json` -- same seeds, same DGPs, same betas, only `rid_metric` differs):
sample size 400, 12 repetitions per beta, beta grid `[0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]`,
12 bootstraps / 6 models-per-class per RID fit, epsilon 0.05, 10 DGPs (6 standard + 4
redundant-driver variants), 5 methods.

**Metric definitions** (`compute_selection_metrics`, `compute_equivalence_recall`,
`compute_credit_split_ratios` in `run_nonlinear_interaction_simulation.py`; k = number of
true drivers for that DGP, so precision@k and recall@k are numerically identical here since
the top-k set is always truncated to exactly k):

| metric | meaning |
|---|---|
| `precision_at_k` / `recall_at_k` | fraction of the top-k ranked features that are true drivers |
| `ndcg_at_k` | rank-aware version -- a true driver ranked #1 counts more than one ranked #k |
| `exact_match` | 1 if the top-k set exactly equals the ground-truth set, else 0 |
| `mean_gt_rank` | average rank assigned to the true drivers (lower is better, unbounded above) |
| `equivalence_recall_at_k` | recall that also credits a substitutable duplicate of a true driver (only defined for the `*_redundant_*` DGPs) |
| `credit_split_ratio_mean` | 0 = winner-take-all, 1 = evenly shared credit across a redundant pair (only defined for the `*_redundant_*` DGPs) |

**Full table** (averaged over all 7 betas x 12 repetitions per cell):

| dgp | method | precision@k | ndcg@k | exact_match | mean_gt_rank | equiv_recall@k | credit_split |
|---|---|---|---|---|---|---|---|
| chen | cross_family_unweighted | 0.973 | 0.981 | 0.893 | 2.554 | - | - |
| chen | cross_family_weighted | 0.973 | 0.981 | 0.893 | 2.554 | - | - |
| chen | rid_tree | 0.976 | 0.983 | 0.905 | 2.554 | - | - |
| chen | stepwise_logreg | 0.970 | 0.980 | 0.881 | 2.562 | - | - |
| chen | stepwise_rf | 0.938 | 0.943 | 0.786 | 2.732 | - | - |
| chen_redundant_r070 | cross_family_unweighted | 0.952 | 0.967 | 0.821 | 2.598 | 0.952 | 0.220 |
| chen_redundant_r070 | cross_family_weighted | 0.952 | 0.967 | 0.821 | 2.598 | 0.952 | 0.213 |
| chen_redundant_r070 | rid_tree | 0.955 | 0.969 | 0.833 | 2.655 | 0.955 | 0.200 |
| chen_redundant_r070 | stepwise_logreg | 0.976 | 0.983 | 0.905 | 2.592 | 0.976 | 0.169 |
| chen_redundant_r070 | stepwise_rf | 0.923 | 0.931 | 0.786 | 2.789 | 0.923 | 0.200 |
| chen_redundant_r095 | cross_family_unweighted | 0.926 | 0.949 | 0.714 | 2.693 | 0.935 | 0.397 |
| chen_redundant_r095 | cross_family_weighted | 0.920 | 0.945 | 0.690 | 2.699 | 0.929 | 0.415 |
| chen_redundant_r095 | rid_tree | 0.845 | 0.888 | 0.393 | 2.795 | 0.851 | 0.356 |
| chen_redundant_r095 | stepwise_logreg | 0.964 | 0.969 | 0.869 | 2.667 | 0.976 | 0.177 |
| chen_redundant_r095 | stepwise_rf | 0.833 | 0.817 | 0.512 | 3.143 | 0.908 | 0.185 |
| custom_nonlinear | cross_family_unweighted | 0.917 | 0.940 | 0.690 | 2.640 | - | - |
| custom_nonlinear | cross_family_weighted | 0.917 | 0.940 | 0.690 | 2.643 | - | - |
| custom_nonlinear | rid_tree | 0.920 | 0.938 | 0.726 | 2.655 | - | - |
| custom_nonlinear | stepwise_logreg | 0.908 | 0.933 | 0.679 | 2.670 | - | - |
| custom_nonlinear | stepwise_rf | 0.872 | 0.890 | 0.583 | 2.798 | - | - |
| custom_sin_log | cross_family_unweighted | 0.923 | 0.937 | 0.726 | 2.679 | - | - |
| custom_sin_log | cross_family_weighted | 0.920 | 0.933 | 0.726 | 2.688 | - | - |
| custom_sin_log | rid_tree | 0.920 | 0.939 | 0.702 | 2.658 | - | - |
| custom_sin_log | stepwise_logreg | 0.780 | 0.838 | 0.190 | 2.905 | - | - |
| custom_sin_log | stepwise_rf | 0.878 | 0.888 | 0.643 | 2.821 | - | - |
| friedman | cross_family_unweighted | 1.000 | 1.000 | 1.000 | 3.000 | - | - |
| friedman | cross_family_weighted | 1.000 | 1.000 | 1.000 | 3.000 | - | - |
| friedman | rid_tree | 0.993 | 0.995 | 0.964 | 3.007 | - | - |
| friedman | stepwise_logreg | 0.886 | 0.925 | 0.429 | 3.114 | - | - |
| friedman | stepwise_rf | 0.995 | 0.997 | 0.976 | 3.007 | - | - |
| friedman_redundant_r070 | cross_family_unweighted | 1.000 | 1.000 | 1.000 | 3.000 | 1.000 | 0.391 |
| friedman_redundant_r070 | cross_family_weighted | 1.000 | 1.000 | 1.000 | 3.000 | 1.000 | 0.387 |
| friedman_redundant_r070 | rid_tree | 0.988 | 0.992 | 0.940 | 3.017 | 0.988 | 0.395 |
| friedman_redundant_r070 | stepwise_logreg | 0.888 | 0.927 | 0.440 | 3.164 | 0.888 | 0.415 |
| friedman_redundant_r070 | stepwise_rf | 0.990 | 0.994 | 0.952 | 3.014 | 0.990 | 0.379 |
| friedman_redundant_r095 | cross_family_unweighted | 0.998 | 0.998 | 0.988 | 3.002 | 0.998 | 0.418 |
| friedman_redundant_r095 | cross_family_weighted | 0.998 | 0.998 | 0.988 | 3.002 | 0.998 | 0.417 |
| friedman_redundant_r095 | rid_tree | 0.981 | 0.988 | 0.905 | 3.021 | 0.981 | 0.413 |
| friedman_redundant_r095 | stepwise_logreg | 0.883 | 0.923 | 0.417 | 3.164 | 0.883 | 0.415 |
| friedman_redundant_r095 | stepwise_rf | 0.981 | 0.981 | 0.905 | 3.052 | 0.981 | 0.368 |
| monk1 | cross_family_unweighted | 0.944 | 0.951 | 0.833 | 2.099 | - | - |
| monk1 | cross_family_weighted | 0.944 | 0.950 | 0.833 | 2.103 | - | - |
| monk1 | rid_tree | 0.940 | 0.950 | 0.821 | 2.099 | - | - |
| monk1 | stepwise_logreg | 0.595 | 0.661 | 0.119 | 3.004 | - | - |
| monk1 | stepwise_rf | 0.567 | 0.625 | 0.238 | 3.087 | - | - |
| monk3 | cross_family_unweighted | 0.905 | 0.930 | 0.714 | 2.115 | - | - |
| monk3 | cross_family_weighted | 0.897 | 0.924 | 0.690 | 2.123 | - | - |
| monk3 | rid_tree | 0.909 | 0.926 | 0.726 | 2.143 | - | - |
| monk3 | stepwise_logreg | 0.714 | 0.781 | 0.238 | 2.627 | - | - |
| monk3 | stepwise_rf | 0.738 | 0.793 | 0.298 | 2.611 | - | - |

`-` = not defined for that DGP (no engineered redundancy). Ground-truth formulas for every
DGP are in `experiments/nonlinear_interaction_simulation/dgp_ground_truth_formulas.md`.

**Reading it**: `monk1`/`monk3` are the standout cases -- both are interaction-heavy by
construction (`monk1`: `1[(X1=X2) v (X5=1)]`; `monk3`: `1[(X5=3^X4=1) v (X5!=4^X2!=3)]`), and
cross-family RID (`0.944`/`0.905` precision) crushes both stepwise methods (`0.595`/`0.567` on
`monk1`) by a wide margin -- the clearest evidence in this repo that cross-family RID captures
structure stepwise selection structurally cannot. `chen_redundant_r095` and `friedman_redundant_r095`/`_r070` are where
`credit_split_ratio_mean` tells its own story. On `chen_redundant_*`, RID variants clearly
lead (0.36-0.42 vs. stepwise's 0.17-0.20 -- close to winner-take-all). On `friedman_redundant_*`,
the picture is more mixed: `stepwise_logreg` actually matches or slightly exceeds the RID
variants (0.415 vs. 0.39-0.42) on both redundancy strengths -- worth noting rather than
glossing over, since it means credit-sharing here isn't an exclusively-RID phenomenon on
every DGP; the effect is real but DGP-dependent, not universal.

Per-beta granularity (signal strength vs. every metric) is in
`metric_table_by_dgp_beta.csv`; raw per-repetition data (including the actual `ranked_top_k`
lists) is in `study_results.csv`, both alongside `metric_table_by_dgp.csv`.

## 3. iLOCO results

### 3a. DGP simulation with `--rid-metric iloco`: same config, full comparison

Identical run configuration to \S2 (confirmed via `study_settings.json`), with `rid_metric`
switched from `sub_mr` to `iloco` for `cross_family_unweighted`, `cross_family_weighted`, and
`rid_tree` (`iloco` doesn't affect `stepwise_logreg`/`stepwise_rf`, which don't use RID at
all -- their rows below are identical to \S2 and included only for context).

| dgp | method | precision@k | ndcg@k | exact_match | mean_gt_rank | equiv_recall@k | credit_split |
|---|---|---|---|---|---|---|---|
| chen | cross_family_unweighted | 0.958 | 0.961 | 0.869 | 2.637 | - | - |
| chen | cross_family_weighted | 0.958 | 0.961 | 0.869 | 2.637 | - | - |
| chen | rid_tree | 0.976 | 0.982 | 0.905 | 2.557 | - | - |
| chen_redundant_r070 | cross_family_unweighted | 0.929 | 0.942 | 0.762 | 2.741 | 0.929 | 0.246 |
| chen_redundant_r070 | cross_family_weighted | 0.929 | 0.942 | 0.762 | 2.741 | 0.929 | 0.243 |
| chen_redundant_r070 | rid_tree | 0.946 | 0.963 | 0.798 | 2.658 | 0.946 | 0.180 |
| chen_redundant_r095 | cross_family_unweighted | 0.920 | 0.942 | 0.726 | 2.732 | 0.938 | 0.470 |
| chen_redundant_r095 | cross_family_weighted | 0.920 | 0.942 | 0.726 | 2.732 | 0.938 | 0.440 |
| chen_redundant_r095 | rid_tree | 0.881 | 0.908 | 0.548 | 2.792 | 0.893 | 0.356 |
| custom_nonlinear | cross_family_unweighted | 0.869 | 0.894 | 0.548 | 2.795 | - | - |
| custom_nonlinear | cross_family_weighted | 0.854 | 0.884 | 0.488 | 2.812 | - | - |
| custom_nonlinear | rid_tree | 0.905 | 0.927 | 0.667 | 2.679 | - | - |
| custom_sin_log | cross_family_unweighted | 0.863 | 0.883 | 0.536 | 2.810 | - | - |
| custom_sin_log | cross_family_weighted | 0.860 | 0.881 | 0.524 | 2.818 | - | - |
| custom_sin_log | rid_tree | 0.911 | 0.932 | 0.667 | 2.670 | - | - |
| friedman | cross_family_unweighted | 0.995 | 0.997 | 0.976 | 3.007 | - | - |
| friedman | cross_family_weighted | 0.990 | 0.994 | 0.952 | 3.012 | - | - |
| friedman | rid_tree | 0.979 | 0.986 | 0.893 | 3.021 | - | - |
| friedman_redundant_r070 | cross_family_unweighted | 0.990 | 0.994 | 0.952 | 3.014 | 0.990 | 0.320 |
| friedman_redundant_r070 | cross_family_weighted | 0.988 | 0.992 | 0.940 | 3.017 | 0.988 | 0.317 |
| friedman_redundant_r070 | rid_tree | 0.981 | 0.987 | 0.905 | 3.029 | 0.981 | 0.400 |
| friedman_redundant_r095 | cross_family_unweighted | 0.981 | 0.987 | 0.905 | 3.029 | 0.981 | 0.313 |
| friedman_redundant_r095 | cross_family_weighted | 0.981 | 0.987 | 0.905 | 3.029 | 0.981 | 0.312 |
| friedman_redundant_r095 | rid_tree | 0.971 | 0.981 | 0.857 | 3.033 | 0.971 | 0.434 |
| monk1 | cross_family_unweighted | 0.905 | 0.922 | 0.738 | 2.183 | - | - |
| monk1 | cross_family_weighted | 0.909 | 0.927 | 0.750 | 2.163 | - | - |
| monk1 | rid_tree | 0.917 | 0.933 | 0.762 | 2.143 | - | - |
| monk3 | cross_family_unweighted | 0.813 | 0.842 | 0.452 | 2.389 | - | - |
| monk3 | cross_family_weighted | 0.802 | 0.834 | 0.429 | 2.405 | - | - |
| monk3 | rid_tree | 0.873 | 0.901 | 0.619 | 2.190 | - | - |

(`stepwise_logreg`/`stepwise_rf` rows omitted here -- identical to \S2, confirmed byte-for-byte
since `iloco` never touches those methods.)

**Result: iLOCO made ground-truth recovery worse almost everywhere**, including on the
interaction-heavy DGPs where it was expected to help most:

| dgp | method | sub_mr precision@k | iloco precision@k | Δ |
|---|---|---|---|---|
| monk3 | cross_family_unweighted | 0.905 | 0.813 | **-0.092** |
| custom_sin_log | cross_family_unweighted | 0.923 | 0.863 | -0.060 |
| custom_nonlinear | cross_family_unweighted | 0.917 | 0.869 | -0.048 |
| monk1 | cross_family_unweighted | 0.944 | 0.905 | -0.039 |
| monk3 | rid_tree | 0.909 | 0.873 | -0.036 |
| monk1 | rid_tree | 0.940 | 0.917 | -0.023 |
| chen | cross_family_unweighted | 0.973 | 0.958 | -0.015 |
| friedman | rid_tree | 0.993 | 0.979 | -0.014 |
| chen | rid_tree | 0.976 | 0.976 | 0.000 |
| chen_redundant_r095 | rid_tree | 0.845 | 0.881 | **+0.036** (only improvement) |

**Why, grounded in the actual DGP formulas**
(`dgp_ground_truth_formulas.md`): `iLOCO_{j,k} = Delta_j + Delta_k - Delta_{j,k}` is exactly
the classical statistical interaction term. When two features act additively (no synergy),
`Delta_{j,k} ~= Delta_j + Delta_k` by definition, so `iLOCO ~= 0` -- the `sum(|iLOCO|)`
reduction used here (`compute_iloco_importance` in `rid/core.py`) cannot credit a feature for
a pure main effect at all, only for genuine synergy above additivity.

- **`chen`**: `f(X) = -2sin(X1) + max(X2,0) + X3 + e^(-X4)` -- 100% additive, zero true
  interactions among the four relevant features. iLOCO has nothing real to detect; its scores
  are essentially noise, which is why cross-family drops here despite `chen` being the
  "easiest" DGP under `sub_mr`.
- **`monk1`**: `f(X) = 1[(X1=X2) v (X5=1)]` -- `X1=X2` is a genuine interaction, but `X5=1` is
  a pure main effect with zero dependence on any other feature. iLOCO can't credit `X5` at
  all, so a noise feature with any spurious pairwise correlation can outrank it.
- **`monk3`**: all-interaction logically (no isolated main effects among its relevant
  features), yet iLOCO did *worst* here of any DGP. This is a 3-way OR-of-ANDs structure
  (`X5` combines with *both* `X4` and `X2`) that pure pairwise (2-way) iLOCO can't fully
  represent, on categorical integer-coded features where zeroing two columns at once (for the
  joint ablation term) is an even less natural operation than zeroing one -- likely dominated
  by estimation noise rather than a real interaction-detection benefit.

**Implication**: the current `sum(|iLOCO|)` reduction is not a general-purpose importance
ranking -- it is structurally an interaction-only signal, and using it as a drop-in
replacement for `sub_mr` will systematically punish any dataset (which is most of them) where
relevant features include genuine main effects. This is exactly why
`run_falcon_cano_top40_comparison.py`'s `--rid-metric` default should probably be reverted
from `iloco` back to `sub_mr` (see \S3b) -- both the walltime evidence and this ground-truth
evidence point the same direction.

Raw data: `experiments/nonlinear_interaction_simulation/results/nonlinear_interaction_simulation_iloco/`.

### 3b. Falcon-Cano with `--rid-metric iloco`: timed out, no results

Submitted at full scale (500 bootstraps, 50-model pool, default `--correlation-threshold 0.8`
-> 138 features, all 5 methods including both cross-family balance modes) as SLURM job
`448982`. **Result: `TIMEOUT` at the full 48-hour walltime limit -- no output produced.**

This was a predictable outcome, not a surprise: direct timing measurements on real Falcon-Cano
data showed `compute_iloco_importance` costs ~6.1s per fitted model at 138 features (9,453
eligible pairs) vs. ~0.03s for `sub_mr` -- roughly 200x more expensive per model, multiplied
across 500 bootstraps x up to 50 candidate models x 6 cross-family model families. The
smoke-tested/validated scale for `iloco` was 40 features (780 pairs, 0.5s/model); 138 features
(~3.4x more features, ~12x more pairs given O(features^2) scaling) was never actually
validated as tractable at the full `.sl` bootstrap/pool scale before this job was submitted --
combined with \S3a's ground-truth evidence that `iloco` isn't even a better ranking in
general, this makes a strong case for treating `iloco` as an opt-in diagnostic metric (useful
where it's cheap, like the DGP experiment's 6-11 features) rather than any dataset's default.

## Summary table: what changed the default `--rid-metric` recommendation

| finding | source | direction |
|---|---|---|
| Falcon-Cano `iloco` at full scale times out at 48h | \S3b, job 448982 | against defaulting to iloco |
| DGP ground-truth recovery gets *worse* under iloco on 9 of 10 DGPs | \S3a | against defaulting to iloco |
| DataSAIL leakage explains part, not all, of stepwise's CV advantage | \S1 | orthogonal -- doesn't bear on iloco, but reframes how much of \S5's gap needs any RID-side fix at all |

Net recommendation: revert `run_falcon_cano_top40_comparison.py`'s `--rid-metric` default
from `iloco` to `sub_mr`; keep `iloco` available as an explicit opt-in for follow-up work on
interaction-specific importance, ideally validated on a low-feature-count setting (like the
DGP experiment) before ever being pointed at a 100+ feature real dataset again.
