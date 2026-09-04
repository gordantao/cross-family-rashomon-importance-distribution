# Cross-Family Rashomon Importance Distribution (RID)

## Purpose

Cross-family RID estimates feature importance as a distribution, not a single score, while accounting for model-class uncertainty across multiple families (for example, linear, kernel, and tree-based models).

The method is designed to answer two questions at once:

1. Which features are important on average?
2. How stable is that importance across many near-optimal models from different families?

## Inputs

- Feature matrix X and target y.
- A model family map model_configs, where each family has:
  - a model class or trainer,
  - model kwargs,
  - optional family-specific search grid.
- Epsilon for Rashomon filtering.
- Number of bootstraps B.
- Candidate models per family per bootstrap M.
- Variable-importance metrics (default: sub_mr, loco, coef).
- Family balancing mode (unweighted or weighted).

## Core Procedure

Before bootstrap iteration, standardize X once (z-score scaling) and run all families on the shared scaled matrix.

For each bootstrap b in {1, ..., B}:

1. Draw a bootstrap sample (X_b, y_b).
2. For each model family f:
   - Train M candidates from that family search space.
   - Compute bootstrap log-loss for each candidate.
   - Keep the family Rashomon set
     $$
     R_{f,b} = \{m : L_{f,b}(m) \le \min_{m'} L_{f,b}(m') + \epsilon\}.
     $$
3. Merge families into a cross-family Rashomon pool:
   $$
   R_b = \bigcup_f R_{f,b}.
   $$
4. Compute feature-importance values for every model in R_b.
5. Store all per-feature values and (optionally) per-model weights for later aggregation.

Only bootstraps with at least one retained model are counted as valid bootstraps.

## Variable-Importance Metrics

RID supports multiple metrics and keeps each metric as its own distribution.

- sub_mr (subtraction-based model reliance):
  permute one feature, measure increase in log-loss.
- loco (leave-one-covariate-out):
  set one feature to zero, measure increase in log-loss.
- coef (coefficient importance):
  normalized absolute coefficients for linear models.

For each feature j and metric, RID collects a sample set
$$
\{\phi^{(b,m)}_j\}
$$
across retained models and valid bootstraps.

## Cross-Family Balancing Modes

### Unweighted

Every retained model contributes equally.

### Weighted

Families with fewer retained Rashomon models are up-weighted, and families with many retained models are down-weighted.

If family f has s_f retained models in bootstrap b, each model gets weight
$$
w_{f,b} = \frac{1}{s_f^2 \sum_g (1/s_g)}.
$$

Therefore the total family weight is
$$
\sum_{m \in R_{f,b}} w_{f,b} = \frac{1/s_f}{\sum_g (1/s_g)},
$$
so smaller-family Rashomon sets carry more influence than larger ones in that bootstrap.

## Distribution Construction

For each metric and feature:

1. Build a common grid from global min to max importance values.
2. Compute a bootstrap-level ECDF on that grid (weighted or unweighted).
3. Average ECDFs over valid bootstraps to obtain the RID curve:
   $$
   F_j(k) = P(\phi_j \le k).
   $$

## Reported Summaries

From each feature distribution, RID reports:

- Expected importance:
  $$
  E[\phi_j].
  $$
- Probability of positive contribution:
  $$
  P(\phi_j > 0) = 1 - F_j(0).
  $$

Feature ranking is typically done by expected importance, with P(phi > 0) as a stability signal.

## Additional Outputs

- Family representation counts: number of retained Rashomon models per family across bootstraps.
- Family performance summaries: bootstrap-mean and bootstrap-std for metrics such as accuracy, F1, and AUPRC over retained models.

## Practical Interpretation

Cross-family RID is useful when several model families fit similarly well but attribute importance differently. Instead of committing to one winner model, it summarizes importance over the whole near-optimal region and exposes uncertainty directly.

In this project, the default cross-family runs use nonlinear and linear-regularized families together (for example Lasso, ElasticNet, Ridge, SVM, Random Forest, and Gradient Boosting) to produce robust, uncertainty-aware feature rankings.