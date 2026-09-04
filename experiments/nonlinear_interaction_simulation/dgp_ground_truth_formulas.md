# DGP Ground-Truth Formulas

This file summarizes the data-generating processes (DGPs) used in the nonlinear interaction simulation experiment.

## Shared Label-Generation Rule (All DGPs)

For each sample $i$:

$$
\epsilon_i \sim \mathcal{N}(0, \sigma^2), \quad z_i = \beta f(X_i) + \epsilon_i
$$

with $\sigma =$ `noise_std` and $f(\cdot)$ the DGP-specific signal.

The binary label is produced with a median cutoff on latent scores:

$$
c = \operatorname{median}(z_1, \ldots, z_n), \quad Y_i = \mathbf{1}[z_i \ge c]
$$

## 1. `custom_nonlinear`

Feature distribution:

$$
X_1,\ldots,X_6 \sim \operatorname{Unif}(0,1)
$$

Ground-truth signal:

$$
f(X) = X_1^2 + 2\,\mathbf{1}[X_2 > 0.5] + 2X_3X_4
$$

Relevant features: $X_1, X_2, X_3, X_4$.

## 2. `custom_sin_log`

Feature distribution:

$$
X_1,\ldots,X_6 \sim \operatorname{Unif}(0,1)
$$

Ground-truth signal:

$$
f(X) = 1.8\sin(\pi X_1) + 1.4\log(1 + 3X_3) + 1.6X_2X_4
$$

Relevant features: $X_1, X_2, X_3, X_4$.

## 3. `chen`

Feature distribution:

$$
X_1,\ldots,X_{10} \sim \mathcal{N}(0,1)
$$

Ground-truth signal:

$$
f(X) = -2\sin(X_1) + \max(X_2, 0) + X_3 + e^{-X_4}
$$

Relevant features: $X_1, X_2, X_3, X_4$.

## 4. `friedman`

Feature distribution:

$$
X_1,\ldots,X_6 \sim \operatorname{Unif}(0,1)
$$

Ground-truth signal:

$$
f(X) = 10\sin(\pi X_1X_2) + 20(X_3 - 0.5)^2 + 10X_4 + 5X_5
$$

Relevant features: $X_1, X_2, X_3, X_4, X_5$.

## 5. `monk1`

Feature domains (uniform categorical sampling):

- $X_1, X_2 \in \{1,2,3\}$
- $X_3 \in \{1,2\}$
- $X_4 \in \{1,2,3\}$
- $X_5 \in \{1,2,3,4\}$
- $X_6 \in \{1,2\}$

Ground-truth signal:

$$
f(X) = \max\left(\mathbf{1}[X_1 = X_2],\, \mathbf{1}[X_5 = 1]\right)
$$

Equivalent logical form:

$$
f(X) = \mathbf{1}\left[(X_1 = X_2) \lor (X_5 = 1)\right]
$$

Relevant features: $X_1, X_2, X_5$.

## 6. `monk3`

Feature domains (uniform categorical sampling):

- $X_1, X_2 \in \{1,2,3\}$
- $X_3 \in \{1,2\}$
- $X_4 \in \{1,2,3\}$
- $X_5 \in \{1,2,3,4\}$
- $X_6 \in \{1,2\}$

Ground-truth signal:

$$
f(X) = \max\left(\mathbf{1}[X_5 = 3 \land X_4 = 1],\, \mathbf{1}[X_5 \ne 4 \land X_2 \ne 3]\right)
$$

Equivalent logical form:

$$
f(X) = \mathbf{1}\left[(X_5 = 3 \land X_4 = 1) \lor (X_5 \ne 4 \land X_2 \ne 3)\right]
$$

Relevant features: $X_2, X_4, X_5$.

## Source

Formulas are implemented in `run_nonlinear_interaction_simulation.py`.
