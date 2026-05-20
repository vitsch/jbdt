"""
§2.5  ECE Calibration Bound for DM-leaf BDTs.

Posterior predictive for DM leaf with N_j observations:
  P(y=c | D_j) = (n_{jc} + α) / (N_j + Cα)          (Dirichlet shrinkage)

MSE calibration per leaf (expectation over data):
  E[(P_c − p_c)²] = Var(P_c) + Bias²(P_c)
                  = [N_j p_c(1−p_c) + α₀²(p⁰_c − p_c)²] / (N_j + α₀)²

For large N_j: dominant term = p_c(1−p_c) / N_j

ECE bound for one leaf (C classes, N_j observations):
  ECE_leaf ≤ √[Σ_c Var(P_c) + Bias²(P_c)]

ECE bound for k-leaf tree (N/k observations per leaf):
  ECE_tree ≤ √[C · σ²_max · k / N]

Prior-averaged ECE under Catalan (E[k]=1.37) vs Chipman (E[k]=2.51):
  ECE_cat / ECE_chip ≤ √(1.37/2.51) ≈ 0.739  → 26% lower for Catalan
"""

import numpy as np
from scipy.special import gammaln, polygamma


# ---------------------------------------------------------------------------
# DM posterior calibration error per leaf
# ---------------------------------------------------------------------------

def dm_mse_bound(
    true_probs: np.ndarray,
    N_leaf: int,
    alpha: float,
) -> float:
    """
    Expected squared calibration error for one DM leaf (analytical).

    E[||P̂ − p_true||²] = Σ_c [Var(P̂_c) + Bias²(P̂_c)]

    Var(P̂_c)   = N_leaf · p_c · (1−p_c) / (N_leaf + α₀)²
    Bias(P̂_c) = α₀ · (p⁰_c − p_c) / (N_leaf + α₀)
                 where p⁰_c = 1/C (uniform prior mean with equal alpha)

    Parameters
    ----------
    true_probs : array (C,) true class probabilities
    N_leaf     : int        number of training observations in this leaf
    alpha      : float      Dirichlet prior concentration per class
    """
    C = len(true_probs)
    a0 = alpha * C
    p = np.asarray(true_probs, float)
    p0 = np.ones(C) / C    # uniform prior mean

    denom = (N_leaf + a0) ** 2
    var_term  = N_leaf * p * (1.0 - p) / denom
    bias_term = (a0 * (p0 - p)) ** 2 / denom
    return float(np.sum(var_term + bias_term))


def dm_ece_bound_leaf(
    true_probs: np.ndarray,
    N_leaf: int,
    alpha: float,
) -> float:
    """ECE bound for one DM leaf: sqrt(MSE bound)."""
    return float(np.sqrt(dm_mse_bound(true_probs, N_leaf, alpha)))


def dm_ece_bound_tree(
    leaf_probs: list,
    N_total: int,
    alpha: float,
    k: int | None = None,
) -> float:
    """
    ECE bound for a k-leaf tree with N_total observations.

    If k is None: use len(leaf_probs) leaves.
    Assumes balanced allocation: N_leaf = N_total // k per leaf.
    """
    k_tree = k if k is not None else len(leaf_probs)
    N_leaf = max(1, N_total // k_tree)
    bounds = [dm_ece_bound_leaf(p, N_leaf, alpha) for p in leaf_probs]
    return float(np.mean(bounds))


def dm_ece_bound_asymptotic(
    max_var: float,
    E_k: float,
    N: int,
    C: int,
) -> float:
    """
    Asymptotic ECE bound (large N, no bias):
    ECE ≤ sqrt(C · max_c[p_c(1-p_c)] · E[k] / N)

    max_var : max over leaves and classes of p_c(1-p_c)
    E_k     : expected number of leaves under prior
    """
    return np.sqrt(C * max_var * E_k / np.asarray(N, float))


# ---------------------------------------------------------------------------
# Empirical ECE (binned calibration)
# ---------------------------------------------------------------------------

def empirical_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    ECE (max-class probability calibration).

    probs  : (n, C) predicted class probabilities
    labels : (n,)   integer true labels
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        acc_b  = correct[mask].mean()
        conf_b = conf[mask].mean()
        ece += mask.sum() / n * abs(conf_b - acc_b)
    return float(ece)


# ---------------------------------------------------------------------------
# Calibration simulation for DM tree
# ---------------------------------------------------------------------------

def calibration_sim(
    leaf_probs: list,
    N_total: int,
    alpha: float = 1.0,
    n_test: int = 500,
    rng: np.random.Generator = None,
) -> float:
    """
    Simulate empirical ECE for a DM-tree predictor.

    Generates N_total training observations and n_test test observations
    from the true model, computes DM posterior means, returns empirical ECE.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    k_true = len(leaf_probs)
    C = len(leaf_probs[0])
    n_leaf = max(1, N_total // k_true)

    # Train: DM posterior means per leaf
    dm_means = []
    for probs in leaf_probs:
        counts = rng.multinomial(n_leaf, probs).astype(float)
        dm_means.append((counts + alpha) / (counts.sum() + C * alpha))

    # Test: generate class labels and predicted probs
    all_probs, all_labels = [], []
    n_test_leaf = max(1, n_test // k_true)
    for j, probs in enumerate(leaf_probs):
        labels = rng.choice(C, size=n_test_leaf, p=probs)
        pred_probs = np.tile(dm_means[j], (n_test_leaf, 1))
        all_probs.append(pred_probs)
        all_labels.append(labels)

    all_probs  = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    return empirical_ece(all_probs, all_labels, n_bins=10)


# ---------------------------------------------------------------------------
# ECE comparison: prior-averaged bound Catalan vs Chipman
# ---------------------------------------------------------------------------

def prior_averaged_ece_bound(
    prior_pmf: np.ndarray,
    max_var: float,
    N_vals: np.ndarray,
    C: int,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Prior-averaged ECE bound: ECE ≤ sqrt(C · max_var · E[k] / N).

    prior_pmf : array (k_max,) prior PMF over k, index 0 = k=1
    max_var   : max_c p_c(1-p_c) over the true leaves
    N_vals    : array of N values to evaluate
    C         : number of classes
    """
    k_vals = np.arange(1, len(prior_pmf) + 1)
    E_k = float(np.dot(k_vals, prior_pmf))
    return dm_ece_bound_asymptotic(max_var, E_k, N_vals, C)
