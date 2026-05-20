"""
Catalan tree-size prior analysis for JBDT vs Chipman (1998) BDT.

Prior on number of leaves k:
  JBDT (Catalan-exponential): p(k) ∝ exp(-γ(k-1)) / S_k,  S_k = Catalan(k-1)
  Chipman (alpha-beta):        p(split at depth d) = α(1+d)^{-β}
  Geometric baseline:          p(k) ∝ r^{k-1},  r = exp(-γ)
"""

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Catalan helpers
# ---------------------------------------------------------------------------

def _log_catalan(k: int) -> float:
    """log S_k = log C(2(k-1), k-1) / k,  S_1=1."""
    n = k - 1
    if n <= 0:
        return 0.0
    return gammaln(2 * n + 1) - 2 * gammaln(n + 1) - np.log(n + 1)


def log_catalan_prior_unnorm(k: int, gamma: float) -> float:
    """log p(k) up to normalizing constant: -γ(k-1) - log S_k."""
    return -gamma * (k - 1) - _log_catalan(k)


# ---------------------------------------------------------------------------
# Catalan PMF (exact, normalized)
# ---------------------------------------------------------------------------

def catalan_pmf(k_max: int = 30, gamma: float = 1.0) -> np.ndarray:
    """
    Normalized PMF over k = 1, ..., k_max.
    p(k) ∝ exp(-γ(k-1)) / S_k.
    Returns array of length k_max (index 0 → k=1).
    """
    log_p = np.array([log_catalan_prior_unnorm(k, gamma) for k in range(1, k_max + 1)])
    log_p -= log_p.max()       # shift for numerical stability
    p = np.exp(log_p)
    p /= p.sum()
    return p


def catalan_moments(k_max: int = 30, gamma: float = 1.0) -> dict:
    """E[k], Var[k], mode, entropy for Catalan prior."""
    k_vals = np.arange(1, k_max + 1)
    p = catalan_pmf(k_max, gamma)
    mean = float(np.dot(k_vals, p))
    var  = float(np.dot(k_vals ** 2, p) - mean ** 2)
    mode = int(k_vals[np.argmax(p)])
    ent  = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    tail = float(1.0 - p[0])    # P(k >= 2)
    return dict(mean=mean, var=var, std=np.sqrt(var), mode=mode,
                entropy=ent, tail_k_ge2=tail, pmf=p)


# ---------------------------------------------------------------------------
# Chipman (1998) alpha-beta prior via recursive DP
# ---------------------------------------------------------------------------

def chipman_pmf(k_max: int = 30, alpha: float = 0.95, beta: float = 2.0,
                d_max: int = 20) -> np.ndarray:
    """
    PMF over k (number of leaves) induced by Chipman alpha-beta prior.

    P(node at depth d splits) = α / (1 + d)^β.

    Recursion: P[k, d] = probability that a subtree rooted at depth d has k leaves.
      P[1, d] = 1 - p_d                     (leaf)
      P[k, d] = p_d * Σ_{j=1}^{k-1} P[j, d+1] * P[k-j, d+1]  (internal node)

    Returns normalized PMF array of length k_max (index 0 → k=1).
    Depth capped at d_max (treated as forced leaf beyond).
    """
    # P[k, d]: k in 1..k_max, d in 0..d_max
    P = np.zeros((k_max + 1, d_max + 2))

    def fill(d: int):
        if d > d_max:
            P[1, d] = 1.0
            return
        if P[:, d].sum() > 0:
            return
        fill(d + 1)
        p_split = alpha / (1.0 + d) ** beta
        p_leaf  = 1.0 - p_split
        P[1, d] = p_leaf
        # convolution for k >= 2
        for k in range(2, k_max + 1):
            for j in range(1, k):
                P[k, d] += p_split * P[j, d + 1] * P[k - j, d + 1]

    fill(0)
    p = P[1:k_max + 1, 0].copy()
    p = np.maximum(p, 0.0)
    s = p.sum()
    if s > 0:
        p /= s
    return p


def chipman_moments(k_max: int = 30, alpha: float = 0.95,
                    beta: float = 2.0, d_max: int = 20) -> dict:
    """E[k], Var[k], mode, entropy for Chipman prior."""
    k_vals = np.arange(1, k_max + 1)
    p = chipman_pmf(k_max, alpha, beta, d_max)
    mean = float(np.dot(k_vals, p))
    var  = float(np.dot(k_vals ** 2, p) - mean ** 2)
    mode = int(k_vals[np.argmax(p)])
    ent  = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    return dict(mean=mean, var=var, std=np.sqrt(var), mode=mode,
                entropy=ent, pmf=p)


# ---------------------------------------------------------------------------
# Geometric baseline prior
# ---------------------------------------------------------------------------

def geometric_pmf(k_max: int = 30, gamma: float = 1.0) -> np.ndarray:
    """Geometric: p(k) ∝ r^{k-1}, r = exp(-γ)."""
    r = np.exp(-gamma)
    k_vals = np.arange(0, k_max)    # k-1 exponents
    p = r ** k_vals
    p /= p.sum()
    return p


# ---------------------------------------------------------------------------
# Tail probability
# ---------------------------------------------------------------------------

def tail_prob(pmf: np.ndarray, k0: int) -> float:
    """P(k >= k0).  pmf is indexed from k=1."""
    if k0 <= 1:
        return 1.0
    idx = k0 - 1    # k0-1 because index 0 = k=1
    if idx >= len(pmf):
        return 0.0
    return float(pmf[idx:].sum())


# ---------------------------------------------------------------------------
# Posterior concentration simulation
# ---------------------------------------------------------------------------

def posterior_concentration(
    prior_pmf: np.ndarray,
    N_vals: np.ndarray,
    k_true: int = 3,
    C: int = 2,
    alpha_dm: float = 1.0,
    n_reps: int = 50,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Simulate P(k = k_true | data, N) for a sequence of sample sizes N.

    True model: k_true leaves with informative class distributions.
    Leaf i has class-0 probability spread evenly from 0.1 to 0.9 (identifiable).

    For candidate k:
      k == k_true : exact true partition
      k  < k_true : greedily merge adjacent leaf pairs that maximize DM ML
      k  > k_true : keep true k_true leaves + (k−k_true) empty leaves
                     (empty leaves add 0 ML; prior penalizes extra leaves)

    Returns array of shape (len(N_vals),) with posterior probability at k_true,
    averaged over n_reps random datasets.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    k_max = len(prior_pmf)
    k_vals = np.arange(1, k_max + 1)

    # Informative leaf class probabilities (leaf i has different p)
    leaf_probs_true = []
    for i in range(k_true):
        p0 = 0.1 + 0.8 * i / max(k_true - 1, 1)
        leaf_probs_true.append(np.array([p0, 1.0 - p0]))

    def _dm_log_ml(counts: np.ndarray, alpha: float) -> float:
        a0 = alpha * len(counts)
        N_leaf = int(counts.sum())
        if N_leaf == 0:
            return 0.0
        return float(gammaln(a0) - gammaln(N_leaf + a0)
                     + np.sum(gammaln(counts + alpha) - gammaln(alpha)))

    def _tree_log_ml(leaf_data: list, k: int) -> float:
        """
        DM log marginal likelihood for a k-leaf tree given observed leaf data.
        k < k_true : greedy optimal merge of adjacent leaf pairs
        k == k_true: exact true partition
        k > k_true : random 50/50 splits of the largest leaf, repeated
                      (incurs the actual DM penalty ≈ (C-1)/2 * log(N_leaf) per split)
        """
        if k == k_true:
            return sum(_dm_log_ml(c, alpha_dm) for c in leaf_data)
        elif k < k_true:
            leaves = [c.copy() for c in leaf_data]
            while len(leaves) > k:
                best_gain, best_i = -np.inf, 0
                for idx in range(len(leaves) - 1):
                    merged = leaves[idx] + leaves[idx + 1]
                    gain = (_dm_log_ml(merged, alpha_dm)
                            - _dm_log_ml(leaves[idx], alpha_dm)
                            - _dm_log_ml(leaves[idx + 1], alpha_dm))
                    if gain > best_gain:
                        best_gain, best_i = gain, idx
                leaves = (leaves[:best_i]
                          + [leaves[best_i] + leaves[best_i + 1]]
                          + leaves[best_i + 2:])
            return sum(_dm_log_ml(lf, alpha_dm) for lf in leaves)
        else:
            # k > k_true: split the largest leaf k - k_true times
            leaves = [c.copy() for c in leaf_data]
            for _ in range(k - k_true):
                sizes = [int(lf.sum()) for lf in leaves]
                j = int(np.argmax(sizes))
                lf = leaves[j]
                # random 50/50 split of each class count
                sub1 = np.array(
                    [rng.binomial(int(cnt), 0.5) for cnt in lf],
                    dtype=float)
                sub2 = lf - sub1
                leaves = ([l for i, l in enumerate(leaves) if i != j]
                          + [sub1, sub2])
            return sum(_dm_log_ml(lf, alpha_dm) for lf in leaves)

    log_prior = np.log(prior_pmf + 1e-300)
    results = np.zeros(len(N_vals))

    for ni, N in enumerate(N_vals):
        prob_acc = 0.0
        for _ in range(n_reps):
            # Generate data from the true model
            n_base = max(1, N // k_true)
            leaf_data = []
            used = 0
            for i, probs in enumerate(leaf_probs_true):
                n_i = n_base if i < k_true - 1 else max(1, N - used)
                counts = rng.multinomial(n_i, probs).astype(float)
                leaf_data.append(counts)
                used += n_i

            log_liks = np.array([_tree_log_ml(leaf_data, k) for k in k_vals])
            log_post = log_prior + log_liks
            log_post -= log_post.max()
            post = np.exp(log_post)
            post /= post.sum()
            prob_acc += float(post[k_true - 1])

        results[ni] = prob_acc / n_reps

    return results


# ---------------------------------------------------------------------------
# Effective decay rate
# ---------------------------------------------------------------------------

def effective_decay_rate(gamma: float, k_max: int = 50) -> float:
    """
    Fit geometric decay rate r such that p(k) ≈ c * r^{k-1}.
    Uses ratio p(k+1)/p(k) averaged over k = 5..10.
    """
    log_p = np.array([log_catalan_prior_unnorm(k, gamma) for k in range(1, k_max + 1)])
    log_p -= log_p.max()
    ratios = np.exp(log_p[5:11] - log_p[4:10])   # p(k+1)/p(k) for k=5..10
    return float(np.mean(ratios))
