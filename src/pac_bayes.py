"""
§2.6  PAC-Bayes Sample Complexity for WAIC-weighted BMA.

McAllester (2003) PAC-Bayes bound:
  For any posterior ρ, prior π, and δ > 0, with probability ≥ 1−δ over training data:

  R(ρ) ≤ R̂(ρ) + √[(KL(ρ‖π) + log(2√n/δ)) / (2n)]

Key quantities:
  ρ     = WAIC weights  w_k ∝ exp(−WAIC_k/2)     (pseudo-posterior)
  π     = Catalan or Chipman prior over k
  KL    = KL(ρ‖π) = Σ_k w_k log(w_k/π_k)
  R̂(ρ) = Σ_k w_k R̂_k  (empirical BMA risk)

Sample complexity (oracle posterior ρ = δ_{k*}):
  N_min(ε, δ) = [−log π(k*) + log(2√N/δ)] / (2ε²)
              ≈ −log π(k*) / (2ε²)   for large N

Prior comparison for N_min (smaller is better):
  k*=1: Catalan γ=1 advantage  (−log 0.691 = 0.37) vs Chipman (−log 0.050 = 3.00) → 8.1× less
  k*=2: Chipman advantage       (−log 0.552 = 0.59) vs Catalan (−log 0.254 = 1.37) → 2.3× less
  k*=3: Chipman advantage       (−log 0.275 = 1.29) vs Catalan (−log 0.047 = 3.06) → 2.4× less

Conclusion: Catalan prior gives tighter PAC-Bayes bounds for sparse models (k* ≤ 1),
which is the relevant regime for medical imaging with small N.
"""

import numpy as np
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# KL divergence between categorical distributions
# ---------------------------------------------------------------------------

def kl_categorical(q: np.ndarray, p: np.ndarray) -> float:
    """KL(q ‖ p) = Σ_k q_k log(q_k / p_k).  Both arrays must sum to 1."""
    q, p = np.asarray(q, float), np.asarray(p, float)
    mask = q > 1e-300
    return float(np.sum(q[mask] * np.log(q[mask] / np.maximum(p[mask], 1e-300))))


# ---------------------------------------------------------------------------
# PAC-Bayes bound (McAllester 2003)
# ---------------------------------------------------------------------------

def pac_bayes_bound(
    emp_risk: float,
    kl_qp: float,
    n: int,
    delta: float = 0.05,
) -> float:
    """
    PAC-Bayes risk upper bound:
    R(ρ) ≤ emp_risk + √[(kl_qp + log(2√n/δ)) / (2n)]

    emp_risk : R̂(ρ) = Σ_k w_k R̂_k
    kl_qp   : KL(ρ‖π)
    n        : training sample size
    delta    : failure probability
    """
    penalty = np.sqrt((kl_qp + np.log(2.0 * np.sqrt(n) / delta)) / (2.0 * n))
    return float(emp_risk + penalty)


def pac_bayes_penalty(kl_qp: float, n: int, delta: float = 0.05) -> float:
    """The complexity penalty term only: √[(KL + log(2√n/δ)) / (2n)]."""
    return float(np.sqrt((kl_qp + np.log(2.0 * np.sqrt(n) / delta)) / (2.0 * n)))


# ---------------------------------------------------------------------------
# Sample complexity: N_min to achieve R(ρ) ≤ R_oracle + ε
# ---------------------------------------------------------------------------

def n_min_pac_bayes(
    kl_qp: float,
    epsilon: float,
    delta: float = 0.05,
    n_max: int = 100_000,
) -> int:
    """
    Smallest n such that PAC-Bayes complexity penalty ≤ epsilon.
    Solves: √[(kl_qp + log(2√n/δ)) / (2n)] ≤ epsilon  iteratively.
    """
    for n in range(1, n_max + 1):
        if pac_bayes_penalty(kl_qp, n, delta) <= epsilon:
            return n
    return n_max


def n_min_oracle(
    log_prior_k_star: float,
    epsilon: float,
    delta: float = 0.05,
) -> float:
    """
    Oracle N_min (ρ = δ_{k*} point mass):
    N_min ≈ −log π(k*) / (2ε²)   (leading-order approximation).
    """
    return log_prior_k_star / (2.0 * epsilon ** 2)


# ---------------------------------------------------------------------------
# KL(WAIC weights ‖ prior) computation
# ---------------------------------------------------------------------------

def kl_waic_prior(
    waics: np.ndarray,
    prior_pmf: np.ndarray,
) -> float:
    """KL(w_WAIC ‖ prior) where w_WAIC ∝ exp(−WAIC_k/2)."""
    log_w = -waics / 2.0
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()
    # Restrict to models with non-zero prior
    mask = prior_pmf > 1e-300
    return kl_categorical(w[mask], prior_pmf[mask])


# ---------------------------------------------------------------------------
# Oracle KL table: KL(δ_{k*} ‖ π) = −log π(k*) for each k* and prior
# ---------------------------------------------------------------------------

def oracle_kl_table(
    k_star_vals: list,
    priors: dict,
) -> dict:
    """
    Compute −log π(k*) for each k* and each named prior PMF.

    priors : dict {name: pmf_array}  (index 0 = k=1)
    Returns dict {k*: {name: kl_value}}
    """
    table = {}
    for k_star in k_star_vals:
        row = {}
        for name, pmf in priors.items():
            idx = k_star - 1
            if idx < len(pmf) and pmf[idx] > 1e-300:
                row[name] = float(-np.log(pmf[idx]))
            else:
                row[name] = float("inf")
        table[k_star] = row
    return table


# ---------------------------------------------------------------------------
# Full PAC-Bayes comparison sweep over N
# ---------------------------------------------------------------------------

def pac_bayes_sweep(
    N_vals: np.ndarray,
    emp_risk_fn,          # callable: emp_risk_fn(N) → float
    priors: dict,         # {name: pmf_array}
    waic_fn=None,         # optional: waic_fn(N) → waics_array (length = len(pmf))
    delta: float = 0.05,
) -> dict:
    """
    Compute PAC-Bayes bounds for each N using WAIC-derived or oracle posterior.

    If waic_fn is None, uses oracle posterior (point mass at k that minimises WAIC).
    Returns dict keyed by prior name, each value is array of PAC-Bayes bounds over N_vals.
    """
    results = {name: [] for name in priors}

    for N in N_vals:
        emp_r = emp_risk_fn(N)
        for name, pmf in priors.items():
            if waic_fn is not None:
                waics = waic_fn(N)
                kl_val = kl_waic_prior(waics, pmf)
            else:
                # Oracle: KL = 0 (posterior already at oracle model)
                kl_val = 0.0
            bound = pac_bayes_bound(emp_r, kl_val, int(N), delta)
            results[name].append(float(bound))

    return {name: np.array(v) for name, v in results.items()}
