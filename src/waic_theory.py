"""
Analytical WAIC and LOO-CV for Dirichlet-Multinomial Leaf BDTs
==============================================================
Implements §2.1 of c_sum_bdt_1.md:
  "WAIC Consistency for Dirichlet-Multinomial Leaf BDTs"

Three building blocks:
  A. Closed-form LOO-CV  — exact Bayesian leave-one-out (Polya-urn formula)
  B. Analytical WAIC     — uses Dirichlet posterior variance via trigamma
  C. Gap theorem         — |WAIC - LOO| = O(1) total, O(1/N) per observation
  D. N_min formula       — minimum N for correct model selection (prob >= 1-delta)

All functions operate on integer class-count vectors and a Dirichlet prior.
No MCMC, no sklearn — pure numpy/scipy.

Reference:
  Watanabe S. (2010). Asymptotic Equivalence of Bayes Cross Validation
  and Widely Applicable Information Criterion in Singular Learning Theory.
  JMLR 11, 3571-3594.

  Gelman A., Hwang J., Vehtari A. (2014). Understanding predictive
  information criteria for Bayesian models. Statistics and Computing 24(6).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, polygamma
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _trigamma(x: np.ndarray) -> np.ndarray:
    """Trigamma function ψ'(x) = d²/dx² log Γ(x).  Vectorised."""
    return polygamma(1, x)


# ---------------------------------------------------------------------------
# A.  Exact LOO-CV for a Dirichlet-Multinomial leaf
# ---------------------------------------------------------------------------

def dm_loocv_score(counts: np.ndarray, alpha: np.ndarray) -> float:
    """
    Exact Bayesian LOO-CV log-score for a single Dirichlet-Multinomial leaf.

    For observation i of class c, the leave-one-out predictive probability is
    (Polya urn / posterior predictive with one draw removed):

        p_LOO(y_i = c) = (n_c - 1 + alpha_c) / (N - 1 + alpha_0)

    Summing over all N observations:

        LOO_score = sum_c  n_c * log((n_c - 1 + alpha_c) / (N - 1 + alpha_0))

    When n_c = 0 the class contributes 0 (nothing to leave out).

    Parameters
    ----------
    counts : array (C,)  integer class counts, sum = N >= 2
    alpha  : array (C,)  Dirichlet prior, all > 0

    Returns
    -------
    float  LOO log-score (higher is better; negate and multiply by -2 for deviance)
    """
    counts = np.asarray(counts, dtype=float)
    alpha  = np.asarray(alpha,  dtype=float)
    N   = counts.sum()
    a0  = alpha.sum()
    mask = counts > 0
    score = np.sum(
        counts[mask] * (
            np.log(counts[mask] - 1.0 + alpha[mask]) - np.log(N - 1.0 + a0)
        )
    )
    return float(score)


def dm_loocv_deviance(counts: np.ndarray, alpha: np.ndarray) -> float:
    """LOO-CV deviance = -2 * LOO_score (lower is better)."""
    return -2.0 * dm_loocv_score(counts, alpha)


# ---------------------------------------------------------------------------
# B.  Analytical WAIC for a Dirichlet-Multinomial leaf
# ---------------------------------------------------------------------------

def dm_waic_components(counts: np.ndarray, alpha: np.ndarray
                        ) -> tuple[float, float, float]:
    """
    Analytical WAIC components for a Dirichlet-Multinomial leaf.

    Uses the *exact* Dirichlet posterior Dirichlet(alpha + counts) to compute:

        lppd     = sum_c n_c * log(p_hat_c)
                   where p_hat_c = (n_c + alpha_c) / (N + alpha_0)
                   [= E_{theta | data}[theta_c]]

        p_waic   = sum_c n_c * [psi'(n_c + alpha_c) + psi'(N + alpha_0)]
                   [= sum_i Var_{theta | data}[log p(y_i | theta)]]
                   where psi' is the trigamma function

        WAIC     = -2 * (lppd - p_waic)

    The variance term uses the identity:
        Var_{theta ~ Dirichlet(a)}[log theta_c] = psi'(a_c) + psi'(a_0)

    Parameters
    ----------
    counts, alpha  same as dm_loocv_score

    Returns
    -------
    (lppd, p_waic, waic_deviance)  all floats
    """
    counts = np.asarray(counts, dtype=float)
    alpha  = np.asarray(alpha,  dtype=float)
    N      = counts.sum()
    a0     = alpha.sum()

    post   = counts + alpha          # posterior parameters
    post_n = N + a0                  # posterior sum

    # lppd: log of posterior predictive mean for each observation
    p_hat = post / post_n
    lppd  = float(np.sum(counts * np.log(p_hat + 1e-300)))

    # p_waic: Var[log theta_c | data] = psi'(post_c) + psi'(post_n)
    # Summed over N observations (n_c observations of class c each contribute)
    p_waic = float(np.sum(counts * (_trigamma(post) + _trigamma(post_n))))

    waic = -2.0 * (lppd - p_waic)
    return lppd, p_waic, waic


def dm_waic_score(counts: np.ndarray, alpha: np.ndarray) -> float:
    """WAIC log-score = lppd - p_waic (higher is better)."""
    lppd, p_waic, _ = dm_waic_components(counts, alpha)
    return lppd - p_waic


def dm_waic_deviance(counts: np.ndarray, alpha: np.ndarray) -> float:
    """WAIC deviance = -2*(lppd - p_waic) (lower is better)."""
    _, _, waic = dm_waic_components(counts, alpha)
    return waic


# ---------------------------------------------------------------------------
# C.  Gap theorem:  |WAIC - LOO| analysis
# ---------------------------------------------------------------------------

def waic_loo_gap(counts: np.ndarray, alpha: np.ndarray) -> dict:
    """
    Compute the WAIC-LOO gap and the analytical O(1/N) bound.

    Theorem (from Taylor expansion to second order):
        LOO_deviance - WAIC_deviance
          = 2 * sum_i [log p_LOO(y_i) - log p_hat(y_i) + Var[log p(y_i|theta)]]
          ≈ -2 * sum_c n_c * (2 / (N + alpha_0)) + O(1/N^2)
          ≈ -4 * N / (N + alpha_0)
          → -4   as N → infinity

    So the TOTAL gap is O(1) (a constant ≈ -4 for large N).
    The PER-OBSERVATION gap is O(1/N) → 0 as N → infinity.

    For model COMPARISON the constant offset cancels:
        [LOO(M1) - LOO(M2)] = [WAIC(M1) - WAIC(M2)] + O(1/N)

    Returns
    -------
    dict with keys:
      'waic_dev'     : WAIC deviance
      'loo_dev'      : LOO-CV deviance
      'gap_total'    : LOO_dev - WAIC_dev  (exact)
      'gap_per_obs'  : gap_total / N
      'gap_bound'    : analytical O(1/N) bound on per-obs gap
      'gap_asymptote': limiting value -4N/(N+alpha_0)
    """
    counts = np.asarray(counts, dtype=float)
    alpha  = np.asarray(alpha,  dtype=float)
    N   = counts.sum()
    a0  = alpha.sum()

    waic_dev = dm_waic_deviance(counts, alpha)
    loo_dev  = dm_loocv_deviance(counts, alpha)

    gap_total   = float(loo_dev - waic_dev)
    gap_per_obs = gap_total / N if N > 0 else 0.0

    # Analytical O(1/N) bound: second-order term from Taylor expansion
    post   = counts + alpha
    post_n = N + a0
    # Leading correction: sum_c n_c / post_c^2  (from second-order Taylor)
    bound_per_obs = float(2.0 * np.sum(counts / post**2) / N)

    # Asymptotic formula
    asymptote = -4.0 * N / post_n

    return {
        "waic_dev":      waic_dev,
        "loo_dev":       loo_dev,
        "gap_total":     gap_total,
        "gap_per_obs":   gap_per_obs,
        "gap_bound":     bound_per_obs,
        "gap_asymptote": asymptote,
    }


def gap_vs_N(p_true: np.ndarray, alpha_strength: float = 1.0,
             N_values: list[int] | None = None,
             rng: np.random.Generator | None = None,
             n_reps: int = 200) -> list[dict]:
    """
    Simulate WAIC-LOO gap for increasing N.

    For each N in N_values:
      1. Draw counts from Multinomial(N, p_true)
      2. Compute exact WAIC and LOO
      3. Record gap_total and gap_per_obs

    Returns list of dicts (one per N) with mean and std of gap.
    """
    if N_values is None:
        N_values = [10, 20, 40, 60, 100, 150, 200, 400, 800, 1600]
    if rng is None:
        rng = np.random.default_rng(42)

    C     = len(p_true)
    alpha = np.full(C, alpha_strength / C)
    rows  = []

    for N in N_values:
        gaps_tot = []
        gaps_per = []
        for _ in range(n_reps):
            counts = rng.multinomial(N, p_true).astype(float)
            while counts.sum() < 2:          # need at least 2 observations
                counts = rng.multinomial(N, p_true).astype(float)
            g = waic_loo_gap(counts, alpha)
            gaps_tot.append(g["gap_total"])
            gaps_per.append(g["gap_per_obs"])
        rows.append({
            "N":             N,
            "gap_mean":      float(np.mean(gaps_tot)),
            "gap_std":       float(np.std(gaps_tot)),
            "gap_per_mean":  float(np.mean(gaps_per)),
            "gap_per_std":   float(np.std(gaps_per)),
            "asymptote":     -4.0 * N / (N + alpha_strength),
        })
    return rows


# ---------------------------------------------------------------------------
# D.  N_min for correct model selection
# ---------------------------------------------------------------------------

def n_min_formula(delta: float, effect_per_obs: float,
                  sigma_per_obs: float) -> float:
    """
    Minimum N for WAIC to select the true model with probability >= 1-delta.

    Derivation (CLT argument):
      Let D_i = WAIC_i(M_wrong) - WAIC_i(M_true)  (per-obs deviance difference)
      Under H_true: E[D_i] = delta > 0, Var[D_i] = sigma^2

      WAIC correctly selects M_true when sum(D_i) > 0.
      By CLT: sum(D_i) ~ N(N*delta, N*sigma^2)

      P(correct) = P(sum(D_i) > 0)
                 = Phi(delta * sqrt(N) / sigma)

      Invert for P(correct) >= 1 - delta_err:
        N_min = (z_{1-delta_err} * sigma / delta)^2

    Parameters
    ----------
    delta          : acceptable error probability (e.g. 0.05)
    effect_per_obs : expected per-obs WAIC difference (M_wrong - M_true), > 0
    sigma_per_obs  : std of per-obs WAIC difference across observations

    Returns
    -------
    N_min as float
    """
    z = norm.ppf(1.0 - delta)
    return float((z * sigma_per_obs / effect_per_obs) ** 2)


def simulate_model_selection(
    p_true_M1: np.ndarray,   # true class probs for correct model
    p_true_M2: np.ndarray,   # "true" class probs for wrong model (misspecified)
    alpha_strength: float = 1.0,
    N_values: list[int] | None = None,
    n_reps: int = 500,
    delta: float = 0.05,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """
    Empirically determine N_min for WAIC to prefer M1 (correct) over M2 (wrong).

    For each N, generate n_reps datasets from p_true_M1, compute WAIC for both
    models, and record the fraction of times M1 is selected (lower WAIC).

    The analytical N_min formula is also computed for comparison.

    Parameters
    ----------
    p_true_M1  : true class distribution (generates the data)
    p_true_M2  : misspecified class distribution for the wrong model
    alpha_strength : total Dirichlet prior mass
    N_values   : list of sample sizes to try
    n_reps     : replications per N
    delta      : target error rate for P(correct) >= 1-delta
    rng        : random generator

    Returns
    -------
    list of dicts, one per N, with keys:
      'N', 'p_correct', 'waic_diff_mean', 'waic_diff_std', 'n_min_formula'
    """
    if N_values is None:
        N_values = [10, 20, 30, 40, 60, 80, 100, 150, 200, 300, 500, 1000]
    if rng is None:
        rng = np.random.default_rng(42)

    C1 = len(p_true_M1)
    C2 = len(p_true_M2)
    alpha1 = np.full(C1, alpha_strength / C1)
    alpha2 = np.full(C2, alpha_strength / C2)

    rows = []
    for N in N_values:
        n_correct   = 0
        diffs       = []

        for _ in range(n_reps):
            # Draw data from true model
            y = rng.choice(C1, size=N, p=p_true_M1)

            # Counts under M1 (correct model)
            cnt1 = np.bincount(y, minlength=C1).astype(float)
            # Counts under M2 (wrong model — project labels if C2 < C1)
            y2   = y % C2
            cnt2 = np.bincount(y2, minlength=C2).astype(float)

            w1 = dm_waic_deviance(cnt1, alpha1)
            w2 = dm_waic_deviance(cnt2, alpha2)

            diff = w1 - w2          # negative → M1 wins (lower WAIC)
            diffs.append(diff)
            if diff < 0:
                n_correct += 1

        diffs = np.array(diffs)
        p_correct = n_correct / n_reps

        # Per-obs effect size and sigma for N_min formula
        effect = -diffs.mean() / N     # positive if M1 wins on average
        sigma  = diffs.std() / N

        n_min = (n_min_formula(delta, max(effect, 1e-10), max(sigma, 1e-10))
                 if effect > 0 else float("inf"))

        rows.append({
            "N":               N,
            "p_correct":       p_correct,
            "waic_diff_mean":  float(diffs.mean()),
            "waic_diff_std":   float(diffs.std()),
            "effect_per_obs":  float(effect),
            "sigma_per_obs":   float(sigma),
            "n_min_formula":   n_min,
        })

    return rows


# ---------------------------------------------------------------------------
# E.  Multi-leaf (2-split) partition WAIC and LOO
# ---------------------------------------------------------------------------

def _leaf_counts(X: np.ndarray, y: np.ndarray, feat: int,
                 thresh: float, C: int) -> tuple[np.ndarray, np.ndarray]:
    """Split X on feature feat<=thresh; return (counts_left, counts_right)."""
    mask  = X[:, feat] <= thresh
    c_l   = np.bincount(y[mask],  minlength=C).astype(float)
    c_r   = np.bincount(y[~mask], minlength=C).astype(float)
    return c_l, c_r


def partition_waic_deviance(X: np.ndarray, y: np.ndarray,
                             feat: int, thresh: float,
                             alpha: np.ndarray) -> float:
    """WAIC deviance for a 2-leaf partition (sum over both leaves)."""
    C = len(alpha)
    c_l, c_r = _leaf_counts(X, y, feat, thresh, C)
    w = 0.0
    for cnt in (c_l, c_r):
        if cnt.sum() >= 2:
            w += dm_waic_deviance(cnt, alpha)
    return w


def partition_loocv_deviance(X: np.ndarray, y: np.ndarray,
                              feat: int, thresh: float,
                              alpha: np.ndarray) -> float:
    """LOO-CV deviance for a 2-leaf partition (sum over both leaves)."""
    C = len(alpha)
    c_l, c_r = _leaf_counts(X, y, feat, thresh, C)
    d = 0.0
    for cnt in (c_l, c_r):
        if cnt.sum() >= 2:
            d += dm_loocv_deviance(cnt, alpha)
    return d


# ---------------------------------------------------------------------------
# F.  Closed-form log marginal likelihood (for reference)
# ---------------------------------------------------------------------------

def dm_log_marginal(counts: np.ndarray, alpha: np.ndarray) -> float:
    """
    Exact log marginal likelihood for a Dirichlet-Multinomial leaf:
        log p(counts | alpha) = log Gamma(a0) - log Gamma(a0+N)
                              + sum_c [log Gamma(alpha_c + n_c) - log Gamma(alpha_c)]
    """
    counts = np.asarray(counts, dtype=float)
    alpha  = np.asarray(alpha,  dtype=float)
    a0 = alpha.sum()
    N  = counts.sum()
    return float(
        gammaln(a0) - gammaln(a0 + N)
        + np.sum(gammaln(alpha + counts) - gammaln(alpha))
    )
