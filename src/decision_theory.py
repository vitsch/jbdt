"""
§2.7  Decision-Theoretic Analysis of JBDT under Asymmetric Loss.

Setup: binary classification (C=2), classes 0 (healthy) and 1 (OA).
Loss matrix:
  L(predict 1, true 0) = C_FP  (false positive — unnecessary treatment)
  L(predict 0, true 1) = C_FN  (false negative — missed OA)
  L(correct)            = 0

Key results:
  1. Optimal threshold:  τ* = C_FP / (C_FP + C_FN)
     Proof: minimise E[L | x] = p(y=1|x)·C_FN·1(d=0) + p(y=0|x)·C_FP·1(d=1)
            → predict 1 iff p(y=1|x) ≥ τ*.

  2. Cost saving from τ* (asymptotic, large N):
     ΔEC = EC(τ=0.5) − EC(τ*)
         = (1/k) Σ_{i: p_i ∈ (τ*, 0.5)} [C_FN·p_i − C_FP·(1−p_i)]   ≥ 0

  3. Miscalibration excess cost bound:
     ΔEC_miscal ≤ (C_FP + C_FN) · ECE_bound
     where ECE_bound = sqrt(C · E[k] · max_var / N)  (§2.5).

  4. Catalan/Chipman ratio: sqrt(E_k_Cat/E_k_Chip) = 0.740  → 26% lower bound.

  5. Double penalty (over-confident model):
     Even at τ*, a miscalibrated model with p̂(leaf_i) < τ* for twilight-zone
     leaves (p_true ∈ (τ*, 0.5)) incurs the SAME cost as argmax — calibration
     error completely erases the threshold-optimisation benefit.

OA context:  r = C_FN/C_FP = 2,  τ* = 1/3.
  Twilight zone: leaves with P(OA|x) ∈ (1/3, 1/2) — mild-OA patients
  incorrectly classified as healthy by argmax but correctly caught by τ*.
"""

import numpy as np
from scipy.special import expit


# ---------------------------------------------------------------------------
# Optimal threshold and expected cost
# ---------------------------------------------------------------------------

def optimal_threshold(c_fp: float, c_fn: float) -> float:
    """Bayes-optimal classification threshold τ* = C_FP / (C_FP + C_FN)."""
    return c_fp / (c_fp + c_fn)


def expected_cost(
    y_true, prob_pos, threshold: float, c_fp: float, c_fn: float,
) -> float:
    """EC = (C_FP·FP + C_FN·FN) / N."""
    y = np.asarray(y_true, int)
    p = np.asarray(prob_pos, float)
    pred = (p >= threshold).astype(int)
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return (c_fp * fp + c_fn * fn) / len(y)


def cost_curve(
    y_true, prob_pos, thresholds, c_fp: float, c_fn: float,
) -> np.ndarray:
    """EC(τ) for an array of thresholds."""
    return np.array(
        [expected_cost(y_true, prob_pos, t, c_fp, c_fn) for t in thresholds]
    )


def cost_saving(y_true, prob_pos, c_fp: float, c_fn: float) -> float:
    """EC(τ=0.5) − EC(τ*): cost saving from using the optimal threshold."""
    tau = optimal_threshold(c_fp, c_fn)
    return (
        expected_cost(y_true, prob_pos, 0.5, c_fp, c_fn)
        - expected_cost(y_true, prob_pos, tau, c_fp, c_fn)
    )


# ---------------------------------------------------------------------------
# DM posterior predictive and miscalibration
# ---------------------------------------------------------------------------

def dm_posterior_prob(n_pos: int, n_total: int, alpha: float = 1.0) -> float:
    """P(y=1 | data) = (n_1 + α) / (n + 2α) under DM with symmetric Dirichlet(α)."""
    return (n_pos + alpha) / (n_total + 2.0 * alpha)


def overcalibrated_prob(p_hat: float, sharpness: float = 3.0) -> float:
    """Over-confident transformation: expit(sharpness · logit(p_hat)).
    Pushes probabilities away from 0.5 — models RF/MLP overconfidence."""
    eps = 1e-9
    logit_p = np.log((p_hat + eps) / (1.0 - p_hat + eps))
    return float(expit(sharpness * logit_p))


# ---------------------------------------------------------------------------
# Calibration-cost bounds (§2.5 → §2.7 bridge)
# ---------------------------------------------------------------------------

def dm_ece_bound(N, E_k: float, max_var: float = 0.25, C: int = 2) -> np.ndarray:
    """ECE ≤ sqrt(C · E[k] · max_var / N)  (§2.5 asymptotic bound)."""
    return np.sqrt(C * E_k * max_var / np.asarray(N, float))


def miscalib_cost_bound(ece_bound, c_fp: float, c_fn: float) -> np.ndarray:
    """ΔEC_miscal ≤ (C_FP + C_FN) · ECE_bound."""
    return (c_fp + c_fn) * np.asarray(ece_bound, float)


# ---------------------------------------------------------------------------
# Analytical (large-N) expected cost for a k-leaf tree
# ---------------------------------------------------------------------------

def theoretical_ec(
    p_true_leaves, threshold: float, c_fp: float, c_fn: float,
) -> float:
    """
    Asymptotic EC (p̂_i → p_true_i):
    EC = (1/k) Σ_i [C_FP(1−p_i)·1(p_i≥τ) + C_FN·p_i·1(p_i<τ)]
    """
    total = 0.0
    for p in p_true_leaves:
        if p >= threshold:
            total += c_fp * (1.0 - p)
        else:
            total += c_fn * p
    return total / len(p_true_leaves)


def theoretical_ec_mismatched(
    p_true_leaves,
    p_hat_leaves,
    threshold: float,
    c_fp: float,
    c_fn: float,
) -> float:
    """
    Asymptotic EC when the decision rule uses p_hat but costs depend on p_true.
    Needed for miscalibrated models: p_hat ≠ p_true.

    EC = (1/k) Σ_i [C_FP(1−p_true_i)·1(p_hat_i≥τ) + C_FN·p_true_i·1(p_hat_i<τ)]
    """
    total = 0.0
    for pt, ph in zip(p_true_leaves, p_hat_leaves):
        if ph >= threshold:
            total += c_fp * (1.0 - pt)
        else:
            total += c_fn * pt
    return total / len(p_true_leaves)


def theoretical_cost_saving(
    p_true_leaves, c_fp: float, c_fn: float,
) -> float:
    """ΔEC = EC(τ=0.5) − EC(τ*) for large N.  Counts twilight-zone leaves."""
    tau = optimal_threshold(c_fp, c_fn)
    return (
        theoretical_ec(p_true_leaves, 0.5, c_fp, c_fn)
        - theoretical_ec(p_true_leaves, tau, c_fp, c_fn)
    )


# ---------------------------------------------------------------------------
# Simulation: four-method cost comparison (DM vs OC, τ* vs argmax)
# ---------------------------------------------------------------------------

def simulate_cost_comparison(
    p_true_leaves,
    N_leaf: int,
    c_fp: float,
    c_fn: float,
    alpha: float = 1.0,
    sharpness: float = 3.0,
    n_reps: int = 2000,
    rng=None,
) -> dict:
    """
    Simulate EC for four decision strategies over n_reps datasets:
      DM + τ*     — DM posterior at optimal threshold  (calibrated + correct rule)
      DM + τ=0.5  — DM posterior at argmax             (calibrated + wrong rule)
      OC + τ*     — over-confident at optimal threshold (miscalibrated + correct rule)
      OC + τ=0.5  — over-confident at argmax            (miscalibrated + wrong rule)

    p_true_leaves : list of P(y=1) per leaf
    N_leaf        : patients per leaf (N_total = N_leaf × k)
    sharpness     : expit(sharpness·logit(·)) over-confidence parameter
    """
    if rng is None:
        rng = np.random.default_rng(0)
    tau_star = optimal_threshold(c_fp, c_fn)
    k = len(p_true_leaves)
    N_total = N_leaf * k

    ec_dm_opt, ec_dm_arg = [], []
    ec_oc_opt, ec_oc_arg = [], []

    for _ in range(n_reps):
        y_true = np.zeros(N_total, int)
        p_hat_dm = np.zeros(N_total, float)
        p_hat_oc = np.zeros(N_total, float)

        for i, p in enumerate(p_true_leaves):
            sl = slice(i * N_leaf, (i + 1) * N_leaf)
            n1 = int(rng.binomial(N_leaf, p))
            y_leaf = np.concatenate([np.ones(n1, int), np.zeros(N_leaf - n1, int)])
            rng.shuffle(y_leaf)
            y_true[sl] = y_leaf

            p_dm = dm_posterior_prob(n1, N_leaf, alpha)
            p_hat_dm[sl] = p_dm
            p_hat_oc[sl] = overcalibrated_prob(p_dm, sharpness)

        ec_dm_opt.append(expected_cost(y_true, p_hat_dm, tau_star, c_fp, c_fn))
        ec_dm_arg.append(expected_cost(y_true, p_hat_dm, 0.5, c_fp, c_fn))
        ec_oc_opt.append(expected_cost(y_true, p_hat_oc, tau_star, c_fp, c_fn))
        ec_oc_arg.append(expected_cost(y_true, p_hat_oc, 0.5, c_fp, c_fn))

    def s(arr):
        return dict(mean=float(np.mean(arr)), std=float(np.std(arr)))

    return dict(
        dm_tau_star=s(ec_dm_opt),
        dm_argmax=s(ec_dm_arg),
        oc_tau_star=s(ec_oc_opt),
        oc_argmax=s(ec_oc_arg),
        saving_dm=float(np.mean(ec_dm_arg) - np.mean(ec_dm_opt)),
        saving_oc=float(np.mean(ec_oc_arg) - np.mean(ec_oc_opt)),
        calib_advantage=float(np.mean(ec_oc_opt) - np.mean(ec_dm_opt)),
    )


def simulate_cost_sweep_N(
    N_leaf_vals,
    p_true_leaves,
    c_fp: float,
    c_fn: float,
    alpha: float = 1.0,
    sharpness: float = 3.0,
    n_reps: int = 2000,
    rng=None,
) -> list:
    """Cost comparison for a range of N_leaf values."""
    out = []
    for N_leaf in N_leaf_vals:
        res = simulate_cost_comparison(
            p_true_leaves, int(N_leaf), c_fp, c_fn, alpha, sharpness, n_reps, rng,
        )
        out.append(dict(
            N_leaf=int(N_leaf),
            N_total=int(N_leaf * len(p_true_leaves)),
            **res,
        ))
    return out


def simulate_cost_sweep_r(
    r_vals,
    p_true_leaves,
    N_leaf: int,
    c_fp_base: float = 1.0,
    alpha: float = 1.0,
    sharpness: float = 3.0,
    n_reps: int = 2000,
    rng=None,
) -> list:
    """Cost comparison for a range of C_FN/C_FP ratios r."""
    out = []
    for r in r_vals:
        c_fn = c_fp_base * r
        res = simulate_cost_comparison(
            p_true_leaves, N_leaf, c_fp_base, c_fn, alpha, sharpness, n_reps, rng,
        )
        tau = optimal_threshold(c_fp_base, c_fn)
        theo_saving = theoretical_cost_saving(p_true_leaves, c_fp_base, c_fn)
        out.append(dict(r=float(r), tau_star=float(tau),
                        theo_saving=theo_saving, **res))
    return out
