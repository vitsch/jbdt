"""
§2.3  BMA Oracle Inequality for WAIC-weighted DM-leaf BDTs.

Main results:
  Jensen bound:  R_BMA ≤ Σ_k w_k R_k                        (Prop 2.3a)
  Excess risk:   R_BMA − R_{k*} ≤ (1−w_{k*}) · M            (Prop 2.3b)
  Convergence:   1−w_{k*} = O(exp(−N·Δ_WAIC/2))             (from §2.1)

WAIC weights: w_k ∝ exp(−WAIC_k / 2)
Oracle model: k* = argmin_k R_k  (best single-model KL risk)
"""

import numpy as np
from scipy.special import gammaln, polygamma

# ---------------------------------------------------------------------------
# DM leaf utilities
# ---------------------------------------------------------------------------

def _dm_log_ml(counts: np.ndarray, alpha: float) -> float:
    """Dirichlet-Multinomial log marginal likelihood for one leaf."""
    C = len(counts)
    a0 = alpha * C
    N = int(counts.sum())
    if N == 0:
        return 0.0
    return float(
        gammaln(a0) - gammaln(N + a0)
        + np.sum(gammaln(counts + alpha) - gammaln(alpha))
    )


def _dm_waic(counts: np.ndarray, alpha: float) -> float:
    """WAIC for a single DM leaf (analytical via trigamma)."""
    C = len(counts)
    a0 = alpha * C
    N = int(counts.sum())
    if N == 0:
        return 0.0
    post = counts + alpha
    post_n = N + a0
    lppd = float(np.sum(counts * np.log(post / post_n + 1e-300)))
    p_waic = float(np.sum(counts * (polygamma(1, post) + polygamma(1, post_n))))
    return -2.0 * (lppd - p_waic)


def dm_posterior_mean(counts: np.ndarray, alpha: float) -> np.ndarray:
    """DM posterior mean class probabilities."""
    a0 = alpha * len(counts)
    return (counts + alpha) / (counts.sum() + a0)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) = Σ_c p_c log(p_c / q_c).  q clipped at 1e-300."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    mask = p > 1e-15
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-300))))


# ---------------------------------------------------------------------------
# Oracle leaf assignment: same greedy-merge / random-split logic as §2.2
# ---------------------------------------------------------------------------

def oracle_assignment(
    true_leaf_data: list,
    k: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple:
    """
    Compute (leaf_data_k, true_to_model) for a k-leaf model.

    true_leaf_data : list of k_true count arrays (training data per true leaf)
    k              : candidate number of leaves
    alpha          : DM prior concentration per class

    Returns
    -------
    leaf_data_k    : list of k count arrays for the model's leaves
    true_to_model  : list of length k_true;  true_to_model[j] = model leaf index
                     that contains true leaf j (for routing test observations)
    """
    k_true = len(true_leaf_data)

    if k == k_true:
        return list(true_leaf_data), list(range(k_true))

    elif k < k_true:
        # Greedy merge: repeatedly merge the adjacent pair with highest combined ML
        leaves = [(c.copy(), [j]) for j, c in enumerate(true_leaf_data)]

        while len(leaves) > k:
            best_gain, best_i = -np.inf, 0
            for i in range(len(leaves) - 1):
                merged = leaves[i][0] + leaves[i + 1][0]
                gain = (_dm_log_ml(merged, alpha)
                        - _dm_log_ml(leaves[i][0], alpha)
                        - _dm_log_ml(leaves[i + 1][0], alpha))
                if gain > best_gain:
                    best_gain, best_i = gain, i
            mc = leaves[best_i][0] + leaves[best_i + 1][0]
            mi = leaves[best_i][1] + leaves[best_i + 1][1]
            leaves = leaves[:best_i] + [(mc, mi)] + leaves[best_i + 2:]

        leaf_data_k = [l[0] for l in leaves]
        true_to_model_dict: dict = {}
        for model_idx, (_, orig_indices) in enumerate(leaves):
            for oj in orig_indices:
                true_to_model_dict[oj] = model_idx
        return leaf_data_k, [true_to_model_dict[j] for j in range(k_true)]

    else:
        # Random 50/50 split of the largest leaf, k - k_true times
        leaves = [(c.copy(), j) for j, c in enumerate(true_leaf_data)]

        for _ in range(k - k_true):
            sizes = [int(l[0].sum()) for l in leaves]
            j = int(np.argmax(sizes))
            lf, orig = leaves[j]
            sub1 = np.array([rng.binomial(int(cnt), 0.5) for cnt in lf], float)
            sub2 = lf - sub1
            leaves = (
                [l for i, l in enumerate(leaves) if i != j]
                + [(sub1, orig), (sub2, orig)]
            )

        leaf_data_k = [l[0] for l in leaves]
        # Route true leaf j to the FIRST model leaf that came from it
        true_to_model_dict = {}
        for model_idx, (_, orig_j) in enumerate(leaves):
            if orig_j not in true_to_model_dict:
                true_to_model_dict[orig_j] = model_idx
        return leaf_data_k, [true_to_model_dict[j] for j in range(k_true)]


# ---------------------------------------------------------------------------
# WAIC and KL risk for a k-leaf model
# ---------------------------------------------------------------------------

def model_waic(leaf_data_k: list, alpha: float) -> float:
    """WAIC for the full k-leaf tree (sum over leaves)."""
    return sum(_dm_waic(c, alpha) for c in leaf_data_k)


def model_kl_risk(
    true_probs: list,
    leaf_data_k: list,
    true_to_model: list,
    alpha: float,
) -> float:
    """
    Average KL risk of model k over the k_true true leaves.

    For each true leaf j: compute KL(p_j || DM_posterior_mean(model_leaf(j)))
    Average over j = 1, ..., k_true.
    """
    k_true = len(true_probs)
    total_kl = 0.0
    for j in range(k_true):
        m_idx = true_to_model[j]
        pred = dm_posterior_mean(leaf_data_k[m_idx], alpha)
        total_kl += kl_divergence(np.asarray(true_probs[j]), pred)
    return total_kl / k_true


# ---------------------------------------------------------------------------
# BMA oracle inequality for one dataset
# ---------------------------------------------------------------------------

def bma_oracle_gap(
    true_leaf_data: list,
    true_probs: list,
    k_vals: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> dict:
    """
    For one training dataset, compute BMA oracle inequality quantities.

    Returns dict with:
      waic_k       : WAIC for each k
      w_k          : normalized WAIC weights
      R_k          : KL risk for each k-leaf model
      R_BMA        : KL risk of the BMA predictor
      R_oracle     : KL risk of oracle model k*
      jensen_bound : Σ_k w_k R_k  (upper bound on R_BMA by Jensen)
      excess_risk  : R_BMA - R_oracle
      theory_bound : (1 - w_{k*}) * max_k R_k  (excess risk bound)
      w_kstar      : weight on oracle model k*
      k_star       : index of oracle model
    """
    k_true = len(true_leaf_data)
    C = len(true_leaf_data[0])

    # Compute per-model WAIC, leaf data, and KL risk
    waics = []
    risks = []
    leaf_assigns = []

    for k in k_vals:
        ld, t2m = oracle_assignment(true_leaf_data, k, alpha, rng)
        waics.append(model_waic(ld, alpha))
        risks.append(model_kl_risk(true_probs, ld, t2m, alpha))
        leaf_assigns.append((ld, t2m))

    waics = np.array(waics)
    risks = np.array(risks)

    # WAIC weights (softmax of -WAIC/2)
    log_w = -waics / 2.0
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()

    # BMA posterior mean for each true leaf j
    k_true_local = len(true_probs)
    bma_preds = []
    for j in range(k_true_local):
        pred_j = np.zeros(C)
        for ki, k in enumerate(k_vals):
            ld, t2m = leaf_assigns[ki]
            pred_j += w[ki] * dm_posterior_mean(ld[t2m[j]], alpha)
        bma_preds.append(pred_j)

    R_BMA = float(np.mean([kl_divergence(np.asarray(true_probs[j]), bma_preds[j])
                            for j in range(k_true_local)]))

    k_star_idx = int(np.argmin(risks))
    R_oracle = float(risks[k_star_idx])
    jensen_bound = float(np.dot(w, risks))
    M = float(risks.max() - R_oracle)
    # Loose bound: (1-w_{k*}) * max_k R_k
    theory_bound = float((1.0 - w[k_star_idx]) * risks.max())
    # Tight bound: Jensen excess = Σ_k w_k R_k − R_oracle ≥ R_BMA − R_oracle
    jensen_excess = float(jensen_bound - R_oracle)

    return dict(
        waic_k=waics,
        w_k=w,
        R_k=risks,
        R_BMA=R_BMA,
        R_oracle=R_oracle,
        jensen_bound=jensen_bound,
        excess_risk=R_BMA - R_oracle,
        theory_bound=theory_bound,
        jensen_excess=jensen_excess,        # tight upper bound on excess_risk
        w_kstar=float(w[k_star_idx]),
        k_star=int(k_vals[k_star_idx]),
        k_waic_best=int(k_vals[np.argmin(waics)]),
    )


# ---------------------------------------------------------------------------
# Simulation: sweep N values
# ---------------------------------------------------------------------------

def bma_oracle_sweep(
    N_vals: np.ndarray,
    k_true: int = 3,
    C: int = 2,
    k_max: int = 8,
    alpha: float = 1.0,
    n_reps: int = 50,
    rng: np.random.Generator = None,
) -> dict:
    """
    Sweep N values, computing BMA oracle gap averaged over n_reps datasets.

    True leaf class probs: spread from 0.1 to 0.9 (identifiable).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    leaf_probs = [
        np.array([0.1 + 0.8 * j / max(k_true - 1, 1), 0.9 - 0.8 * j / max(k_true - 1, 1)])
        for j in range(k_true)
    ]
    k_vals = np.arange(1, k_max + 1)

    tracked_keys = ["w_kstar", "R_BMA", "R_oracle", "jensen_bound",
                    "excess_risk", "theory_bound", "jensen_excess"]
    out = {
        "N_vals": N_vals.tolist(),
        "k_vals": k_vals.tolist(),
        "k_waic_best_frac": [],
    }
    for key in tracked_keys:
        out[key] = []

    for N in N_vals:
        acc = {k: [] for k in tracked_keys}
        correct_waic = 0

        for _ in range(n_reps):
            n_base = max(1, N // k_true)
            train_data = []
            used = 0
            for i, probs in enumerate(leaf_probs):
                n_i = n_base if i < k_true - 1 else max(1, N - used)
                train_data.append(rng.multinomial(n_i, probs).astype(float))
                used += n_i

            res = bma_oracle_gap(train_data, leaf_probs, k_vals, alpha, rng)
            for key in tracked_keys:
                if key in res:
                    acc[key].append(res[key])
            if res["k_waic_best"] == k_true:
                correct_waic += 1

        for key in tracked_keys:
            out[key].append(float(np.mean(acc[key])) if acc[key] else 0.0)
        out["k_waic_best_frac"].append(correct_waic / n_reps)

    return out
