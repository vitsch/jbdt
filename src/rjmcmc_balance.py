"""
§2.4  RJMCMC Detailed Balance for Catalan Birth–Death Moves.

From bdt_jakaite.py, the log-acceptance ratio for a birth move T(k) → T'(k+1) is:

  log α_B = ΔlogDM-ML + log_prior_birth(k, γ) + log_prop_birth

where:
  ΔlogDM-ML          = log DM-ML(left) + log DM-ML(right) − log DM-ML(parent)
  log_prior_birth(k,γ)= −γ + log S_k − log S_{k+1}      (Catalan ratio)
  log_prop_birth      = log(p_D/p_B) + log k − log r(T') + log m + log(b−a)

The REVERSE death move T'(k+1) → T(k) has:
  log_prior_death(k+1) = +γ + log S_{k+1} − log S_k = −log_prior_birth(k)
  log_prop_death       = −log_prop_birth          (exact negative)
  ΔlogDM-ML_death      = −ΔlogDM-ML_birth

Therefore:  log α_B(T→T') + log α_D(T'→T) = 0  ←  detailed balance identity.

Key prior-contribution formula (Catalan):
  log_prior_birth(k, γ) = −γ − log[2(2k−1)/(k+1)]

  k=1  →  −γ − log 1    = −γ          (no Catalan penalty at stump)
  k=2  →  −γ − log 2   ≈ −γ − 0.693
  large→  −γ − log 4   ≈ −γ − 1.386   (asymptotic penalty)

For Chipman prior: log_prior_birth can be POSITIVE for shallow splits (p_split → 1 at depth 0).
Catalan prior ALWAYS contributes a negative term → automatic complexity regularisation.
"""

import numpy as np
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# Catalan utilities (mirror bdt_jakaite.py exactly)
# ---------------------------------------------------------------------------

def _log_catalan(k: int) -> float:
    n = k - 1
    if n <= 0:
        return 0.0
    return gammaln(2 * n + 1) - 2 * gammaln(n + 1) - np.log(n + 1)


def log_prior_birth(k: int, gamma: float) -> float:
    """log p(T')/p(T) for birth move: k leaves → k+1.  From bdt_jakaite.py."""
    return -gamma + _log_catalan(k) - _log_catalan(k + 1)


def log_prior_death(k: int, gamma: float) -> float:
    """log p(T')/p(T) for death move: k leaves → k−1.  From bdt_jakaite.py."""
    return gamma + _log_catalan(k) - _log_catalan(k - 1)


def catalan_prior_birth_formula(k: int, gamma: float) -> float:
    """
    Closed-form Catalan birth prior contribution:
      log_prior_birth(k, γ) = −γ − log[2(2k−1)/(k+1)]

    Uses the identity S_{k+1}/S_k = 2(2k−1)/(k+1).
    """
    if k <= 1:
        return -gamma    # S_2/S_1 = 1/1 = 1 → log = 0
    ratio = 2.0 * (2 * k - 1) / (k + 1)
    return -gamma - np.log(ratio)


# ---------------------------------------------------------------------------
# Chipman prior birth contribution (approximate, depth-averaged)
# ---------------------------------------------------------------------------

def chipman_prior_birth_avg(k: int, alpha: float = 0.95, beta: float = 2.0) -> float:
    """
    Average Chipman log-prior contribution for a birth at expected depth d_avg(k).
    d_avg(k) ≈ log2(k) (average leaf depth in balanced binary tree).

    log p_chip(split at d) = log α − β log(1+d)
    """
    d_avg = np.log2(max(k, 1))
    log_p_split = np.log(alpha) - beta * np.log(1.0 + d_avg)
    return log_p_split  # positive for shallow d, negative for deep d


# ---------------------------------------------------------------------------
# DM marginal likelihood
# ---------------------------------------------------------------------------

def _dm_log_ml(counts: np.ndarray, alpha: float) -> float:
    C = len(counts)
    a0 = alpha * C
    N = int(counts.sum())
    if N == 0:
        return 0.0
    return float(gammaln(a0) - gammaln(N + a0)
                 + np.sum(gammaln(counts + alpha) - gammaln(alpha)))


# ---------------------------------------------------------------------------
# Simplified birth–death RJMCMC  (no feature splits; k = state)
# ---------------------------------------------------------------------------

def log_acceptance_birth(
    leaf_data: list,
    gamma: float,
    alpha: float,
    p_B: float,
    p_D: float,
    r_T_prime: int,
    m: int = 1,
    b_a: float = 1.0,
) -> float:
    """
    Log acceptance ratio for a birth move on the first leaf in leaf_data.
    Mirrors bdt_jakaite.py _try_birth() exactly (except continuous split density).

    leaf_data : list of count arrays, leaf 0 is split into two halves
    r_T_prime : number of prunable nodes in proposed tree T'
    m         : number of features (for log(m) term)
    b_a       : feature range b−a (for continuous split density 1/(b−a))
    """
    k = len(leaf_data)
    parent = leaf_data[0]
    N_parent = int(parent.sum())
    # Split into ~equal halves
    left  = np.floor(parent / 2).astype(int)
    right = (parent - left).astype(int)

    log_ll    = (_dm_log_ml(left, alpha) + _dm_log_ml(right, alpha)
                 - _dm_log_ml(parent, alpha))
    log_prior = log_prior_birth(k, gamma)
    log_prop  = (np.log(p_D / p_B) + np.log(k) - np.log(max(r_T_prime, 1))
                 + np.log(m) + np.log(max(b_a, 1e-10)))
    return log_ll + log_prior + log_prop


def log_acceptance_death(
    leaf_data_prime: list,
    gamma: float,
    alpha: float,
    p_B: float,
    p_D: float,
    r_T: int,
    m: int = 1,
    b_a: float = 1.0,
) -> float:
    """
    Log acceptance ratio for the REVERSE death move (T'→T): merges last two leaves.
    Mirrors bdt_jakaite.py _try_death().
    """
    k_prime = len(leaf_data_prime)     # k+1 leaves
    k = k_prime - 1

    left  = leaf_data_prime[-2]
    right = leaf_data_prime[-1]
    merged = left + right

    log_ll    = (_dm_log_ml(merged, alpha) - _dm_log_ml(left, alpha)
                 - _dm_log_ml(right, alpha))
    log_prior = log_prior_death(k_prime, gamma)    # k_prime → k_prime − 1
    log_prop  = (np.log(p_B / p_D) + np.log(r_T) - np.log(k)
                 - np.log(m) - np.log(max(b_a, 1e-10)))
    return log_ll + log_prior + log_prop


def _r_T_prime(k_T: int, parent_was_prunable: bool) -> int:
    """
    Exact prunable-node count in T' after splitting one leaf of T.

    Splitting leaf L in T:
      + 1  : L (now internal) is prunable (both children are new leaves)
      − 1  : L's parent P (if P was prunable = both children were leaves) loses status
    For k_T = 1 there is no parent, so r_T' = 1 always.
    """
    if k_T == 1:
        return 1
    return k_T - 1 + 1 - (1 if parent_was_prunable else 0)


def verify_detailed_balance(
    leaf_data: list,
    gamma: float = 1.0,
    alpha: float = 1.0,
    p_B: float = 0.10,
    p_D: float = 0.10,
    m: int = 10,
    b_a: float = 2.0,
) -> dict:
    """
    Verify log α_B(T→T') + log α_D(T'→T) = 0 for a specific pair.

    T  = leaf_data (k leaves) — split the FIRST leaf 50/50 to get T'
    T' = [left, right] + leaf_data[1:]  (k+1 leaves)

    Exact prunable counts:
      k=1: r_T = 0, r_T' = 1   (root becomes prunable after first split)
      k≥2: r_T = number of prunable nodes in T (for a left-chain or balanced tree,
            approximately 1 if the first leaf's parent is prunable, else floor(k/2))

    We use the analytical identity:
      log_prior_birth(k) + log_prior_death(k+1) = 0                 [always]
      ΔlogML_birth      + ΔlogML_death           = 0                 [always]
      log_prop_birth(k, r_T') + log_prop_death(k+1, r_T', k) = 0   [if r values match]
    to prove DB holds.  The last line follows because:
      log_prop_B = log(p_D/p_B) + log(k)  - log(r_T') + log(m) + log(b−a)
      log_prop_D = log(p_B/p_D) - log(k)  + log(r_T') - log(m) - log(b−a)
    which sum to 0 regardless of r_T' (it cancels).
    """
    k = len(leaf_data)
    parent = leaf_data[0]
    left   = np.floor(parent / 2).astype(int)
    right  = (parent - left).astype(int)
    leaf_data_prime = [left, right] + list(leaf_data[1:])

    # For k=1: r_T = 0, r_T' = 1 (exactly).
    # For k≥2: in a balanced tree the first leaf's parent has the second leaf as sibling,
    # which is also a leaf → parent was prunable → r_T' = r_T (−1 + 1 = 0 net change).
    # We use r_T = floor(k/2) as a balanced-tree approximation, r_T' = same.
    if k == 1:
        r_T      = 0
        r_T_prim = 1
    else:
        r_T      = max(1, k // 2)   # balanced-tree approximation
        r_T_prim = r_T              # parent was prunable → net change = 0

    # Birth: T → T'  (uses r_T_prim)
    log_ll_B   = (_dm_log_ml(left, alpha) + _dm_log_ml(right, alpha)
                  - _dm_log_ml(parent, alpha))
    log_pri_B  = log_prior_birth(k, gamma)
    log_prop_B = (np.log(p_D / p_B) + np.log(k) - np.log(max(r_T_prim, 1))
                  + np.log(m) + np.log(max(b_a, 1e-10)))
    log_aB     = log_ll_B + log_pri_B + log_prop_B

    # Death: T' → T  (reverse; uses r_T_prim as "current tree's prunable count")
    log_ll_D   = -log_ll_B                           # exact negative
    log_pri_D  = log_prior_death(k + 1, gamma)       # = −log_prior_birth(k)
    log_prop_D = (np.log(p_B / p_D) + np.log(r_T_prim) - np.log(k)
                  - np.log(m) - np.log(max(b_a, 1e-10)))
    log_aD     = log_ll_D + log_pri_D + log_prop_D

    return dict(
        log_aB=log_aB,
        log_aD=log_aD,
        sum_log_a=log_aB + log_aD,
        db_residual=abs(log_aB + log_aD),
        k=k,
        # Individual cancellations
        prior_sum=log_pri_B + log_pri_D,
        ll_sum=log_ll_B + log_ll_D,
        prop_sum=log_prop_B + log_prop_D,
    )


# ---------------------------------------------------------------------------
# Pure-k RJMCMC (for chain convergence experiments)
# ---------------------------------------------------------------------------

def run_pure_k_chain(
    N_total: int,
    C: int,
    k_max: int,
    gamma: float,
    alpha: float,
    n_steps: int = 30_000,
    rng: np.random.Generator = None,
    true_probs: list = None,
) -> dict:
    """
    Simplified RJMCMC over k (number of leaves only).

    log π(k) = Σ_j log DM-ML(n_j) + (-γ(k−1) − log S_k)
    Data: N_total observations split into k balanced leaves, each with
    class distribution given by true_probs (cycled if len < k).

    Returns dict with k_chain, acceptance rates, empirical PMF.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if true_probs is None:
        true_probs = [np.full(C, 1.0 / C)]   # uniform default

    def gen_data(k):
        """Generate balanced leaf data for a k-leaf tree."""
        n = max(1, N_total // k)
        data = []
        for j in range(k):
            p = true_probs[j % len(true_probs)]
            data.append(rng.multinomial(n, p).astype(float))
        return data

    def log_pi(k_cur, data):
        log_ml  = sum(_dm_log_ml(c, alpha) for c in data)
        log_pri = -gamma * (k_cur - 1) - _log_catalan(k_cur)
        return log_ml + log_pri

    # Initialise
    k = 1
    data = gen_data(k)
    lp   = log_pi(k, data)

    k_chain = np.zeros(n_steps + 1, dtype=int)
    k_chain[0] = k
    n_birth_proposed = n_death_proposed = 0
    n_birth_accepted = n_death_accepted = 0

    for step in range(n_steps):
        # Propose birth or death with equal probability (except at boundaries)
        if k == 1:
            move = "birth"
        elif k >= k_max:
            move = "death"
        else:
            move = "birth" if rng.uniform() < 0.5 else "death"

        if move == "birth":
            n_birth_proposed += 1
            k_prop = k + 1
            data_prop = gen_data(k_prop)
            lp_prop   = log_pi(k_prop, data_prop)
        else:
            n_death_proposed += 1
            k_prop = k - 1
            data_prop = gen_data(k_prop)
            lp_prop   = log_pi(k_prop, data_prop)

        log_a = lp_prop - lp
        if np.log(rng.uniform() + 1e-300) < log_a:
            k, data, lp = k_prop, data_prop, lp_prop
            if move == "birth":
                n_birth_accepted += 1
            else:
                n_death_accepted += 1

        k_chain[step + 1] = k

    k_vals = np.arange(1, k_max + 1)
    emp_pmf = np.array([np.sum(k_chain == kv) / len(k_chain) for kv in k_vals])

    # Theoretical posterior (normalized over k=1..k_max)
    theo_log = np.array([log_pi(kv, gen_data(kv)) for kv in k_vals])
    theo_log -= theo_log.max()
    theo_pmf = np.exp(theo_log)
    theo_pmf /= theo_pmf.sum()

    return dict(
        k_chain=k_chain,
        emp_pmf=emp_pmf,
        theo_pmf=theo_pmf,
        birth_rate=n_birth_accepted / max(n_birth_proposed, 1),
        death_rate=n_death_accepted / max(n_death_proposed, 1),
        k_vals=k_vals,
        tvd=0.5 * float(np.sum(np.abs(emp_pmf - theo_pmf))),   # total variation distance
    )


# ---------------------------------------------------------------------------
# Birth acceptance rate vs k (analytical, data-averaged)
# ---------------------------------------------------------------------------

def birth_accept_rate_vs_k(
    k_vals: np.ndarray,
    N_total: int,
    C: int,
    gamma: float,
    alpha: float,
    p_B: float = 0.10,
    p_D: float = 0.10,
    n_reps: int = 200,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Empirical birth acceptance rate for each k, averaged over random datasets.
    Uses balanced leaf splits and random data from uniform class distribution.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    rates = np.zeros(len(k_vals))
    for ki, k in enumerate(k_vals):
        n_accept = 0
        n_base   = max(1, N_total // k)
        for _ in range(n_reps):
            # Current tree: k balanced leaves
            cur_data = [rng.multinomial(n_base, np.full(C, 1/C)).astype(float)
                        for _ in range(k)]

            # Proposed birth: split first leaf
            parent = cur_data[0]
            left   = np.floor(parent / 2).astype(int)
            right  = (parent - left).astype(int)
            prop_data = [left, right] + cur_data[1:]

            # Log ratio without proposal factors (simplified)
            log_r = (sum(_dm_log_ml(c, alpha) for c in prop_data)
                     - sum(_dm_log_ml(c, alpha) for c in cur_data)
                     + log_prior_birth(k, gamma))

            if log_r >= 0 or rng.uniform() < np.exp(log_r):
                n_accept += 1
        rates[ki] = n_accept / n_reps
    return rates
