"""
Jakaite & Schetinin (2025) Bayesian Decision Tree — Python implementation
Source: Schetinin V., Jakaite L. "Bayesian Learning Strategies for Reducing
        Uncertainty of Decision-Making in Case of Missing Values."
        Machine Learning and Knowledge Extraction, 2025, 7, 106.
        https://doi.org/10.3390/make7030106
MATLAB repo: https://github.com/ljakaite/Bayesian-Decision-Trees-Benchmarking

Algorithm: RJ-MCMC over binary DT structures with four move types:
    Birth        — split a leaf (Grow in Chipman 1998)
    Death        — merge a prunable internal node (Prune in Chipman)
    Change-Split — replace (feature, threshold) at any internal node
    Change-Rule  — perturb threshold via truncated Gaussian at any internal node

Key differences from BDTClassifier (Chipman 1998):
1. Tree-size prior: p(T) ∝ exp(−γ(k−1)) / S_k  where S_k = Catalan(k−1)
2. Change-Split and Change-Rule apply to ANY internal node (not just prunable)
3. Change-Rule proposes q' ~ N'(q, σ², [a,b]) — local threshold perturbation
4. Asymmetric move probabilities: p_birth=p_death=0.1, p_chg_split=0.2, p_chg_rule=0.6
5. Sweeping strategy: proposals creating leaves with < p_min samples are rejected

sklearn-compatible: JBDTClassifier
"""

from __future__ import annotations

import numpy as np
from copy import deepcopy
from scipy.special import gammaln, ndtr   # ndtr = standard normal CDF Φ
from scipy.stats import truncnorm as _truncnorm_dist
from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

_LOG_EPS = -700.0


# ---------------------------------------------------------------------------
# Tree node  (identical to bdt.py)
# ---------------------------------------------------------------------------

class _Node:
    def __init__(self, indices: np.ndarray | None = None, depth: int = 0):
        self.feature   = -1
        self.threshold = 0.0
        self.left      = None
        self.right     = None
        self.indices   = indices
        self.counts    = None
        self.depth     = depth

    @property
    def is_leaf(self) -> bool:
        return self.left is None


# ---------------------------------------------------------------------------
# Tree traversal
# ---------------------------------------------------------------------------

def _leaves(root: _Node):
    if root.is_leaf:
        yield root
    else:
        yield from _leaves(root.left)
        yield from _leaves(root.right)


def _all_internals(root: _Node):
    if not root.is_leaf:
        yield root
        yield from _all_internals(root.left)
        yield from _all_internals(root.right)


def _prunable(root: _Node):
    if not root.is_leaf:
        if root.left.is_leaf and root.right.is_leaf:
            yield root
        else:
            yield from _prunable(root.left)
            yield from _prunable(root.right)


def _collect_indices(node: _Node) -> np.ndarray:
    """All training-sample indices in a subtree."""
    if node.is_leaf:
        return node.indices
    return np.concatenate([_collect_indices(node.left),
                           _collect_indices(node.right)])


def _reroute(node: _Node, X: np.ndarray, idx: np.ndarray) -> None:
    """Route idx samples through node's subtree, updating leaf.indices in-place."""
    if node.is_leaf:
        node.indices = idx
    else:
        mask = X[idx, node.feature] <= node.threshold
        _reroute(node.left,  X, idx[mask])
        _reroute(node.right, X, idx[~mask])


# ---------------------------------------------------------------------------
# Leaf model — Dirichlet-Multinomial marginal likelihood
# ---------------------------------------------------------------------------

def _log_dir_mult(counts: np.ndarray, prior: np.ndarray) -> float:
    a0 = prior.sum()
    n  = counts.sum()
    return float(
        gammaln(a0) - gammaln(a0 + n)
        + (gammaln(prior + counts) - gammaln(prior)).sum()
    )


def _subtree_ll(node: _Node, y: np.ndarray, prior: np.ndarray) -> float:
    """Sum of DirMult log-likelihood for all leaves in a subtree."""
    return sum(
        _log_dir_mult(np.bincount(y[l.indices], minlength=len(prior)), prior)
        for l in _leaves(node)
        if l.indices is not None and len(l.indices) > 0
    )


# ---------------------------------------------------------------------------
# Tree-size prior: p(T) ∝ exp(−γ(k−1)) / S_k,  S_k = Catalan(k−1)
# ---------------------------------------------------------------------------

def _log_catalan(k: int) -> float:
    """log S_k = log Catalan(k−1).  S_1=1, S_2=1, S_3=2, S_4=5, ..."""
    n = k - 1
    if n <= 0:
        return 0.0
    return gammaln(2 * n + 1) - 2 * gammaln(n + 1) - np.log(n + 1)


def _log_prior_birth(k: int, gamma: float) -> float:
    """log p(T_birth) / p(T_current): k leaves → k+1 leaves."""
    return -gamma + _log_catalan(k) - _log_catalan(k + 1)


def _log_prior_death(k: int, gamma: float) -> float:
    """log p(T_death) / p(T_current): k leaves → k-1 leaves."""
    return gamma + _log_catalan(k) - _log_catalan(k - 1)


# ---------------------------------------------------------------------------
# Truncated-Gaussian helpers for Change-Rule
# ---------------------------------------------------------------------------

def _log_truncnorm_z(mu: float, sigma: float, a: float, b: float) -> float:
    """log[Φ((b−μ)/σ) − Φ((a−μ)/σ)] — log normalisation constant."""
    z_lo = (a - mu) / sigma
    z_hi = (b - mu) / sigma
    p_lo = float(ndtr(z_lo))
    p_hi = float(ndtr(z_hi))
    return np.log(max(p_hi - p_lo, 1e-300))


def _sample_truncnorm(mu: float, sigma: float, a: float, b: float,
                      rng: np.random.Generator) -> float | None:
    """Sample q' ~ N'(μ, σ², [a,b]).  Returns None if interval is degenerate."""
    if b - a < 1e-10:
        return None
    z_lo = (a - mu) / sigma
    z_hi = (b - mu) / sigma
    try:
        q = float(_truncnorm_dist.rvs(z_lo, z_hi, loc=mu, scale=sigma,
                                      random_state=int(rng.integers(int(1e9)))))
    except Exception:
        return None
    return q if a < q < b else None


# ---------------------------------------------------------------------------
# Move 1 — Birth  (equations 5–7)
# ---------------------------------------------------------------------------

def _try_birth(
    tree: _Node, X: np.ndarray, y: np.ndarray, prior: np.ndarray,
    rng: np.random.Generator, p_min: int,
    p_birth: float, p_death: float, gamma: float,
) -> tuple[_Node | None, float]:
    """Propose a Birth move: split a randomly chosen leaf."""
    all_leaves = list(_leaves(tree))
    k = len(all_leaves)                        # current leaf count
    leaf = all_leaves[rng.integers(k)]

    if leaf.indices is None or len(leaf.indices) <= 2 * p_min:
        return None, _LOG_EPS                  # insufficient data (Algorithm 1 line 4)

    idx = leaf.indices
    m   = X.shape[1]

    # Draw split: v ~ U(1,m),  q ~ U(min xᵥ, max xᵥ)
    feat  = int(rng.integers(m))
    x_col = X[idx, feat]
    vals  = np.unique(x_col)
    if len(vals) < 2:
        return None, _LOG_EPS
    thresh = float(rng.choice(vals[:-1]))
    a, b   = float(x_col.min()), float(x_col.max())

    mask      = x_col <= thresh
    left_idx  = idx[mask]
    right_idx = idx[~mask]

    # Sweeping strategy: reject if either child < p_min (Algorithm 1 lines 9–18)
    if len(left_idx) < p_min or len(right_idx) < p_min:
        return None, _LOG_EPS

    # Build proposed tree
    new_tree   = deepcopy(tree)
    new_leaves = list(_leaves(new_tree))
    pos        = next(i for i, l in enumerate(all_leaves) if l is leaf)
    new_leaf   = new_leaves[pos]

    new_leaf.feature   = feat
    new_leaf.threshold = thresh
    new_leaf.left      = _Node(indices=left_idx,  depth=leaf.depth + 1)
    new_leaf.right     = _Node(indices=right_idx, depth=leaf.depth + 1)
    new_leaf.indices   = None

    r_T_prime = len(list(_prunable(new_tree)))   # r(T')

    # Likelihood ratio: ΔlogDirMult
    ll_old   = _log_dir_mult(np.bincount(y[idx],       minlength=len(prior)), prior)
    ll_left  = _log_dir_mult(np.bincount(y[left_idx],  minlength=len(prior)), prior)
    ll_right = _log_dir_mult(np.bincount(y[right_idx], minlength=len(prior)), prior)
    log_ll   = ll_left + ll_right - ll_old

    # Prior ratio (eq 2): log p(T')/p(T)
    log_prior = _log_prior_birth(k, gamma)

    # Proposal ratio (eqs 5–7): [p_death/p_birth] · [k/r(T')] · m · 1/g_j(q)
    #   g_j(q) = 1/(b−a)  →  1/g_j(q) = (b−a)
    log_prop = (np.log(p_death / p_birth)
                + np.log(k)
                - np.log(max(r_T_prime, 1))
                + np.log(m)
                + np.log(max(b - a, 1e-10)))

    return new_tree, log_ll + log_prior + log_prop


# ---------------------------------------------------------------------------
# Move 2 — Death  (equations 8–10)
# ---------------------------------------------------------------------------

def _try_death(
    tree: _Node, X: np.ndarray, y: np.ndarray, prior: np.ndarray,
    rng: np.random.Generator, p_min: int,
    p_birth: float, p_death: float, gamma: float,
) -> tuple[_Node | None, float]:
    """Propose a Death move: collapse a randomly chosen prunable internal node."""
    all_leaves = list(_leaves(tree))
    k          = len(all_leaves)
    if k <= 1:
        return None, _LOG_EPS

    prunable = list(_prunable(tree))
    r_T      = len(prunable)
    if r_T == 0:
        return None, _LOG_EPS

    node = prunable[rng.integers(r_T)]

    left_idx   = node.left.indices
    right_idx  = node.right.indices
    merged_idx = np.concatenate([left_idx, right_idx])

    feat   = node.feature
    thresh = node.threshold
    x_col  = X[merged_idx, feat]
    a, b   = float(x_col.min()), float(x_col.max())
    m      = X.shape[1]

    # Build proposed tree
    new_tree     = deepcopy(tree)
    new_prunable = list(_prunable(new_tree))
    pos          = next(i for i, n in enumerate(prunable) if n is node)
    new_node     = new_prunable[pos]

    new_node.feature   = -1
    new_node.left      = None
    new_node.right     = None
    new_node.indices   = merged_idx.copy()

    # Likelihood ratio
    ll_left   = _log_dir_mult(np.bincount(y[left_idx],   minlength=len(prior)), prior)
    ll_right  = _log_dir_mult(np.bincount(y[right_idx],  minlength=len(prior)), prior)
    ll_merged = _log_dir_mult(np.bincount(y[merged_idx], minlength=len(prior)), prior)
    log_ll    = ll_merged - ll_left - ll_right

    # Prior ratio: log p(T')/p(T)  (k→k−1)
    log_prior = _log_prior_death(k, gamma)

    # Proposal ratio (eqs 8–10): [p_birth/p_death] · [r(T)/(k−1)] · (1/m) · g_j(q)
    #   g_j(q) = 1/(b−a)
    log_prop = (np.log(p_birth / p_death)
                + np.log(r_T)
                - np.log(k - 1)
                - np.log(m)
                - np.log(max(b - a, 1e-10)))

    return new_tree, log_ll + log_prior + log_prop


# ---------------------------------------------------------------------------
# Move 3 — Change-Split  (equation 11)
# ---------------------------------------------------------------------------

def _try_change_split(
    tree: _Node, X: np.ndarray, y: np.ndarray, prior: np.ndarray,
    rng: np.random.Generator, p_min: int,
) -> tuple[_Node | None, float]:
    """Replace (feature, threshold) at any internal node (Algorithm 2, variant 1)."""
    internals = list(_all_internals(tree))
    if not internals:
        return None, _LOG_EPS

    node    = internals[rng.integers(len(internals))]
    all_idx = _collect_indices(node)

    # Range of OLD feature at this node (for proposal ratio g(q))
    a_old = float(X[all_idx, node.feature].min())
    b_old = float(X[all_idx, node.feature].max())
    ll_old = _subtree_ll(node, y, prior)

    # Propose new feature v' and threshold q'
    feat_new  = int(rng.integers(X.shape[1]))
    x_new     = X[all_idx, feat_new]
    vals_new  = np.unique(x_new)
    if len(vals_new) < 2:
        return None, _LOG_EPS
    thresh_new = float(rng.choice(vals_new[:-1]))
    a_new = float(x_new.min())
    b_new = float(x_new.max())

    # Build proposed tree: deepcopy, update split, reroute subtree
    new_tree     = deepcopy(tree)
    new_internals = list(_all_internals(new_tree))
    pos          = next(i for i, n in enumerate(internals) if n is node)
    new_node     = new_internals[pos]

    new_node.feature   = feat_new
    new_node.threshold = thresh_new
    _reroute(new_node, X, all_idx)

    # Sweeping: reject if any leaf in subtree < p_min
    if any(len(l.indices) < p_min for l in _leaves(new_node)):
        return None, _LOG_EPS

    ll_new = _subtree_ll(new_node, y, prior)

    # MH ratio (eq 11): likelihood ratio · [g(q)/g'(q')]
    #   g(q)  = 1/(b_old − a_old),  g'(q') = 1/(b_new − a_new)
    #   log g(q)/g'(q') = log(b_new − a_new) − log(b_old − a_old)
    log_g_ratio = (np.log(max(b_new - a_new, 1e-10))
                   - np.log(max(b_old - a_old, 1e-10)))

    return new_tree, (ll_new - ll_old) + log_g_ratio


# ---------------------------------------------------------------------------
# Move 4 — Change-Rule  (equation 12)
# ---------------------------------------------------------------------------

def _try_change_rule(
    tree: _Node, X: np.ndarray, y: np.ndarray, prior: np.ndarray,
    rng: np.random.Generator, p_min: int, sigma: float,
) -> tuple[_Node | None, float]:
    """Perturb threshold via truncated Gaussian at any internal node (Algorithm 2, variant 2)."""
    internals = list(_all_internals(tree))
    if not internals:
        return None, _LOG_EPS

    node    = internals[rng.integers(len(internals))]
    all_idx = _collect_indices(node)
    feat    = node.feature
    q_old   = node.threshold
    x_col   = X[all_idx, feat]
    a, b    = float(x_col.min()), float(x_col.max())

    if b - a < 1e-10:
        return None, _LOG_EPS

    # Sample q' ~ N'(q_old, σ², [a, b])
    q_new = _sample_truncnorm(q_old, sigma, a, b, rng)
    if q_new is None:
        return None, _LOG_EPS

    ll_old     = _subtree_ll(node, y, prior)
    log_z_old  = _log_truncnorm_z(q_old, sigma, a, b)  # normalisation for q'~N'(q_old,…)

    # Build proposed tree
    new_tree      = deepcopy(tree)
    new_internals = list(_all_internals(new_tree))
    pos           = next(i for i, n in enumerate(internals) if n is node)
    new_node      = new_internals[pos]

    new_node.threshold = q_new
    _reroute(new_node, X, all_idx)

    if any(len(l.indices) < p_min for l in _leaves(new_node)):
        return None, _LOG_EPS

    ll_new    = _subtree_ll(new_node, y, prior)
    log_z_new = _log_truncnorm_z(q_new, sigma, a, b)   # normalisation for q~N'(q_new,…)

    # MH ratio (eq 12): likelihood · [φ_N'(q; q', σ²,[a,b]) / φ_N'(q'; q, σ²,[a,b])]
    # = likelihood · [Z(q_new) / Z(q_old)]   (Gaussian kernels cancel; see derivation)
    return new_tree, (ll_new - ll_old) + (log_z_old - log_z_new)


# ---------------------------------------------------------------------------
# One MCMC step
# ---------------------------------------------------------------------------

def _mcmc_step_jakaite(
    tree: _Node, X: np.ndarray, y: np.ndarray, prior: np.ndarray,
    rng: np.random.Generator,
    p_min: int, p_birth: float, p_death: float,
    p_chg_split: float, p_chg_rule: float,
    gamma: float, sigma: float,
) -> _Node:
    u = rng.random()
    c1 = p_birth
    c2 = c1 + p_death
    c3 = c2 + p_chg_split

    if u < c1:
        proposed, log_a = _try_birth(tree, X, y, prior, rng, p_min,
                                     p_birth, p_death, gamma)
    elif u < c2:
        proposed, log_a = _try_death(tree, X, y, prior, rng, p_min,
                                     p_birth, p_death, gamma)
    elif u < c3:
        proposed, log_a = _try_change_split(tree, X, y, prior, rng, p_min)
    else:
        proposed, log_a = _try_change_rule(tree, X, y, prior, rng, p_min, sigma)

    if proposed is None:
        return tree
    return proposed if np.log(rng.random() + 1e-300) <= log_a else tree


# ---------------------------------------------------------------------------
# Finalise and predict
# ---------------------------------------------------------------------------

def _finalize_tree(root: _Node, y: np.ndarray, n_classes: int) -> _Node:
    t = deepcopy(root)
    for leaf in _leaves(t):
        idx = leaf.indices
        leaf.counts  = np.bincount(y[idx], minlength=n_classes) if idx is not None else np.zeros(n_classes)
        leaf.indices = None
    return t


def _predict_proba_one(root: _Node, X: np.ndarray, prior: np.ndarray) -> np.ndarray:
    proba  = np.empty((len(X), len(prior)))
    alpha0 = prior.sum()
    for k in range(len(X)):
        node = root
        while not node.is_leaf:
            node = node.left if X[k, node.feature] <= node.threshold else node.right
        counts   = node.counts if node.counts is not None else np.zeros(len(prior))
        proba[k] = (prior + counts) / (alpha0 + counts.sum())
    return proba


# ---------------------------------------------------------------------------
# Single-chain runner  (module-level for joblib)
# ---------------------------------------------------------------------------

def _run_chain_jakaite(
    seed: int, X: np.ndarray, y: np.ndarray, prior: np.ndarray,
    p_min: int, p_birth: float, p_death: float,
    p_chg_split: float, p_chg_rule: float,
    gamma: float, sigma: float,
    n_burnin: int, n_samples: int,
    verbose: bool, chain_id: int,
) -> list[_Node]:
    rng = np.random.default_rng(seed)
    n_classes = len(prior)

    # Initialise with a single valid split (k=2)
    root = _init_tree(X, y, p_min, n_classes, rng)

    samples: list[_Node] = []
    for i in range(n_burnin + n_samples):
        root = _mcmc_step_jakaite(
            root, X, y, prior, rng,
            p_min, p_birth, p_death, p_chg_split, p_chg_rule,
            gamma, sigma,
        )
        if i >= n_burnin:
            samples.append(_finalize_tree(root, y, n_classes))

    if verbose:
        n_leaves = sum(1 for _ in _leaves(root))
        print(f"  JChain {chain_id}: done — final tree has {n_leaves} leaves",
              flush=True)
    return samples


def _init_tree(X: np.ndarray, y: np.ndarray, p_min: int,
               n_classes: int, rng: np.random.Generator) -> _Node:
    """Initialise with one splitting node (k=2) as in the paper."""
    n, m = X.shape
    idx  = np.arange(n)
    for _ in range(200):
        feat  = int(rng.integers(m))
        vals  = np.unique(X[:, feat])
        if len(vals) < 2:
            continue
        thresh = float(rng.choice(vals[:-1]))
        mask   = X[:, feat] <= thresh
        if mask.sum() >= p_min and (~mask).sum() >= p_min:
            root            = _Node(depth=0)
            root.feature    = feat
            root.threshold  = thresh
            root.left       = _Node(indices=idx[mask],  depth=1)
            root.right      = _Node(indices=idx[~mask], depth=1)
            return root
    # Fallback: single-leaf root
    return _Node(indices=idx, depth=0)


# ---------------------------------------------------------------------------
# Public estimator
# ---------------------------------------------------------------------------

class JBDTClassifier(BaseEstimator, ClassifierMixin):
    """
    Jakaite & Schetinin (2025) Bayesian Decision Tree Classifier.

    RJ-MCMC over binary decision tree structures with four move types:
    Birth, Death, Change-Split, and Change-Rule.  Pools n_chains independent
    chains run in parallel; predictions are the mean posterior class probability
    across all pooled samples.

    Parameters
    ----------
    n_samples : int, default=500
        Post-burn-in samples per chain.
    n_burnin : int, default=250
        Burn-in iterations per chain (discarded).
    n_chains : int, default=4
        Independent MCMC chains (parallelised via joblib).
    p_birth : float, default=0.10
        Proposal probability for Birth move.
    p_death : float, default=0.10
        Proposal probability for Death move.
    p_chg_split : float, default=0.20
        Proposal probability for Change-Split move.
    p_chg_rule : float, default=0.60
        Proposal probability for Change-Rule move.
        (p_birth + p_death + p_chg_split + p_chg_rule must equal 1.)
    p_min : int, default=2
        Minimum samples per leaf (sweeping strategy).
    sigma : float, default=1.41  (≈ sqrt(2))
        Standard deviation for the truncated-Gaussian Change-Rule proposal.
        The paper uses variance σ²=2.0, i.e. σ≈1.41.
    gamma : float, default=1.0
        Tree-size prior decay: p(k) ∝ exp(−γ(k−1)).
        Higher γ → stronger preference for smaller trees.
    prior_strength : float, default=1.0
        Total Dirichlet prior mass at each leaf: α_c = prior_strength/n_classes.
    n_jobs : int, default=-1
        Parallel workers for running chains.
    random_state : int, default=42
    verbose : int, default=0
    """

    def __init__(
        self,
        n_samples: int     = 500,
        n_burnin: int      = 250,
        n_chains: int      = 4,
        p_birth: float     = 0.10,
        p_death: float     = 0.10,
        p_chg_split: float = 0.20,
        p_chg_rule: float  = 0.60,
        p_min: int         = 2,
        sigma: float       = np.sqrt(2.0),   # σ²=2.0 as in paper
        gamma: float       = 1.0,
        prior_strength: float = 1.0,
        n_jobs: int        = -1,
        random_state: int  = 42,
        verbose: int       = 0,
    ):
        self.n_samples      = n_samples
        self.n_burnin       = n_burnin
        self.n_chains       = n_chains
        self.p_birth        = p_birth
        self.p_death        = p_death
        self.p_chg_split    = p_chg_split
        self.p_chg_rule     = p_chg_rule
        self.p_min          = p_min
        self.sigma          = sigma
        self.gamma          = gamma
        self.prior_strength = prior_strength
        self.n_jobs         = n_jobs
        self.random_state   = random_state
        self.verbose        = verbose

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_   = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        y_enc           = np.searchsorted(self.classes_, y)
        self.prior_     = np.full(self.n_classes_,
                                  self.prior_strength / self.n_classes_)

        rng   = np.random.default_rng(self.random_state)
        seeds = rng.integers(int(1e9), size=self.n_chains)

        if self.verbose:
            print(f"JBDTClassifier: {self.n_chains} chains × "
                  f"{self.n_burnin + self.n_samples} iterations  "
                  f"move probs=(B={self.p_birth}, D={self.p_death}, "
                  f"CS={self.p_chg_split}, CR={self.p_chg_rule})",
                  flush=True)

        chain_results = Parallel(n_jobs=self.n_jobs)(
            delayed(_run_chain_jakaite)(
                int(seeds[c]), X, y_enc, self.prior_,
                self.p_min, self.p_birth, self.p_death,
                self.p_chg_split, self.p_chg_rule,
                self.gamma, self.sigma,
                self.n_burnin, self.n_samples,
                bool(self.verbose), c,
            )
            for c in range(self.n_chains)
        )
        self.samples_ = [s for chain in chain_results for s in chain]
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X     = check_array(X)
        proba = sum(_predict_proba_one(t, X, self.prior_) for t in self.samples_)
        proba /= len(self.samples_)
        return proba

    def predict(self, X):
        check_is_fitted(self)
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    @property
    def n_posterior_samples_(self) -> int:
        check_is_fitted(self)
        return len(self.samples_)


# ---------------------------------------------------------------------------
# Benchmark: compare JBDTClassifier vs BDTClassifier (Chipman 1998)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, time, os
    sys.path.insert(0, os.path.dirname(__file__))
    from bdt import BDTClassifier

    from sklearn.datasets import load_iris, load_breast_cancer
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    SETTINGS = dict(n_samples=200, n_burnin=100, n_chains=4,
                    n_jobs=-1, verbose=1, random_state=42)
    CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def cv_run(clf, X, y, binary=False):
        accs, aucs = [], []
        for tr, te in CV.split(X, y):
            clf.fit(X[tr], y[tr])
            proba = clf.predict_proba(X[te])
            pred  = clf.predict(X[te])
            accs.append((pred == y[te]).mean())
            if binary:
                aucs.append(roc_auc_score(y[te], proba[:, 1]))
            else:
                aucs.append(roc_auc_score(y[te], proba, multi_class="ovr"))
        return np.array(accs), np.array(aucs)

    results = {}
    for ds_name, (X, y), binary in [
        ("Iris",          load_iris(return_X_y=True),          False),
        ("BreastCancer",  load_breast_cancer(return_X_y=True), True),
    ]:
        results[ds_name] = {}
        for clf_name, clf in [
            ("Chipman1998", BDTClassifier(**SETTINGS)),
            ("Jakaite2025", JBDTClassifier(**SETTINGS)),
        ]:
            print(f"\n=== {clf_name} on {ds_name} ===")
            t0 = time.perf_counter()
            accs, aucs = cv_run(clf, X, y, binary=binary)
            dt = time.perf_counter() - t0
            results[ds_name][clf_name] = dict(
                acc_mean=accs.mean(), acc_std=accs.std(),
                auc_mean=aucs.mean(), auc_std=aucs.std(),
                time=dt,
            )
            print(f"  Accuracy : {accs.mean():.4f} ± {accs.std():.4f}")
            print(f"  AUC      : {aucs.mean():.4f} ± {aucs.std():.4f}")
            print(f"  Time     : {dt:.1f}s")

    # Print summary table
    print("\n" + "="*70)
    print(f"{'Dataset':<15} {'Method':<15} {'Accuracy':>12} {'AUC':>12} {'Time(s)':>8}")
    print("-"*70)
    for ds, methods in results.items():
        for clf_name, r in methods.items():
            print(f"{ds:<15} {clf_name:<15} "
                  f"{r['acc_mean']:.4f}±{r['acc_std']:.4f}  "
                  f"{r['auc_mean']:.4f}±{r['auc_std']:.4f}  "
                  f"{r['time']:>7.1f}")
    print("="*70)
