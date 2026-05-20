"""
Bayesian Decision Tree (BDT) Classifier — multichain implementation
Based on: Chipman, George & McCulloch (1998)
"Bayesian CART model search"
J. American Statistical Association 93(443), 935–948.

Algorithm
---------
Places a prior over binary decision tree structures (Chipman 1998 prior) and
a Dirichlet-Categorical model at each leaf.  Samples from the posterior over
tree structures using Metropolis-Hastings with three move types:
    Grow   — split a leaf into two child leaves
    Prune  — merge two sibling leaves back into their parent
    Change — replace the split rule at a prunable internal node

n_chains independent chains are run in parallel via joblib; their posterior
samples are pooled for prediction.  Predictive probabilities are the mean of
the Dirichlet posterior at the leaf across all posterior samples — naturally
calibrated and uncertainty-aware.

sklearn-compatible: BDTClassifier

Typical use
-----------
    from bdt import BDTClassifier
    clf = BDTClassifier(n_samples=500, n_burnin=250, n_chains=4, n_jobs=-1)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)   # shape (N, n_classes)
"""

from __future__ import annotations

import numpy as np
from copy import deepcopy
from scipy.special import gammaln
from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class _Node:
    """A node in the Bayesian decision tree.

    Leaf:     left is None; indices holds training sample indices (during MCMC),
              counts holds class counts (in stored posterior samples).
    Internal: left / right are child _Nodes; feature / threshold define the split.
    """
    # No __slots__: loky (joblib's default backend) pickles results back to
    # the main process; __slots__ without __dict__ breaks pickle in Python 3.12+.

    def __init__(self, indices: np.ndarray | None = None, depth: int = 0):
        self.feature   = -1
        self.threshold = 0.0
        self.left      = None
        self.right     = None
        self.indices   = indices   # sample indices — only set at leaves during MCMC
        self.counts    = None      # class counts  — only set in finalized samples
        self.depth     = depth

    @property
    def is_leaf(self) -> bool:
        return self.left is None


# ---------------------------------------------------------------------------
# Tree traversal helpers
# ---------------------------------------------------------------------------

def _leaves(root: _Node):
    """Yield all leaf nodes (DFS, left-first)."""
    if root.is_leaf:
        yield root
    else:
        yield from _leaves(root.left)
        yield from _leaves(root.right)


def _prunable(root: _Node):
    """Yield internal nodes whose both children are leaves."""
    if not root.is_leaf:
        if root.left.is_leaf and root.right.is_leaf:
            yield root
        else:
            yield from _prunable(root.left)
            yield from _prunable(root.right)


def _collect_indices(node: _Node) -> np.ndarray:
    """Gather all sample indices in a subtree."""
    if node.is_leaf:
        return node.indices
    return np.concatenate([_collect_indices(node.left),
                           _collect_indices(node.right)])


# ---------------------------------------------------------------------------
# Bayesian leaf model — Dirichlet-Multinomial marginal likelihood
# ---------------------------------------------------------------------------

def _log_dir_mult(counts: np.ndarray, prior: np.ndarray) -> float:
    """
    Log marginal likelihood of class counts under a Dirichlet-Multinomial model.
    Integrates out the class probability vector θ:
        p(counts | α) = Γ(α₀)/Γ(α₀+n) · Π_k Γ(α_k+n_k)/Γ(α_k)
    """
    alpha0 = prior.sum()
    n      = counts.sum()
    return float(
        gammaln(alpha0) - gammaln(alpha0 + n) +
        (gammaln(prior + counts) - gammaln(prior)).sum()
    )


# ---------------------------------------------------------------------------
# Tree structure prior (Chipman 1998)
# ---------------------------------------------------------------------------

_LOG_EPS = -700.0   # substitute for log(0)

def _log_p_split(depth: int, alpha: float, beta: float) -> float:
    p = alpha * (1.0 + depth) ** (-beta)
    p = float(np.clip(p, 1e-300, 1.0 - 1e-300))
    return np.log(p)


def _log_p_nosplit(depth: int, alpha: float, beta: float) -> float:
    p = alpha * (1.0 + depth) ** (-beta)
    p = float(np.clip(p, 1e-300, 1.0 - 1e-300))
    return np.log(1.0 - p)


# ---------------------------------------------------------------------------
# Split enumeration helpers
# ---------------------------------------------------------------------------

def _valid_splits(idx: np.ndarray, X: np.ndarray) -> list[tuple[int, float]]:
    """Return all (feature, threshold) pairs that yield a non-empty binary split."""
    pairs = []
    for f in range(X.shape[1]):
        vals = np.unique(X[idx, f])
        for t in vals[:-1]:       # threshold = value; left ≤ t, right > t
            pairs.append((f, float(t)))
    return pairs


# ---------------------------------------------------------------------------
# MCMC move proposals
# ---------------------------------------------------------------------------

def _try_grow(
    tree: _Node, X: np.ndarray, y: np.ndarray,
    prior: np.ndarray, alpha: float, beta: float,
    max_depth: int, rng: np.random.Generator,
) -> tuple[_Node | None, float]:
    """Propose a Grow move: split a randomly chosen growable leaf."""

    leaves     = list(_leaves(tree))
    growable   = [l for l in leaves
                  if l.depth < max_depth and l.indices is not None and len(l.indices) >= 2]
    if not growable:
        return None, _LOG_EPS

    leaf     = growable[rng.integers(len(growable))]
    splits   = _valid_splits(leaf.indices, X)
    n_splits = len(splits)
    if n_splits == 0:
        return None, _LOG_EPS

    feat, thresh = splits[rng.integers(n_splits)]
    mask          = X[leaf.indices, feat] <= thresh
    left_idx      = leaf.indices[mask]
    right_idx     = leaf.indices[~mask]
    if len(left_idx) == 0 or len(right_idx) == 0:
        return None, _LOG_EPS

    # Build proposed tree: deepcopy then modify the corresponding leaf
    new_tree    = deepcopy(tree)
    old_leaves  = leaves                          # already computed
    new_leaves  = list(_leaves(new_tree))
    pos         = next(i for i, l in enumerate(old_leaves) if l is leaf)
    new_leaf    = new_leaves[pos]

    new_leaf.feature   = feat
    new_leaf.threshold = thresh
    new_leaf.left      = _Node(indices=left_idx,  depth=leaf.depth + 1)
    new_leaf.right     = _Node(indices=right_idx, depth=leaf.depth + 1)
    new_leaf.indices   = None

    # ---------- log MH ratio ----------
    # Likelihood ratio
    ll_old   = _log_dir_mult(np.bincount(y[leaf.indices], minlength=len(prior)), prior)
    ll_left  = _log_dir_mult(np.bincount(y[left_idx],    minlength=len(prior)), prior)
    ll_right = _log_dir_mult(np.bincount(y[right_idx],   minlength=len(prior)), prior)
    log_ll   = ll_left + ll_right - ll_old

    # Prior ratio: P(internal at d) · P(leaf at d+1)^2 / P(leaf at d)
    d               = leaf.depth
    log_prior       = (_log_p_split(d, alpha, beta)
                       + 2 * _log_p_nosplit(d + 1, alpha, beta)
                       - _log_p_nosplit(d, alpha, beta))

    # Proposal ratio: q_reverse / q_forward
    #   forward  = (1/3) · (1/n_growable) · (1/n_splits)
    #   reverse  = (1/3) · (1/n_prunable(T'))
    n_prunable_new  = len(list(_prunable(new_tree)))
    log_proposal    = (np.log(len(growable)) + np.log(n_splits)
                       - np.log(max(1, n_prunable_new)))

    return new_tree, log_ll + log_prior + log_proposal


def _try_prune(
    tree: _Node, X: np.ndarray, y: np.ndarray,
    prior: np.ndarray, alpha: float, beta: float,
    max_depth: int, rng: np.random.Generator,
) -> tuple[_Node | None, float]:
    """Propose a Prune move: collapse a randomly chosen prunable internal node."""

    prunable   = list(_prunable(tree))
    if not prunable:
        return None, _LOG_EPS

    node       = prunable[rng.integers(len(prunable))]
    merged_idx = np.concatenate([node.left.indices, node.right.indices])

    # Build proposed tree
    new_tree      = deepcopy(tree)
    old_prunable  = prunable
    new_prunable  = list(_prunable(new_tree))
    pos           = next(i for i, n in enumerate(old_prunable) if n is node)
    new_node      = new_prunable[pos]

    new_node.feature   = -1
    new_node.left      = None
    new_node.right     = None
    new_node.indices   = merged_idx.copy()

    # ---------- log MH ratio (exact inverse of Grow) ----------
    ll_merged = _log_dir_mult(np.bincount(y[merged_idx],          minlength=len(prior)), prior)
    ll_left   = _log_dir_mult(np.bincount(y[node.left.indices],   minlength=len(prior)), prior)
    ll_right  = _log_dir_mult(np.bincount(y[node.right.indices],  minlength=len(prior)), prior)
    log_ll    = ll_merged - ll_left - ll_right

    d          = node.depth
    log_prior  = (_log_p_nosplit(d, alpha, beta)
                  - _log_p_split(d, alpha, beta)
                  - 2 * _log_p_nosplit(d + 1, alpha, beta))

    # q_reverse / q_forward
    #   forward  = (1/3) · (1/n_prunable(T))
    #   reverse  = (1/3) · (1/n_growable(T')) · (1/n_splits_at_merged)
    n_splits_merged = len(_valid_splits(merged_idx, X))
    new_leaves      = list(_leaves(new_tree))
    n_growable_new  = sum(1 for l in new_leaves
                          if l.depth < max_depth
                          and l.indices is not None
                          and len(l.indices) >= 2)
    log_proposal    = (np.log(len(prunable))
                       - np.log(max(1, n_growable_new))
                       - np.log(max(1, n_splits_merged)))

    return new_tree, log_ll + log_prior + log_proposal


def _try_change(
    tree: _Node, X: np.ndarray, y: np.ndarray,
    prior: np.ndarray, rng: np.random.Generator,
) -> tuple[_Node | None, float]:
    """Propose a Change move: replace the split rule at a prunable internal node.

    Restricted to prunable internals (both children leaves) to avoid
    re-routing entire subtrees.  Prior ratio = 0 (structure unchanged);
    proposal ratio = 0 (symmetric — same n_prunable and same n_splits pool).
    """
    prunable = list(_prunable(tree))
    if not prunable:
        return None, _LOG_EPS

    node     = prunable[rng.integers(len(prunable))]
    all_idx  = np.concatenate([node.left.indices, node.right.indices])
    splits   = _valid_splits(all_idx, X)
    if len(splits) < 2:     # need at least an alternative split to change to
        return None, _LOG_EPS

    # Exclude current split to ensure we actually change something
    current = (node.feature, node.threshold)
    alts    = [s for s in splits if s != current]
    if not alts:
        return None, _LOG_EPS

    feat, thresh = alts[rng.integers(len(alts))]
    mask          = X[all_idx, feat] <= thresh
    new_left_idx  = all_idx[mask]
    new_right_idx = all_idx[~mask]
    if len(new_left_idx) == 0 or len(new_right_idx) == 0:
        return None, _LOG_EPS

    # Build proposed tree
    new_tree     = deepcopy(tree)
    new_prunable = list(_prunable(new_tree))
    pos          = next(i for i, n in enumerate(prunable) if n is node)
    new_node     = new_prunable[pos]

    new_node.feature        = feat
    new_node.threshold      = thresh
    new_node.left.indices   = new_left_idx
    new_node.right.indices  = new_right_idx

    # ---------- log MH ratio ----------
    ll_old = (_log_dir_mult(np.bincount(y[node.left.indices],  minlength=len(prior)), prior) +
              _log_dir_mult(np.bincount(y[node.right.indices], minlength=len(prior)), prior))
    ll_new = (_log_dir_mult(np.bincount(y[new_left_idx],  minlength=len(prior)), prior) +
              _log_dir_mult(np.bincount(y[new_right_idx], minlength=len(prior)), prior))

    return new_tree, ll_new - ll_old     # prior ratio = 0, proposal ratio ≈ 0


# ---------------------------------------------------------------------------
# One MCMC step
# ---------------------------------------------------------------------------

def _mcmc_step(
    tree: _Node, X: np.ndarray, y: np.ndarray,
    prior: np.ndarray, alpha: float, beta: float,
    max_depth: int, rng: np.random.Generator,
) -> _Node:
    move = rng.integers(3)
    if move == 0:
        proposed, log_alpha = _try_grow(tree, X, y, prior, alpha, beta, max_depth, rng)
    elif move == 1:
        proposed, log_alpha = _try_prune(tree, X, y, prior, alpha, beta, max_depth, rng)
    else:
        proposed, log_alpha = _try_change(tree, X, y, prior, rng)

    if proposed is None:
        return tree

    log_u = np.log(rng.random() + 1e-300)
    return proposed if log_u <= log_alpha else tree


# ---------------------------------------------------------------------------
# Chain finalisation and prediction helpers
# ---------------------------------------------------------------------------

def _finalize_tree(root: _Node, y: np.ndarray, n_classes: int) -> _Node:
    """Deep-copy of tree with leaf.counts set (class counts) and indices freed."""
    t = deepcopy(root)
    for leaf in _leaves(t):
        idx = leaf.indices
        leaf.counts  = np.bincount(y[idx], minlength=n_classes) if idx is not None else np.zeros(n_classes)
        leaf.indices = None
    return t


def _predict_proba_one(root: _Node, X: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Posterior predictive from a single sampled tree."""
    proba = np.empty((len(X), len(prior)))
    alpha0 = prior.sum()
    for k in range(len(X)):
        node = root
        while not node.is_leaf:
            node = node.left if X[k, node.feature] <= node.threshold else node.right
        counts   = node.counts if node.counts is not None else np.zeros(len(prior))
        n        = counts.sum()
        proba[k] = (prior + counts) / (alpha0 + n)
    return proba


# ---------------------------------------------------------------------------
# Single-chain runner (module-level for joblib serialisation)
# ---------------------------------------------------------------------------

def _run_chain(
    seed: int,
    X: np.ndarray, y: np.ndarray,
    prior: np.ndarray,
    alpha: float, beta: float,
    max_depth: int,
    n_burnin: int, n_samples: int,
    verbose: bool,
    chain_id: int,
) -> list[_Node]:
    rng  = np.random.default_rng(seed)
    root = _Node(indices=np.arange(len(X)), depth=0)
    samples: list[_Node] = []

    for i in range(n_burnin + n_samples):
        root = _mcmc_step(root, X, y, prior, alpha, beta, max_depth, rng)
        if i >= n_burnin:
            samples.append(_finalize_tree(root, y, len(prior)))

    if verbose:
        n_leaves = len(list(_leaves(root)))
        print(f"  Chain {chain_id}: done — final tree has {n_leaves} leaves", flush=True)

    return samples


# ---------------------------------------------------------------------------
# Public estimator
# ---------------------------------------------------------------------------

class BDTClassifier(BaseEstimator, ClassifierMixin):
    """
    Bayesian Decision Tree classifier.

    Places a Chipman (1998) prior over binary tree structures and a
    Dirichlet-Categorical model at each leaf.  Posterior inference is via
    Metropolis-Hastings MCMC with Grow / Prune / Change moves.

    n_chains independent chains run in parallel; their samples are pooled.
    Predictions are the mean posterior class probabilities across all
    pooled samples — naturally calibrated uncertainty estimates.

    Parameters
    ----------
    n_samples : int, default=500
        Posterior samples collected per chain (post-burn-in).
    n_burnin : int, default=250
        Burn-in iterations per chain (discarded).
    n_chains : int, default=4
        Number of independent MCMC chains (run in parallel).
    alpha : float, default=0.95
        Chipman prior: P(node at depth d is internal) = α · (1+d)^{-β}.
        Higher α → deeper trees.
    beta : float, default=0.5
        Chipman prior depth penalty.  Higher β → shallower trees.
    prior_strength : float, default=1.0
        Total Dirichlet prior mass at each leaf: α_k = prior_strength / n_classes.
        Lower → weaker prior; higher → more regularised leaf estimates.
    max_depth : int, default=6
        Hard maximum tree depth (prevents unbounded growth).
    n_jobs : int, default=-1
        Parallel workers for running chains.  -1 = all logical cores.
    random_state : int, default=42
    verbose : int, default=0
        Set to 1 to print per-chain progress.

    Attributes
    ----------
    samples_ : list[_Node]
        Pooled posterior tree samples (length = n_chains * n_samples).
    classes_ : ndarray
    prior_ : ndarray, shape (n_classes,)
    n_classes_ : int
    """

    def __init__(
        self,
        n_samples: int    = 500,
        n_burnin: int     = 250,
        n_chains: int     = 4,
        alpha: float      = 0.95,
        beta: float       = 0.5,
        prior_strength: float = 1.0,
        max_depth: int    = 6,
        n_jobs: int       = -1,
        random_state: int = 42,
        verbose: int      = 0,
    ):
        self.n_samples      = n_samples
        self.n_burnin       = n_burnin
        self.n_chains       = n_chains
        self.alpha          = alpha
        self.beta           = beta
        self.prior_strength = prior_strength
        self.max_depth      = max_depth
        self.n_jobs         = n_jobs
        self.random_state   = random_state
        self.verbose        = verbose

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_   = unique_labels(y)
        self.n_classes_ = len(self.classes_)

        # Map labels → contiguous 0-based integers
        y_enc = np.searchsorted(self.classes_, y)

        # Symmetric Dirichlet prior
        self.prior_ = np.full(self.n_classes_,
                              self.prior_strength / self.n_classes_)

        rng   = np.random.default_rng(self.random_state)
        seeds = rng.integers(int(1e9), size=self.n_chains)

        if self.verbose:
            print(f"BDTClassifier: {self.n_chains} chains × "
                  f"{self.n_burnin + self.n_samples} iterations "
                  f"(n_jobs={self.n_jobs})", flush=True)

        chain_results = Parallel(n_jobs=self.n_jobs)(
            delayed(_run_chain)(
                int(seeds[c]),
                X, y_enc,
                self.prior_,
                self.alpha, self.beta,
                self.max_depth,
                self.n_burnin, self.n_samples,
                bool(self.verbose),
                c,
            )
            for c in range(self.n_chains)
        )

        self.samples_ = [s for chain in chain_results for s in chain]
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
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
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    from sklearn.datasets import load_iris, load_breast_cancer
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    print("=== BDTClassifier — Iris (4 features, 3 classes) ===")
    X, y = load_iris(return_X_y=True)
    clf  = BDTClassifier(n_samples=200, n_burnin=100, n_chains=4,
                         n_jobs=-1, verbose=1, random_state=42)
    t0   = time.perf_counter()
    scores = cross_val_score(clf, X, y, cv=StratifiedKFold(5),
                             scoring="accuracy")
    dt   = time.perf_counter() - t0
    print(f"5-fold accuracy: {scores.mean():.4f} ± {scores.std():.4f}  "
          f"({dt:.1f}s)\n")

    print("=== BDTClassifier — Breast Cancer (30 features, binary) ===")
    X, y = load_breast_cancer(return_X_y=True)
    clf  = BDTClassifier(n_samples=200, n_burnin=100, n_chains=4,
                         n_jobs=-1, verbose=1, random_state=42)
    t0   = time.perf_counter()
    aucs = []
    for tr, te in StratifiedKFold(5).split(X, y):
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    dt = time.perf_counter() - t0
    scores = np.array(aucs)
    print(f"5-fold AUC: {scores.mean():.4f} ± {scores.std():.4f}  "
          f"({dt:.1f}s)")
