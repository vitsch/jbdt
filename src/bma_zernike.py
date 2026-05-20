"""
BMA over Zernike radial orders, WAIC-weighted, backed by JBDTClassifier.

Model space : M_k = "Zernike moments up to radial order n_max=k" + Haralick.
Evidence    : WAIC computed from each model's MCMC posterior samples.
Weights     : pseudo-BMA (Vehtari et al. 2017), w_k ∝ exp(-WAIC_k / 2).
Uncertainty : two-level — within-model BDT posterior std + between-model BMA variance.

Reliable when N/d ≥ 16 (c_summary_17). Use PSIS-LOO for smaller N (c_summary_16 §6).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from bdt_jakaite import JBDTClassifier, _predict_proba_one

# Default candidate radial orders
_ORDERS_DEFAULT = [4, 6, 8, 10, 12, 14, 16]


# ---------------------------------------------------------------------------
# Zernike index helpers
# ---------------------------------------------------------------------------

def n_zernike(order: int) -> int:
    """Number of Zernike moments with radial degree ≤ order."""
    return sum(
        1 for n in range(order + 1)
        for m in range(n + 1)
        if (n - m) % 2 == 0
    )


# ---------------------------------------------------------------------------
# WAIC and BMA weights
# ---------------------------------------------------------------------------

def waic(clf: JBDTClassifier, X: np.ndarray, y: np.ndarray):
    """
    WAIC from the fitted clf's MCMC posterior samples evaluated on (X, y).

    Returns
    -------
    waic_val : float  — lower = better fit with complexity penalty
    lppd     : float  — log pointwise predictive density
    p_waic   : float  — effective parameter count (posterior variance)
    """
    y_enc = np.searchsorted(clf.classes_, y)
    N = len(X)
    log_liks = np.array([
        np.log(
            _predict_proba_one(tree, X, clf.prior_)[np.arange(N), y_enc] + 1e-300
        )
        for tree in clf.samples_
    ])  # (T, N)
    lppd   = np.log(np.exp(log_liks).mean(axis=0)).sum()
    p_waic = log_liks.var(axis=0).sum()
    return -2.0 * (lppd - p_waic), lppd, p_waic


def bma_weights_from_waic(waic_vals: np.ndarray) -> np.ndarray:
    """Pseudo-BMA weights: w_k ∝ exp(-WAIC_k / 2), normalised."""
    log_w = -0.5 * waic_vals
    log_w -= log_w.max()
    w = np.exp(log_w)
    return w / w.sum()


# ---------------------------------------------------------------------------
# Within-model uncertainty
# ---------------------------------------------------------------------------

def bdt_posterior_std(
    clf: JBDTClassifier, X: np.ndarray, class_idx: int = 1
) -> np.ndarray:
    """
    Std of P(class_idx | x) across MCMC trees — within-model epistemic uncertainty.

    Captures tree-structure uncertainty given the chosen Zernike order.
    Combine with BMAZernikeBDT.predict_uncertainty() for the full two-level
    uncertainty decomposition (c_summary_15 §5).

    Returns (N,) array.
    """
    probs = np.array([
        _predict_proba_one(tree, X, clf.prior_)[:, class_idx]
        for tree in clf.samples_
    ])  # (T, N)
    return probs.std(axis=0)


# ---------------------------------------------------------------------------
# BMA classifier
# ---------------------------------------------------------------------------

class BMAZernikeBDT(BaseEstimator, ClassifierMixin):
    """
    Bayesian Model Averaging over Zernike radial orders.

    Feature layout of X
    -------------------
    Columns 0 : n_zernike(max_order)   — Zernike moments (D=16 → 81 columns)
    Columns n_zernike_total : end       — Haralick / other features (optional)

    For each candidate order k in `orders`, model M_k uses columns
    [0 : n_zernike(k)] + [n_zernike_total : end]. WAIC from each model's
    own MCMC samples gives evidence; pseudo-BMA weights follow.

    If X has no Haralick columns (X.shape[1] == n_zernike_total), the
    feature slice is just X[:, :n_zernike(k)].

    Parameters
    ----------
    orders          : list of int — radial orders to include in the BMA ensemble
    n_zernike_total : int — Zernike column count in X (default 81 for D=16)
    n_samples       : MCMC posterior samples per JBDTClassifier
    n_burnin        : MCMC burn-in steps
    n_chains        : MCMC chains (pooled into n_samples)
    n_jobs          : parallelism for JBDTClassifier (-1 = all cores)
    random_state    : base random seed; each order gets seed + k
    """

    def __init__(
        self,
        orders: list = None,
        n_zernike_total: int = 81,
        n_samples: int = 200,
        n_burnin: int = 100,
        n_chains: int = 4,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.orders          = orders if orders is not None else _ORDERS_DEFAULT
        self.n_zernike_total = n_zernike_total
        self.n_samples       = n_samples
        self.n_burnin        = n_burnin
        self.n_chains        = n_chains
        self.n_jobs          = n_jobs
        self.random_state    = random_state

    # ------------------------------------------------------------------
    def _feature_slice(self, order: int, X: np.ndarray) -> np.ndarray:
        n_k      = n_zernike(order)
        haralick = X[:, self.n_zernike_total:]
        if haralick.shape[1] == 0:
            return X[:, :n_k]
        return np.hstack([X[:, :n_k], haralick])

    def _make_bdt(self, k: int) -> JBDTClassifier:
        return JBDTClassifier(
            n_samples=self.n_samples,
            n_burnin=self.n_burnin,
            n_chains=self.n_chains,
            n_jobs=self.n_jobs,
            random_state=self.random_state + k,
            verbose=0,
        )

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BMAZernikeBDT":
        self.classes_   = np.unique(y)
        self.clfs_      : list[JBDTClassifier] = []
        self.waic_vals_ : np.ndarray
        self.n_moments_ : list[int] = []

        waic_list: list[float] = []
        for k, order in enumerate(self.orders):
            Xk  = self._feature_slice(order, X)
            clf = self._make_bdt(k)
            clf.fit(Xk, y)
            w_val, _, _ = waic(clf, Xk, y)
            self.clfs_.append(clf)
            waic_list.append(w_val)
            self.n_moments_.append(n_zernike(order))

        self.waic_vals_ = np.array(waic_list)
        self.weights_   = bma_weights_from_waic(self.waic_vals_)
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "clfs_")
        P = np.zeros((len(X), len(self.classes_)))
        for w, order, clf in zip(self.weights_, self.orders, self.clfs_):
            Xk = self._feature_slice(order, X)
            P += w * clf.predict_proba(Xk)
        return P

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_uncertainty(
        self, X: np.ndarray, class_idx: int = 1
    ) -> np.ndarray:
        """
        BMA between-model variance for the given class.

        Returns (N,) — epistemic uncertainty from Zernike-order ambiguity.
        Add bdt_posterior_std(clf_k, X) for within-model layer (c_summary_15 §5).
        """
        check_is_fitted(self, "clfs_")
        P_bma = self.predict_proba(X)[:, class_idx]
        var   = np.zeros(len(X))
        for w, order, clf in zip(self.weights_, self.orders, self.clfs_):
            Xk = self._feature_slice(order, X)
            Pk = clf.predict_proba(Xk)[:, class_idx]
            var += w * (Pk - P_bma) ** 2
        return var

    def print_summary(self) -> None:
        """Print the per-order WAIC and BMA weight table."""
        check_is_fitted(self, "clfs_")
        print(f"\n  {'Order':>6} {'n_mom':>7} {'WAIC':>12} {'weight':>10}  bar")
        print("  " + "-" * 52)
        for order, nm, wv, ww in zip(
            self.orders, self.n_moments_, self.waic_vals_, self.weights_
        ):
            bar = "█" * int(round(ww * 24))
            print(f"  D={order:<3}  {nm:>6d}   {wv:>11.2f}   {ww:>9.4f}  {bar}")
        dominant = [o for o, w in zip(self.orders, self.weights_) if w > 0.05]
        print(f"\n  Dominant orders (w > 0.05): {dominant}")
        print(f"  Effective model count (1/Σw²): "
              f"{1.0 / (self.weights_ ** 2).sum():.2f}")
