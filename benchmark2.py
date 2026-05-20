"""
BMA over Zernike Orders — Benchmark 2
Two independent datasets, both with N >> d.

Dataset A — Synthetic Gaussian texture (controlled)
    N = 1000 images (500 smooth + 500 rough), 64×64 px
    Smooth: Gaussian-blurred noise (σ=12) → low-order Zernike moments large
    Rough : Gaussian-blurred noise (σ=2)  → high-order Zernike moments large
    Ground truth: D=6-8 should be most informative
    Purpose: controlled verification that WAIC finds the right order

Dataset B — Olivetti faces (sklearn builtin, real images)
    N = 400, 64×64 px grayscale, 40 subjects × 10 images
    Binary: subjects 0-19 (y=0) vs subjects 20-39 (y=1)
    N/d = 400/25 ≈ 16 >> 1 (reliable WAIC regime, unlike N=40 knee dataset)
    Purpose: real-image validation

Both use:
  - Zernike orders D ∈ {2, 4, 6, 8}
  - Radius = 30 px
  - BDT with WAIC-weighted BMA
  - 5-fold stratified CV (not LOOCV — N large enough)
"""

from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import zernike_moments
from bdt_jakaite import JBDTClassifier, _predict_proba_one

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ORDERS   = [2, 4, 6, 8]          # radial orders to include in BMA
RADIUS   = 30                     # px — fits in 64×64 image
N_FOLDS  = 5
BDT_FAST = dict(n_samples=200, n_burnin=100, n_chains=4, n_jobs=-1, verbose=0)
RNG_SEED = 42

# ---------------------------------------------------------------------------
# Zernike helpers (Zernike-only, no Haralick)
# ---------------------------------------------------------------------------

def _n_zernike(order: int) -> int:
    return sum(1 for n in range(order + 1)
               for m in range(n + 1) if (n - m) % 2 == 0)


def extract_zernike(img: np.ndarray, degree: int = 8) -> np.ndarray:
    """Extract Zernike moments up to given degree from a 64×64 image."""
    return zernike_moments(img.astype(np.float64), radius=RADIUS, degree=degree)


def feature_slice(X_full: np.ndarray, order: int) -> np.ndarray:
    """Columns 0 : n_zernike(order) of the full D=8 feature matrix."""
    return X_full[:, : _n_zernike(order)]


# ---------------------------------------------------------------------------
# WAIC
# ---------------------------------------------------------------------------

def waic(clf: JBDTClassifier, X: np.ndarray, y: np.ndarray) -> float:
    y_enc    = np.searchsorted(clf.classes_, y)
    N        = len(X)
    log_liks = np.array([
        np.log(_predict_proba_one(t, X, clf.prior_)[np.arange(N), y_enc] + 1e-300)
        for t in clf.samples_
    ])
    lppd   = np.log(np.exp(log_liks).mean(0)).sum()
    p_waic = log_liks.var(0).sum()
    return float(-2.0 * (lppd - p_waic))


def bma_weights(waic_vals: np.ndarray) -> np.ndarray:
    log_w  = -0.5 * waic_vals
    log_w -= log_w.max()
    w      = np.exp(log_w)
    return w / w.sum()

# ---------------------------------------------------------------------------
# Dataset A — Synthetic Gaussian texture
# ---------------------------------------------------------------------------

def make_texture_dataset(n_per_class: int = 500, size: int = 64,
                          sigma_smooth: float = 12.0,
                          sigma_rough:  float = 2.0,
                          seed: int = RNG_SEED) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate two-class Gaussian texture dataset.

    Class 0 (smooth): Gaussian-blurred random field (σ=sigma_smooth).
        Power concentrated in low Zernike orders.
    Class 1 (rough):  Lightly blurred random field (σ=sigma_rough).
        Power spread across low and high Zernike orders.
    """
    rng   = np.random.default_rng(seed)
    imgs  = []
    for cls, sigma in [(0, sigma_smooth), (1, sigma_rough)]:
        for _ in range(n_per_class):
            noise = rng.standard_normal((size, size))
            blurred = gaussian_filter(noise, sigma=sigma)
            # Normalise to [0, 1]
            lo, hi = blurred.min(), blurred.max()
            imgs.append((blurred - lo) / (hi - lo + 1e-8))

    X_raw = np.array(imgs)                              # (2N, H, W)
    y     = np.array([0] * n_per_class + [1] * n_per_class)
    return X_raw, y


def extract_features_batch(images: np.ndarray, degree: int = 8) -> np.ndarray:
    """Extract Zernike features from all images up to given degree."""
    return np.stack([extract_zernike(img, degree) for img in images])

# ---------------------------------------------------------------------------
# Dataset B — Olivetti faces
# ---------------------------------------------------------------------------

def load_olivetti() -> tuple[np.ndarray, np.ndarray]:
    """
    Load Olivetti faces.  Binary: subjects 0-19 (y=0) vs 20-39 (y=1).
    Returns raw images (N, 64, 64) and labels.
    """
    data = fetch_olivetti_faces()
    imgs = data.images                          # (400, 64, 64) float32 [0,1]
    y    = (data.target >= 20).astype(int)      # 0 = subjects 0-19, 1 = 20-39
    return imgs.astype(np.float64), y

# ---------------------------------------------------------------------------
# CV benchmark runner
# ---------------------------------------------------------------------------

def run_cv(X_full: np.ndarray, y: np.ndarray,
           dataset_name: str, n_splits: int = N_FOLDS,
           seed: int = RNG_SEED) -> dict:
    """
    5-fold stratified CV.
    For each fold, on the training set:
      1. Fit one BDT per Zernike order
      2. Compute WAIC → BMA weights
      3. Predict validation set with each BDT and with BMA ensemble

    Also computes full-data WAIC for weight visualisation.
    """
    skf   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    N     = len(y)
    K     = len(ORDERS)
    max_d = max(_n_zernike(o) for o in ORDERS)

    # Accumulators
    accs     = {f"bdt_d{o}": [] for o in ORDERS}
    accs["bma"] = []
    aucs     = {f"bdt_d{o}": [] for o in ORDERS}
    aucs["bma"] = []
    fold_weights = np.zeros((n_splits, K))

    print(f"\n  {dataset_name}: N={N}, d(D=8)={_n_zernike(8)}, "
          f"N/d={N//_n_zernike(8)} >> 1")

    t_start = time.perf_counter()
    for fold_idx, (tr, te) in enumerate(skf.split(X_full, y)):
        X_tr, y_tr = X_full[tr], y[tr]
        X_te, y_te = X_full[te], y[te]

        clfs, waic_vals = [], []
        for k, order in enumerate(ORDERS):
            Xk_tr = feature_slice(X_tr, order)
            clf   = JBDTClassifier(random_state=seed + fold_idx * K + k, **BDT_FAST)
            clf.fit(Xk_tr, y_tr)
            waic_vals.append(waic(clf, Xk_tr, y_tr))
            clfs.append(clf)

        w_arr = bma_weights(np.array(waic_vals))
        fold_weights[fold_idx] = w_arr

        # Predict
        p_bma = np.zeros((len(X_te), 2))
        for w, order, clf in zip(w_arr, ORDERS, clfs):
            Xk_te  = feature_slice(X_te, order)
            p_k    = clf.predict_proba(Xk_te)
            pred_k = p_k.argmax(1)

            accs[f"bdt_d{order}"].append((pred_k == y_te).mean())
            aucs[f"bdt_d{order}"].append(
                roc_auc_score(y_te, p_k[:, 1]) if len(np.unique(y_te)) > 1 else 0.5)

            p_bma += w * p_k

        bma_pred = p_bma.argmax(1)
        accs["bma"].append((bma_pred == y_te).mean())
        aucs["bma"].append(
            roc_auc_score(y_te, p_bma[:, 1]) if len(np.unique(y_te)) > 1 else 0.5)

        print(f"    Fold {fold_idx+1}/{n_splits}: "
              f"BMA acc={accs['bma'][-1]:.3f}  "
              f"BDT-D8={accs['bdt_d8'][-1]:.3f}  "
              f"weights={w_arr.round(3)}", flush=True)

    t_cv = time.perf_counter() - t_start

    # Full-data WAIC for display
    print(f"\n  Full-data WAIC (all {N} samples):")
    waic_full = []
    for k, order in enumerate(ORDERS):
        Xk = feature_slice(X_full, order)
        clf = JBDTClassifier(random_state=seed + 9999 + k, **BDT_FAST)
        clf.fit(Xk, y)
        wv = waic(clf, Xk, y)
        waic_full.append(wv)
    waic_full  = np.array(waic_full)
    w_full     = bma_weights(waic_full)

    return {
        "dataset":       dataset_name,
        "N":             N,
        "cv_acc":        {k: (np.mean(v), np.std(v)) for k, v in accs.items()},
        "cv_auc":        {k: (np.mean(v), np.std(v)) for k, v in aucs.items()},
        "fold_weights":  fold_weights,
        "waic_full":     waic_full,
        "weights_full":  w_full,
        "cv_time_s":     t_cv,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(res: dict, waic_theory: dict | None = None) -> None:
    ds    = res["dataset"]
    acc   = res["cv_acc"]
    auc   = res["cv_auc"]
    wfull = res["waic_full"]
    ww    = res["weights_full"]

    print(f"\n{'='*60}")
    print(f"  {ds}")
    print(f"{'='*60}")
    print(f"\n  {'Method':<20} {'Acc (mean±std)':>18} {'AUC (mean±std)':>18}")
    print(f"  {'-'*58}")
    for order in ORDERS:
        k   = f"bdt_d{order}"
        m_a, s_a = acc[k]
        m_u, s_u = auc[k]
        print(f"  BDT  D={order:<2}           {m_a*100:6.1f} ± {s_a*100:.1f}%"
              f"       {m_u:.3f} ± {s_u:.3f}")
    m_a, s_a = acc["bma"]
    m_u, s_u = auc["bma"]
    print(f"  BMA-BDT (WAIC)      {m_a*100:6.1f} ± {s_a*100:.1f}%"
          f"       {m_u:.3f} ± {s_u:.3f}  ← proposed")

    print(f"\n  Full-data WAIC per order:")
    print(f"  {'Order':<8} {'n_mom':>6} {'WAIC':>12} {'weight':>10}")
    print(f"  {'-'*38}")
    for order, wv, w in zip(ORDERS, wfull, ww):
        bar  = "█" * int(round(w * 30))
        print(f"  D={order:<2}      {_n_zernike(order):>6d}   {wv:>11.2f}   {w:>8.4f}  {bar}")
    best_order = ORDERS[int(np.argmin(wfull))]
    print(f"  → WAIC selects D={best_order} as best-fitting order")

    if waic_theory:
        print(f"\n  Expected best order (from data design): D={waic_theory['expected']}")
        match = (best_order == waic_theory["expected"])
        print(f"  WAIC matches expectation: {'YES ✓' if match else 'NO ✗'}")

    mean_fw = res["fold_weights"].mean(0)
    print(f"\n  Mean CV-fold BMA weights:")
    for order, w in zip(ORDERS, mean_fw):
        bar = "█" * int(round(w * 40))
        print(f"    D={order:<2}  {w:.4f}  {bar}")

    print(f"\n  CV time: {res['cv_time_s']:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    all_results = {}

    # ── Dataset A: Synthetic ─────────────────────────────────────────────
    print("\n" + "#"*60)
    print("  Dataset A: Synthetic Gaussian Texture")
    print("  Class 0 (smooth): σ=12  →  expect low-order Zernike (D≤4)")
    print("  Class 1 (rough):  σ=2   →  expect high-order Zernike (D≥6)")
    print("#"*60)

    t0 = time.perf_counter()
    X_raw, y_synth = make_texture_dataset(n_per_class=500)
    print(f"  Generated {len(y_synth)} images in {time.perf_counter()-t0:.2f}s")

    print("  Extracting Zernike D=8 features …")
    t0 = time.perf_counter()
    X_synth = extract_features_batch(X_raw, degree=8)     # (1000, 25)
    print(f"  X_synth shape: {X_synth.shape}  [{time.perf_counter()-t0:.1f}s]")

    # Quick diagnostic: mean Zernike energy per order per class
    print("\n  Zernike energy (mean |A_nm|) per class per order:")
    print(f"  {'Order':<8} {'n_mom':>6} {'Class0 energy':>14} {'Class1 energy':>14}")
    prev = 0
    for order in ORDERS:
        nk = _n_zernike(order)
        e0 = X_synth[y_synth == 0][:, prev:nk].mean()
        e1 = X_synth[y_synth == 1][:, prev:nk].mean()
        ratio = e1 / (e0 + 1e-8)
        print(f"  D={order:<2}      {nk - prev:>6d}       {e0:>12.3f}       {e1:>12.3f}   ratio={ratio:.2f}")
        prev = nk

    res_synth = run_cv(X_synth, y_synth, "Synthetic Gaussian Texture",
                       seed=RNG_SEED)
    print_report(res_synth, waic_theory={"expected": 8})

    # ── Dataset B: Olivetti faces ────────────────────────────────────────
    print("\n" + "#"*60)
    print("  Dataset B: Olivetti Faces")
    print("  Binary: subjects 0-19 (y=0) vs subjects 20-39 (y=1)")
    print("#"*60)

    t0 = time.perf_counter()
    imgs_oliv, y_oliv = load_olivetti()
    print(f"  Loaded {len(y_oliv)} images in {time.perf_counter()-t0:.2f}s")

    print("  Extracting Zernike D=8 features …")
    t0 = time.perf_counter()
    X_oliv = extract_features_batch(imgs_oliv, degree=8)   # (400, 25)
    print(f"  X_oliv shape: {X_oliv.shape}  [{time.perf_counter()-t0:.1f}s]")

    res_oliv = run_cv(X_oliv, y_oliv, "Olivetti Faces (binary)",
                      seed=RNG_SEED)
    print_report(res_oliv)

    # ── Save JSON ────────────────────────────────────────────────────────
    def _ser(d):
        if isinstance(d, np.ndarray): return d.tolist()
        if isinstance(d, dict):       return {k: _ser(v) for k, v in d.items()}
        if isinstance(d, tuple):      return list(d)
        return d

    out = Path(__file__).parent / "results" / "bma_benchmark2.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(_ser({"synthetic": res_synth, "olivetti": res_oliv}), f, indent=2)
    print(f"\nResults saved → {out}")
