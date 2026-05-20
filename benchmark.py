"""
BMA over Zernike Orders — Benchmark on Jakaite et al. (2021) Dataset
Data: 40 knee X-ray ROIs (20 Control + 20 Case), Lateral and Medial compartments
      https://doi.org/10.6084/m9.figshare.8303996
Goal: verify that WAIC-weighted BMA over Zernike orders (D=4…16) improves
      on fixed D=16, and that BMA weights concentrate on high-order moments
      as reported in Figure 1 of the paper.
"""

from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))
from features import zernike_moments, haralick_features
from bdt_jakaite import JBDTClassifier, _predict_proba_one

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR  = Path(__file__).parent / "data"
CTRL_DIR  = DATA_DIR / "control"
CASE_DIR  = DATA_DIR / "case"

ORDERS    = [4, 6, 8, 10, 12, 14, 16]
N_ZERNIKE = 81                          # moments at D=16

BDT_KWARGS = dict(n_samples=100, n_burnin=50, n_chains=2, n_jobs=2,
                  verbose=0)

GLCM_LEVELS = 64

# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _n_zernike(order: int) -> int:
    return sum(1 for n in range(order + 1)
               for m in range(n + 1) if (n - m) % 2 == 0)


def load_image(path: Path) -> np.ndarray:
    """Load 16-bit TIFF; normalise to float64 [0, 1]."""
    arr = np.array(Image.open(path))            # uint16
    lo, hi = float(arr.min()), float(arr.max())
    return (arr.astype(np.float64) - lo) / (hi - lo + 1e-8)


def extract_features(img: np.ndarray) -> np.ndarray:
    """Return 95-d feature vector: 81 Zernike (D=16) + 14 Haralick."""
    radius = min(img.shape) // 2
    z = zernike_moments(img, radius=radius, degree=16)          # (81,)
    h = haralick_features(img, levels=GLCM_LEVELS)              # (14,)
    return np.concatenate([z, h])


def feature_slice(X: np.ndarray, order: int) -> np.ndarray:
    """Select Zernike columns up to radial order + all Haralick columns."""
    k = _n_zernike(order)
    return np.hstack([X[:, :k], X[:, N_ZERNIKE:]])


# ---------------------------------------------------------------------------
# WAIC
# ---------------------------------------------------------------------------

def waic(clf: JBDTClassifier, X: np.ndarray, y: np.ndarray):
    """WAIC from MCMC posterior samples; lower = better."""
    y_enc    = np.searchsorted(clf.classes_, y)
    N        = len(X)
    log_liks = np.array([
        np.log(
            _predict_proba_one(t, X, clf.prior_)[np.arange(N), y_enc] + 1e-300
        )
        for t in clf.samples_
    ])                                           # (T, N)
    lppd   = np.log(np.exp(log_liks).mean(0)).sum()
    p_waic = log_liks.var(0).sum()
    return -2.0 * (lppd - p_waic), lppd, p_waic


def bma_weights(waic_vals: np.ndarray) -> np.ndarray:
    log_w  = -0.5 * waic_vals
    log_w -= log_w.max()
    w      = np.exp(log_w)
    return w / w.sum()


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(roi_type: str):
    """
    Load all images for one ROI type ('lateral' or 'medial').
    Returns X (40, 95), y (40,), patient_ids list.
    """
    roi = roi_type.upper()
    ctrl_imgs = sorted((CTRL_DIR / roi_type).glob(f"*_{roi}.tiff"))
    case_imgs = sorted((CASE_DIR / roi_type).glob(f"*_{roi}.tiff"))

    assert len(ctrl_imgs) == 20, f"Expected 20 control images, got {len(ctrl_imgs)}"
    assert len(case_imgs) == 20, f"Expected 20 case images,    got {len(case_imgs)}"

    feats, labels, ids = [], [], []
    for path, label in [(p, 0) for p in ctrl_imgs] + [(p, 1) for p in case_imgs]:
        img = load_image(path)
        feats.append(extract_features(img))
        labels.append(label)
        ids.append(path.stem)

    return np.stack(feats), np.array(labels), ids


# ---------------------------------------------------------------------------
# LOOCV runner
# ---------------------------------------------------------------------------

def loocv(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """
    Leave-one-out CV.  For each fold:
      - Fit 7 BDTs (one per Zernike order) on training set
      - Compute WAIC → BMA weights
      - Predict test sample using BMA ensemble and each individual model

    Returns dict with per-sample predictions for each condition.
    """
    N = len(y)
    results = {
        "bma":    np.zeros(N, dtype=int),
        "bma_p":  np.zeros((N, 2)),
        "bma_w":  np.zeros((N, len(ORDERS))),   # per-fold weights
        "bma_waic": np.zeros((N, len(ORDERS))), # per-fold WAIC values
    }
    for order in ORDERS:
        results[f"bdt_d{order}"] = np.zeros(N, dtype=int)

    for i in range(N):
        tr = [j for j in range(N) if j != i]
        X_tr, y_tr = X[tr], y[tr]
        X_te        = X[i:i+1]

        clfs, waic_vals = [], []
        for k, order in enumerate(ORDERS):
            Xk_tr = feature_slice(X_tr, order)
            clf   = JBDTClassifier(random_state=random_state + k, **BDT_KWARGS)
            clf.fit(Xk_tr, y_tr)

            w_val, _, _ = waic(clf, Xk_tr, y_tr)
            clfs.append(clf)
            waic_vals.append(w_val)

            # Store individual BDT prediction
            Xk_te = feature_slice(X_te, order)
            results[f"bdt_d{order}"][i] = clf.predict(Xk_te)[0]

        # BMA
        w_arr = bma_weights(np.array(waic_vals))
        p_bma = np.zeros(2)
        for w, order, clf in zip(w_arr, ORDERS, clfs):
            p_bma += w * clf.predict_proba(feature_slice(X_te, order))[0]

        results["bma"][i]      = int(p_bma.argmax())
        results["bma_p"][i]    = p_bma
        results["bma_w"][i]    = w_arr
        results["bma_waic"][i] = waic_vals

        if (i + 1) % 10 == 0:
            print(f"  Fold {i+1:2d}/{N} done", flush=True)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def accuracy(y_true, y_pred):
    return 100.0 * (y_true == y_pred).mean()


def print_results(roi_type: str, y: np.ndarray, results: dict,
                  waic_full: np.ndarray, weights_full: np.ndarray):
    print(f"\n{'='*60}")
    print(f"  ROI type : {roi_type.upper()}")
    print(f"{'='*60}")

    print(f"\n{'Condition':<22} {'LOOCV Acc (%)':>14}")
    print("-" * 38)

    # Table 1 reference from the paper
    ref = {"lateral": {"RF": 80.0, "SVM": 82.5, "ANN": 80.0, "GMDH": 85.0},
           "medial":  {"RF": 72.5, "SVM": 75.0, "ANN": 75.0, "GMDH": 77.5}}
    for name, acc in ref[roi_type.lower()].items():
        print(f"  Paper: {name:<15} {acc:>12.1f}")
    print()

    for order in ORDERS:
        acc = accuracy(y, results[f"bdt_d{order}"])
        print(f"  BDT  D={order:2d}            {acc:>12.1f}")
    acc_bma = accuracy(y, results["bma"])
    print(f"  BMA-BDT (WAIC)       {acc_bma:>12.1f}  ← proposed")

    print(f"\n  WAIC per order (full-data, lower=better):")
    print(f"  {'Order':<8} {'n_mom':>6} {'WAIC':>12} {'weight':>10}")
    print(f"  {'-'*38}")
    for order, wv, ww in zip(ORDERS, waic_full, weights_full):
        nk = _n_zernike(order)
        print(f"  D={order:2d}     {nk:>6d}   {wv:>11.2f}   {ww:>9.4f}")

    # Mean per-fold weights
    mean_w = results["bma_w"].mean(axis=0)
    print(f"\n  Mean per-fold BMA weights:")
    for order, w in zip(ORDERS, mean_w):
        bar = "█" * int(round(w * 40))
        print(f"    D={order:2d}  {w:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Full-data WAIC (for display only; not used in LOOCV)
# ---------------------------------------------------------------------------

def full_data_waic(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    waic_vals = []
    for k, order in enumerate(ORDERS):
        Xk  = feature_slice(X, order)
        clf = JBDTClassifier(random_state=random_state + k, **BDT_KWARGS)
        clf.fit(Xk, y)
        wv, _, _ = waic(clf, Xk, y)
        waic_vals.append(wv)
        print(f"    Full-data WAIC D={order:2d}: {wv:.2f}", flush=True)
    return np.array(waic_vals)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    all_results = {}

    for roi_type in ["lateral", "medial"]:
        print(f"\n{'#'*60}")
        print(f"  Loading {roi_type.upper()} images …")
        t0 = time.perf_counter()
        X, y, ids = load_dataset(roi_type)
        print(f"  Feature matrix: {X.shape}  labels: {y.sum()} case / {(y==0).sum()} ctrl")
        print(f"  Feature extraction: {time.perf_counter()-t0:.1f}s")

        print(f"\n  Computing full-data WAIC …")
        waic_full    = full_data_waic(X, y, random_state=42)
        weights_full = bma_weights(waic_full)

        print(f"\n  Running LOOCV (N=40, {len(ORDERS)} orders × 40 folds) …")
        t0 = time.perf_counter()
        res = loocv(X, y, random_state=42)
        print(f"  LOOCV time: {time.perf_counter()-t0:.1f}s")

        print_results(roi_type, y, res, waic_full, weights_full)

        # Save summary for c_summary_16
        summary = {
            "roi_type":     roi_type,
            "n_total":      int(len(y)),
            "n_case":       int(y.sum()),
            "n_ctrl":       int((y == 0).sum()),
            "waic_per_order": {
                f"D{o}": {"waic": float(wv), "weight": float(ww), "n_moments": _n_zernike(o)}
                for o, wv, ww in zip(ORDERS, waic_full, weights_full)
            },
            "loocv_accuracy": {
                **{f"BDT_D{o}": float(accuracy(y, res[f"bdt_d{o}"])) for o in ORDERS},
                "BMA_BDT": float(accuracy(y, res["bma"])),
            },
            "mean_fold_weights": {
                f"D{o}": float(w) for o, w in zip(ORDERS, res["bma_w"].mean(0))
            },
        }
        all_results[roi_type] = summary

    # Save JSON for c_summary_16
    out = Path(__file__).parent / "results" / "bma_benchmark.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out}")
