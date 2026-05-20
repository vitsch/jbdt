"""
Feature extraction for knee X-ray classification.

Implements the feature set from Jakaite, Schetinin et al. (2021):
    - Zernike moments  (order 16 → 81 features)
    - Haralick GLCM features (14 features, averaged over 4 directions)
    Total: 95 features per image

Pure numpy/scipy implementation — no mahotas or skimage required.
Batch extraction is parallelised via joblib.

Usage
-----
    from features import ZernikeHaralickExtractor
    ext = ZernikeHaralickExtractor(radius=100, zernike_degree=16)
    X = ext.extract_batch(images, n_jobs=-1)   # shape (N, 95)
"""

from __future__ import annotations

import numpy as np
from math import factorial
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# Zernike moments
# ---------------------------------------------------------------------------

def _zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """
    Radial polynomial R_n^|m|(rho) of the Zernike basis.

    R_n^m(rho) = sum_{k=0}^{(n-m)//2}
                    [(-1)^k (n-k)!] / [k! ((n+m)/2-k)! ((n-m)/2-k)!]
                    * rho^(n-2k)
    """
    m = abs(m)
    assert (n - m) % 2 == 0, "n-|m| must be even"
    R = np.zeros_like(rho, dtype=np.float64)
    for k in range((n - m) // 2 + 1):
        num = (-1) ** k * factorial(n - k)
        den = factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        R += (num / den) * rho ** (n - 2 * k)
    return R


def zernike_moments(
    image: np.ndarray,
    radius: int = 100,
    degree: int = 16,
) -> np.ndarray:
    """
    Compute Zernike moment magnitudes |A_n^m| up to given degree.

    Parameters
    ----------
    image : (H, W) array, grayscale uint8 or float in [0, 1]
    radius : pixels — defines the unit circle within which moments are computed
    degree : maximum Zernike order (default 16 → 81 moments)

    Returns
    -------
    moments : ndarray, shape (81,) for degree=16
        Number of moments = sum_{n=0}^{degree} floor(n/2) + 1
    """
    if image.ndim == 3:
        image = _to_gray(image)
    img = image.astype(np.float64)

    h, w = img.shape
    cy, cx = h / 2.0, w / 2.0

    # Polar coordinates normalised to unit circle
    y_idx, x_idx = np.mgrid[:h, :w]
    y = (y_idx - cy) / radius
    x = (x_idx - cx) / radius
    rho   = np.hypot(x, y)
    theta = np.arctan2(y, x)

    mask    = rho <= 1.0
    rho_m   = rho[mask]
    theta_m = theta[mask]
    f_m     = img[mask]

    moments = []
    for n in range(degree + 1):
        for m in range(n + 1):
            if (n - m) % 2 != 0:
                continue
            R = _zernike_radial(n, m, rho_m)
            Z = R * np.exp(1j * m * theta_m)
            # Moment magnitude, normalised by unit-circle area
            A = np.abs(np.dot(f_m, np.conj(Z))) * (n + 1) / np.pi
            moments.append(A)

    return np.array(moments, dtype=np.float64)


# ---------------------------------------------------------------------------
# Haralick GLCM features
# ---------------------------------------------------------------------------

_EPS = 1e-12

# Directions: (dy, dx) for 0°, 45°, 90°, 135°
_DIRECTIONS = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]


def _build_glcm(image: np.ndarray, dy: int, dx: int, levels: int) -> np.ndarray:
    """
    Build a normalised, symmetric GLCM for one (dy, dx) displacement.
    Only pixels with valid neighbours in both directions are counted.
    """
    glcm = np.zeros((levels, levels), dtype=np.float64)

    r0 = slice(max(0, -dy), image.shape[0] + min(0, -dy))
    c0 = slice(max(0, -dx), image.shape[1] + min(0, -dx))
    r1 = slice(max(0,  dy), image.shape[0] + min(0,  dy))
    c1 = slice(max(0,  dx), image.shape[1] + min(0,  dx))

    i_vals = image[r0, c0].ravel().astype(int)
    j_vals = image[r1, c1].ravel().astype(int)

    np.add.at(glcm, (i_vals, j_vals), 1)

    # Symmetrise (count both (i→j) and (j→i) transitions)
    glcm = glcm + glcm.T
    total = glcm.sum()
    if total > 0:
        glcm /= total
    return glcm


def _haralick_from_glcm(P: np.ndarray) -> np.ndarray:
    """
    Compute all 14 Haralick texture features from a normalised GLCM P.

    Haralick, R.M., Shanmugam, K., Dinstein, I. (1973).
    IEEE Trans. Systems Man Cybernetics, 3(6), 610–621.
    """
    N   = P.shape[0]
    idx = np.arange(N, dtype=np.float64)

    # Marginal distributions
    px = P.sum(axis=1)   # shape (N,)
    py = P.sum(axis=0)   # shape (N,)

    # Grid of (i, j) index differences / sums
    I, J = np.meshgrid(idx, idx, indexing="ij")  # (N, N)

    mu_x    = (px * idx).sum()
    mu_y    = (py * idx).sum()
    sigma_x = np.sqrt(((idx - mu_x) ** 2 * px).sum()) + _EPS
    sigma_y = np.sqrt(((idx - mu_y) ** 2 * py).sum()) + _EPS

    # P_{x+y}[k] for k = 0 … 2N-2
    P_xpy = np.zeros(2 * N - 1)
    for k in range(2 * N - 1):
        mask = (I + J).astype(int) == k
        P_xpy[k] = P[mask].sum()
    xpy_idx = np.arange(2 * N - 1, dtype=np.float64)

    # P_{x-y}[k] for k = 0 … N-1
    P_xmy = np.zeros(N)
    for k in range(N):
        mask = np.abs(I - J).astype(int) == k
        P_xmy[k] = P[mask].sum()
    xmy_idx = np.arange(N, dtype=np.float64)

    # Entropy helpers
    HXY  = -(P    * np.log(P    + _EPS)).sum()
    HX   = -(px   * np.log(px   + _EPS)).sum()
    HY   = -(py   * np.log(py   + _EPS)).sum()

    pxpy = np.outer(px, py)
    HXY1 = -(P    * np.log(pxpy + _EPS)).sum()
    HXY2 = -(pxpy * np.log(pxpy + _EPS)).sum()

    # --- 14 features ---

    # 1. Angular Second Moment (Energy)
    f1 = (P ** 2).sum()

    # 2. Contrast
    f2 = ((I - J) ** 2 * P).sum()

    # 3. Correlation
    f3 = ((I * J * P).sum() - mu_x * mu_y) / (sigma_x * sigma_y)

    # 4. Sum of Squares (Variance)
    mu = (I * P).sum()
    f4 = (((I - mu) ** 2) * P).sum()

    # 5. Inverse Difference Moment (Homogeneity)
    f5 = (P / (1 + (I - J) ** 2)).sum()

    # 6. Sum Average
    f6 = (xpy_idx * P_xpy).sum()

    # 7. Sum Variance
    SA = f6
    f7 = ((xpy_idx - SA) ** 2 * P_xpy).sum()

    # 8. Sum Entropy
    f8 = -(P_xpy * np.log(P_xpy + _EPS)).sum()

    # 9. Entropy
    f9 = HXY

    # 10. Difference Variance
    mean_xmy = (xmy_idx * P_xmy).sum()
    f10 = ((xmy_idx - mean_xmy) ** 2 * P_xmy).sum()

    # 11. Difference Entropy
    f11 = -(P_xmy * np.log(P_xmy + _EPS)).sum()

    # 12. Information Measure of Correlation 1
    denom12 = max(HX, HY)
    f12 = (HXY - HXY1) / (denom12 + _EPS)

    # 13. Information Measure of Correlation 2
    val13 = 1.0 - np.exp(-2.0 * max(0.0, HXY2 - HXY))
    f13 = np.sqrt(val13)

    # 14. Maximal Correlation Coefficient
    # Q_ij = sum_k P[i,k]*P[j,k] / (px[i]*py[k]+eps)
    # MCC = sqrt(2nd-largest eigenvalue of Q)
    # Numerically stabilised: use Q = (P / (px[:,None]*py[None,:]+eps))
    Q = (P / (pxpy + _EPS)) @ (P / (pxpy + _EPS)).T
    eigvals = np.sort(np.linalg.eigvalsh(Q))[::-1]
    f14 = np.sqrt(max(0.0, eigvals[1] if len(eigvals) > 1 else 0.0))

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14])


def haralick_features(
    image: np.ndarray,
    levels: int = 64,
) -> np.ndarray:
    """
    Compute 14 Haralick features averaged over 4 directions (0°, 45°, 90°, 135°).

    Parameters
    ----------
    image : (H, W) grayscale uint8 array
    levels : number of grey levels for GLCM quantisation (default 64)

    Returns
    -------
    features : ndarray, shape (14,)
    """
    if image.ndim == 3:
        image = _to_gray(image)

    # Quantise to [0, levels-1]
    img_q = _quantise(image, levels)

    feats = np.stack([
        _haralick_from_glcm(_build_glcm(img_q, dy, dx, levels))
        for dy, dx in _DIRECTIONS
    ])
    return feats.mean(axis=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA to grayscale via luminosity formula."""
    if image.shape[2] == 4:
        image = image[..., :3]
    return (0.2989 * image[..., 0] +
            0.5870 * image[..., 1] +
            0.1140 * image[..., 2])


def _quantise(image: np.ndarray, levels: int) -> np.ndarray:
    """Map pixel intensities linearly to [0, levels-1] as uint16."""
    img = image.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return (img * (levels - 1)).astype(np.uint16)


def _extract_one(image: np.ndarray, radius: int, degree: int, levels: int) -> np.ndarray:
    """Extract Zernike + Haralick features from one image (worker for joblib)."""
    if image.ndim == 3:
        image = _to_gray(image)
    z = zernike_moments(image, radius=radius, degree=degree)
    h = haralick_features(image.astype(np.uint8), levels=levels)
    return np.concatenate([z, h])


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class ZernikeHaralickExtractor:
    """
    Extract Zernike + Haralick features from a batch of grayscale images.

    Reproduces the 95-feature vector (81 Zernike + 14 Haralick) used in
    Jakaite, Schetinin et al. (2021) for knee OA classification.

    Parameters
    ----------
    radius : int, default=100
        Radius (pixels) of the unit circle for Zernike computation.
        Should be chosen so that the ROI fits within the circle.
    zernike_degree : int, default=16
        Maximum Zernike polynomial order.  Degree 16 → 81 moments.
    glcm_levels : int, default=64
        Grey levels for GLCM quantisation.  64 balances speed vs. resolution.
    n_jobs : int, default=-1
        Parallel workers for batch extraction.

    Attributes
    ----------
    n_features_ : int   — total features per image (81 + 14 = 95)
    feature_names_ : list[str]
    """

    def __init__(
        self,
        radius: int = 100,
        zernike_degree: int = 16,
        glcm_levels: int = 64,
        n_jobs: int = -1,
    ):
        self.radius         = radius
        self.zernike_degree = zernike_degree
        self.glcm_levels    = glcm_levels
        self.n_jobs         = n_jobs

        # Derive feature names
        z_names = [
            f"Z{n}_{m}"
            for n in range(zernike_degree + 1)
            for m in range(n + 1)
            if (n - m) % 2 == 0
        ]
        h_names = [
            "H_energy", "H_contrast", "H_correlation",
            "H_variance", "H_homogeneity",
            "H_sum_avg", "H_sum_var", "H_sum_entropy",
            "H_entropy", "H_diff_var", "H_diff_entropy",
            "H_imc1", "H_imc2", "H_mcc",
        ]
        self.feature_names_: list[str] = z_names + h_names
        self.n_features_: int = len(self.feature_names_)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract features from a single image. Returns shape (95,)."""
        return _extract_one(
            image, self.radius, self.zernike_degree, self.glcm_levels
        )

    def extract_batch(self, images: list | np.ndarray) -> np.ndarray:
        """
        Extract features from a list/array of images in parallel.

        Parameters
        ----------
        images : list of (H, W) or (H, W, C) arrays

        Returns
        -------
        X : ndarray, shape (N, 95)
        """
        rows = Parallel(n_jobs=self.n_jobs)(
            delayed(_extract_one)(
                img, self.radius, self.zernike_degree, self.glcm_levels
            )
            for img in images
        )
        return np.vstack(rows)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    rng = np.random.default_rng(42)
    # Simulate 10 grayscale knee X-ray patches, 224×224
    images = [rng.integers(0, 256, (224, 224), dtype=np.uint8) for _ in range(10)]

    ext = ZernikeHaralickExtractor(radius=100, zernike_degree=16, n_jobs=-1)
    print(f"Feature vector length: {ext.n_features_}  "
          f"(expected 95 = 81 Zernike + 14 Haralick)")
    print(f"Names[:5]: {ext.feature_names_[:5]}")
    print(f"Names[-5:]: {ext.feature_names_[-5:]}")

    t0 = time.perf_counter()
    X  = ext.extract_batch(images)
    dt = time.perf_counter() - t0

    print(f"\nExtracted {X.shape} in {dt:.2f}s ({dt/len(images):.2f}s/image)")
    print(f"Feature range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"NaN count: {np.isnan(X).sum()}")
