"""
§2.3 + §2.5 Experiments
  §2.3  BMA Oracle Inequality: excess risk bound for WAIC-weighted BMA
  §2.5  Calibration ECE Bound: posterior predictive calibration under Catalan-DM

Experiments:
  A. WAIC weight convergence: w_{k*} → 1 as N grows; WAIC selection rate
  B. Jensen bound verification: R_BMA ≤ Σ_k w_k R_k
  C. Excess risk convergence: R_BMA − R_{k*} → 0, theory bound tightness
  D. ECE analytical bounds: Catalan vs Chipman vs Geometric (E[k] comparison)
  E. Empirical ECE vs N: simulated calibration curves for Catalan vs Chipman trees
"""

import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from bma_oracle import bma_oracle_sweep, kl_divergence
from ece_calibration import (
    dm_mse_bound, dm_ece_bound_leaf, dm_ece_bound_tree,
    dm_ece_bound_asymptotic, prior_averaged_ece_bound, calibration_sim
)
from catalan_prior import catalan_pmf, catalan_moments, chipman_pmf

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(42)
K_MAX = 20
results = {}

# True model parameters
K_TRUE = 3
C = 2
ALPHA = 1.0

# True leaf probs used in ALL experiments for consistency
LEAF_PROBS = [
    np.array([0.1, 0.9]),
    np.array([0.5, 0.5]),
    np.array([0.9, 0.1]),
]

# ---------------------------------------------------------------------------
# Experiment A + C: WAIC weight convergence and excess risk
# (Single sweep, extract all quantities)
# ---------------------------------------------------------------------------
print("=" * 60)
print("§2.3 Experiments A + C: BMA oracle sweep")
print("=" * 60)

N_vals_bma = np.array([20, 40, 80, 150, 300, 500, 800])
sweep = bma_oracle_sweep(
    N_vals_bma, k_true=K_TRUE, C=C, k_max=8,
    alpha=ALPHA, n_reps=80, rng=rng
)

print(f"\n  True k*={K_TRUE}, C={C}, leaf probs=[0.1/0.9, 0.5/0.5, 0.9/0.1]")
print(f"\n  Experiment A: WAIC weight on k* and WAIC selection rate")
print(f"  {'N':>5}  {'w_{k*}':>10}  {'1-w_{k*}':>10}  {'P(WAIC=k*)':>12}")
for i, N in enumerate(N_vals_bma):
    w_ks = sweep["w_kstar"][i]
    frac  = sweep["k_waic_best_frac"][i]
    print(f"  {N:>5}  {w_ks:>10.4f}  {1-w_ks:>10.4e}  {frac:>12.4f}")

print(f"\n  Experiment C: BMA excess risk and oracle bounds")
print(f"  {'N':>5}  {'R_BMA':>9}  {'R_orac':>9}  {'excess':>9}  "
      f"{'Jensen_ex':>11}  {'Je/ex':>7}  {'R_BMA/R_orac':>13}")
for i, N in enumerate(N_vals_bma):
    R_b = sweep["R_BMA"][i]
    R_o = sweep["R_oracle"][i]
    ex  = sweep["excess_risk"][i]
    je  = sweep["jensen_excess"][i]
    je_ratio = je / (ex + 1e-12)
    rr  = R_b / (R_o + 1e-12)
    print(f"  {N:>5}  {R_b:>9.5f}  {R_o:>9.5f}  {ex:>9.6f}  "
          f"{je:>11.6f}  {je_ratio:>7.3f}  {rr:>13.4f}")

print(f"\n  Experiment B: Jensen bound verification")
print(f"  {'N':>5}  {'R_BMA':>9}  {'J_bound':>9}  {'J_bound/R_BMA':>15}  {'J holds?':>10}")
for i, N in enumerate(N_vals_bma):
    R_b = sweep["R_BMA"][i]
    J_b = sweep["jensen_bound"][i]
    print(f"  {N:>5}  {R_b:>9.5f}  {J_b:>9.5f}  {J_b/R_b:>15.4f}  {'YES' if J_b >= R_b - 1e-8 else 'NO':>10}")

results["exp_A_C_B_sweep"] = sweep

# ---------------------------------------------------------------------------
# Experiment D: ECE analytical bounds
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("§2.5 Experiment D: Analytical ECE bounds (Catalan vs Chipman)")
print("=" * 60)

N_vals_ece = np.array([20, 40, 80, 150, 300, 500, 1000])

# max_var = max over leaves and classes of p_c(1-p_c)
max_var = max(p[0] * p[1] for p in LEAF_PROBS)  # p[0]*(1-p[0]) for C=2

# Prior moments
cat_p1 = catalan_pmf(K_MAX, gamma=1.0)
cat_p2 = catalan_pmf(K_MAX, gamma=2.0)
chip_p = chipman_pmf(K_MAX, alpha=0.95, beta=2.0, d_max=20)

E_k_cat1 = float(np.dot(np.arange(1, K_MAX + 1), cat_p1))
E_k_cat2 = float(np.dot(np.arange(1, K_MAX + 1), cat_p2))
E_k_chip = float(np.dot(np.arange(1, K_MAX + 1), chip_p))

print(f"\n  True leaf probs: {[list(np.round(p, 2)) for p in LEAF_PROBS]}")
print(f"  max_var = max_c p_c(1-p_c) = {max_var:.4f}")
print(f"  E[k]: Catalan γ=1={E_k_cat1:.3f}, Catalan γ=2={E_k_cat2:.3f}, Chipman={E_k_chip:.3f}")
print(f"  ECE ratio Catalan/Chipman ≤ sqrt(E_cat/E_chip) = {np.sqrt(E_k_cat1/E_k_chip):.4f}")

ece_cat1 = prior_averaged_ece_bound(cat_p1, max_var, N_vals_ece, C)
ece_cat2 = prior_averaged_ece_bound(cat_p2, max_var, N_vals_ece, C)
ece_chip = prior_averaged_ece_bound(chip_p, max_var, N_vals_ece, C)

print(f"\n  ECE upper bound: sqrt(C · max_var · E[k] / N)")
print(f"  {'N':>6}  {'Cat γ=1':>10}  {'Cat γ=2':>10}  {'Chipman':>10}  "
      f"{'Cat/Chip':>10}")
exp_d = {}
for i, N in enumerate(N_vals_ece):
    ratio = ece_cat1[i] / (ece_chip[i] + 1e-12)
    print(f"  {N:>6}  {ece_cat1[i]:>10.5f}  {ece_cat2[i]:>10.5f}  {ece_chip[i]:>10.5f}  "
          f"{ratio:>10.4f}")
    exp_d[int(N)] = dict(cat_g1=float(ece_cat1[i]), cat_g2=float(ece_cat2[i]),
                          chipman=float(ece_chip[i]), ratio_cat_chip=float(ratio))

results["exp_D_ece_bounds"] = exp_d
results["exp_D_E_k"] = dict(catalan_g1=E_k_cat1, catalan_g2=E_k_cat2, chipman=E_k_chip,
                              ece_ratio_cat1_chip=float(np.sqrt(E_k_cat1/E_k_chip)))

# Per-leaf ECE analysis (finite-sample, no asymptotics)
print(f"\n  Per-leaf finite-sample ECE bound (actual DM formula, N=80):")
N_demo = 80
k_demo = 3
N_leaf_demo = N_demo // k_demo
print(f"  N={N_demo}, k={k_demo} leaves, N_leaf={N_leaf_demo}")
for probs in LEAF_PROBS:
    mse = dm_mse_bound(probs, N_leaf_demo, ALPHA)
    ece = dm_ece_bound_leaf(probs, N_leaf_demo, ALPHA)
    print(f"    p={list(np.round(probs, 2))}: MSE={mse:.5f}, ECE_bound={ece:.5f}")

# ---------------------------------------------------------------------------
# Experiment E: Empirical ECE vs N (simulated)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("§2.5 Experiment E: Empirical ECE calibration curves")
print("=" * 60)

N_vals_sim = np.array([20, 40, 80, 150, 300, 500])
n_reps_ece = 80

# For Catalan: use k=E[k] rounded to nearest integer (expected tree size)
# For Chipman: use k=E[k] rounded

# True k*=3 leaves (oracle model)
# Compare ECE for: oracle k=3 tree, Catalan-expected k=1 tree, Chipman-expected k=3 tree

print(f"\n  Empirical ECE averaged over {n_reps_ece} datasets")
print(f"  {'N':>5}  {'k=1 tree':>10}  {'k=2 tree':>10}  {'k=3 (true)':>12}  "
      f"{'k=5 tree':>10}  {'theory(k=3)':>13}")

exp_e = {}
for N in N_vals_sim:
    row = {}
    for k_sim in [1, 2, 3, 5]:
        n_leaf_sim = max(1, N // k_sim)
        ece_vals = []
        for _ in range(n_reps_ece):
            # Use k_sim leaves with the true leaf probs (cycled if k_sim < k_true)
            probs_sim = [LEAF_PROBS[j % K_TRUE] for j in range(k_sim)]
            ece_vals.append(calibration_sim(probs_sim, N, ALPHA, n_test=400, rng=rng))
        row[k_sim] = float(np.mean(ece_vals))
    # Theory bound for k=3 (true model)
    theo = dm_ece_bound_tree(LEAF_PROBS, N, ALPHA, k=3)
    row["theory_k3"] = theo
    exp_e[int(N)] = row
    print(f"  {N:>5}  {row[1]:>10.4f}  {row[2]:>10.4f}  {row[3]:>12.4f}  "
          f"{row[5]:>10.4f}  {theo:>13.5f}")

results["exp_E_empirical_ece"] = exp_e

# Ratio: k=1 vs k=3 ECE (more leaves = worse calibration per obs)
print(f"\n  ECE ratio k=3/k=1 (larger k → less data per leaf → worse calibration):")
for N in N_vals_sim:
    r = exp_e[N][3] / (exp_e[N][1] + 1e-10)
    theory_r = np.sqrt(3.0 / 1.0)
    print(f"  N={N:>4}: empirical={r:.3f}, theory=sqrt(3/1)={theory_r:.3f}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
out_path = os.path.join(RESULTS_DIR, "bma_ece.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved → {out_path}")

# ---------------------------------------------------------------------------
# Summary of all key numbers
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("§2.3 + §2.5 KEY FINDINGS")
print("=" * 60)

print(f"\n§2.3 BMA Oracle Inequality:")
# N where 1-w_{k*} < 0.01 (99% weight on oracle)
idx_99 = next((i for i, w in enumerate(sweep["w_kstar"]) if w >= 0.99), None)
N_99 = int(N_vals_bma[idx_99]) if idx_99 is not None else None
idx_90 = next((i for i, w in enumerate(sweep["w_kstar"]) if w >= 0.90), None)
N_90 = int(N_vals_bma[idx_90]) if idx_90 is not None else None
print(f"  N for w_{{k*}} ≥ 0.90: {N_90}   N for w_{{k*}} ≥ 0.99: {N_99}")
print(f"  WAIC selection rate P(WAIC=k*): "
      + "  ".join(f"N={N}:{f:.3f}" for N, f in zip(N_vals_bma, sweep["k_waic_best_frac"])))
print(f"  Jensen bound ALWAYS holds: {all(sweep['jensen_bound'][i] >= sweep['R_BMA'][i] - 1e-7 for i in range(len(N_vals_bma)))}")
print(f"  Theory bound ALWAYS ≥ excess risk: {all(sweep['theory_bound'][i] >= sweep['excess_risk'][i] - 1e-7 for i in range(len(N_vals_bma)))}")
print(f"  Excess risk at N=800: {sweep['excess_risk'][-1]:.6f}")

print(f"\n§2.5 ECE Calibration:")
print(f"  Catalan/Chipman ECE ratio ≤ sqrt(E_cat/E_chip) = sqrt({E_k_cat1:.3f}/{E_k_chip:.3f}) = {np.sqrt(E_k_cat1/E_k_chip):.4f}")
print(f"  Catalan γ=1 is {(1-np.sqrt(E_k_cat1/E_k_chip))*100:.1f}% lower ECE than Chipman (asymptotic)")
print(f"  Catalan γ=2 is {(1-np.sqrt(E_k_cat2/E_k_chip))*100:.1f}% lower ECE than Chipman (asymptotic)")
N_example = 300
i_ex = list(N_vals_ece).index(N_example)
print(f"  At N={N_example}: Cat γ=1={ece_cat1[i_ex]:.5f}, Cat γ=2={ece_cat2[i_ex]:.5f}, Chipman={ece_chip[i_ex]:.5f}")
print(f"  Theory: ECE ∝ sqrt(E[k]/N) → Cat/Chip = {np.sqrt(E_k_cat1/E_k_chip):.4f}")
