"""
§2.2  Catalan Prior Concentration
Five experiments comparing JBDT Catalan-exponential prior vs Chipman (1998)
alpha-beta prior and geometric baseline.

Experiments:
  A. PMF comparison: Catalan vs Chipman vs Geometric for γ ∈ {0.5,1,2,3}
  B. Tail probabilities P(k ≥ k₀) for k₀ ∈ {3,5,10,20}
  C. Effective decay rates vs theoretical bound exp(-γ)/4
  D. Posterior concentration: P(k=k*|data,N) vs N for Catalan vs Chipman
  E. Moment table: E[k], Var[k], mode, entropy across priors
"""

import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from catalan_prior import (
    catalan_pmf, catalan_moments,
    chipman_pmf, chipman_moments,
    geometric_pmf,
    tail_prob, posterior_concentration,
    effective_decay_rate,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

K_MAX = 30
results = {}

# ---------------------------------------------------------------------------
# Experiment A: PMF comparison
# ---------------------------------------------------------------------------
print("=" * 60)
print("Experiment A: PMF comparison (Catalan vs Chipman vs Geometric)")
print("=" * 60)

gamma_vals = [0.5, 1.0, 2.0, 3.0]
exp_a = {}

# Chipman default (α=0.95, β=2)
chipman_p = chipman_pmf(K_MAX, alpha=0.95, beta=2.0, d_max=20)
chipman_m = chipman_moments(K_MAX, alpha=0.95, beta=2.0, d_max=20)

for gamma in gamma_vals:
    cat_p  = catalan_pmf(K_MAX, gamma)
    geo_p  = geometric_pmf(K_MAX, gamma)
    cat_m  = catalan_moments(K_MAX, gamma)

    r_eff  = effective_decay_rate(gamma)
    r_theo = np.exp(-gamma) / 4.0

    exp_a[str(gamma)] = {
        "catalan_pmf_k1to10": cat_p[:10].tolist(),
        "geometric_pmf_k1to10": geo_p[:10].tolist(),
        "catalan_E_k": cat_m["mean"],
        "catalan_mode": cat_m["mode"],
        "catalan_std": cat_m["std"],
        "catalan_entropy": cat_m["entropy"],
        "r_effective": r_eff,
        "r_theoretical": r_theo,
        "ratio_r_eff_theo": r_eff / (r_theo + 1e-15),
    }
    print(f"\n  γ={gamma}:")
    print(f"    Catalan:  E[k]={cat_m['mean']:.3f}, mode={cat_m['mode']}, "
          f"std={cat_m['std']:.3f}, H={cat_m['entropy']:.3f}")
    print(f"    r_eff={r_eff:.4f}, r_theo=exp(-γ)/4={r_theo:.4f}, "
          f"ratio={r_eff/r_theo:.3f}")
    print(f"    Catalan P(k=1..5): " +
          "  ".join(f"p({k})={cat_p[k-1]:.4f}" for k in range(1, 6)))

print(f"\n  Chipman (α=0.95, β=2): E[k]={chipman_m['mean']:.3f}, "
      f"mode={chipman_m['mode']}, std={chipman_m['std']:.3f}")
print(f"    Chipman P(k=1..5): " +
      "  ".join(f"p({k})={chipman_p[k-1]:.4f}" for k in range(1, 6)))

results["exp_A_pmf"] = exp_a
results["chipman_default"] = {
    "E_k": chipman_m["mean"],
    "mode": chipman_m["mode"],
    "std": chipman_m["std"],
    "entropy": chipman_m["entropy"],
    "pmf_k1to10": chipman_p[:10].tolist(),
}

# ---------------------------------------------------------------------------
# Experiment B: Tail probabilities
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Experiment B: Tail probabilities P(k ≥ k₀)")
print("=" * 60)

k0_vals = [3, 5, 10, 20]
exp_b = {}

for gamma in [1.0, 2.0]:
    cat_p = catalan_pmf(K_MAX, gamma)
    geo_p = geometric_pmf(K_MAX, gamma)
    row = {}
    for k0 in k0_vals:
        t_cat = tail_prob(cat_p, k0)
        t_geo = tail_prob(geo_p, k0)
        t_chip = tail_prob(chipman_p, k0)
        row[k0] = dict(catalan=t_cat, geometric=t_geo, chipman=t_chip)
    exp_b[str(gamma)] = row
    print(f"\n  γ={gamma} (r_cat={np.exp(-gamma)/4:.4f}, r_geo={np.exp(-gamma):.4f}):")
    print(f"  {'k0':>4}  {'Catalan':>10}  {'Geometric':>10}  {'Chipman':>10}")
    for k0 in k0_vals:
        r = row[k0]
        print(f"  {k0:>4}  {r['catalan']:>10.4e}  {r['geometric']:>10.4e}  "
              f"{r['chipman']:>10.4e}")

results["exp_B_tails"] = exp_b

# ---------------------------------------------------------------------------
# Experiment C: Effective decay rates
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Experiment C: Effective decay rates r_eff vs exp(-γ)/4")
print("=" * 60)

gamma_sweep = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])
exp_c = {}
print(f"  {'γ':>5}  {'r_theo':>8}  {'r_eff':>8}  {'ratio':>7}  {'E[k]':>7}")
for g in gamma_sweep:
    r_t = np.exp(-g) / 4.0
    r_e = effective_decay_rate(g)
    cat_m = catalan_moments(K_MAX, g)
    exp_c[str(round(g, 2))] = dict(
        r_theoretical=r_t, r_effective=r_e,
        ratio=r_e / r_t if r_t > 0 else None,
        E_k=cat_m["mean"], mode=cat_m["mode"]
    )
    print(f"  {g:>5.2f}  {r_t:>8.5f}  {r_e:>8.5f}  {r_e/r_t:>7.4f}  "
          f"{cat_m['mean']:>7.3f}")

results["exp_C_decay"] = exp_c

# ---------------------------------------------------------------------------
# Experiment D: Posterior concentration P(k=k*|data,N) vs N
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Experiment D: Posterior concentration P(k=k*|data,N) vs N")
print("=" * 60)

k_true = 3
C = 2
N_vals = np.array([5, 10, 20, 40, 80, 150, 300, 500, 800])

cat_p_gamma1 = catalan_pmf(K_MAX, gamma=1.0)
cat_p_gamma2 = catalan_pmf(K_MAX, gamma=2.0)

print(f"  True model: k*={k_true} leaves, C={C} classes per leaf")
print(f"  Posterior P(k=k*|data,N):")

conc_cat1 = posterior_concentration(cat_p_gamma1, N_vals, k_true, C)
conc_cat2 = posterior_concentration(cat_p_gamma2, N_vals, k_true, C)
conc_chip = posterior_concentration(chipman_p,    N_vals, k_true, C)
conc_geo  = posterior_concentration(geometric_pmf(K_MAX, 1.0), N_vals, k_true, C)

exp_d = {}
print(f"\n  {'N':>5}  {'Cat γ=1':>10}  {'Cat γ=2':>10}  {'Chipman':>10}  {'Geo γ=1':>10}")
for i, N in enumerate(N_vals):
    print(f"  {N:>5}  {conc_cat1[i]:>10.4f}  {conc_cat2[i]:>10.4f}  "
          f"{conc_chip[i]:>10.4f}  {conc_geo[i]:>10.4f}")
    exp_d[int(N)] = dict(
        catalan_g1=float(conc_cat1[i]),
        catalan_g2=float(conc_cat2[i]),
        chipman=float(conc_chip[i]),
        geometric_g1=float(conc_geo[i]),
    )

results["exp_D_concentration"] = exp_d

# Compute N_90: smallest N with P(k=k*|data,N) >= 0.90
def n_90(conc_arr, N_vals):
    idx = np.argmax(conc_arr >= 0.90)
    if conc_arr[idx] < 0.90:
        return None
    return int(N_vals[idx])

n90_cat1 = n_90(conc_cat1, N_vals)
n90_cat2 = n_90(conc_cat2, N_vals)
n90_chip = n_90(conc_chip, N_vals)
print(f"\n  N_90 (first N with P(k=k*)≥0.90):")
print(f"    Catalan γ=1: {n90_cat1}")
print(f"    Catalan γ=2: {n90_cat2}")
print(f"    Chipman:     {n90_chip}")
results["exp_D_n90"] = dict(catalan_g1=n90_cat1, catalan_g2=n90_cat2, chipman=n90_chip)

# N_95 and N_99: first N with P(k=k*)>=0.95 / 0.99
def n_thresh(conc_arr, N_vals, thresh):
    idx = np.argmax(conc_arr >= thresh)
    return int(N_vals[idx]) if conc_arr[idx] >= thresh else None

results["exp_D_n95"] = dict(
    catalan_g1=n_thresh(conc_cat1, N_vals, 0.95),
    catalan_g2=n_thresh(conc_cat2, N_vals, 0.95),
    chipman=n_thresh(conc_chip, N_vals, 0.95),
)
results["exp_D_n99"] = dict(
    catalan_g1=n_thresh(conc_cat1, N_vals, 0.99),
    catalan_g2=n_thresh(conc_cat2, N_vals, 0.99),
    chipman=n_thresh(conc_chip, N_vals, 0.99),
)

# Crossover: first N where Catalan γ=1 > Chipman
diff_cat1_chip = conc_cat1 - conc_chip
cross_idx = np.argmax(diff_cat1_chip > 0)
if diff_cat1_chip[cross_idx] > 0 and cross_idx > 0:
    # linear interpolation
    N_a, N_b = int(N_vals[cross_idx - 1]), int(N_vals[cross_idx])
    d_a, d_b = float(diff_cat1_chip[cross_idx - 1]), float(diff_cat1_chip[cross_idx])
    N_cross = int(N_a + (N_b - N_a) * (-d_a) / (d_b - d_a))
else:
    N_cross = None
results["exp_D_crossover_cat1_vs_chip"] = N_cross
print(f"\n  Catalan γ=1 overtakes Chipman at N ≈ {N_cross}")

# ---------------------------------------------------------------------------
# Experiment E: Moment table
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Experiment E: Moment table across priors and γ values")
print("=" * 60)

exp_e = {}
header = f"  {'Prior':>20}  {'E[k]':>7}  {'Var[k]':>7}  {'mode':>5}  {'H':>6}"
print(header)
print("  " + "-" * (len(header) - 2))

for gamma in [0.5, 1.0, 2.0, 3.0]:
    m = catalan_moments(K_MAX, gamma)
    label = f"Catalan γ={gamma}"
    exp_e[label] = {k: m[k] for k in ["mean", "var", "mode", "entropy"]}
    print(f"  {label:>20}  {m['mean']:>7.3f}  {m['var']:>7.3f}  {m['mode']:>5}  "
          f"{m['entropy']:>6.3f}")

for gamma in [0.5, 1.0, 2.0, 3.0]:
    geo_p    = geometric_pmf(K_MAX, gamma)
    k_v      = np.arange(1, K_MAX + 1)
    geo_mean = float(np.dot(k_v, geo_p))
    geo_var  = float(np.dot(k_v ** 2, geo_p) - geo_mean ** 2)
    geo_e    = -float(np.sum(geo_p[geo_p > 1e-15] * np.log(geo_p[geo_p > 1e-15])))
    label    = f"Geometric γ={gamma}"
    exp_e[label] = dict(mean=geo_mean, var=geo_var, mode=1, entropy=geo_e)
    print(f"  {label:>20}  {geo_mean:>7.3f}  {geo_var:>7.3f}  {'1':>5}  {geo_e:>6.3f}")

for (a, b) in [(0.95, 2.0), (0.5, 1.0)]:
    m = chipman_moments(K_MAX, a, b)
    label = f"Chipman α={a},β={b}"
    exp_e[label] = {k: m[k] for k in ["mean", "var", "mode", "entropy"]}
    print(f"  {label:>20}  {m['mean']:>7.3f}  {m['var']:>7.3f}  {m['mode']:>5}  "
          f"{m['entropy']:>6.3f}")

results["exp_E_moments"] = exp_e

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
out_path = os.path.join(RESULTS_DIR, "catalan_concentration.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved → {out_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("§2.2 KEY FINDINGS")
print("=" * 60)

cat1 = catalan_moments(K_MAX, 1.0)
chip = chipman_moments(K_MAX, 0.95, 2.0)
r_cat = np.exp(-1.0) / 4.0
# Chipman effective decay rate: mean p(k+1)/p(k) for k=5..10
chip_ratios = [chipman_p[k] / chipman_p[k-1] for k in range(5, 11) if chipman_p[k-1] > 1e-10]
r_chip = float(np.mean(chip_ratios)) if chip_ratios else None

print(f"\n1. Tail decay (γ=1): r_Catalan ≈ exp(-1)/4 ≈ {r_cat:.4f}")
print(f"   vs Chipman effective ratio p(2)/p(1) ≈ {r_chip:.4f}")
print(f"   Catalan suppresses large trees {r_chip/r_cat:.1f}× faster (per leaf)")

print(f"\n2. Prior means: E_cat[k]={cat1['mean']:.2f}, E_chip[k]={chip['mean']:.2f}")
print(f"   Catalan mode={cat1['mode']}, Chipman mode={chip['mode']}")

print(f"\n3. Tail P(k≥5):")
cat_tail5 = tail_prob(catalan_pmf(K_MAX, 1.0), 5)
chip_tail5 = tail_prob(chipman_p, 5)
print(f"   Catalan γ=1: {cat_tail5:.4e},  Chipman α=0.95,β=2: {chip_tail5:.4f}")
print(f"   Ratio: {chip_tail5/cat_tail5:.1f}× more probability in tails for Chipman")

print(f"\n4. Posterior concentration (k*=3, C=2):")
n95_c1 = results["exp_D_n95"]["catalan_g1"]
n95_c2 = results["exp_D_n95"]["catalan_g2"]
n95_ch = results["exp_D_n95"]["chipman"]
print(f"   N_90: Cat γ=1={n90_cat1}, Cat γ=2={n90_cat2}, Chipman={n90_chip}")
print(f"   N_95: Cat γ=1={n95_c1}, Cat γ=2={n95_c2}, Chipman={n95_ch}")
print(f"   Crossover (Cat γ=1 overtakes Chipman): N≈{results['exp_D_crossover_cat1_vs_chip']}")
print(f"   At N=800: Cat γ=1={conc_cat1[-1]:.4f}, Cat γ=2={conc_cat2[-1]:.4f}, "
      f"Chipman={conc_chip[-1]:.4f}")
print(f"\n5. Asymptotic concentration (N→∞ limit from prior tails):")
cat1_m = catalan_moments(K_MAX, 1.0)
cat1_asym = cat1_m['pmf'][k_true - 1] / cat1_m['pmf'][k_true - 1:].sum()
cat2_m = catalan_moments(K_MAX, 2.0)
cat2_asym = cat2_m['pmf'][k_true - 1] / cat2_m['pmf'][k_true - 1:].sum()
chip_asym = chipman_p[k_true - 1] / chipman_p[k_true - 1:].sum()
print(f"   P(k=k*|∞) = p(k*) / P_prior(k≥k*):")
print(f"   Cat γ=1: {cat1_asym:.4f}  Cat γ=2: {cat2_asym:.4f}  Chipman: {chip_asym:.4f}")
