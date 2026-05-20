"""
§2.7 Experiments — Decision-Theoretic Analysis of JBDT under Asymmetric Loss

Experiments:
  A. Optimal threshold τ*(r) for r = C_FN/C_FP ∈ {1, 1.5, 2, 3, 5, 10}
  B. 3-leaf OA example at N_leaf=13 (N_total=40): theoretical + simulated EC
  C. Cost sweep over N: DM+τ* vs DM+argmax vs OC+τ* vs OC+argmax
  D. Calibration-cost bound: (C_FP+C_FN)·ECE_bound — Catalan vs Chipman
  E. Cost saving ΔEC(r) sweep — grows with asymmetry ratio r
"""

import json, sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from decision_theory import (
    optimal_threshold, expected_cost, cost_curve, cost_saving,
    dm_posterior_prob, overcalibrated_prob,
    dm_ece_bound, miscalib_cost_bound,
    theoretical_ec, theoretical_ec_mismatched, theoretical_cost_saving,
    simulate_cost_comparison, simulate_cost_sweep_N, simulate_cost_sweep_r,
)
from catalan_prior import catalan_pmf, chipman_pmf

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(42)

# OA detection setup
C_FP   = 1.0   # cost of false positive (unnecessary referral)
C_FN   = 2.0   # cost of false negative (missed OA)
R      = C_FN / C_FP   # asymmetry ratio = 2
TAU_STAR = optimal_threshold(C_FP, C_FN)

# 3-leaf OA example (N=40 knee OA context)
P_TRUE_LEAVES = [0.15, 0.40, 0.75]
#   Leaf 1: healthy-dominant (P(OA)=0.15, below τ*=0.333)
#   Leaf 2: TWILIGHT ZONE   (P(OA)=0.40, between τ*=0.333 and 0.5)
#   Leaf 3: OA-dominant     (P(OA)=0.75, above 0.5)

# Catalan/Chipman E[k] from §2.5
K_MAX = 20
cat_pmf  = catalan_pmf(K_MAX, gamma=1.0)
chip_pmf = chipman_pmf(K_MAX, alpha=0.95, beta=2.0, d_max=20)
k_vals   = np.arange(1, K_MAX + 1)
E_K_CAT  = float(np.dot(k_vals, cat_pmf))
E_K_CHIP = float(np.dot(k_vals, chip_pmf))

results = {}

# ---------------------------------------------------------------------------
# Experiment A: Optimal threshold τ*(r)
# ---------------------------------------------------------------------------
print("=" * 60)
print("§2.7 Experiment A: Optimal threshold τ*(r)")
print("=" * 60)

r_vals_A = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
print(f"\n  Argmax uses τ=0.5 (only optimal when r=1 — symmetric loss)")
print(f"  {'r=C_FN/C_FP':>14}  {'τ*(r)':>8}  {'Δτ=0.5−τ*':>12}  {'context':>20}")

exp_a = {}
for r in r_vals_A:
    tau = optimal_threshold(C_FP, C_FP * r)
    delta_tau = 0.5 - tau
    if r == 1.0:
        ctx = "symmetric (argmax OK)"
    elif r == 2.0:
        ctx = "OA detection (τ*=1/3)"
    elif r == 5.0:
        ctx = "cancer screening"
    else:
        ctx = ""
    print(f"  {r:>14.1f}  {tau:>8.4f}  {delta_tau:>12.4f}  {ctx:>20}")
    exp_a[r] = dict(tau_star=tau, delta_tau=delta_tau)

print(f"\n  Medical OA (r={R:.0f}): τ*={TAU_STAR:.4f}, argmax threshold error={0.5-TAU_STAR:.4f}")
results["exp_A_threshold"] = {str(r): v for r, v in exp_a.items()}

# ---------------------------------------------------------------------------
# Experiment B: 3-leaf OA example — theoretical and simulated EC
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("§2.7 Experiment B: 3-leaf OA example (N_leaf=13, N_total=40)")
print("=" * 60)

N_LEAF_OA = 13   # N_total = 39 ≈ 40

print(f"\n  Leaf structure: P(OA|leaf) = {P_TRUE_LEAVES}")
print(f"  C_FP={C_FP:.0f}, C_FN={C_FN:.0f}, r={R:.0f}, τ*={TAU_STAR:.4f}")
print(f"\n  --- Theoretical (large-N) expected cost ---")

# DM posterior probs at large-N limit: p_hat → p_true
p_hat_dm_theory = P_TRUE_LEAVES   # asymptotic
p_hat_oc_theory = [overcalibrated_prob(p) for p in P_TRUE_LEAVES]

print(f"\n  Leaf probs at large N:")
print(f"  {'Leaf':>5}  {'p_true':>8}  {'p_hat_DM':>10}  {'p_hat_OC':>10}")
for i, (pt, poc) in enumerate(zip(P_TRUE_LEAVES, p_hat_oc_theory)):
    print(f"  {i+1:>5}  {pt:>8.3f}  {pt:>10.3f}  {poc:>10.3f}")

ec_dm_opt_th  = theoretical_ec(P_TRUE_LEAVES, TAU_STAR, C_FP, C_FN)
ec_dm_arg_th  = theoretical_ec(P_TRUE_LEAVES, 0.5, C_FP, C_FN)
# OC model: decision from p_hat_oc, but costs from p_true
ec_oc_opt_th  = theoretical_ec_mismatched(P_TRUE_LEAVES, p_hat_oc_theory, TAU_STAR, C_FP, C_FN)
ec_oc_arg_th  = theoretical_ec_mismatched(P_TRUE_LEAVES, p_hat_oc_theory, 0.5, C_FP, C_FN)

print(f"\n  Theoretical EC (large N):")
print(f"  {'Method':>22}  {'EC':>7}  {'vs DM+τ*':>10}")
for label, ec_th in [
    ("DM + τ*",  ec_dm_opt_th),
    ("DM + τ=0.5", ec_dm_arg_th),
    ("OC + τ*",  ec_oc_opt_th),
    ("OC + τ=0.5", ec_oc_arg_th),
]:
    diff = ec_th - ec_dm_opt_th
    note = "OPTIMAL" if diff < 1e-6 else f"+{diff:.4f} (+{100*diff/ec_dm_opt_th:.1f}%)"
    print(f"  {label:>22}  {ec_th:>7.4f}  {note}")

saving_th = ec_dm_arg_th - ec_dm_opt_th
calib_penalty_th = ec_oc_opt_th - ec_dm_opt_th
print(f"\n  Cost saving DM (τ* vs argmax): {saving_th:.4f} ({100*saving_th/ec_dm_arg_th:.1f}% of argmax EC)")
print(f"  Calibration penalty (OC vs DM at τ*): {calib_penalty_th:.4f}")
print(f"  Double-penalty insight: OC+τ* = DM+argmax = {ec_oc_opt_th:.4f}  (miscalibration erases τ* benefit)")

# Simulate at N_leaf=13
print(f"\n  --- Simulated (N_leaf={N_LEAF_OA}, 2000 reps) ---")
sim_B = simulate_cost_comparison(
    P_TRUE_LEAVES, N_LEAF_OA, C_FP, C_FN, alpha=1.0, sharpness=3.0, n_reps=2000, rng=rng)

print(f"\n  {'Method':>22}  {'EC mean':>8}  {'EC std':>8}  {'vs DM+τ*':>10}")
base_mean = sim_B['dm_tau_star']['mean']
for key, label in [
    ("dm_tau_star",  "DM + τ*"),
    ("dm_argmax",    "DM + τ=0.5"),
    ("oc_tau_star",  "OC + τ*"),
    ("oc_argmax",    "OC + τ=0.5"),
]:
    m = sim_B[key]['mean']
    s = sim_B[key]['std']
    diff = m - base_mean
    note = "BEST" if diff < 1e-4 else f"+{diff:.4f}"
    print(f"  {label:>22}  {m:>8.4f}  {s:>8.4f}  {note}")

print(f"\n  Simulated cost saving (DM τ* vs argmax): {sim_B['saving_dm']:.4f}")
print(f"  Simulated calibration advantage (DM vs OC at τ*): {sim_B['calib_advantage']:.4f}")

results["exp_B_oa_example"] = dict(
    theoretical=dict(
        dm_tau_star=ec_dm_opt_th, dm_argmax=ec_dm_arg_th,
        oc_tau_star=ec_oc_opt_th, oc_argmax=ec_oc_arg_th,
        saving=saving_th, calib_penalty=calib_penalty_th,
    ),
    simulated=sim_B,
    N_leaf=N_LEAF_OA,
    p_true_leaves=P_TRUE_LEAVES,
)

# ---------------------------------------------------------------------------
# Experiment C: Cost sweep over N
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("§2.7 Experiment C: Cost sweep over N (C_FN/C_FP=2, OA leaves)")
print("=" * 60)

N_leaf_vals_C = [5, 10, 13, 20, 40, 80, 200]
print(f"\n  p_true_leaves = {P_TRUE_LEAVES},  τ* = {TAU_STAR:.4f}")
print(f"  (Asymptotic ΔEC theory = {saving_th:.4f} per patient)")
print(f"\n  {'N_leaf':>8}  {'N_total':>8}  {'DM+τ*':>8}  {'DM+0.5':>8}  "
      f"{'OC+τ*':>8}  {'OC+0.5':>8}  {'ΔEC_DM':>8}  {'ΔEC_OC':>8}")

sweep_C = simulate_cost_sweep_N(
    N_leaf_vals_C, P_TRUE_LEAVES, C_FP, C_FN, n_reps=2000, rng=rng)

for row in sweep_C:
    dm_opt  = row['dm_tau_star']['mean']
    dm_arg  = row['dm_argmax']['mean']
    oc_opt  = row['oc_tau_star']['mean']
    oc_arg  = row['oc_argmax']['mean']
    print(f"  {row['N_leaf']:>8}  {row['N_total']:>8}  {dm_opt:>8.4f}  {dm_arg:>8.4f}  "
          f"{oc_opt:>8.4f}  {oc_arg:>8.4f}  {dm_arg-dm_opt:>8.4f}  {oc_arg-oc_opt:>8.4f}")

print(f"\n  Note: ΔEC_DM → {saving_th:.4f} as N→∞ (Leaf-2 always in twilight zone for large N)")
print(f"  ΔEC_DM at N_leaf=13 (N=40 knee OA): {sweep_C[2]['dm_argmax']['mean']-sweep_C[2]['dm_tau_star']['mean']:.4f}")
results["exp_C_sweep_N"] = sweep_C

# ---------------------------------------------------------------------------
# Experiment D: Calibration-cost bound Catalan vs Chipman
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("§2.7 Experiment D: Calibration-cost bound — Catalan vs Chipman")
print("=" * 60)

N_vals_D = np.array([20, 40, 80, 150, 300, 500, 1000])
MAX_VAR = 0.25   # max leaf variance for p_c(1−p_c)

print(f"\n  ΔEC_bound = (C_FP+C_FN) × ECE_bound(N)")
print(f"  ECE_bound = sqrt(C·E[k]·max_var/N),  C=2, max_var={MAX_VAR}")
print(f"  E[k]: Catalan γ=1 = {E_K_CAT:.3f},  Chipman = {E_K_CHIP:.3f}")
print(f"  Cost sum C_FP+C_FN = {C_FP+C_FN:.0f}  (OA: C_FP=1, C_FN=2)")
print(f"\n  {'N':>6}  {'ECE Cat':>10}  {'ECE Chip':>10}  {'ΔEC_bnd Cat':>13}  "
      f"{'ΔEC_bnd Chip':>14}  {'Ratio':>7}")

exp_d = {}
for N in N_vals_D:
    ece_cat  = float(dm_ece_bound(N, E_K_CAT, MAX_VAR, C=2))
    ece_chip = float(dm_ece_bound(N, E_K_CHIP, MAX_VAR, C=2))
    bnd_cat  = float(miscalib_cost_bound(ece_cat, C_FP, C_FN))
    bnd_chip = float(miscalib_cost_bound(ece_chip, C_FP, C_FN))
    ratio    = bnd_cat / bnd_chip
    print(f"  {N:>6}  {ece_cat:>10.4f}  {ece_chip:>10.4f}  {bnd_cat:>13.4f}  "
          f"{bnd_chip:>14.4f}  {ratio:>7.4f}")
    exp_d[int(N)] = dict(ece_cat=ece_cat, ece_chip=ece_chip,
                         bound_cat=bnd_cat, bound_chip=bnd_chip, ratio=ratio)

ratio_exact = np.sqrt(E_K_CAT / E_K_CHIP)
print(f"\n  Exact ratio = sqrt(E_k_Cat/E_k_Chip) = sqrt({E_K_CAT:.3f}/{E_K_CHIP:.3f}) = {ratio_exact:.4f}")
print(f"  → Catalan prior gives {(1-ratio_exact)*100:.1f}% tighter cost-miscalibration bound at all N")
print(f"  At N=40 (knee OA): ΔEC_bound Cat={exp_d[40]['bound_cat']:.3f}, Chip={exp_d[40]['bound_chip']:.3f}")
results["exp_D_cost_bound"] = exp_d

# ---------------------------------------------------------------------------
# Experiment E: Cost saving ΔEC(r) sweep
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("§2.7 Experiment E: Cost saving ΔEC vs asymmetry ratio r=C_FN/C_FP")
print("=" * 60)

r_vals_E = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
print(f"\n  N_leaf=13 (N_total=40), p_true_leaves={P_TRUE_LEAVES}")
print(f"\n  {'r':>5}  {'τ*':>6}  {'ΔEC_theo':>10}  {'ΔEC_DM_sim':>12}  {'ΔEC_OC_sim':>12}  "
      f"{'DM adv over OC':>15}")

sweep_E = simulate_cost_sweep_r(
    r_vals_E, P_TRUE_LEAVES, N_leaf=N_LEAF_OA, c_fp_base=C_FP,
    n_reps=2000, rng=rng)

exp_e = {}
for row in sweep_E:
    r = row['r']
    dm_adv = row['saving_dm'] - row['saving_oc']
    print(f"  {r:>5.1f}  {row['tau_star']:>6.4f}  {row['theo_saving']:>10.4f}  "
          f"{row['saving_dm']:>12.4f}  {row['saving_oc']:>12.4f}  {dm_adv:>15.4f}")
    exp_e[str(r)] = dict(
        tau_star=row['tau_star'],
        theo_saving=row['theo_saving'],
        saving_dm=row['saving_dm'],
        saving_oc=row['saving_oc'],
        dm_advantage_over_oc=dm_adv,
    )

print(f"\n  r=1 (symmetric): ΔEC≈0 for both (τ*=0.5 same as argmax)")
print(f"  r=2 (OA): DM gains {sweep_E[2]['saving_dm']:.4f}, OC gains {sweep_E[2]['saving_oc']:.4f}")
print(f"  r=5: DM gains {sweep_E[4]['saving_dm']:.4f}, OC gains {sweep_E[4]['saving_oc']:.4f}")
print(f"  DM calibration advantage (over OC at τ*) grows with r")
results["exp_E_sweep_r"] = exp_e

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
out_path = os.path.join(RESULTS_DIR, "decision_theory.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved → {out_path}")

# ---------------------------------------------------------------------------
# KEY FINDINGS
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("§2.7 KEY FINDINGS")
print("=" * 60)

print(f"""
§2.7 Decision-Theoretic Analysis:

  1. Optimal threshold: τ* = C_FP/(C_FP+C_FN) = {TAU_STAR:.4f} for OA (r=2)
     Argmax (τ=0.5) is suboptimal for ALL r≠1 — medical context always has r>1.

  2. 3-leaf OA example (N_total≈40, p_true=[0.15,0.40,0.75]):
     DM + τ*   = {ec_dm_opt_th:.4f}  ← OPTIMAL
     DM + τ=0.5 = {ec_dm_arg_th:.4f}  (+{100*(ec_dm_arg_th-ec_dm_opt_th)/ec_dm_opt_th:.1f}% excess cost)
     OC + τ*   = {ec_oc_opt_th:.4f}  (SAME as DM+argmax — calibration erases τ* benefit)
     OC + τ=0.5 = {ec_oc_arg_th:.4f}

     Key: Leaf 2 (P(OA)=0.40 ∈ (τ*,0.5)) is the "twilight zone":
       DM+τ* correctly predicts OA; argmax and OC both predict healthy.
       Cost saving per patient in Leaf 2: C_FN×0.40 − C_FP×0.60 = {2*0.40-1*0.60:.2f}

  3. Calibration-cost bound (§2.5 → §2.7):
     ΔEC_miscal ≤ (C_FP+C_FN) × ECE_bound = {C_FP+C_FN:.0f} × sqrt(C·E[k]·max_var/N)
     Catalan/Chipman ratio: {ratio_exact:.4f} ({100*(1-ratio_exact):.1f}% tighter — same as ECE ratio)
     At N=40: Catalan bound={exp_d[40]['bound_cat']:.3f}, Chipman={exp_d[40]['bound_chip']:.3f}

  4. Double penalty for discriminative classifiers:
     (a) Cost of wrong threshold (τ=0.5 instead of τ*): {saving_th:.4f} per patient
     (b) Cost of miscalibration (OC vs DM at τ*): {calib_penalty_th:.4f} per patient  [theoretical]
     (c) Sum: {saving_th+calib_penalty_th:.4f} — both penalties avoidable with DM-BDT + τ*
""")
