#!/usr/bin/env bash
# Reproduce all experiments from the JBDT paper.
# Results are written to results/*.json (pre-computed copies already present).
# Run time: ~10 minutes on a modern laptop.
set -euo pipefail

mkdir -p results

echo "=== §2.1  WAIC-LOO gap ==="
python waic_consistency.py

echo "=== §2.2  Catalan prior concentration ==="
python catalan_concentration.py

echo "=== §2.3 + §2.5  BMA oracle + ECE ==="
python bma_ece_experiments.py

echo "=== §2.4 + §2.6  RJMCMC + PAC-Bayes ==="
python rjmcmc_pacbayes_experiments.py

echo "=== §2.7  Decision-theoretic analysis ==="
python decision_theory_experiments.py

echo "=== Benchmark 2 (synthetic + Olivetti) ==="
python benchmark2.py

echo ""
echo "All results written to results/.  Done."
echo ""
echo "NOTE: benchmark.py (knee X-ray) requires the figshare dataset"
echo "      (https://doi.org/10.6084/m9.figshare.8303996) — skipped here."
