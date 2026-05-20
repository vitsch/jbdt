# JBDT — Bayesian Decision Trees with Catalan-Exponential Prior

Code and paper for:

> Jakaite, J. & Schetinin, V. (2025). Bayesian Decision Tree Classifier with Catalan Prior
> for Medical Image Analysis. *Machine Learning and Knowledge Extraction*, 7(3), 106.
> https://doi.org/10.3390/make7030106

This repository accompanies the Bayesian Analysis submission that provides the first
analytical characterisation of the Catalan-exponential tree-size prior: tail bounds,
effective decay constant, posterior concentration rates, and PAC-Bayes sample complexity.

---

## Repository layout

```
jbdt/
├── src/                        # Core library modules
│   ├── bdt.py                  # Chipman (1998) BCART baseline
│   ├── bdt_jakaite.py          # JBDT: RJMCMC + Catalan prior + WAIC BMA
│   ├── catalan_prior.py        # Catalan-exponential prior, tail bounds, reff
│   ├── waic_theory.py          # Closed-form WAIC/LOO for DM leaf model
│   ├── bma_oracle.py           # BMA oracle inequality, excess-risk bound
│   ├── pac_bayes.py            # PAC-Bayes sample complexity
│   ├── rjmcmc_balance.py       # Detailed balance verification utilities
│   ├── ece_calibration.py      # ECE bounds (Catalan vs Chipman)
│   ├── decision_theory.py      # Asymmetric-loss decision theory
│   ├── bma_zernike.py          # BMA over Zernike feature orders
│   └── features.py             # Zernike moments, Haralick features
│
├── waic_consistency.py         # §2.1 WAIC-LOO gap experiments
├── catalan_concentration.py    # §2.2 prior concentration experiments
├── bma_ece_experiments.py      # §2.3 + §2.5 BMA oracle + ECE
├── rjmcmc_pacbayes_experiments.py  # §2.4 + §2.6 RJMCMC + PAC-Bayes
├── decision_theory_experiments.py  # §2.7 decision-theoretic analysis
├── benchmark.py                # BMA over Zernike orders on knee X-ray data*
├── benchmark2.py               # BMA on synthetic textures + Olivetti faces
│
├── results/                    # Pre-computed JSON results (all experiments)
│
├── paper/                      # LaTeX submission package
│   ├── main.tex
│   ├── supplement.tex
│   ├── cover_letter.tex
│   ├── references.bib
│   ├── Makefile
│   └── README.md               # Paper build instructions
│
├── run_experiments.sh          # Reproduce all results from scratch
├── requirements.txt
└── .gitignore
```

*`benchmark.py` requires proprietary knee X-ray data; see [Data](#data) section.

---

## Installation

```bash
git clone https://github.com/vitsch/jbdt.git
cd jbdt
pip install -r requirements.txt
```

Python 3.9+ required.

---

## Reproducing experiments

Run all experiments in order (saves to `results/`):

```bash
bash run_experiments.sh
```

Or run individual experiments:

```bash
python waic_consistency.py           # ~30 s   → results/waic_consistency.json
python catalan_concentration.py      # ~10 s   → results/catalan_concentration.json
python bma_ece_experiments.py        # ~60 s   → results/bma_ece.json
python rjmcmc_pacbayes_experiments.py  # ~120 s → results/rjmcmc_pacbayes.json
python decision_theory_experiments.py  # ~10 s  → results/decision_theory.json
python benchmark2.py                 # ~5 min  → results/bma_benchmark2.json
```

All experiments use `numpy.random.seed(42)` for reproducibility.

---

## Data

**`benchmark2.py`** requires no external data: it generates synthetic Gaussian
texture images and downloads the Olivetti faces dataset automatically via
`sklearn.datasets.fetch_olivetti_faces`.

**`benchmark.py`** requires 40 knee X-ray ROI images from:

> Jakaite et al. (2021). *Knee X-ray ROI dataset.*
> https://doi.org/10.6084/m9.figshare.8303996

Download and extract so that the directory structure is:
```
data/
├── control/
│   └── lateral/  *.tiff
│   └── medial/   *.tiff
└── case/
    └── lateral/  *.tiff
    └── medial/   *.tiff
```

The patient data is not included in this repository.

---

## Compiling the paper

See `paper/README.md` for full instructions. Quick start:

```bash
cd paper
# Download imsart.cls from https://www.e-publications.org/ims/support/imsart.html
make all        # builds main.pdf, supplement.pdf, cover_letter.pdf
```

---

## Key results (pre-computed)

| Result | Value |
|--------|-------|
| WAIC-LOO gap | $-4N/(N+\alpha_0) + O(N^{-1})$ |
| $N_{\min}$ formula | $\approx 5.41/\Delta$ |
| Catalan $\gamma=1$: $\mathbb{E}[k]$ | 1.373 |
| Chipman: $\mathbb{E}[k]$ | 2.509 |
| $P(k\geq 5)$ ratio | $29.7\times$ (Catalan vs Chipman) |
| $N_{95}$ ratio | Catalan=300, Chipman>800 |
| PAC-Bayes $N_{\min}$ ratio ($k^*=1$) | $8.1\times$ (Catalan better) |
| Decision cost advantage | $17.4\%$ per patient (OA, $r=2$) |

---

## Contact

Vitaly Schetinin · vitaly.schetinin@gmail.com
