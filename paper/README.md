# Submission Package — Bayesian Analysis

## Files

| File | Description |
|------|-------------|
| `main.tex` | Main paper (~25 pages, Sections 1–8 + References) |
| `supplement.tex` | Supplementary appendices (Sections A–C) |
| `references.bib` | BibTeX bibliography (25 entries) |
| `cover_letter.tex` | Cover letter to the editor |
| `Makefile` | Build all PDFs with `make` |

## Prerequisites

### 1. Download `imsart.cls`

Bayesian Analysis is published by IMS and requires their LaTeX class.

Download `imsart.cls` and related files from:
```
https://www.e-publications.org/ims/support/imsart.html
```

Place `imsart.cls`, `imsart.sty`, and `ba.bst` in the same directory as `main.tex`.

### 2. Fallback (if `imsart.cls` is unavailable)

Replace the first line of `main.tex` and `supplement.tex`:
```latex
% Replace:
\documentclass[ba]{imsart}

% With:
\documentclass[12pt]{article}
```
And add to the preamble:
```latex
\usepackage{amsthm,amsmath,amssymb}
% Manually define theorem environments (already in main.tex as fallback comments)
```

### 3. Build

```bash
cd paper/
make all
```

Or manually:
```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplement && bibtex supplement && pdflatex supplement
pdflatex cover_letter
```

## Bayesian Analysis Submission Checklist

- [ ] Download and add `imsart.cls`, `imsart.sty`, `ba.bst`
- [ ] Verify PDF compiles without errors (`make all`)
- [ ] Check page limit (BA target: ≤30 pages in imsart double-spaced format)
- [ ] Add MSC 2020 subject codes to frontmatter (see below)
- [ ] Anonymise for double-blind review (remove author name/institution from frontmatter)
- [ ] Upload to ScholarOne Manuscripts: https://mc.manuscriptcentral.com/ba-stat
- [ ] Upload `main.pdf`, `supplement.pdf`, source `.tex` files, and `cover_letter.pdf`

## MSC 2020 Subject Classifications

Add to `main.tex` frontmatter inside `\begin{frontmatter}...\end{frontmatter}`:
```latex
\MSC[2020]{62F15, 68T05, 62C10, 62C12}
```

- **62F15** Bayesian inference
- **68T05** Learning and adaptive systems
- **62C10** Bayesian problems; characterization of Bayes procedures
- **62C12** Empirical decision procedures

## Key Numbers (for author verification)

| Result | Value |
|--------|-------|
| WAIC-LOO gap | $-4N/(N+\alpha_0) + O(N^{-1})$ |
| $N_{\min}$ formula | $\approx 5.41/\Delta$ |
| Catalan $\gamma=1$: $E[k]$ | 1.373 |
| Chipman: $E[k]$ | 2.509 |
| $P(k\geq 5)$ ratio | $29.7\times$ (Catalan vs Chipman) |
| $N_{95}$ ratio | Catalan=300, Chipman>800 |
| PAC-Bayes $N_{\min}$ ratio ($k^*=1$) | $8.1\times$ (Catalan better) |
| ECE ratio | $\sqrt{1.373/2.509} = 0.740$ |
| Decision cost advantage | $17.4\%$ per patient (OA, $r=2$) |

## Contact

Vitaly Schetinin · vitaly.schetinin@gmail.com
