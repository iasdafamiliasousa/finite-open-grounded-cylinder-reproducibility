# Finite Open Grounded Cylinder — Reproducibility Package

This repository contains the numerical code and tabulated data used for the convergence analysis of the weighted Nyström solution for a point charge on the symmetry axis of a finite, open, grounded cylindrical shell.

## Contents

- `src/grounded_cylinder_convergence_extrapolated.py` — weighted Nyström convergence and continuum extrapolation code.
- `data/convergence_table_extrapolated.csv` — numerical values for the convergence table.
- `data/convergence_table_extrapolated_SWP.tex` — LaTeX version of the convergence table.
- `data/extrapolation_summary.txt` — fitted continuum limits and convergence exponents.

## Reference calculation

The default calculation reproduces the convergence study for

- `L/a = 6`
- `z0/a = 0`
- `N = 60, 80, 120, 180, 260, 360, 500, 700`

The code evaluates the total induced charge, the dimensionless surface-charge density at the source plane, the two rim-singularity amplitudes, and an independent off-grid residual. Continuum values are estimated from

`F(N) = F_inf + c N^(-p)`

using the five finest discretizations.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Reproduce the convergence table

From the repository root:

```bash
python src/grounded_cylinder_convergence_extrapolated.py --outdir results
```

The script writes:

- `results/convergence_table_extrapolated.csv`
- `results/convergence_table_extrapolated_SWP.tex`
- `results/extrapolation_summary.txt`

## Numerical reconstruction

Values of the regular auxiliary amplitude away from the Gauss–Chebyshev nodes are reconstructed by global Chebyshev–Lagrange interpolation implemented in barycentric form. The endpoint values are used to obtain the rim-singularity amplitudes. The off-grid residual is evaluated at independent angular test points, with the logarithmically singular integral split at each test point and integrated on the two subintervals by Gauss–Legendre quadrature.

## Reproducibility scope

This package reproduces the convergence calculations underlying the manuscript's numerical convergence tables.

## Citation

A persistent DOI will be added after archiving a release on Zenodo.
