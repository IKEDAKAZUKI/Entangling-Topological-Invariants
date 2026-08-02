# Entangling Topological Invariants

This repository contains the numerical implementation, distributed data, and
vector figures for the mixed-Chern, cross-pump, and fixed-rank factorization
calculations in *Entangling Topological Invariants*.

## Repository layout

The command-line drivers in `code/` are intentionally small. Their numerical
work is organized in `code/eti/`:

- `parameters.py` — named deformation endpoints and the exact mixed-label direction
- `mixed_chern.py` — QWZ blocks, mixed Chern curvature, and Fig. 1--2 data
- `label_resolution.py` — projected-label maps and hierarchical diagnostics
- `pump.py` — ribbon Hamiltonians and finite-cylinder time evolution
- `pump_figures.py` — Fig. 3 data assembly and rendering
- `s4_geometry.py` — Yang projectors, second-Chern quadrature, and patch frames
- `s4_pipeline.py` — structured Pauli-string model and clutching-grid data
- `tomography.py` — finite-shot and readout-error tomography
- `robustness.py` — pump ensembles, control metadata, and final figure assembly

The `data/` directory contains the arrays used in the article. The corresponding
PDF figures are in `figures/`. Deterministic seeds are recorded in
`config/seeds.json`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Check the schemas, reference values, and figure files distributed with the
repository:

```bash
python code/check_outputs.py
```

Rebuild the figures from the included data:

```bash
./reproduce_all.sh
```

Regenerate every numerical data set before rebuilding the figures:

```bash
REGENERATE_ALL=1 ./reproduce_all.sh
```

The complete regeneration includes dense parameter scans, finite-time
Schrödinger evolution, disorder ensembles, second-Chern integration, and
finite-shot tomography. The default command reuses the heavier distributed
data and regenerates the light calculations and all published figures.

## Numerical conventions

The driver scripts set BLAS thread counts explicitly. Heat-map grids,
quadrature orders, finite-cylinder dimensions, and shot budgets appear in the
corresponding modules and in the manuscript. The deformation paths shown in the manuscript are named in
`code/eti/parameters.py`. The mixed-label direction is stored there as an exact
rational vector, and the endpoint couplings are written as $6/5$ and $5/4$.

## Citation

Please cite the associated article when using this software or data. Citation
metadata are provided in `CITATION.cff`.
