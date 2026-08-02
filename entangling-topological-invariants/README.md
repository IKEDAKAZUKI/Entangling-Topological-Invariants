# Entangling Topological Invariants

This repository contains the numerical code, data, and generated figures used
to study mixed Chern topology, finite-time cross pumping, and the fixed-rank
factorization obstruction of rank-four Berry bundles.

## Contents

- `code/` — numerical implementations and validation checks
- `config/seeds.json` — deterministic random seeds
- `data/` — numerical data used by the calculations and figures
- `figures/` — generated figures in vector format
- `requirements.txt` — Python dependencies
- `environment.yml` — equivalent conda environment

## Quick start

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Validate the distributed data and figures:

```bash
python code/validate_results.py
```

Regenerate figures from the included data and run the validation checks:

```bash
./reproduce_all.sh
```

Regenerate all numerical data before rebuilding the figures:

```bash
REGENERATE_ALL=1 ./reproduce_all.sh
```

The full regeneration includes dense parameter scans, finite-time evolution,
disorder ensembles, second-Chern integration, and finite-shot tomography, and
therefore requires substantially more time and memory than the default run.

## Numerical conventions

Random seeds are stored in `config/seeds.json`. BLAS thread counts are fixed by
the driver scripts for consistent execution. Figures are written as vector PDF
files; heat maps are evaluated on the numerical grids documented by the code.

## Citation

Please cite the associated article when using this software or data. Citation
metadata are provided in `CITATION.cff`.
