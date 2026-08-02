# Changelog

## 5.0.0 - 2026-08-02

- Separated the mixed-Chern, label-resolution, pump, second-Chern, and tomography calculations into focused modules.
- Moved manuscript-level deformation paths to `code/eti/parameters.py`, with exact rational choices for the mixed-label direction and coupling endpoints.
- Removed exploratory random-model, bootstrap, point-noise, and smooth-noise routines that are not used by the article.
- Renamed the mixed-label data and figure files to match their role in the manuscript.
- Replaced the old validator name with `check_outputs.py`, which performs schema and regression checks on the distributed results.
- Kept the published PDF figures as the only rendered figure format in the repository.
