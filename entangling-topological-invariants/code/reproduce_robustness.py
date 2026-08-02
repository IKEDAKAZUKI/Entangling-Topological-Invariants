from __future__ import annotations

from pathlib import Path
from repro_config import seed
import argparse
import csv
import importlib.util
import os
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Tinos', 'Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FIG = ROOT / 'figures'
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

spec_s4 = importlib.util.spec_from_file_location('structured_s4', Path(__file__).with_name('reproduce_structured_s4.py'))
s4 = importlib.util.module_from_spec(spec_s4)
assert spec_s4.loader is not None
spec_s4.loader.exec_module(s4)
r = s4.r


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _equatorial_hamiltonian_grid(kind: str, matrices: np.ndarray, ngrid: int) -> tuple[np.ndarray, ...]:
    ch, th, ph, wch, wth, wph = r.tomography_quadrature_axes(ngrid)
    shape = (ngrid, ngrid, 2 * ngrid)
    nvec = np.empty(shape + (5,), float)
    h = np.empty(shape + (8, 8), complex)
    epsilon = 1.25 if kind == 'odd_mixed' else 0.0
    for i, cv in enumerate(ch):
        for j, tv in enumerate(th):
            for k, pv in enumerate(ph):
                n = r.s3_equator_vector(float(cv), float(tv), float(pv))
                nvec[i, j, k] = n
                if kind == 'odd_mixed':
                    h[i, j, k] = r.odd_flat_hamiltonian(n) + epsilon * r.generic_s4_perturbation(n, matrices)
                else:
                    h[i, j, k] = r.even_flat_hamiltonian(n)
    return ch, th, ph, wch, wth, wph, nvec, h


def _transition_winding_from_hamiltonian_grid(
    h: np.ndarray, rn: np.ndarray, rs: np.ndarray,
    ch: np.ndarray, th: np.ndarray, ph: np.ndarray,
    wch: np.ndarray, wth: np.ndarray, wph: float,
) -> float:
    """Reconstruct one clean transition family using the same batch shape
    as the noisy data.  Keeping the LAPACK batch shape fixed avoids severe
    mixed-batch slowdowns in some OpenBLAS builds.
    """
    _, u = np.linalg.eigh(h)
    occ = u[..., :4]
    p = occ @ np.swapaxes(occ.conj(), -1, -2)
    un, _ = s4._batch_patch_frame(p, rn)
    us, _ = s4._batch_patch_frame(p, rs)
    gg = np.swapaxes(un.conj(), -1, -2) @ us
    uu, _, vh = np.linalg.svd(gg)
    gg = uu @ vh
    return float(s4._grid_winding_batch(gg[None], ch, th, ph, wch, wth, wph)[0])



def _pauli_tomography_design():
    """Local three-qubit Pauli-basis tomography design.

    Twenty-seven settings, X/Y/Z on each of the three two-level factors,
    provide multinomial eight-outcome data.  All 63 nontrivial Pauli
    coefficients are reconstructed by averaging over compatible settings.
    """
    local = (r.I2, r.sx, r.sy, r.sz)
    axis_names = ('I', 'X', 'Y', 'Z')
    pauli_indices = [(a, b, c) for a in range(4) for b in range(4) for c in range(4)]
    pauli_labels = [''.join(axis_names[x] for x in idx) for idx in pauli_indices]
    pauli_matrices = np.asarray([
        np.kron(np.kron(local[a], local[b]), local[c]) for a, b, c in pauli_indices
    ])

    settings = []
    denominator = np.zeros(64, float)
    for a in range(1, 4):
        for b in range(1, 4):
            for c in range(1, 4):
                eigenvalues, eigenvectors = [], []
                for axis in (a, b, c):
                    ev, vec = np.linalg.eigh(local[axis])
                    eigenvalues.append(ev)
                    eigenvectors.append(vec)
                basis = np.kron(np.kron(eigenvectors[0], eigenvectors[1]), eigenvectors[2])
                outcomes = np.asarray([
                    (x, y, z)
                    for x in eigenvalues[0]
                    for y in eigenvalues[1]
                    for z in eigenvalues[2]
                ])
                weights = np.zeros((64, 8), float)
                compatible = np.zeros(64, float)
                for index, pauli_index in enumerate(pauli_indices):
                    if all(p_axis == 0 or p_axis == setting_axis
                           for p_axis, setting_axis in zip(pauli_index, (a, b, c))):
                        outcome_weight = np.ones(8, float)
                        for subsystem, p_axis in enumerate(pauli_index):
                            if p_axis != 0:
                                outcome_weight *= outcomes[:, subsystem]
                        weights[index] = outcome_weight
                        compatible[index] = 1.0
                denominator += compatible
                settings.append((
                    ''.join(axis_names[x] for x in (a, b, c)),
                    basis,
                    weights,
                    compatible,
                ))
    if np.any(denominator <= 0):
        raise RuntimeError('incomplete Pauli tomography design')
    return pauli_matrices, pauli_labels, tuple(settings), denominator


def _pauli_setting_probabilities(p_true: np.ndarray, settings) -> np.ndarray:
    """Probabilities for rho_occ=P/4 in each local Pauli basis."""
    rho = p_true.reshape((-1, 8, 8)) / 4.0
    probabilities = []
    for _, basis, _, _ in settings:
        rotated = np.einsum('ia,pij,jb->pab', basis.conj(), rho, basis, optimize=True)
        pvals = np.real(np.diagonal(rotated, axis1=-2, axis2=-1))
        pvals = np.clip(pvals, 0.0, None)
        pvals /= np.sum(pvals, axis=-1, keepdims=True)
        probabilities.append(pvals)
    return np.asarray(probabilities)


def _rank4_projectors_from_pauli_shots(
    probabilities: np.ndarray,
    shots_per_setting: int,
    batch: int,
    rng: np.random.Generator,
    pauli_matrices: np.ndarray,
    settings,
    denominator: np.ndarray,
) -> np.ndarray:
    """Linear inversion followed by rank-four spectral projection.

    Multinomial counts are linearly inverted into a density matrix.  The
    four leading eigenvectors define the nearest rank-four projector in
    Frobenius norm, which is the object needed by the transition protocol.
    """
    npoints = probabilities.shape[1]
    coefficients = np.zeros((batch, npoints, 64), float)
    for setting_index, (_, _, weights, _) in enumerate(settings):
        counts = rng.multinomial(
            int(shots_per_setting), probabilities[setting_index], size=(batch, npoints)
        )
        frequencies = counts / float(shots_per_setting)
        coefficients += np.einsum('bpo,so->bps', frequencies, weights, optimize=True)
    coefficients /= denominator[None, None, :]
    coefficients[..., 0] = 1.0
    rho_linear = np.einsum('bps,sij->bpij', coefficients, pauli_matrices, optimize=True) / 8.0
    rho_linear = (rho_linear + np.swapaxes(rho_linear.conj(), -1, -2)) / 2.0
    _, eigenvectors = np.linalg.eigh(rho_linear)
    occupied = eigenvectors[..., -4:]
    return occupied @ np.swapaxes(occupied.conj(), -1, -2)



def _apply_symmetric_readout_confusion(probabilities: np.ndarray, error_rate: float) -> np.ndarray:
    """Apply independent symmetric binary readout confusion to three hardware factors.

    `probabilities` has axes (setting, parameter_point, true_outcome).  The
    returned probabilities are indexed by observed outcomes.  Reconstruction
    intentionally assumes the ideal Pauli readout model, so this tests an
    unmitigated SPAM systematic rather than a calibrated correction.
    """
    rate = float(error_rate)
    if not (0.0 <= rate < 0.5):
        raise ValueError('readout error rate must lie in [0, 1/2)')
    local = np.asarray([[1.0 - rate, rate], [rate, 1.0 - rate]])
    channel = np.kron(np.kron(local, local), local)
    observed = np.einsum('ot,spt->spo', channel, probabilities, optimize=True)
    observed = np.clip(observed, 0.0, None)
    observed /= np.sum(observed, axis=-1, keepdims=True)
    return observed


def _parity_margin(winding: np.ndarray) -> np.ndarray:
    """Distance to the nearest half-integer decision boundary."""
    values = np.asarray(winding, dtype=float)
    return np.abs((values - 0.5) - np.rint(values - 0.5))


def _projector_metrics_and_winding(
    reconstructed_flat: np.ndarray,
    clean_flat: np.ndarray,
    ngrid: int,
    rank: int,
    rn: np.ndarray,
    rs: np.ndarray,
    ch: np.ndarray,
    th: np.ndarray,
    ph: np.ndarray,
    wch: np.ndarray,
    wth: np.ndarray,
    wph: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gauge-invariant subspace metrics and one-grid winding for a batch."""
    fidelity_grid = np.real(np.einsum(
        'pij,bpji->bp', clean_flat, reconstructed_flat, optimize=True
    )) / rank
    fidelity_grid = np.clip(fidelity_grid, 0.0, 1.0)
    distance = np.sqrt(np.mean(np.maximum(0.0, 1.0 - fidelity_grid), axis=1))
    fidelity = np.mean(fidelity_grid, axis=1)
    batch = reconstructed_flat.shape[0]
    reconstructed = reconstructed_flat.reshape(
        (batch, ngrid, ngrid, 2 * ngrid, 8, 8)
    )
    un, en = s4._batch_patch_frame(reconstructed, rn)
    us, es = s4._batch_patch_frame(reconstructed, rs)
    transition = np.swapaxes(un.conj(), -1, -2) @ us
    uu, _, vh = np.linalg.svd(transition)
    transition = uu @ vh
    winding = s4._grid_winding_batch(transition, ch, th, ph, wch, wth, wph)
    minimum_overlap = np.minimum(
        np.min(en, axis=(1, 2, 3)), np.min(es, axis=(1, 2, 3))
    )
    return winding, minimum_overlap, distance, fidelity


def _draw_pauli_counts(
    probabilities: np.ndarray,
    shots_per_setting: int,
    batch: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw multinomial count arrays with shape (batch, setting, point, outcome)."""
    settings_count, npoints, _ = probabilities.shape
    out = np.empty((batch, settings_count, npoints, 8), dtype=np.int32)
    for setting_index in range(settings_count):
        out[:, setting_index] = rng.multinomial(
            int(shots_per_setting), probabilities[setting_index], size=(batch, npoints)
        )
    return out


def _rank4_projectors_from_pauli_counts(
    counts: np.ndarray,
    pauli_matrices: np.ndarray,
    settings,
    denominator: np.ndarray,
) -> np.ndarray:
    """Linear inversion followed by rank-four spectral projection from sampled counts."""
    batch, settings_count, npoints, _ = counts.shape
    if settings_count != len(settings):
        raise ValueError('count/settings mismatch')
    coefficients = np.zeros((batch, npoints, 64), float)
    totals = np.sum(counts, axis=-1, keepdims=True)
    frequencies = counts / np.maximum(totals, 1)
    for setting_index, (_, _, weights, _) in enumerate(settings):
        coefficients += np.einsum(
            'bpo,so->bps', frequencies[:, setting_index], weights, optimize=True
        )
    coefficients /= denominator[None, None, :]
    coefficients[..., 0] = 1.0
    rho_linear = np.einsum(
        'bps,sij->bpij', coefficients, pauli_matrices, optimize=True
    ) / 8.0
    rho_linear = (rho_linear + np.swapaxes(rho_linear.conj(), -1, -2)) / 2.0
    _, eigenvectors = np.linalg.eigh(rho_linear)
    occupied = eigenvectors[..., -4:]
    return occupied @ np.swapaxes(occupied.conj(), -1, -2)


def finite_shot_tomography_data() -> None:
    """Finite-shot occupied-subspace tomography and parity classification.

    The three binary factors of the structured eight-level model are measured
    in the 27 local Pauli bases.  Each setting has eight outcomes and receives
    an equal number of shots.  Pole reference subspaces are treated as two
    one-time pre-calibrations; the stochastic budget used below concerns
    the equatorial projector reconstructions.
    """
    matrices = s4.structured_s4_matrices()
    ngrid = 4
    rank = 4
    realizations = 100
    shots_values = np.asarray([40, 50, 60, 75, 100, 125, 150, 175, 200, 300, 500, 1000])
    batch_size = 10
    rng = np.random.default_rng(seed("finite_shot_tomography"))
    pauli_matrices, pauli_labels, settings, denominator = _pauli_tomography_design()

    with open(DATA / 'tomography_pauli_settings.csv', 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([
            'setting_label', 'basis_kappa', 'basis_tau', 'basis_sigma',
            'outcomes', 'compatible_Pauli_coefficients'
        ])
        for label, _, _, compatible in settings:
            writer.writerow([label, label[0], label[1], label[2], 8, int(np.sum(compatible))])

    with open(DATA / 'tomography_pauli_coefficients.csv', 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['coefficient_index', 'Pauli_label', 'compatible_settings'])
        for index, (label, count) in enumerate(zip(pauli_labels, denominator)):
            writer.writerow([index, label, int(count)])

    realization_rows: list[list[float]] = []
    summary_rows: list[list[float]] = []
    settings_per_point = len(settings)
    equatorial_points = 2 * ngrid**3
    pole_calibrations = 2

    for kind, target in (('odd_mixed', 1), ('even', 2)):
        ch, th, ph, wch, wth, wph, _, base = _equatorial_hamiltonian_grid(kind, matrices, ngrid)
        epsilon = 1.25 if kind == 'odd_mixed' else 0.0
        rn, rs = r.tomography_references(kind, epsilon, matrices)
        clean_winding = _transition_winding_from_hamiltonian_grid(
            base, rn, rs, ch, th, ph, wch, wth, wph
        )
        _, clean_vectors = np.linalg.eigh(base)
        clean_occupied = clean_vectors[..., :rank]
        clean_projector = clean_occupied @ np.swapaxes(clean_occupied.conj(), -1, -2)
        clean_flat = clean_projector.reshape((-1, 8, 8))
        probabilities = _pauli_setting_probabilities(clean_projector, settings)
        kind_code = 0 if kind == 'odd_mixed' else 1

        for shots_per_setting in shots_values:
            winding_values: list[float] = []
            minimum_overlaps: list[float] = []
            projector_distances: list[float] = []
            subspace_fidelities: list[float] = []

            for start in range(0, realizations, batch_size):
                batch = min(batch_size, realizations - start)
                reconstructed_flat = _rank4_projectors_from_pauli_shots(
                    probabilities, int(shots_per_setting), batch, rng,
                    pauli_matrices, settings, denominator,
                )
                fidelity_grid = np.real(np.einsum(
                    'pij,bpji->bp', clean_flat, reconstructed_flat, optimize=True
                )) / rank
                fidelity_grid = np.clip(fidelity_grid, 0.0, 1.0)
                distance = np.sqrt(np.mean(np.maximum(0.0, 1.0 - fidelity_grid), axis=1))
                fidelity = np.mean(fidelity_grid, axis=1)

                reconstructed = reconstructed_flat.reshape(
                    (batch, ngrid, ngrid, 2 * ngrid, 8, 8)
                )
                un, en = s4._batch_patch_frame(reconstructed, rn)
                us, es = s4._batch_patch_frame(reconstructed, rs)
                transition = np.swapaxes(un.conj(), -1, -2) @ us
                uu, _, vh = np.linalg.svd(transition)
                transition = uu @ vh
                winding = s4._grid_winding_batch(
                    transition, ch, th, ph, wch, wth, wph
                )
                minimum_overlap = np.minimum(
                    np.min(en, axis=(1, 2, 3)), np.min(es, axis=(1, 2, 3))
                )

                winding_values.extend(winding.tolist())
                minimum_overlaps.extend(minimum_overlap.tolist())
                projector_distances.extend(distance.tolist())
                subspace_fidelities.extend(fidelity.tolist())

            windings = np.asarray(winding_values)
            overlaps = np.asarray(minimum_overlaps)
            distances = np.asarray(projector_distances)
            fidelities = np.asarray(subspace_fidelities)
            nearest = np.rint(windings).astype(int)
            parity_margin = np.abs((windings - 0.5) - np.rint(windings - 0.5))
            parity_correct = (nearest % 2) == (target % 2)
            integer_correct = nearest == target
            ci_low, ci_high = wilson_interval(int(np.sum(parity_correct)), realizations)
            shots_per_point = settings_per_point * int(shots_per_setting)
            total_shots = equatorial_points * shots_per_point

            for realization, (winding, overlap, distance, fidelity) in enumerate(zip(
                    windings, overlaps, distances, fidelities)):
                realization_rows.append([
                    kind_code, target, int(shots_per_setting), settings_per_point,
                    shots_per_point, equatorial_points, total_shots, pole_calibrations,
                    realization, clean_winding, winding, overlap, distance, fidelity,
                    nearest[realization], parity_margin[realization],
                    int(not parity_correct[realization]), int(not integer_correct[realization]),
                ])

            summary_rows.append([
                kind_code, target, int(shots_per_setting), settings_per_point,
                shots_per_point, equatorial_points, total_shots, pole_calibrations,
                clean_winding, float(np.mean(windings)), float(np.std(windings, ddof=1)),
                float(np.mean(distances)), float(np.std(distances, ddof=1)),
                float(np.mean(fidelities)), float(np.std(fidelities, ddof=1)),
                float(np.mean(overlaps)), float(np.min(overlaps)),
                float(np.mean(parity_margin)), float(np.min(parity_margin)),
                float(np.quantile(parity_margin, 0.10)),
                float(np.mean(parity_correct)), ci_low, ci_high,
                float(np.mean(integer_correct)),
            ])

    np.savetxt(
        DATA / 'tomography_finite_shot_realizations.csv', np.asarray(realization_rows), delimiter=',',
        header=(
            'kind_code,target,shots_per_setting,settings_per_parameter_point,'
            'shots_per_parameter_point,equatorial_points_per_family,total_equatorial_shots_per_family,'
            'one_time_pole_calibrations,realization,W3_clean_grid,W3_reconstructed,min_patch_overlap,'
            'projector_distance_rms,mean_subspace_fidelity,nearest_integer,parity_margin_to_half_integer,parity_error,integer_error'
        ), comments=''
    )
    np.savetxt(
        DATA / 'tomography_finite_shot_summary.csv', np.asarray(summary_rows), delimiter=',',
        header=(
            'kind_code,target,shots_per_setting,settings_per_parameter_point,'
            'shots_per_parameter_point,equatorial_points_per_family,total_equatorial_shots_per_family,'
            'one_time_pole_calibrations,W3_clean_grid,mean_winding,std_winding,'
            'mean_projector_distance_rms,std_projector_distance_rms,'
            'mean_subspace_fidelity,std_subspace_fidelity,mean_min_patch_overlap,minimum_patch_overlap,'
            'mean_parity_margin,minimum_parity_margin,parity_margin_10th_percentile,'
            'parity_correct_rate,parity_correct_ci_low,parity_correct_ci_high,integer_correct_rate'
        ), comments=''
    )


def readout_confusion_tomography_data() -> None:
    """Finite-shot parity classification with unmitigated readout confusion.

    Each of the three ambient hardware bits is independently reported
    incorrectly with probability r.  State preparation, control, and the
    Hamiltonian model remain ideal; the reconstruction continues to assume
    ideal Pauli readout.  This therefore isolates one concrete SPAM systematic.
    """
    matrices = s4.structured_s4_matrices()
    ngrid = 4
    rank = 4
    realizations = 100
    shots_values = np.asarray([100, 150, 200, 300, 500])
    readout_rates = np.asarray([0.0, 0.005, 0.01, 0.02])
    batch_size = 10
    rng = np.random.default_rng(seed("readout_confusion_tomography"))
    pauli_matrices, _, settings, denominator = _pauli_tomography_design()
    settings_per_point = len(settings)
    equatorial_points = 2 * ngrid**3
    rows: list[list[float]] = []
    summaries: list[list[float]] = []

    for kind, target in (('odd_mixed', 1), ('even', 2)):
        ch, th, ph, wch, wth, wph, _, base = _equatorial_hamiltonian_grid(
            kind, matrices, ngrid
        )
        epsilon = 1.25 if kind == 'odd_mixed' else 0.0
        rn, rs = r.tomography_references(kind, epsilon, matrices)
        clean_winding = _transition_winding_from_hamiltonian_grid(
            base, rn, rs, ch, th, ph, wch, wth, wph
        )
        _, clean_vectors = np.linalg.eigh(base)
        clean_occ = clean_vectors[..., :rank]
        clean_projector = clean_occ @ np.swapaxes(clean_occ.conj(), -1, -2)
        clean_flat = clean_projector.reshape((-1, 8, 8))
        ideal_probabilities = _pauli_setting_probabilities(clean_projector, settings)
        kind_code = 0 if kind == 'odd_mixed' else 1

        for readout_rate in readout_rates:
            probabilities = _apply_symmetric_readout_confusion(
                ideal_probabilities, float(readout_rate)
            )
            for shots_per_setting in shots_values:
                winding_values: list[float] = []
                overlaps: list[float] = []
                distances: list[float] = []
                fidelities: list[float] = []
                for start in range(0, realizations, batch_size):
                    batch = min(batch_size, realizations - start)
                    reconstructed_flat = _rank4_projectors_from_pauli_shots(
                        probabilities, int(shots_per_setting), batch, rng,
                        pauli_matrices, settings, denominator,
                    )
                    winding, overlap, distance, fidelity = _projector_metrics_and_winding(
                        reconstructed_flat, clean_flat, ngrid, rank, rn, rs,
                        ch, th, ph, wch, wth, wph,
                    )
                    winding_values.extend(winding.tolist())
                    overlaps.extend(overlap.tolist())
                    distances.extend(distance.tolist())
                    fidelities.extend(fidelity.tolist())

                windings = np.asarray(winding_values)
                overlap_values = np.asarray(overlaps)
                distance_values = np.asarray(distances)
                fidelity_values = np.asarray(fidelities)
                nearest = np.rint(windings).astype(int)
                margins = _parity_margin(windings)
                parity_correct = (nearest % 2) == (target % 2)
                integer_correct = nearest == target
                ci_low, ci_high = wilson_interval(int(np.sum(parity_correct)), realizations)
                shots_per_point = settings_per_point * int(shots_per_setting)
                total_shots = equatorial_points * shots_per_point

                for realization in range(realizations):
                    rows.append([
                        kind_code, target, float(readout_rate), int(shots_per_setting),
                        settings_per_point, shots_per_point, equatorial_points, total_shots,
                        realization, clean_winding, windings[realization],
                        overlap_values[realization], distance_values[realization],
                        fidelity_values[realization], nearest[realization],
                        margins[realization], int(not parity_correct[realization]),
                        int(not integer_correct[realization]),
                    ])

                summaries.append([
                    kind_code, target, float(readout_rate), int(shots_per_setting),
                    settings_per_point, shots_per_point, equatorial_points, total_shots,
                    clean_winding, float(np.mean(windings)), float(np.std(windings, ddof=1)),
                    float(np.mean(distance_values)), float(np.mean(fidelity_values)),
                    float(np.mean(overlap_values)), float(np.min(overlap_values)),
                    float(np.mean(margins)), float(np.min(margins)),
                    float(np.quantile(margins, 0.10)), float(np.mean(parity_correct)),
                    ci_low, ci_high, float(np.mean(integer_correct)),
                ])

    np.savetxt(
        DATA / 'tomography_readout_confusion_realizations.csv', np.asarray(rows), delimiter=',',
        header=(
            'kind_code,target,readout_flip_probability,shots_per_setting,'
            'settings_per_parameter_point,shots_per_parameter_point,equatorial_points_per_family,'
            'total_equatorial_shots_per_family,realization,W3_clean_grid,W3_reconstructed,'
            'min_patch_overlap,projector_distance_rms,mean_subspace_fidelity,nearest_integer,'
            'parity_margin_to_half_integer,parity_error,integer_error'
        ), comments=''
    )
    np.savetxt(
        DATA / 'tomography_readout_confusion_summary.csv', np.asarray(summaries), delimiter=',',
        header=(
            'kind_code,target,readout_flip_probability,shots_per_setting,'
            'settings_per_parameter_point,shots_per_parameter_point,equatorial_points_per_family,'
            'total_equatorial_shots_per_family,W3_clean_grid,mean_winding,std_winding,'
            'mean_projector_distance_rms,mean_subspace_fidelity,mean_min_patch_overlap,'
            'minimum_patch_overlap,mean_parity_margin,minimum_parity_margin,'
            'parity_margin_10th_percentile,parity_correct_rate,parity_correct_ci_low,'
            'parity_correct_ci_high,integer_correct_rate'
        ), comments=''
    )


def bootstrap_tomography_data() -> None:
    """Parametric multinomial bootstrap for one fitted finite-shot data set per family."""
    matrices = s4.structured_s4_matrices()
    ngrid = 4
    rank = 4
    shots_per_setting = 500
    bootstrap_replicates = 200
    batch_size = 10
    rng_data = np.random.default_rng(seed("bootstrap_observed_dataset"))
    rng_boot = np.random.default_rng(seed("bootstrap_resampling"))
    pauli_matrices, _, settings, denominator = _pauli_tomography_design()
    realization_rows: list[list[float]] = []
    summary_rows: list[list[float]] = []

    for kind, target in (('odd_mixed', 1), ('even', 2)):
        ch, th, ph, wch, wth, wph, _, base = _equatorial_hamiltonian_grid(
            kind, matrices, ngrid
        )
        epsilon = 1.25 if kind == 'odd_mixed' else 0.0
        rn, rs = r.tomography_references(kind, epsilon, matrices)
        _, clean_vectors = np.linalg.eigh(base)
        clean_occ = clean_vectors[..., :rank]
        clean_projector = clean_occ @ np.swapaxes(clean_occ.conj(), -1, -2)
        clean_flat = clean_projector.reshape((-1, 8, 8))
        probabilities = _pauli_setting_probabilities(clean_projector, settings)
        observed_counts = _draw_pauli_counts(
            probabilities, shots_per_setting, 1, rng_data
        )
        observed_flat = _rank4_projectors_from_pauli_counts(
            observed_counts, pauli_matrices, settings, denominator
        )
        observed_w, observed_overlap, observed_distance, observed_fidelity = (
            _projector_metrics_and_winding(
                observed_flat, clean_flat, ngrid, rank, rn, rs,
                ch, th, ph, wch, wth, wph,
            )
        )
        observed_grid = observed_flat[0].reshape((ngrid, ngrid, 2 * ngrid, 8, 8))
        empirical = _pauli_setting_probabilities(observed_grid, settings)
        boot_values: list[float] = []
        boot_overlaps: list[float] = []
        for start in range(0, bootstrap_replicates, batch_size):
            batch = min(batch_size, bootstrap_replicates - start)
            boot_counts = _draw_pauli_counts(
                empirical, shots_per_setting, batch, rng_boot
            )
            boot_flat = _rank4_projectors_from_pauli_counts(
                boot_counts, pauli_matrices, settings, denominator
            )
            winding, overlap, _, _ = _projector_metrics_and_winding(
                boot_flat, clean_flat, ngrid, rank, rn, rs,
                ch, th, ph, wch, wth, wph,
            )
            boot_values.extend(winding.tolist())
            boot_overlaps.extend(overlap.tolist())

        windings = np.asarray(boot_values)
        overlaps = np.asarray(boot_overlaps)
        nearest = np.rint(windings).astype(int)
        margins = _parity_margin(windings)
        observed_value = float(observed_w[0])
        observed_nearest = int(np.rint(observed_value))
        target_hits = (nearest % 2) == (target % 2)
        observed_hits = (nearest % 2) == (observed_nearest % 2)
        target_stability = float(np.mean(target_hits))
        observed_stability = float(np.mean(observed_hits))
        target_ci_low, target_ci_high = wilson_interval(int(np.sum(target_hits)), bootstrap_replicates)
        observed_ci_low, observed_ci_high = wilson_interval(int(np.sum(observed_hits)), bootstrap_replicates)
        q025 = float(np.quantile(windings, 0.025))
        q500 = float(np.median(windings))
        q975 = float(np.quantile(windings, 0.975))
        bootstrap_mean = float(np.mean(windings))
        bootstrap_bias = bootstrap_mean - observed_value
        bias_corrected_point = observed_value - bootstrap_bias
        basic_interval_low = 2 * observed_value - q975
        basic_interval_high = 2 * observed_value - q025
        kind_code = 0 if kind == 'odd_mixed' else 1
        for index in range(bootstrap_replicates):
            realization_rows.append([
                kind_code, target, shots_per_setting, index, windings[index],
                nearest[index], margins[index], overlaps[index],
                int((nearest[index] % 2) == (target % 2)),
            ])
        summary_rows.append([
            kind_code, target, shots_per_setting, bootstrap_replicates,
            observed_value, observed_nearest, _parity_margin(observed_w)[0],
            observed_overlap[0], observed_distance[0], observed_fidelity[0],
            bootstrap_mean, float(np.std(windings, ddof=1)), bootstrap_bias,
            bias_corrected_point, q025, q500, q975,
            basic_interval_low, basic_interval_high, float(np.mean(margins)),
            float(np.quantile(margins, 0.10)), target_stability,
            target_ci_low, target_ci_high, observed_stability,
            observed_ci_low, observed_ci_high,
        ])

    np.savetxt(
        DATA / 'tomography_bootstrap_realizations.csv', np.asarray(realization_rows), delimiter=',',
        header=(
            'kind_code,target,shots_per_setting,bootstrap_replicate,W3_bootstrap,'
            'nearest_integer,parity_margin_to_half_integer,min_patch_overlap,target_parity_correct'
        ), comments=''
    )
    np.savetxt(
        DATA / 'tomography_bootstrap_summary.csv', np.asarray(summary_rows), delimiter=',',
        header=(
            'kind_code,target,shots_per_setting,bootstrap_replicates,W3_observed,'
            'observed_nearest_integer,observed_parity_margin,min_patch_overlap_observed,'
            'projector_distance_observed,subspace_fidelity_observed,mean_bootstrap_winding,'
            'std_bootstrap_winding,bootstrap_bias,bias_corrected_point_estimate,'
            'bootstrap_q025,bootstrap_median,bootstrap_q975,basic_interval_low,basic_interval_high,'
            'mean_bootstrap_parity_margin,bootstrap_margin_10th_percentile,'
            'target_parity_stability,target_parity_ci_low,target_parity_ci_high,'
            'observed_parity_stability,observed_parity_ci_low,observed_parity_ci_high'
        ), comments=''
    )


POINT_NOISE_DELTAS = np.asarray([
    1e-3, 1e-2, 5e-2, .1, .2, .4, .6, .8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0
])


def _pointwise_tomography_noise_chunk(kind_code: int, delta_index: int) -> tuple[np.ndarray, np.ndarray]:
    """One independent point-noise ensemble in a fresh process.

    Running every (family, amplitude) task in a short-lived interpreter avoids
    the long-lived batched-LAPACK memory growth observed on some OpenBLAS
    builds.  Scientific parameters remain unchanged: 100 realizations on the
    N=6 equatorial grid for each family and amplitude.
    """
    if kind_code not in (0, 1):
        raise ValueError('kind_code must be 0 (odd) or 1 (even)')
    if not (0 <= delta_index < len(POINT_NOISE_DELTAS)):
        raise ValueError('delta_index out of range')
    kind, target = ('odd_mixed', 1) if kind_code == 0 else ('even', 2)
    delta = float(POINT_NOISE_DELTAS[delta_index])
    matrices = s4.structured_s4_matrices()
    ngrid = 6
    rank = 4
    realizations = 100
    batch_size = 20
    rng = np.random.default_rng(seed('pointwise_tomography_noise') + 1000 * kind_code + delta_index)

    ch, th, ph, wch, wth, wph, _, base = _equatorial_hamiltonian_grid(kind, matrices, ngrid)
    epsilon = 1.25 if kind == 'odd_mixed' else 0.0
    rn, rs = r.tomography_references(kind, epsilon, matrices)
    clean_w = _transition_winding_from_hamiltonian_grid(base, rn, rs, ch, th, ph, wch, wth, wph)
    _, clean_u = np.linalg.eigh(base)
    clean_occ = clean_u[..., :rank]
    clean_p = clean_occ @ np.swapaxes(clean_occ.conj(), -1, -2)

    values: list[float] = []
    overlaps: list[float] = []
    projector_distances: list[float] = []
    subspace_fidelities: list[float] = []
    for start_index in range(0, realizations, batch_size):
        batch = min(batch_size, realizations - start_index)
        noise = rng.normal(size=(batch,) + base.shape) + 1j * rng.normal(size=(batch,) + base.shape)
        noise = (noise + np.swapaxes(noise.conj(), -1, -2)) / 2
        noise /= np.linalg.norm(noise, axis=(-2, -1), keepdims=True)
        h = base[None] + delta * noise
        _, u = np.linalg.eigh(h)
        occ = u[..., :rank]
        p_rec = occ @ np.swapaxes(occ.conj(), -1, -2)

        f_grid = np.real(np.einsum('...ij,b...ji->b...', clean_p, p_rec, optimize=True)) / rank
        f_grid = np.clip(f_grid, 0.0, 1.0)
        d2_grid = np.maximum(0.0, 1.0 - f_grid)
        d_rms = np.sqrt(np.mean(d2_grid, axis=(1, 2, 3)))
        f_mean = np.mean(f_grid, axis=(1, 2, 3))

        un, en = s4._batch_patch_frame(p_rec, rn)
        us, es = s4._batch_patch_frame(p_rec, rs)
        gg = np.swapaxes(un.conj(), -1, -2) @ us
        uu, _, vh = np.linalg.svd(gg)
        gg = uu @ vh
        winding = s4._grid_winding_batch(gg, ch, th, ph, wch, wth, wph)
        minov = np.minimum(np.min(en, axis=(1, 2, 3)), np.min(es, axis=(1, 2, 3)))

        values.extend(winding.tolist())
        overlaps.extend(minov.tolist())
        projector_distances.extend(d_rms.tolist())
        subspace_fidelities.extend(f_mean.tolist())

    vals = np.asarray(values)
    ovs = np.asarray(overlaps)
    dps = np.asarray(projector_distances)
    fids = np.asarray(subspace_fidelities)
    nearest = np.rint(vals).astype(int)
    margins = _parity_margin(vals)
    correct = (nearest % 2) == (target % 2)
    integer_correct = nearest == target
    low, high = wilson_interval(int(np.sum(correct)), realizations)
    metric_corr = float(np.corrcoef(np.abs(vals - clean_w), dps)[0, 1])

    rows = np.column_stack([
        np.full(realizations, kind_code), np.full(realizations, delta), np.arange(realizations),
        vals, np.full(realizations, clean_w), ovs, dps, fids, nearest, margins,
        (~correct).astype(int), (~integer_correct).astype(int),
    ])
    summary = np.asarray([[
        kind_code, target, delta, clean_w,
        float(np.mean(vals)), float(np.std(vals, ddof=1)),
        float(np.sqrt(np.mean((vals - clean_w) ** 2))),
        float(np.mean(dps)), float(np.std(dps, ddof=1)),
        float(np.mean(fids)), float(np.std(fids, ddof=1)),
        float(np.mean(ovs)), float(np.min(ovs)),
        float(np.mean(margins)), float(np.min(margins)),
        float(np.mean(correct)), low, high,
        float(np.mean(integer_correct)), metric_corr,
    ]])
    return rows, summary


def _write_point_noise_chunk(kind_code: int, delta_index: int) -> None:
    rows, summary = _pointwise_tomography_noise_chunk(kind_code, delta_index)
    chunk_dir = DATA / '.point_noise_chunks'
    chunk_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        chunk_dir / f'kind{kind_code}_delta{delta_index:02d}.npz', rows=rows, summary=summary
    )


def pointwise_tomography_noise_data() -> None:
    """Classifier robustness for independently corrupted reconstruction points.

    The 30 independent ensembles are generated in fresh subprocesses to keep
    memory use bounded and the full-regeneration path reliable.  These data do
    not define a smooth perturbed Hamiltonian family; they test the discrete
    transition-winding estimator under pointwise reconstruction errors.
    """
    import shutil
    import subprocess
    import sys

    chunk_dir = DATA / '.point_noise_chunks'
    shutil.rmtree(chunk_dir, ignore_errors=True)
    chunk_dir.mkdir(exist_ok=True)
    for kind_code in (0, 1):
        for delta_index, delta in enumerate(POINT_NOISE_DELTAS):
            print(f'[point-noise] family={kind_code} delta={float(delta):.3g}', flush=True)
            subprocess.check_call([
                sys.executable, __file__, '--part', 'point_noise_chunk',
                '--kind-code', str(kind_code), '--delta-index', str(delta_index),
            ])

    all_rows, summaries = [], []
    for kind_code in (0, 1):
        for delta_index in range(len(POINT_NOISE_DELTAS)):
            path = chunk_dir / f'kind{kind_code}_delta{delta_index:02d}.npz'
            with np.load(path) as archive:
                all_rows.append(archive['rows'])
                summaries.append(archive['summary'])
    rows = np.concatenate(all_rows, axis=0)
    summary = np.concatenate(summaries, axis=0)

    np.savetxt(
        DATA / 'tomography_point_noise_realizations.csv', rows, delimiter=',',
        header=(
            'kind_code,delta,realization,W3_reconstructed,W3_clean_grid,min_patch_overlap,'
            'projector_distance_rms,mean_subspace_fidelity,nearest_integer,'
            'parity_margin_to_half_integer,parity_error,integer_error'
        ), comments=''
    )
    np.savetxt(
        DATA / 'tomography_point_noise_summary.csv', summary, delimiter=',',
        header=(
            'kind_code,target,delta,clean_grid_winding,mean_winding,std_winding,'
            'rms_shift_from_clean,mean_projector_distance_rms,std_projector_distance_rms,'
            'mean_subspace_fidelity,std_subspace_fidelity,mean_min_patch_overlap,minimum_patch_overlap,'
            'mean_parity_margin,minimum_parity_margin,parity_correct_rate,'
            'parity_correct_ci_low,parity_correct_ci_high,integer_correct_rate,'
            'projector_distance_error_correlation'
        ), comments=''
    )
    shutil.rmtree(chunk_dir)


def _smooth_noise_matrix(nvec: np.ndarray, coeff: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    scale = abs(coeff[0]) + np.linalg.norm(coeff[1:]) / np.sqrt(5)
    scale = max(float(scale), 1e-12)
    return (
        coeff[0] * matrices[0]
        + np.einsum('...i,ijk,i->...jk', nvec, matrices[1:], coeff[1:]) / np.sqrt(5)
    ) / scale


def smooth_control_noise_data() -> None:
    """Smooth, low-harmonic Hamiltonian perturbations with a global gap bound.

    Every realization has sup-norm at most one by construction.  Hence the
    actual Chern residue is invariant whenever the tabulated Weyl lower bound
    stays positive.  The transition estimator is evaluated independently.
    """
    matrices = s4.structured_s4_matrices()
    ngrid = 6
    realizations = 30
    deltas = np.asarray([.02, .05, .08, .10, .12, .14])
    rng = np.random.default_rng(seed("smooth_tomography_noise"))
    rows: list[list[float]] = []
    summaries: list[list[float]] = []

    for kind, target, base_bound in (('odd_mixed', 1, .3125), ('even', 2, 2.0)):
        ch, th, ph, wch, wth, wph, nvec, base = _equatorial_hamiltonian_grid(kind, matrices, ngrid)
        epsilon = 1.25 if kind == 'odd_mixed' else 0.0
        rn, rs = r.tomography_references(kind, epsilon, matrices)
        clean_w = _transition_winding_from_hamiltonian_grid(base, rn, rs, ch, th, ph, wch, wth, wph)
        for delta in deltas:
            values, overlaps = [], []
            coeffs = []
            for realization in range(realizations):
                coeff = rng.normal(size=6)
                coeffs.append(coeff)
                smooth = _smooth_noise_matrix(nvec, coeff, matrices)
                h = base + float(delta) * smooth
                _, u = np.linalg.eigh(h)
                occ = u[..., :4]
                p = occ @ np.swapaxes(occ.conj(), -1, -2)
                un, en = s4._batch_patch_frame(p, rn)
                us, es = s4._batch_patch_frame(p, rs)
                gg = np.swapaxes(un.conj(), -1, -2) @ us
                uu, _, vh = np.linalg.svd(gg)
                gg = uu @ vh
                winding = float(s4._grid_winding_batch(gg[None], ch, th, ph, wch, wth, wph)[0])
                overlap = float(min(np.min(en), np.min(es)))
                values.append(winding)
                overlaps.append(overlap)
                rows.append([
                    0 if kind == 'odd_mixed' else 1, target, float(delta), realization,
                    winding, overlap, *coeff
                ])
            vals = np.asarray(values)
            ovs = np.asarray(overlaps)
            nearest = np.rint(vals).astype(int)
            correct = (nearest % 2) == (target % 2)
            low, high = wilson_interval(int(np.sum(correct)), realizations)
            summaries.append([
                0 if kind == 'odd_mixed' else 1, target, float(delta), clean_w,
                float(np.mean(vals)), float(np.std(vals, ddof=1)),
                float(np.min(ovs)), float(np.mean(correct)), low, high,
                float(base_bound - 2 * delta)
            ])

    np.savetxt(
        DATA / 'tomography_smooth_noise_realizations.csv', np.asarray(rows), delimiter=',',
        header=(
            'kind_code,target,delta,realization,W3_reconstructed,min_patch_overlap,'
            'c0,c1,c2,c3,c4,c5'
        ), comments=''
    )
    np.savetxt(
        DATA / 'tomography_smooth_noise_summary.csv', np.asarray(summaries), delimiter=',',
        header=(
            'kind_code,target,delta,clean_grid_winding,mean_winding,std_winding,'
            'minimum_patch_overlap,parity_correct_rate,parity_correct_ci_low,'
            'parity_correct_ci_high,rigorous_global_gap_lower_bound'
        ), comments=''
    )


def _full_disorder_endpoint(seed: int, total_time: float = 100.0, length: int = 8,
                            circumference: int = 4, nsteps: int = 360,
                            disorder: float = .02) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    profile = rng.normal(size=(length, circumference))
    profile -= np.mean(profile)
    profile /= np.max(np.abs(profile))
    theta0 = .37
    dim = 2 * length * circumference
    right = np.zeros((dim, dim), complex)
    for x in range(length // 2, length):
        for y in range(circumference):
            pos = 2 * (x * circumference + y)
            right[pos:pos + 2, pos:pos + 2] = r.I2
    total = acharge = bcharge = 0.0
    for a in (1, -1):
        for b in (1, -1):
            h0 = r.full_realspace_cylinder_hamiltonian(
                theta0, a, b, length, circumference, profile, disorder
            )
            _, u0 = np.linalg.eigh(h0)
            psi = u0[:, :length * circumference]
            q0 = float(np.real(np.trace(psi.conj().T @ right @ psi)))
            for j in range(nsteps):
                theta_mid = theta0 + (j + .5) * 2 * np.pi / nsteps
                h = r.full_realspace_cylinder_hamiltonian(
                    theta_mid, a, b, length, circumference, profile, disorder
                )
                ev, u = np.linalg.eigh(h)
                coeff = u.conj().T @ psi
                coeff *= np.exp(-1j * ev * total_time / nsteps)[:, None]
                psi = u @ coeff
                if j % 40 == 39:
                    psi, _ = np.linalg.qr(psi)
            q = float(np.real(np.trace(psi.conj().T @ right @ psi))) - q0
            total += q
            acharge += a * q
            bcharge += b * q
    return float(total), float(acharge), float(bcharge)


def pump_robustness_data() -> None:
    size_rows = []
    for lx, ly in ((8, 4), (10, 4), (12, 6), (14, 6)):
        out = r.finite_time_cross_pump(
            np.asarray([100.0]), length=lx, circumference=ly, nsteps=360,
            disorder=.03, edge_potential=.06, representative_time=100.0
        )
        size_rows.append([lx, ly, out[5][0], out[6][0], out[7][0]])
    np.savetxt(
        DATA / 'finite_time_size_convergence.csv', np.asarray(size_rows), delimiter=',',
        header='Lx,Ly,total_transfer,A_transfer,B_transfer', comments=''
    )

    offset_rows = []
    for theta0 in (0.15, 0.30, 0.37, 0.45, 0.60):
        out = r.finite_time_cross_pump(
            np.asarray([100.0]), length=10, circumference=4, nsteps=360,
            disorder=.03, edge_potential=.06, representative_time=100.0,
            theta0=theta0,
        )
        offset_rows.append([theta0, out[5][0], out[6][0], out[7][0]])
    np.savetxt(
        DATA / 'finite_time_initial_flux_offset_scan.csv', np.asarray(offset_rows), delimiter=',',
        header='theta0,total_transfer,A_transfer,B_transfer', comments=''
    )

    ensemble = []
    for idx in range(12):
        seed_value = seed("pump_disorder_ensemble_base") + idx
        endpoint = _full_disorder_endpoint(seed_value)
        ensemble.append([seed_value, *endpoint])
    ensemble = np.asarray(ensemble)
    np.savetxt(
        DATA / 'finite_time_disorder_ensemble.csv', ensemble, delimiter=',',
        header='seed,total_transfer,A_transfer,B_transfer', comments=''
    )
    mean = np.mean(ensemble[:, 1:], axis=0)
    std = np.std(ensemble[:, 1:], axis=0, ddof=1)
    np.savetxt(
        DATA / 'finite_time_disorder_ensemble_summary.csv',
        np.asarray([[12, .02, *mean, *std]]), delimiter=',',
        header=(
            'realizations,disorder_strength,mean_total,mean_A,mean_B,'
            'std_total,std_A,std_B'
        ), comments=''
    )

    lengths = np.arange(4, 17)
    gaps = []
    for length in lengths:
        ev = np.linalg.eigvalsh(r.ribbon_hamiltonian(
            0.0, -.5, int(length), disorder=0.0, edge_potential=0.0, left_offset=0.0
        ))
        gaps.append(float(ev[length] - ev[length - 1]))
    gaps = np.asarray(gaps)
    fit = np.polyfit(lengths, np.log(gaps), 1)
    pred = np.polyval(fit, lengths)
    ss_res = float(np.sum((np.log(gaps) - pred) ** 2))
    ss_tot = float(np.sum((np.log(gaps) - np.mean(np.log(gaps))) ** 2))
    xi = float(-1 / fit[0])
    r2 = float(1 - ss_res / ss_tot)
    np.savetxt(
        DATA / 'edge_hybridization_scaling.csv', np.c_[lengths, gaps, np.exp(pred)], delimiter=',',
        header='Lx,edge_hybridization_gap,exponential_fit', comments=''
    )
    np.savetxt(
        DATA / 'edge_hybridization_fit.csv', np.asarray([[fit[0], fit[1], xi, r2]]), delimiter=',',
        header='log_slope,log_intercept,localization_length,R_squared', comments=''
    )


def coupling_graph_data() -> None:
    matrices = s4.structured_s4_matrices()
    names = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']
    rows = []
    for term, mat in zip(names, matrices):
        for i in range(8):
            for j in range(i + 1, 8):
                if abs(mat[i, j]) > .5:
                    rows.append([
                        int(term[1:]), i, j, abs(mat[i, j]), np.angle(mat[i, j])
                    ])
    np.savetxt(
        DATA / 'structured_s4_coupling_edges.csv', np.asarray(rows), delimiter=',',
        header='term_index,state_i,state_j,absolute_matrix_element,phase_radians', comments=''
    )

    # Full control graph at a representative point n_mu=1/sqrt(5), lambda=1.25.
    # family_code=0 denotes the flattened Yang/spectator Hamiltonian and
    # family_code=1 denotes the Pauli-string coupling terms.
    p0 = (r.I2 + r.sz) / 2
    base_matrices = [
        r.kron3(p0, r.sx, r.sx),
        r.kron3(p0, r.sx, r.sy),
        r.kron3(p0, r.sx, r.sz),
        r.kron3(p0, r.sy, r.I2),
    ]
    base_coefficients = np.full(4, 1 / np.sqrt(5))
    mixing_coefficients = np.asarray([
        1.25 * 0.35 / 2,
        *([1.25 / 10] * 5),
    ])
    full_rows = []
    for term_index, (mat, coefficient) in enumerate(zip(base_matrices, base_coefficients)):
        for i in range(8):
            for j in range(i + 1, 8):
                value = coefficient * mat[i, j]
                if abs(value) > 1e-12:
                    full_rows.append([0, term_index, i, j, abs(value), np.angle(value)])
    for term_index, (mat, coefficient) in enumerate(zip(matrices, mixing_coefficients)):
        for i in range(8):
            for j in range(i + 1, 8):
                value = coefficient * mat[i, j]
                if abs(value) > 1e-12:
                    full_rows.append([1, term_index, i, j, abs(value), np.angle(value)])
    np.savetxt(
        DATA / 'structured_s4_full_control_edges.csv', np.asarray(full_rows), delimiter=',',
        header=(
            'family_code,term_index,state_i,state_j,'
            'absolute_representative_coupling,phase_radians'
        ), comments=''
    )


def _plot_coupling_graph(ax) -> None:
    positions = {}
    for kappa in (0, 1):
        for tau in (0, 1):
            for sigma in (0, 1):
                idx = 4 * kappa + 2 * tau + sigma
                positions[idx] = (2.35 * kappa + .75 * sigma, 1.15 * (1 - tau))
    for idx, (x, y) in positions.items():
        kappa, tau, sigma = idx // 4, (idx // 2) % 2, idx % 2
        ax.scatter([x], [y], s=28, zorder=5, facecolor='black', edgecolor='black', linewidth=.6)
        dy = .10 if tau == 0 else -.12
        va = 'bottom' if tau == 0 else 'top'
        ax.text(x, y + dy, rf'$|{kappa}{tau}{sigma}\rangle$', ha='center', va=va, fontsize=6.0, zorder=6)
    edges = np.genfromtxt(DATA / 'structured_s4_full_control_edges.csv', delimiter=',', names=True)
    handles = []
    for family_code, ls, rad, label, width in (
            (0, '-', 0.00, 'flattened Yang / spectator couplings', 1.35),
            (1, '--', .12, 'Pauli-string interblock coupling', 1.0)):
        selected = np.atleast_1d(edges[edges['family_code'] == family_code])
        seen = set()
        for row in selected:
            pair = (int(row['state_i']), int(row['state_j']))
            if pair in seen:
                continue
            seen.add(pair)
            p1, p2 = positions[pair[0]], positions[pair[1]]
            patch = FancyArrowPatch(
                p1, p2, arrowstyle='-', connectionstyle=f'arc3,rad={rad}',
                linewidth=width, linestyle=ls, alpha=.82, zorder=2
            )
            ax.add_patch(patch)
        handle, = ax.plot([], [], ls=ls, lw=width, label=label)
        handles.append(handle)
    ax.text(.38, 1.48, r'$\kappa=0$ (Yang block)', ha='center', fontsize=6.7)
    ax.text(2.73, 1.48, r'$\kappa=1$ (spectator doublets)', ha='center', fontsize=6.7)
    ax.legend(handles=handles, frameon=False, fontsize=5.7, loc='lower center', ncol=1,
              bbox_to_anchor=(.5, -.02))
    ax.set_xlim(-.35, 3.45); ax.set_ylim(-.47, 1.62); ax.axis('off')


def plot_figures() -> None:
    c2 = np.genfromtxt(DATA / 'structured_s4_c2_direct.csv', delimiter=',', names=True)
    conv = np.genfromtxt(DATA / 'structured_s4_c2_convergence.csv', delimiter=',', names=True)
    gap = np.genfromtxt(DATA / 'structured_s4_gap_bound.csv', delimiter=',', names=True)
    mix = np.genfromtxt(DATA / 'structured_s4_block_mixing.csv', delimiter=',', names=True)
    tomo = np.genfromtxt(DATA / 'clutching_tomography_convergence.csv', delimiter=',', names=True)
    finite_shot = np.genfromtxt(DATA / 'tomography_finite_shot_summary.csv', delimiter=',', names=True)
    finite_real = np.genfromtxt(DATA / 'tomography_finite_shot_realizations.csv', delimiter=',', names=True)
    readout = np.genfromtxt(DATA / 'tomography_readout_confusion_summary.csv', delimiter=',', names=True)
    sizes = np.genfromtxt(DATA / 'finite_time_size_convergence.csv', delimiter=',', names=True)
    ensemble = np.genfromtxt(DATA / 'finite_time_disorder_ensemble.csv', delimiter=',', names=True)
    ensemble_summary = np.genfromtxt(DATA / 'finite_time_disorder_ensemble_summary.csv', delimiter=',', names=True)
    edge = np.genfromtxt(DATA / 'edge_hybridization_scaling.csv', delimiter=',', names=True)
    edgefit = np.genfromtxt(DATA / 'edge_hybridization_fit.csv', delimiter=',', names=True)
    offsets = np.genfromtxt(DATA / 'finite_time_initial_flux_offset_scan.csv', delimiter=',', names=True)

    # Main Fig. 4
    fig = plt.figure(figsize=(7.3, 5.2), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(c2['lambda'], c2['C2_order6'], 'o--', ms=3.1, label='quadrature order 6')
    ax.plot(c2['lambda'], c2['C2_order8'], 's-', ms=3.1, label='quadrature order 8')
    ax.axhline(1, lw=.55)
    ax.set_xlabel(r'Pauli-string coupling $\lambda$'); ax.set_ylabel(r'$C_2$')
    ax.set_ylim(.94, 1.055); ax.legend(frameon=True, facecolor='white', framealpha=1.0, edgecolor='none', fontsize=6.0, loc='upper left')
    ins = ax.inset_axes([.58, .10, .37, .33])
    ins.set_facecolor('white')
    ins.patch.set_alpha(1.0)
    ins.semilogy(conv['quadrature_order'], np.abs(conv['C2_lambda1p25'] - 1), 'o-', ms=2.8)
    ins.set_title(r'convergence at $\lambda=1.25$', fontsize=5.2, pad=1)
    ins.set_xticks([4,6,8])
    ins.tick_params(labelsize=5.0)
    ax.set_title('(a) Second Chern number', loc='left', fontsize=8.7)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(gap['lambda'], gap['min_sampled_gap'], label='sampled minimum gap')
    ax.plot(gap['lambda'], gap['Weyl_global_lower_bound'], '--', label='analytic lower bound')
    ax.set_xlabel(r'Pauli-string coupling $\lambda$'); ax.set_ylabel('occupied/unoccupied gap')
    ax2 = ax.twinx(); ax2.plot(mix['lambda'], mix['mean_block_commutator_norm'], '-.', label='block-commutator norm')
    ax2.set_ylabel(r'$\langle\|[P,Q_{\rm block}]\|_F/2\rangle$')
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], frameon=False, fontsize=6.0, loc='center left')
    ax.set_title('(b) Gap and interblock mixing', loc='left', fontsize=8.7)

    ax = fig.add_subplot(gs[1, 0])
    samples = tomo['parameter_points_per_family']
    ax.semilogx(samples, tomo['W3_odd_bundle'], 'o-', label=r'odd-$C_2$ bundle')
    ax.semilogx(samples, tomo['W3_factorized_reference'], 's--', label='factorized reference')
    ax.axhline(1, lw=.5); ax.axhline(2, lw=.5)
    ax.set_xlabel('equatorial projector reconstructions per bundle'); ax.set_ylabel(r'reconstructed $W_3[G_{NS}]$')
    ax.legend(frameon=False, fontsize=6.0)
    ax.set_title('(c) Clutching winding', loc='left', fontsize=8.7)

    ax = fig.add_subplot(gs[1, 1])
    for code, marker, linestyle, label in ((0, 'o', '-', 'odd'), (1, 's', '--', 'even')):
        sub = finite_shot[finite_shot['kind_code'] == code]
        x = sub['shots_per_parameter_point']
        y = sub['parity_correct_rate']
        yerr = np.maximum(0.0, np.vstack([
            y - sub['parity_correct_ci_low'], sub['parity_correct_ci_high'] - y
        ]))
        ax.errorbar(
            x, y, yerr=yerr, fmt=marker + linestyle, ms=3.0, lw=.9,
            capsize=1.5, label=label + ' winding parity'
        )
    ax.set_xscale('log')
    ax.set_ylim(-.04, 1.05)
    ax.set_xlabel('shots per parameter point (27 Pauli bases)')
    ax.set_ylabel('correct parity probability')
    ax.legend(frameon=False, fontsize=5.8, loc='lower right')
    ax.set_title('(d) Parity classification', loc='left', fontsize=8.7)
    fig.savefig(FIG / 'fig4_full_reduction_tomography.pdf', bbox_inches='tight')
    fig.savefig(FIG / 'fig4_full_reduction_tomography.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Pump robustness
    fig = plt.figure(figsize=(7.25, 5.0), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    labels = [rf'${int(x)}\!\times\!{int(y)}$' for x, y in zip(np.atleast_1d(sizes['Lx']), np.atleast_1d(sizes['Ly']))]
    xx = np.arange(len(labels))
    ax.plot(xx, np.atleast_1d(sizes['B_transfer']), 'o-', label=r'$\Delta Q_B^R$')
    ax.plot(xx, np.atleast_1d(sizes['total_transfer']), 's--', ms=3, label=r'$\Delta Q^R$')
    ax.plot(xx, np.atleast_1d(sizes['A_transfer']), '^-.', ms=3, label=r'$\Delta Q_A^R$')
    ax.axhline(4, lw=.45); ax.axhline(0, lw=.45)
    ax.set_xticks(xx, labels); ax.set_xlabel(r'cylinder $L_x\times L_y$'); ax.set_ylabel('one-cycle transfer')
    ax.set_ylim(-.12, 4.12); ax.legend(frameon=False, fontsize=5.8)
    ax.set_title('(a) Direct-evolution size sequence', loc='left', fontsize=8.3)

    ax = fig.add_subplot(gs[0, 1])
    xj = np.arange(len(np.atleast_1d(ensemble['B_transfer'])))
    ax.plot(xj, np.atleast_1d(ensemble['B_transfer']), 'o', ms=3, label=r'$\Delta Q_B^R$')
    ax.plot(xj, np.atleast_1d(ensemble['total_transfer']), 's', ms=2.6, label=r'$\Delta Q^R$')
    ax.plot(xj, np.atleast_1d(ensemble['A_transfer']), '^', ms=2.6, label=r'$\Delta Q_A^R$')
    ax.axhline(4, lw=.45); ax.axhline(0, lw=.45)
    ax.set_xlabel('two-dimensional disorder realization'); ax.set_ylabel('one-cycle transfer')
    ax.set_ylim(-.12, 4.12); ax.legend(frameon=False, fontsize=5.7)
    ax.text(.04, .08,
            rf'$\overline{{\Delta Q_B^R}}={float(ensemble_summary["mean_B"]):.2f}$' + '\n' +
            rf'$\sigma_B={float(ensemble_summary["std_B"]):.1e}$',
            transform=ax.transAxes, fontsize=5.9)
    ax.set_title('(b) Two-dimensional disorder ensemble', loc='left', fontsize=8.3)

    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(edge['Lx'], edge['edge_hybridization_gap'], 'o', ms=3, label='clean ribbon')
    ax.semilogy(edge['Lx'], edge['exponential_fit'], '-', label='exponential fit')
    ax.set_xlabel(r'ribbon width $L_x$'); ax.set_ylabel(r'opposite-edge gap $\Delta_{\rm edge}$')
    ax.legend(frameon=False, fontsize=5.9)
    ax.text(.05, .08,
            rf'$\xi={float(edgefit["localization_length"]):.2f}$, $R^2>0.999$',
            transform=ax.transAxes, fontsize=5.9)
    ax.set_title('(c) Exponential edge hybridization', loc='left', fontsize=8.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(offsets['theta0'], offsets['B_transfer'], 'o-', label=r'$\Delta Q_B^R$')
    ax.plot(offsets['theta0'], offsets['total_transfer'], 's--', ms=3, label=r'$\Delta Q^R$')
    ax.plot(offsets['theta0'], offsets['A_transfer'], '^-.', ms=3, label=r'$\Delta Q_A^R$')
    ax.axhline(4, lw=.45); ax.axhline(0, lw=.45)
    ax.set_xlabel(r'initial flux offset $\theta_0$'); ax.set_ylabel('one-cycle transfer')
    ax.set_ylim(-.75, 4.12); ax.legend(frameon=False, fontsize=5.8)
    ax.text(.04, .08, r'plateau for $0.30\leq\theta_0\leq0.60$', transform=ax.transAxes, fontsize=5.9)
    ax.set_title('(d) Initial-offset robustness window', loc='left', fontsize=8.3)
    fig.savefig(FIG / 'figS2_pump_robustness.pdf', bbox_inches='tight')
    fig.savefig(FIG / 'figS2_pump_robustness.png', dpi=240, bbox_inches='tight')
    plt.close(fig)


    # Shot-limited winding bias, parity margins, and readout errors.
    fig = plt.figure(figsize=(7.2, 2.75), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(.45, 2.15, 36)
    for code, ls, label in ((0, '-', 'odd'), (1, '--', 'even')):
        sub = finite_real[(finite_real['kind_code'] == code) &
                          (finite_real['shots_per_setting'] == 200)]
        ax.hist(sub['W3_reconstructed'], bins=bins, density=True, histtype='step',
                linewidth=1.2, linestyle=ls, label=label)
    for boundary in (.5, 1.5):
        ax.axvline(boundary, lw=.55, ls=':')
    ax.set_xlabel(r'reconstructed $\widehat W_3$')
    ax.set_ylabel('probability density')
    ax.legend(frameon=False, fontsize=6.0)
    ax.set_title('(a) Winding estimator', loc='left', fontsize=8.4)

    ax = fig.add_subplot(gs[0, 1])
    for code, marker, ls, label in ((0, 'o', '-', 'odd'), (1, 's', '--', 'even')):
        sub = finite_shot[finite_shot['kind_code'] == code]
        ax.semilogx(sub['shots_per_setting'], sub['mean_parity_margin'],
                    marker + ls, ms=3, label=label + ' mean')
        ax.semilogx(sub['shots_per_setting'], sub['parity_margin_10th_percentile'],
                    ls, lw=.75, alpha=.7)
    ax.set_xlabel('shots per Pauli basis')
    ax.set_ylabel('distance to half-integer')
    ax.set_ylim(-.01, .48)
    ax.legend(frameon=False, fontsize=5.5)
    ax.set_title('(b) Parity margin', loc='left', fontsize=8.4)

    ax = fig.add_subplot(gs[0, 2])
    for code, marker, family in ((0, 'o', 'odd'), (1, 's', 'even')):
        for shots, ls in ((200, '-'), (300, '--')):
            sub = readout[(readout['kind_code'] == code) &
                          (readout['shots_per_setting'] == shots)]
            x = 100 * sub['readout_flip_probability']
            y = sub['parity_correct_rate']
            yerr = np.maximum(0.0, np.vstack([
                y - sub['parity_correct_ci_low'], sub['parity_correct_ci_high'] - y
            ]))
            ax.errorbar(x, y, yerr=yerr, fmt=marker + ls, ms=3, lw=.85,
                        capsize=1.4, label=f'{family}, {shots}')
    ax.set_xlabel('readout-flip probability (%)')
    ax.set_ylabel('correct parity probability')
    ax.set_ylim(-.04, 1.05)
    ax.legend(frameon=False, fontsize=5.0, loc='lower left')
    ax.set_title('(c) Readout errors', loc='left', fontsize=8.4)

    fig.savefig(FIG / 'figS5_tomography_systematics.pdf', bbox_inches='tight')
    fig.savefig(FIG / 'figS5_tomography_systematics.png', dpi=240, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    import subprocess
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--part',
        choices=['noise', 'point_noise', 'point_noise_chunk', 'finite_shot', 'readout', 'bootstrap', 'smooth_noise', 'pump', 'coupling', 'plot', 'all'],
        default='all'
    )
    parser.add_argument('--kind-code', type=int)
    parser.add_argument('--delta-index', type=int)
    args = parser.parse_args()
    started = time.time()

    # Batched point-noise diagonalization and single-family smooth-noise
    # diagonalization use different LAPACK batch shapes.  Some BLAS builds
    # slow down dramatically when both are executed in one interpreter, so
    # aggregate modes deliberately use fresh subprocesses.
    if args.part in ('noise', 'all'):
        subprocess.check_call([sys.executable, __file__, '--part', 'point_noise'])
        subprocess.check_call([sys.executable, __file__, '--part', 'finite_shot'])
        subprocess.check_call([sys.executable, __file__, '--part', 'readout'])
        subprocess.check_call([sys.executable, __file__, '--part', 'bootstrap'])
        subprocess.check_call([sys.executable, __file__, '--part', 'smooth_noise'])
    if args.part == 'point_noise':
        pointwise_tomography_noise_data()
    if args.part == 'point_noise_chunk':
        if args.kind_code is None or args.delta_index is None:
            parser.error('--kind-code and --delta-index are required for point_noise_chunk')
        _write_point_noise_chunk(args.kind_code, args.delta_index)
    if args.part == 'finite_shot':
        finite_shot_tomography_data()
    if args.part == 'readout':
        readout_confusion_tomography_data()
    if args.part == 'bootstrap':
        bootstrap_tomography_data()
    if args.part == 'smooth_noise':
        smooth_control_noise_data()
    if args.part in ('pump', 'all'):
        pump_robustness_data()
    if args.part in ('coupling', 'all'):
        coupling_graph_data()
    if args.part in ('plot', 'all'):
        plot_figures()
    print(f'robustness calculations {args.part} complete in {time.time()-started:.2f} s', flush=True)


if __name__ == '__main__':
    main()
