from __future__ import annotations

import csv

import numpy as np

from repro_config import seed
from .common import DATA, I2, sx, sy, sz
from . import s4_geometry as geom
from . import s4_pipeline as pipeline
from .parameters import S4_COUPLING_ENDPOINT

def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, center - half)), float(min(1.0, center + half))

def _equatorial_hamiltonian_grid(kind: str, matrices: np.ndarray, ngrid: int) -> tuple[np.ndarray, ...]:
    ch, th, ph, wch, wth, wph = geom.tomography_quadrature_axes(ngrid)
    shape = (ngrid, ngrid, 2 * ngrid)
    nvec = np.empty(shape + (5,), float)
    h = np.empty(shape + (8, 8), complex)
    epsilon = S4_COUPLING_ENDPOINT if kind == 'odd_mixed' else 0.0
    for i, cv in enumerate(ch):
        for j, tv in enumerate(th):
            for k, pv in enumerate(ph):
                n = geom.s3_equator_vector(float(cv), float(tv), float(pv))
                nvec[i, j, k] = n
                if kind == 'odd_mixed':
                    h[i, j, k] = geom.odd_bundle_hamiltonian(n) + epsilon * geom.structured_coupling(n, matrices)
                else:
                    h[i, j, k] = geom.even_flat_hamiltonian(n)
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
    un, _ = pipeline.batch_patch_frame(p, rn)
    us, _ = pipeline.batch_patch_frame(p, rs)
    gg = np.swapaxes(un.conj(), -1, -2) @ us
    uu, _, vh = np.linalg.svd(gg)
    gg = uu @ vh
    return float(pipeline.grid_winding_batch(gg[None], ch, th, ph, wch, wth, wph)[0])

def _pauli_tomography_design():
    """Local three-qubit Pauli-basis tomography design.

    Twenty-seven settings, X/Y/Z on each of the three two-level factors,
    provide multinomial eight-outcome data.  All 63 nontrivial Pauli
    coefficients are reconstructed by averaging over compatible settings.
    """
    local = (I2, sx, sy, sz)
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
    un, en = pipeline.batch_patch_frame(reconstructed, rn)
    us, es = pipeline.batch_patch_frame(reconstructed, rs)
    transition = np.swapaxes(un.conj(), -1, -2) @ us
    uu, _, vh = np.linalg.svd(transition)
    transition = uu @ vh
    winding = pipeline.grid_winding_batch(transition, ch, th, ph, wch, wth, wph)
    minimum_overlap = np.minimum(
        np.min(en, axis=(1, 2, 3)), np.min(es, axis=(1, 2, 3))
    )
    return winding, minimum_overlap, distance, fidelity

def finite_shot_tomography_data() -> None:
    """Finite-shot occupied-subspace tomography and parity classification.

    The three binary factors of the structured eight-level model are measured
    in the 27 local Pauli bases.  Each setting has eight outcomes and receives
    an equal number of shots.  Pole reference subspaces are treated as two
    one-time pre-calibrations; the stochastic budget used below concerns
    the equatorial projector reconstructions.
    """
    matrices = pipeline.coupling_matrices()
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
        epsilon = S4_COUPLING_ENDPOINT if kind == 'odd_mixed' else 0.0
        rn, rs = geom.tomography_references(kind, epsilon, matrices)
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
                un, en = pipeline.batch_patch_frame(reconstructed, rn)
                us, es = pipeline.batch_patch_frame(reconstructed, rs)
                transition = np.swapaxes(un.conj(), -1, -2) @ us
                uu, _, vh = np.linalg.svd(transition)
                transition = uu @ vh
                winding = pipeline.grid_winding_batch(
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
    matrices = pipeline.coupling_matrices()
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
        epsilon = S4_COUPLING_ENDPOINT if kind == 'odd_mixed' else 0.0
        rn, rs = geom.tomography_references(kind, epsilon, matrices)
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

