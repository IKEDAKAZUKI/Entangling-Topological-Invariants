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
from scipy.special import ndtri
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FIG = ROOT / 'figures'
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location('base_reproduce', Path(__file__).with_name('reproduce_all.py'))
r = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(r)


def structured_s4_matrices() -> np.ndarray:
    """Six norm-one Pauli strings on kappa x tau x sigma."""
    return np.asarray([
        r.kron3(r.sx, r.I2, r.I2),
        r.kron3(r.sy, r.sx, r.I2),
        r.kron3(r.sy, r.sz, r.sx),
        r.kron3(r.sx, r.sy, r.sz),
        r.kron3(r.sz, r.sx, r.sy),
        r.kron3(r.I2, r.sy, r.sx),
    ])


def structured_mixing_data() -> None:
    matrices = structured_s4_matrices()
    np.save(DATA / 'structured_s4_pauli_matrices.npy', matrices)
    (DATA / 'structured_s4_pauli_strings.txt').write_text(
        'M0=kappa_x\n'
        'M1=kappa_y tau_x\n'
        'M2=kappa_y tau_z sigma_x\n'
        'M3=kappa_x tau_y sigma_z\n'
        'M4=kappa_z tau_x sigma_y\n'
        'M5=tau_y sigma_x\n'
    )

    rng = np.random.default_rng(seed("structured_s4_sampling"))
    nvec = rng.normal(size=(1600, 5))
    nvec /= np.linalg.norm(nvec, axis=1)[:, None]
    nvec = np.vstack([nvec, np.eye(5), -np.eye(5)])
    h0 = np.asarray([r.odd_flat_hamiltonian(n) for n in nvec])
    v = np.asarray([r.generic_s4_perturbation(n, matrices) for n in nvec])

    # The additive Pauli-string coupling generally splits the four occupied
    # energies.  Its isolated occupied projector P_lambda defines the topology.
    # Reflattening I-2P_lambda restores exact Wilczek--Zee degeneracy without
    # changing that projector or any topological invariant.
    mixed_endpoint = h0 + 1.25 * v
    _, mixed_vectors = np.linalg.eigh(mixed_endpoint)
    mixed_occ = mixed_vectors[:, :, :4]
    mixed_projector = np.einsum('nai,nbi->nab', mixed_occ, mixed_occ.conj())
    reflattened = np.eye(8, dtype=complex)[None] - 2 * mixed_projector
    ref_evals, ref_vectors = np.linalg.eigh(reflattened)
    ref_occ = ref_vectors[:, :, :4]
    ref_projector = np.einsum('nai,nbi->nab', ref_occ, ref_occ.conj())
    reflattening_metrics = np.asarray([[
        1.25,
        float(np.max(np.ptp(ref_evals[:, :4], axis=1))),
        float(np.max(np.ptp(ref_evals[:, 4:], axis=1))),
        float(np.min(ref_evals[:, 4] - ref_evals[:, 3])),
        float(np.max(np.linalg.norm(ref_projector - mixed_projector, axis=(1, 2)))),
        float(np.max(np.linalg.norm(reflattened @ reflattened - np.eye(8), axis=(1, 2)))),
    ]])
    np.savetxt(
        DATA / 'structured_s4_reflattening_check.csv', reflattening_metrics, delimiter=',',
        header=(
            'lambda,max_occupied_energy_spread,max_unoccupied_energy_spread,'
            'minimum_reflattened_gap,max_projector_frobenius_difference,'
            'max_involution_residual'
        ), comments=''
    )

    # Full Hamiltonian decomposition in the fixed ambient kappa x tau x sigma
    # control basis.  This record distinguishes the hardware tensor encoding
    # from the emergent two-by-two factorization tested inside the occupied
    # rank-four bundle.
    terms = [
        ('B1', 'I tau_x sigma_x', 'n1/2', 'Yang pair coupling'),
        ('B2', 'kappa_z tau_x sigma_x', 'n1/2', 'kappa-conditioned Yang pair coupling'),
        ('B3', 'I tau_x sigma_y', 'n2/2', 'Yang pair coupling'),
        ('B4', 'kappa_z tau_x sigma_y', 'n2/2', 'kappa-conditioned Yang pair coupling'),
        ('B5', 'I tau_x sigma_z', 'n3/2', 'Yang pair coupling'),
        ('B6', 'kappa_z tau_x sigma_z', 'n3/2', 'kappa-conditioned Yang pair coupling'),
        ('B7', 'I tau_y I', 'n4/2', 'Yang pair coupling'),
        ('B8', 'kappa_z tau_y I', 'n4/2', 'kappa-conditioned Yang pair coupling'),
        ('D1', 'I tau_z I', '(n5-1)/2', 'diagonal detuning'),
        ('D2', 'kappa_z tau_z I', '(n5+1)/2', 'conditional diagonal detuning'),
        ('M0', 'kappa_x', '0.175 lambda', 'structured interblock pair coupling'),
        ('M1', 'kappa_y tau_x', 'lambda n1/(2 sqrt(5))', 'structured interblock pair coupling'),
        ('M2', 'kappa_y tau_z sigma_x', 'lambda n2/(2 sqrt(5))', 'structured interblock pair coupling'),
        ('M3', 'kappa_x tau_y sigma_z', 'lambda n3/(2 sqrt(5))', 'structured interblock pair coupling'),
        ('M4', 'kappa_z tau_x sigma_y', 'lambda n4/(2 sqrt(5))', 'structured interblock pair coupling'),
        ('M5', 'tau_y sigma_x', 'lambda n5/(2 sqrt(5))', 'structured pair coupling'),
    ]
    with open(DATA / 'structured_s4_full_control_terms.csv', 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['term', 'Pauli_string', 'coefficient', 'control_role'])
        writer.writerows(terms)

    eps = np.linspace(0, 1.25, 26)
    sampled_gap = []
    for value in eps:
        ev = np.linalg.eigvalsh(h0 + value * v)
        sampled_gap.append(float(np.min(ev[:, 4] - ev[:, 3])))
    sampled_gap = np.asarray(sampled_gap)
    bound = 2 - 2 * eps * r.S4_NORM_BOUND
    np.savetxt(
        DATA / 'structured_s4_gap_bound.csv', np.c_[eps, sampled_gap, bound], delimiter=',',
        header='lambda,min_sampled_gap,Weyl_global_lower_bound', comments=''
    )

    ptriv = r.block_diag(np.eye(4), np.zeros((4, 4)))
    emix = np.linspace(0, 1.25, 21)
    mixing = []
    for value in emix:
        _, u = np.linalg.eigh((h0 + value * v)[:320])
        occ = u[:, :, :4]
        p = np.einsum('nai,nbi->nab', occ, occ.conj())
        comm = p @ ptriv - ptriv @ p
        mixing.append(float(np.mean(np.linalg.norm(comm, axis=(1, 2)) / 2)))
    np.savetxt(
        DATA / 'structured_s4_block_mixing.csv', np.c_[emix, np.asarray(mixing)], delimiter=',',
        header='lambda,mean_block_commutator_norm', comments=''
    )

    eps_c2 = np.asarray([0.0, 0.4, 0.8, 1.1, 1.25])
    rows = []
    for value in eps_c2:
        c6, _ = r.direct_c2_perturbed(float(value), 6, matrices)
        c8, gap = r.direct_c2_perturbed(float(value), 8, matrices)
        rows.append([value, c6, c8, gap])
    np.savetxt(
        DATA / 'structured_s4_c2_direct.csv', np.asarray(rows), delimiter=',',
        header='lambda,C2_order6,C2_order8,min_quadrature_gap', comments=''
    )

    conv = []
    for order in (4, 5, 6, 7, 8):
        conv.append([order, *r.direct_c2_perturbed(1.25, order, matrices)])
    np.savetxt(
        DATA / 'structured_s4_c2_convergence.csv', np.asarray(conv), delimiter=',',
        header='quadrature_order,C2_lambda1p25,min_quadrature_gap', comments=''
    )



def tomography_grid_data() -> None:
    """Clean convergence from the same single-grid estimator used for noise tests."""
    matrices = structured_s4_matrices()
    rows = []
    grids = {}
    for order in (4, 5, 6, 8, 10, 12):
        odd = r.tomography_grid(order, 'odd_mixed', 1.25, matrices)
        even = r.tomography_grid(order, 'even', 0.0, matrices)
        ch, th, ph, wch, wth, wph, godd, min_no, min_so = odd
        _, _, _, _, _, _, geven, min_ne, min_se = even
        wodd = float(_grid_winding_batch(godd[None], ch, th, ph, wch, wth, wph)[0])
        weven = float(_grid_winding_batch(geven[None], ch, th, ph, wch, wth, wph)[0])
        samples = 2 * int(order) ** 3
        rows.append([
            order, samples, samples, samples,
            wodd, min_no, min_so, weven, min_ne, min_se,
        ])
        if order == 8:
            grids['axes'] = (ch, th, ph, wch, wth, wph)
            grids['odd'] = godd
            grids['even'] = geven
    np.savetxt(
        DATA / 'clutching_tomography_convergence.csv', np.asarray(rows), delimiter=',',
        header=(
            'grid_N,parameter_points_per_family,projector_reconstructions_per_family,'
            'transition_matrices_per_family,W3_odd_bundle,min_overlap_N_odd,'
            'min_overlap_S_odd,W3_factorized_reference,min_overlap_N_even,min_overlap_S_even'
        ), comments=''
    )

    ch, th, ph, wch, wth, wph = grids['axes']
    godd = grids['odd']
    geven = grids['even']
    phase_odd = np.angle(np.linalg.det(godd))
    phase_even = np.angle(np.linalg.det(geven))
    np.savez_compressed(
        DATA / 'clutching_tomography_grid_N8.npz',
        chi=ch, theta=th, phi=ph,
        weight_chi=wch, weight_theta=wth, weight_phi=np.asarray([wph]),
        G_odd_bundle=godd, G_factorized_reference=geven,
        det_phase_odd=phase_odd, det_phase_even=phase_even,
        estimator=np.asarray(['Gauss-Legendre polar differentiation and Fourier azimuthal differentiation'])
    )
    np.savetxt(
        DATA / 'transition_function_u4_diagnostics.csv',
        np.asarray([
            [0, phase_odd.min(), phase_odd.max(), np.max(np.abs(phase_odd))],
            [1, phase_even.min(), phase_even.max(), np.max(np.abs(phase_even))],
        ]), delimiter=',',
        header='kind_code,min_arg_detG,max_arg_detG,max_abs_arg_detG', comments=''
    )

    accounting = []
    for order in (4, 5, 6, 8, 10, 12):
        points = 2 * int(order) ** 3
        accounting.append([order, points, points, points, 2, 27, 8, 63, 32])
    np.savetxt(
        DATA / 'tomography_measurement_accounting.csv', np.asarray(accounting), delimiter=',',
        header=(
            'grid_N,equatorial_parameter_points_per_family,projector_reconstructions_per_family,'
            'transition_matrices_per_family,one_time_pole_calibrations,'
            'local_Pauli_basis_settings_per_point,outcomes_per_setting,'
            'nontrivial_Pauli_coefficients,intrinsic_real_parameters_rank4_projector_in_dim8'
        ), comments=''
    )


def _odd_hamiltonian_batch(nvec: np.ndarray, epsilon: float, matrices: np.ndarray) -> np.ndarray:
    zero2 = np.zeros((2, 2), complex)
    derivatives = np.asarray([r.block_diag(r.GAMMA[i], zero2, zero2) for i in range(5)])
    constant = r.block_diag(np.zeros((4, 4), complex), -r.I2, r.I2)
    h = constant[None] + np.einsum('ni,ijk->njk', nvec, derivatives)
    v = (
        0.35 * matrices[0][None]
        + np.einsum('ni,ijk->njk', nvec, matrices[1:]) / np.sqrt(5)
    ) / r.S4_PERTURBATION_SCALE
    return h + epsilon * v


def hemisphere_patch_scan() -> None:
    matrices = structured_s4_matrices()
    sampler = qmc.Sobol(d=5, scramble=True, seed=seed("structured_s4_hemisphere_sobol"))
    u = np.clip(sampler.random_base2(16), 1e-12, 1 - 1e-12)
    x = ndtri(u)
    x /= np.linalg.norm(x, axis=1)[:, None]
    north = x.copy(); north[:, 4] = np.abs(north[:, 4])
    south = x.copy(); south[:, 4] = -np.abs(south[:, 4])
    equator = x[::2].copy(); equator[:, 4] = 0.0
    equator /= np.linalg.norm(equator, axis=1)[:, None]
    north = np.vstack([north, equator])
    south = np.vstack([south, equator])
    ref_n, ref_s = r.tomography_references('odd_mixed', 1.25, matrices)

    def scan(vectors: np.ndarray, reference: np.ndarray):
        best = 1.0
        location = vectors[0].copy()
        for start in range(0, len(vectors), 4096):
            nv = vectors[start:start + 4096]
            h = _odd_hamiltonian_batch(nv, 1.25, matrices)
            _, eigvec = np.linalg.eigh(h)
            occ = eigvec[:, :, :4]
            p = np.einsum('nai,nbi->nab', occ, occ.conj())
            pr = np.einsum('nij,jk->nik', p, reference)
            overlap = np.einsum('ji,njk->nik', reference.conj(), pr)
            overlap = (overlap + np.swapaxes(overlap.conj(), -1, -2)) / 2
            minev = np.linalg.eigvalsh(overlap)[:, 0]
            idx = int(np.argmin(minev))
            if float(minev[idx]) < best:
                best = float(minev[idx])
                location = nv[idx].copy()
        return best, location

    min_n, loc_n = scan(north, ref_n)
    min_s, loc_s = scan(south, ref_s)
    np.savetxt(
        DATA / 'structured_s4_hemisphere_patch_scan.csv',
        np.asarray([[len(north) + len(south), min_n, min_s, *loc_n, *loc_s]]), delimiter=',',
        header=(
            'sampled_points_total,min_north_overlap,min_south_overlap,'
            'north_n1,north_n2,north_n3,north_n4,north_n5,'
            'south_n1,south_n2,south_n3,south_n4,south_n5'
        ), comments=''
    )


def _batch_patch_frame(p: np.ndarray, reference: np.ndarray):
    pr = np.einsum('...ij,jk->...ik', p, reference)
    overlap = np.einsum('ji,...jk->...ik', reference.conj(), pr)
    overlap = (overlap + np.swapaxes(overlap.conj(), -1, -2)) / 2
    ev, u = np.linalg.eigh(overlap)
    invsqrt = np.einsum('...ij,...j,...kj->...ik', u, 1 / np.sqrt(ev), u.conj())
    return pr @ invsqrt, ev[..., 0]


def _barycentric_derivative_matrix(nodes: np.ndarray) -> np.ndarray:
    """Polynomial spectral derivative on arbitrary distinct one-dimensional nodes."""
    x = np.asarray(nodes, dtype=float)
    diff = x[:, None] - x[None, :]
    safe = diff.copy()
    np.fill_diagonal(safe, 1.0)
    weights = 1.0 / np.prod(safe, axis=1)
    derivative = (weights[None, :] / weights[:, None]) / safe
    np.fill_diagonal(derivative, 0.0)
    np.fill_diagonal(derivative, -np.sum(derivative, axis=1))
    return derivative


def _grid_winding_batch(
    g: np.ndarray,
    ch: np.ndarray,
    th: np.ndarray,
    ph: np.ndarray,
    wch: np.ndarray,
    wth: np.ndarray,
    wph: float,
) -> np.ndarray:
    """Winding from one discrete transition-function data set.

    The polar derivatives use global barycentric differentiation on the
    Gauss--Legendre nodes.  The periodic azimuthal derivative uses a Fourier
    spectral derivative.  No off-grid transition evaluation is made.
    """
    if g.ndim != 6:
        raise ValueError('g must have shape (batch,N,N,2N,4,4)')
    dc = _barycentric_derivative_matrix(ch)
    dt = _barycentric_derivative_matrix(th)
    gc = np.einsum('ia,bajkmn->bijkmn', dc, g, optimize=True)
    gt = np.einsum('ja,biakmn->bijkmn', dt, g, optimize=True)

    nphi = len(ph)
    period = 2.0 * np.pi
    kphi = 2.0 * np.pi * np.fft.fftfreq(nphi, d=period / nphi)
    fourier = np.fft.fft(g, axis=3)
    gp = np.fft.ifft(
        1j * kphi[None, None, None, :, None, None] * fourier,
        axis=3,
    )

    gd = np.swapaxes(g.conj(), -1, -2)
    aa, bb, cc = gd @ gc, gd @ gt, gd @ gp
    tr = np.trace(aa @ (bb @ cc - cc @ bb), axis1=-2, axis2=-1)
    quadrature = wch[None, :, None, None] * wth[None, None, :, None] * float(wph)
    return -np.real(np.sum(tr * quadrature, axis=(1, 2, 3)) / (8 * np.pi**2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['base', 'grid', 'hemisphere', 'all'], default='all')
    args = parser.parse_args()
    started = time.time()
    if args.part in ('base', 'all'):
        structured_mixing_data()
    if args.part in ('grid', 'all'):
        tomography_grid_data()
    if args.part in ('hemisphere', 'all'):
        hemisphere_patch_scan()
    print(f'structured S4 calculations {args.part} complete in {time.time()-started:.2f} s', flush=True)


if __name__ == '__main__':
    main()
