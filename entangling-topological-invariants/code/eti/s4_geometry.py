from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import roots_legendre

from .common import I2, block_diag, sx, sy, sz

GAMMA = np.array([
    np.kron(sx, sx), np.kron(sx, sy), np.kron(sx, sz),
    np.kron(sy, I2), np.kron(sz, I2),
])

COUPLING_SCALE = 2.0
COUPLING_NORM_BOUND = (7 / 20 + 1.0) / COUPLING_SCALE

@lru_cache(maxsize=None)
def _leggauss_cached(order: int):
    return roots_legendre(order)

def yang_projector(n: np.ndarray) -> np.ndarray:
    return .5 * (np.eye(4) - np.einsum('i,ijk->jk', n, GAMMA))

def s4_coordinates(a: float, b: float, c: float, d: float) -> tuple[np.ndarray, list[np.ndarray]]:
    sa, ca = np.sin(a), np.cos(a)
    sb, cb = np.sin(b), np.cos(b)
    sc, cc = np.sin(c), np.cos(c)
    sd, cd = np.sin(d), np.cos(d)
    n = np.array([sa*sb*sc*cd, sa*sb*sc*sd, sa*sb*cc, sa*cb, ca])
    da = np.array([ca*sb*sc*cd, ca*sb*sc*sd, ca*sb*cc, ca*cb, -sa])
    db = np.array([sa*cb*sc*cd, sa*cb*sc*sd, sa*cb*cc, -sa*sb, 0])
    dc = np.array([sa*sb*cc*cd, sa*sb*cc*sd, -sa*sb*sc, 0, 0])
    dd = np.array([-sa*sb*sc*sd, sa*sb*sc*cd, 0, 0, 0])
    return n, [da, db, dc, dd]

def odd_bundle_hamiltonian(n: np.ndarray) -> np.ndarray:
    return block_diag(np.tensordot(n, GAMMA, axes=(0,0)), -I2, I2)

def structured_coupling(n: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    return ((7 / 20) * matrices[0] + np.tensordot(n, matrices[1:], axes=(0,0)) / np.sqrt(5)) / COUPLING_SCALE

def second_chern_quadrature(epsilon: float, order: int, matrices: np.ndarray) -> tuple[float, float]:
    nodes, weights = _leggauss_cached(int(order))
    angles = .5 * (nodes + 1) * np.pi
    weights = .5 * np.pi * weights
    azimuths = (np.arange(2*order) + .5) * np.pi / order
    waz = np.pi / order
    zero2 = np.zeros((2,2), complex)
    derivatives_n = np.asarray([block_diag(GAMMA[i], zero2, zero2) for i in range(5)])
    derivatives_n += epsilon * matrices[1:] / (np.sqrt(5) * COUPLING_SCALE)
    total = 0j
    minimum_gap = np.inf
    for ia, aa in enumerate(angles):
        for ib, bb in enumerate(angles):
            for ic, cc in enumerate(angles):
                weight = weights[ia] * weights[ib] * weights[ic] * waz
                for dd in azimuths:
                    n, dn = s4_coordinates(aa, bb, cc, dd)
                    h = odd_bundle_hamiltonian(n) + epsilon * structured_coupling(n, matrices)
                    ev, u = np.linalg.eigh(h)
                    minimum_gap = min(minimum_gap, float(ev[4]-ev[3]))
                    dp = []
                    for tangent in dn:
                        dh = np.tensordot(tangent, derivatives_n, axes=(0,0))
                        dh_eigen = u.conj().T @ dh @ u
                        x = np.zeros((8,8), complex)
                        denominator = ev[:4][None,:] - ev[4:][:,None]
                        off = dh_eigen[4:,:4] / denominator
                        x[4:,:4] = off
                        x[:4,4:] = off.conj().T
                        dp.append(x)
                    curvature = {}
                    for mu in range(4):
                        for nu in range(mu+1,4):
                            curvature[mu,nu] = (dp[mu] @ dp[nu] - dp[nu] @ dp[mu])[:4,:4]
                    density = np.trace(
                        curvature[0,1] @ curvature[2,3]
                        - curvature[0,2] @ curvature[1,3]
                        + curvature[0,3] @ curvature[1,2]
                    )
                    total += weight * density
    c2 = float(np.real(-total / (4*np.pi**2)))
    return c2, float(minimum_gap)

def even_flat_hamiltonian(n: np.ndarray) -> np.ndarray:
    p = np.kron(yang_projector(n), I2)
    return np.eye(8, dtype=complex) - 2 * p

def occupied_projector_and_frame(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, u = np.linalg.eigh(h)
    frame = u[:, :4]
    return frame @ frame.conj().T, frame

def polar_patch_frame(p: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float]:
    overlap = reference.conj().T @ p @ reference
    overlap = (overlap + overlap.conj().T) / 2
    ev, u = np.linalg.eigh(overlap)
    if float(np.min(ev)) <= 1e-11:
        raise RuntimeError('Patch reference lost rank on the equator')
    invsqrt = u @ np.diag(1 / np.sqrt(ev)) @ u.conj().T
    return p @ reference @ invsqrt, float(np.min(ev))

def s3_equator_vector(ch: float, th: float, ph: float) -> np.ndarray:
    return np.array([
        np.sin(ch) * np.sin(th) * np.cos(ph),
        np.sin(ch) * np.sin(th) * np.sin(ph),
        np.sin(ch) * np.cos(th),
        np.cos(ch),
        0.0,
    ])

def tomography_references(kind: str, epsilon: float, matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    north = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    south = -north
    if kind == 'odd_mixed':
        hn = odd_bundle_hamiltonian(north) + epsilon * structured_coupling(north, matrices)
        hs = odd_bundle_hamiltonian(south) + epsilon * structured_coupling(south, matrices)
    elif kind == 'even':
        hn = even_flat_hamiltonian(north)
        hs = even_flat_hamiltonian(south)
    else:
        raise ValueError(kind)
    return occupied_projector_and_frame(hn)[1], occupied_projector_and_frame(hs)[1]

def reconstructed_transition(
    ch: float,
    th: float,
    ph: float,
    kind: str,
    epsilon: float,
    matrices: np.ndarray,
    ref_n: np.ndarray,
    ref_s: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    n = s3_equator_vector(ch, th, ph)
    if kind == 'odd_mixed':
        h = odd_bundle_hamiltonian(n) + epsilon * structured_coupling(n, matrices)
    elif kind == 'even':
        h = even_flat_hamiltonian(n)
    else:
        raise ValueError(kind)
    p, _ = occupied_projector_and_frame(h)
    un, min_n = polar_patch_frame(p, ref_n)
    us, min_s = polar_patch_frame(p, ref_s)
    g = un.conj().T @ us
    uu, _, vh = np.linalg.svd(g)
    return uu @ vh, min_n, min_s

def tomography_quadrature_axes(
    ngrid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Single-dataset equatorial tomography grid.

    The two polar coordinates use mapped Gauss--Legendre nodes, while the
    azimuthal coordinate uses an equispaced periodic grid.  A family is
    reconstructed exactly once at each of the 2*N^3 parameter points.
    """
    if ngrid < 3:
        raise ValueError('ngrid must be at least 3')
    nodes, weights = roots_legendre(int(ngrid))
    polar = 0.5 * np.pi * (nodes + 1.0)
    polar_weights = 0.5 * np.pi * weights
    nphi = 2 * int(ngrid)
    phi = np.arange(nphi, dtype=float) * (2.0 * np.pi / nphi)
    phi_weight = 2.0 * np.pi / nphi
    return (
        polar.copy(), polar.copy(), phi,
        polar_weights.copy(), polar_weights.copy(), float(phi_weight)
    )

def tomography_grid(
    ngrid: int,
    kind: str,
    epsilon: float,
    matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, float, float]:
    """Reconstruct one U(4) transition matrix per parameter-space point."""
    ref_n, ref_s = tomography_references(kind, epsilon, matrices)
    chvals, thvals, phvals, wch, wth, wph = tomography_quadrature_axes(ngrid)
    grid = np.empty((ngrid, ngrid, 2 * ngrid, 4, 4), dtype=complex)
    min_n = 1.0
    min_s = 1.0
    for ic, ch in enumerate(chvals):
        for it, th in enumerate(thvals):
            for ip, ph in enumerate(phvals):
                transition, overlap_n, overlap_s = reconstructed_transition(
                    float(ch), float(th), float(ph), kind, epsilon, matrices, ref_n, ref_s
                )
                grid[ic, it, ip] = transition
                min_n = min(min_n, overlap_n)
                min_s = min(min_s, overlap_s)
    return chvals, thvals, phvals, wch, wth, wph, grid, float(min_n), float(min_s)

