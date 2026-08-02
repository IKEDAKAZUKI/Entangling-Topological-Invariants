from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar

from repro_config import seed
from .common import I2, block_diag, sx, sy, sz
from .parameters import PUMP_PHASE_OFFSET

TAU_X4 = np.kron(sx, I2)
ZA4 = np.kron(sz, I2)
ZB4 = np.kron(I2, sz)

def x_disorder_profile(length: int, seed_value: int | None = None) -> np.ndarray:
    if seed_value is None:
        seed_value = seed("pump_x_disorder_profile")
    rng = np.random.default_rng(seed_value)
    v = rng.normal(size=length)
    v -= np.mean(v)
    v /= np.max(np.abs(v))
    return v

def sector_momentum_offset(a: int, b: int) -> float:
    # A pi shift for the negative-mass sectors separates otherwise coincident
    # edge crossings.  For even circumference it is a gauge-equivalent
    # relabeling of transverse momenta and does not change any invariant.
    return 0.0 if a * b == 1 else np.pi

def ribbon_hamiltonian(
    ky: float,
    mass: float,
    length: int,
    disorder: float = 0.0,
    edge_potential: float = 0.0,
    seed_value: int | None = None,
    left_offset: float = 0.0,
) -> np.ndarray:
    onsite = np.sin(ky) * sy + (mass + np.cos(ky)) * sz
    hop = 0.5 * (sz - 1j * sx)
    profile = x_disorder_profile(length, seed_value)
    h = np.zeros((2 * length, 2 * length), dtype=complex)
    for x in range(length):
        scalar = disorder * profile[x]
        scalar += edge_potential * ((1.0 if x == 0 else 0.0) - 0.7 * (1.0 if x == length - 1 else 0.0))
        scalar += left_offset * (1.0 if x == 0 else 0.0)
        h[2*x:2*x+2, 2*x:2*x+2] = onsite + scalar * I2
        if x < length - 1:
            h[2*x:2*x+2, 2*(x+1):2*(x+1)+2] = hop
            h[2*(x+1):2*(x+1)+2, 2*x:2*x+2] = hop.conj().T
    return h

def edge_chain_operator(length: int, orbital: np.ndarray = I2, width: int = 3) -> np.ndarray:
    op = np.zeros((2 * length, 2 * length), dtype=complex)
    for x in range(length):
        if x < width or x >= length - width:
            op[2*x:2*x+2, 2*x:2*x+2] = orbital
    return op

def full_four_sector_ribbon(
    ky: float,
    length: int,
    breaking: float = 0.0,
    disorder: float = 0.03,
    edge_potential: float = 0.06,
) -> np.ndarray:
    masses = [1.0, -1.0, -1.0, 1.0]
    shifts = [sector_momentum_offset(1,1), sector_momentum_offset(1,-1),
              sector_momentum_offset(-1,1), sector_momentum_offset(-1,-1)]
    offsets = [0.08, -0.05, 0.03, -0.06]
    blocks = [
        ribbon_hamiltonian(ky + shift, mass, length, disorder, edge_potential, 19, off)
        for mass, shift, off in zip(masses, shifts, offsets)
    ]
    h = block_diag(*blocks)
    if breaking:
        h += breaking * np.kron(TAU_X4, edge_chain_operator(length, width=3))
    return h

def full_ribbon_edge_dataset(
    breaking: float,
    length: int = 14,
    nk: int = 181,
) -> tuple[np.ndarray, float]:
    kys = np.linspace(-np.pi, np.pi, nk)
    edge_op = np.kron(np.eye(4), edge_chain_operator(length, width=3))
    za_op = np.kron(ZA4, np.eye(2 * length))
    zb_op = np.kron(ZB4, np.eye(2 * length))
    data = []
    min_edge_gap = np.inf
    for ky in kys:
        ev, u = np.linalg.eigh(full_four_sector_ribbon(ky, length, breaking))
        ew = np.real(np.einsum('ai,ab,bi->i', u.conj(), edge_op, u))
        za = np.real(np.einsum('ai,ab,bi->i', u.conj(), za_op, u))
        zb = np.real(np.einsum('ai,ab,bi->i', u.conj(), zb_op, u))
        strong = ew > 0.35
        if np.any(strong):
            min_edge_gap = min(min_edge_gap, float(np.min(np.abs(ev[strong]))))
        keep = (ew > 0.08) & (np.abs(ev) < 0.95)
        for energy, weight, aa, bb in zip(ev[keep], ew[keep], za[keep], zb[keep]):
            data.append((ky, energy, weight, aa, bb))
    return np.asarray(data), min_edge_gap

def right_half_operator(length: int) -> np.ndarray:
    op = np.zeros((2 * length, 2 * length), dtype=complex)
    for x in range(length // 2, length):
        op[2*x:2*x+2, 2*x:2*x+2] = I2
    return op

def x_density(states: np.ndarray, length: int) -> np.ndarray:
    return np.asarray([
        float(np.sum(np.abs(states[2*x:2*x+2, :])**2)) for x in range(length)
    ])

def transport_sector_on_cylinder(
    a: int,
    b: int,
    length: int = 20,
    circumference: int = 8,
    ntheta: int = 141,
    disorder: float = 0.03,
    edge_potential: float = 0.06,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    theta = np.linspace(PUMP_PHASE_OFFSET, 2 * np.pi + PUMP_PHASE_OFFSET, ntheta)
    right = right_half_operator(length)
    charge_right = np.zeros(ntheta)
    rho_initial = np.zeros(length)
    rho_final = np.zeros(length)
    minimum_subspace_singular_value = 1.0
    for transverse_mode in range(circumference):
        shift = sector_momentum_offset(a, b)
        ky0 = (2 * np.pi * transverse_mode + a * theta[0]) / circumference - np.pi + shift
        h0 = ribbon_hamiltonian(
            ky0, float(a * b), length, disorder, edge_potential, 19, 0.02 * (a + 2 * b)
        )
        _, vecs = np.linalg.eigh(h0)
        initial = vecs[:, :length]
        previous = initial.copy()
        charge_right[0] += float(np.real(np.trace(previous.conj().T @ right @ previous)))
        rho_initial += x_density(initial, length)
        for it, value in enumerate(theta[1:], start=1):
            ky = (2 * np.pi * transverse_mode + a * value) / circumference - np.pi + shift
            h = ribbon_hamiltonian(
                ky, float(a * b), length, disorder, edge_potential, 19, 0.02 * (a + 2 * b)
            )
            _, instantaneous = np.linalg.eigh(h)
            overlap = np.abs(previous.conj().T @ instantaneous)**2
            rows, cols = linear_sum_assignment(-overlap)
            cols = cols[np.argsort(rows)]
            current = instantaneous[:, cols]
            overlap_selected = previous.conj().T @ current
            minimum_subspace_singular_value = min(
                minimum_subspace_singular_value,
                float(np.min(np.linalg.svd(overlap_selected, compute_uv=False)))
            )
            diagonal_overlap = np.sum(previous.conj() * current, axis=0)
            current *= np.exp(-1j * np.angle(diagonal_overlap))[None, :]
            previous = current
            charge_right[it] += float(np.real(np.trace(current.conj().T @ right @ current)))
        rho_final += x_density(previous, length)
    return theta, charge_right, rho_final - rho_initial, minimum_subspace_singular_value

def full_cylinder_cross_pump(
    length: int = 20,
    circumference: int = 8,
    ntheta: int = 141,
    disorder: float = 0.03,
    edge_potential: float = 0.06,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = None
    acharge = None
    bcharge = None
    rho_b = None
    sector_endpoints = []
    min_overlap = 1.0
    theta_ref = None
    for a in (1, -1):
        for b in (1, -1):
            theta, qr, rho, overlap = transport_sector_on_cylinder(
                a, b, length, circumference, ntheta, disorder, edge_potential
            )
            theta_ref = theta
            delta = qr - qr[0]
            total = delta if total is None else total + delta
            acharge = a * delta if acharge is None else acharge + a * delta
            bcharge = b * delta if bcharge is None else bcharge + b * delta
            rho_b = b * rho if rho_b is None else rho_b + b * rho
            min_overlap = min(min_overlap, overlap)
            sector_endpoints.append([a, b, delta[-1], b * delta[-1], overlap])
    flux = (theta_ref - theta_ref[0]) / (2 * np.pi)
    return flux, total, acharge, bcharge, rho_b, np.asarray(sector_endpoints)

def finite_time_cross_pump(
    ramp_times: np.ndarray,
    length: int = 10,
    circumference: int = 4,
    nsteps: int = 720,
    disorder: float = 0.03,
    edge_potential: float = 0.06,
    snapshot_time: float = 100.0,
    theta0: float = PUMP_PHASE_OFFSET,
) -> tuple[np.ndarray, ...]:
    """Direct single-particle Schrödinger evolution of the half-filled cylinder.

    The calculation keeps every one-particle state.  The x-dependent scalar
    disorder preserves the two label charges and transverse translation, so
    the full cylinder is exactly block diagonalized into ky sectors.  This is
    a basis transformation, not a projected-edge approximation.
    """
    times = np.asarray(ramp_times, dtype=float)
    theta0 = float(theta0)
    flux = np.linspace(0.0, 1.0, nsteps + 1)
    snapshot_index = int(np.argmin(np.abs(times - snapshot_time)))
    snapshot_time = float(times[snapshot_index])
    right = right_half_operator(length)

    endpoint_total = np.zeros(len(times))
    endpoint_a = np.zeros(len(times))
    endpoint_b = np.zeros(len(times))
    trajectory_total = np.zeros(nsteps + 1)
    trajectory_a = np.zeros(nsteps + 1)
    trajectory_b = np.zeros(nsteps + 1)
    density_b = np.zeros(length)

    for a in (1, -1):
        for b in (1, -1):
            shift = sector_momentum_offset(a, b)
            for transverse_mode in range(circumference):
                ky0 = (2 * np.pi * transverse_mode + a * theta0) / circumference - np.pi + shift
                h0 = ribbon_hamiltonian(
                    ky0, float(a * b), length, disorder, edge_potential, 19,
                    0.02 * (a + 2 * b)
                )
                _, vec0 = np.linalg.eigh(h0)
                initial = vec0[:, :length]
                q0 = float(np.real(np.trace(initial.conj().T @ right @ initial)))
                rho0 = x_density(initial, length)

                evals = np.empty((nsteps, 2 * length), dtype=float)
                evecs = np.empty((nsteps, 2 * length, 2 * length), dtype=complex)
                for j in range(nsteps):
                    theta_mid = theta0 + (j + 0.5) * 2 * np.pi / nsteps
                    ky = (2 * np.pi * transverse_mode + a * theta_mid) / circumference - np.pi + shift
                    h = ribbon_hamiltonian(
                        ky, float(a * b), length, disorder, edge_potential, 19,
                        0.02 * (a + 2 * b)
                    )
                    evals[j], evecs[j] = np.linalg.eigh(h)

                states = [initial.copy() for _ in times]
                trajectory_total[0] += 0.0
                trajectory_a[0] += 0.0
                trajectory_b[0] += 0.0
                for j in range(nsteps):
                    u = evecs[j]
                    ud = u.conj().T
                    for it, total_time in enumerate(times):
                        psi = states[it]
                        coeff = ud @ psi
                        coeff *= np.exp(-1j * evals[j] * total_time / nsteps)[:, None]
                        psi = u @ coeff
                        if j % 80 == 79:
                            psi, _ = np.linalg.qr(psi)
                        states[it] = psi
                    rep = states[snapshot_index]
                    qrep = float(np.real(np.trace(rep.conj().T @ right @ rep))) - q0
                    trajectory_total[j + 1] += qrep
                    trajectory_a[j + 1] += a * qrep
                    trajectory_b[j + 1] += b * qrep

                for it, psi in enumerate(states):
                    q = float(np.real(np.trace(psi.conj().T @ right @ psi))) - q0
                    endpoint_total[it] += q
                    endpoint_a[it] += a * q
                    endpoint_b[it] += b * q
                density_b += b * (x_density(states[snapshot_index], length) - rho0)

    return (
        flux, trajectory_total, trajectory_a, trajectory_b,
        times, endpoint_total, endpoint_a, endpoint_b,
        density_b, np.asarray([snapshot_time])
    )

def finite_time_gap_scales(
    length: int = 10,
    circumference: int = 4,
    disorder: float = 0.03,
    edge_potential: float = 0.06,
    theta0: float = PUMP_PHASE_OFFSET,
) -> tuple[float, float]:
    """Minimum opposite-edge anticrossing and bulk-like direct gap."""
    theta0 = float(theta0)
    edge_gap = np.inf
    bulk_half_gap = np.inf
    edge_op = edge_chain_operator(length, width=3)
    coarse = np.linspace(theta0, theta0 + 2 * np.pi, 181)
    for a in (1, -1):
        for b in (1, -1):
            shift = sector_momentum_offset(a, b)
            for transverse_mode in range(circumference):
                def spectrum(theta: float) -> tuple[np.ndarray, np.ndarray]:
                    ky = (2 * np.pi * transverse_mode + a * theta) / circumference - np.pi + shift
                    h = ribbon_hamiltonian(
                        ky, float(a * b), length, disorder, edge_potential, 19,
                        0.02 * (a + 2 * b)
                    )
                    return np.linalg.eigh(h)

                coarse_gaps = []
                for theta in coarse:
                    ev, u = spectrum(float(theta))
                    coarse_gaps.append(float(ev[length] - ev[length - 1]))
                    ew = np.real(np.einsum('ai,ab,bi->i', u.conj(), edge_op, u))
                    bulk = np.abs(ev[ew < 0.35])
                    if len(bulk):
                        bulk_half_gap = min(bulk_half_gap, float(np.min(bulk)))
                coarse_gaps = np.asarray(coarse_gaps)
                for idx in np.argsort(coarse_gaps)[:4]:
                    lo = coarse[max(0, idx - 1)]
                    hi = coarse[min(len(coarse) - 1, idx + 1)]
                    if hi <= lo:
                        continue
                    result = minimize_scalar(
                        lambda th: spectrum(float(th))[0][length] - spectrum(float(th))[0][length - 1],
                        bounds=(float(lo), float(hi)), method='bounded',
                        options={'xatol': 1e-12, 'maxiter': 160}
                    )
                    edge_gap = min(edge_gap, float(result.fun))
    return float(edge_gap), float(2 * bulk_half_gap)

def full_realspace_cylinder_hamiltonian(
    theta: float,
    a: int,
    b: int,
    length: int,
    circumference: int,
    disorder_profile: np.ndarray,
    disorder: float = 0.02,
    edge_potential: float = 0.04,
) -> np.ndarray:
    """Full x-y real-space cylinder with label-preserving two-dimensional disorder."""
    dim = 2 * length * circumference
    h = np.zeros((dim, dim), dtype=complex)
    tx = 0.5 * (sz - 1j * sx)
    ty = 0.5 * (sz - 1j * sy)
    shift = sector_momentum_offset(a, b)

    def site(x: int, y: int) -> slice:
        pos = 2 * (x * circumference + y)
        return slice(pos, pos + 2)

    for x in range(length):
        for y in range(circumference):
            scalar = disorder * disorder_profile[x, y]
            scalar += edge_potential * (1.0 if x == 0 else 0.0)
            scalar -= 0.75 * edge_potential * (1.0 if x == length - 1 else 0.0)
            h[site(x, y), site(x, y)] = float(a * b) * sz + scalar * I2
            if x < length - 1:
                h[site(x, y), site(x + 1, y)] = tx
                h[site(x + 1, y), site(x, y)] = tx.conj().T
            yp = (y + 1) % circumference
            phase = np.exp(1j * shift)
            if y == circumference - 1:
                phase *= np.exp(1j * a * theta)
            h[site(x, y), site(x, yp)] += phase * ty
            h[site(x, yp), site(x, y)] += np.conj(phase) * ty.conj().T
    return h

def finite_time_cross_pump_2d_disorder(
    total_time: float = 100.0,
    length: int = 8,
    circumference: int = 4,
    nsteps: int = 720,
    disorder: float = 0.02,
) -> tuple[float, float, float]:
    """One full-real-space check with generic v(x,y) disorder."""
    rng = np.random.default_rng(seed("pump_full_2d_disorder_single"))
    profile = rng.normal(size=(length, circumference))
    profile -= np.mean(profile)
    profile /= np.max(np.abs(profile))
    theta0 = PUMP_PHASE_OFFSET
    dim = 2 * length * circumference
    right = np.zeros((dim, dim), dtype=complex)
    for x in range(length // 2, length):
        for y in range(circumference):
            pos = 2 * (x * circumference + y)
            right[pos:pos + 2, pos:pos + 2] = I2
    total = acharge = bcharge = 0.0
    for a in (1, -1):
        for b in (1, -1):
            h0 = full_realspace_cylinder_hamiltonian(
                theta0, a, b, length, circumference, profile, disorder
            )
            _, u0 = np.linalg.eigh(h0)
            psi = u0[:, :length * circumference]
            q0 = float(np.real(np.trace(psi.conj().T @ right @ psi)))
            for j in range(nsteps):
                theta_mid = theta0 + (j + 0.5) * 2 * np.pi / nsteps
                h = full_realspace_cylinder_hamiltonian(
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
