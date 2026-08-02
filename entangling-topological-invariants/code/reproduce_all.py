from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from repro_config import seed
import json
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.special import roots_legendre

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
DATA = ROOT / 'data'
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)


def kron3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.kron(a, np.kron(b, c))


def block_diag(*mats: np.ndarray) -> np.ndarray:
    n = sum(m.shape[0] for m in mats)
    out = np.zeros((n, n), dtype=complex)
    pos = 0
    for m in mats:
        q = pos + m.shape[0]
        out[pos:q, pos:q] = m
        pos = q
    return out


def spectral_derivative(arr: np.ndarray, axis: int) -> np.ndarray:
    n = arr.shape[axis]
    q = np.fft.fftfreq(n, d=1 / n)
    shape = [1] * arr.ndim
    shape[axis] = n
    q = q.reshape(shape)
    return np.fft.ifft(1j * q * np.fft.fft(arr, axis=axis), axis=axis)


def qwz_hamiltonian(kx: float, ky: float, mass: float) -> np.ndarray:
    return np.sin(kx) * sx + np.sin(ky) * sy + (mass + np.cos(kx) + np.cos(ky)) * sz


def qwz_projector(kx: float, ky: float, mass: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(qwz_hamiltonian(kx, ky, mass))
    v = vecs[:, [0]]
    return v @ v.conj().T


@lru_cache(maxsize=None)
def projector_grid(mass: float, n: int) -> np.ndarray:
    ks = 2 * np.pi * np.arange(n) / n - np.pi
    p = np.empty((n, n, 2, 2), dtype=complex)
    for ix, kx in enumerate(ks):
        for iy, ky in enumerate(ks):
            p[ix, iy] = qwz_projector(kx, ky, mass)
    return p


def chern_density_from_projector(p: np.ndarray) -> tuple[float, np.ndarray]:
    n = p.shape[0]
    px = spectral_derivative(p, 0)
    py = spectral_derivative(p, 1)
    comm = px @ py - py @ px
    tr = np.trace(p @ comm, axis1=-2, axis2=-1)
    density = np.real(tr / (2j * np.pi))
    dk = 2 * np.pi / n
    return float(density.sum() * dk * dk), density


def qwz_chern_piecewise(m: np.ndarray | float) -> np.ndarray:
    x = np.asarray(m, dtype=float)
    out = np.zeros_like(x)
    out[(x > -2) & (x < 0)] = -1
    out[(x > 0) & (x < 2)] = 1
    close = np.isclose(x, -2) | np.isclose(x, 0) | np.isclose(x, 2)
    out = out.astype(float)
    out[close] = np.nan
    return out


# -----------------------------------------------------------------------------
# Fig. 1: resolved zero-marginal phase
# -----------------------------------------------------------------------------

def generate_fig1() -> None:
    mvals = np.linspace(-3, 3, 601)
    cp = qwz_chern_piecewise(mvals)
    cm = qwz_chern_piecewise(-mvals)
    ctot = 2 * (cp + cm)
    ca = np.zeros_like(mvals)
    cb = np.zeros_like(mvals)
    chi = 2 * (cp - cm)
    gap = 2 * np.minimum.reduce([np.abs(mvals + 2), np.abs(mvals), np.abs(mvals - 2)])
    np.savetxt(
        DATA / 'phase_scan_2x2.csv',
        np.c_[mvals, cp, cm, ctot, ca, cb, chi, gap], delimiter=',',
        header='m,C_ab_plus,C_ab_minus,C_total,C_A,C_B,chi,direct_gap', comments=''
    )

    cplus, fplus = chern_density_from_projector(projector_grid(1.0, 101))
    cminus, fminus = chern_density_from_projector(projector_grid(-1.0, 101))
    fchi = 2 * (fplus - fminus)
    np.savez_compressed(DATA / 'cross_curvature_density.npz', density=fchi, C_plus=cplus, C_minus=cminus)

    fig = plt.figure(figsize=(7.3, 3.30), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 0.040], wspace=0.08)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)

    ax1.plot(mvals, ctot, label=r'$C_{\rm tot}$')
    ax1.plot(mvals, ca, '--', label=r'$C_A=C_B$')
    ax1.plot(mvals, chi, label=r'$\chi$')
    ax1.fill_between(mvals, -4.6, 4.6, where=gap < 0.14, alpha=0.12)
    ax1.set_xlim(-3, 3); ax1.set_ylim(-4.6, 4.6)
    ax1.set_xlabel(r'mass parameter $m$'); ax1.set_ylabel('integer response')
    ax1.legend(frameon=True, facecolor='white', framealpha=1.0, edgecolor='none', fontsize=7.3, loc='upper left')
    ax1.set_title('(a) One-label Chern responses', loc='left', fontsize=9, pad=4)
    ax1.tick_params(labelsize=7.2)

    im = ax2.imshow(fchi.T, origin='lower', extent=(-1,1,-1,1), aspect='equal')
    ax2.set_xlabel(r'$k_x/\pi$'); ax2.set_ylabel(r'$k_y/\pi$')
    ax2.set_title('(b) Mixed Berry curvature', loc='left', fontsize=9, pad=4)
    ax2.tick_params(labelsize=7.2)
    cbx = fig.colorbar(im, cax=cax)
    cbx.set_label(r'$f_{AB}(\mathbf{k})$', fontsize=8)
    cbx.ax.tick_params(labelsize=7.0)
    fig.savefig(FIG / 'fig1_zero_marginal.pdf', bbox_inches='tight')
    fig.savefig(FIG / 'fig1_zero_marginal.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# -----------------------------------------------------------------------------
# Fig. 2: generic momentum-dependent mixing and Gauss--Codazzi curvature
# -----------------------------------------------------------------------------

ZA = kron3(sz, I2, I2)
ZB = kron3(I2, sz, I2)
PROJECTED_LABEL = 2 * ZA + ZB
CROSS_WEIGHTS = np.array([1, -1, -1, 1], dtype=float)
V1 = kron3(sx, I2, sz)
V2 = kron3(I2, sx, sx)
V3 = kron3(sy, sy, sy)
V4 = kron3(sx, sx, I2)

# Coefficient vector used for the representative in Eq. (9):
# (v_A, v_B, v_AB^(sigma), v_AB^(0)).
MIXING_COEFFICIENTS = np.asarray([0.55, 0.50, 0.35, 0.20], dtype=float)


def four_block_hamiltonian(kx: float, ky: float, m: float = 1.0) -> np.ndarray:
    masses = [m, -m, -m, m]
    return block_diag(*(qwz_hamiltonian(kx, ky, mm) for mm in masses))


def generic_mixing(kx: float, ky: float, coefficients: np.ndarray = MIXING_COEFFICIENTS) -> np.ndarray:
    v_a, v_b, v_ab_sigma, v_ab_zero = np.asarray(coefficients, dtype=float)
    return (
        v_a * np.sin(kx) * V1
        + v_b * np.sin(ky) * V2
        + v_ab_sigma * (np.cos(kx) - np.cos(ky)) * V3
        + v_ab_zero * np.sin(kx + ky) * V4
    )


def mixed_resolved_grid(eps: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    ks = 2 * np.pi * np.arange(n) / n - np.pi
    p = np.zeros((n, n, 8, 8), dtype=complex)
    q = np.zeros((4, n, n, 8, 8), dtype=complex)
    lines = np.zeros((4, n, n, 8), dtype=complex)
    energy_gap = np.inf
    label_gap = np.inf
    for ix, kx in enumerate(ks):
        for iy, ky in enumerate(ks):
            h = four_block_hamiltonian(kx, ky) + eps * generic_mixing(kx, ky)
            ev, u = np.linalg.eigh(h)
            energy_gap = min(energy_gap, float(ev[4] - ev[3]))
            occ = u[:, :4]
            pij = occ @ occ.conj().T
            p[ix, iy] = pij
            rv, ru = np.linalg.eigh(occ.conj().T @ PROJECTED_LABEL @ occ)
            order = np.argsort(rv)[::-1]
            rv = rv[order]
            ru = ru[:, order]
            label_gap = min(label_gap, float(np.min(np.abs(np.diff(rv)))))
            for s in range(4):
                v = occ @ ru[:, s]
                lines[s, ix, iy] = v
                q[s, ix, iy] = np.outer(v, v.conj())
    return p, q, lines, energy_gap, label_gap


def fhs_line_chern(lines: np.ndarray) -> np.ndarray:
    ns, n, _, _ = lines.shape
    out = np.zeros(ns)
    for s in range(ns):
        v = lines[s]
        phase_sum = 0.0
        for ix in range(n):
            for iy in range(n):
                ux = np.vdot(v[ix, iy], v[(ix + 1) % n, iy]); ux /= abs(ux)
                uy = np.vdot(v[ix, iy], v[ix, (iy + 1) % n]); uy /= abs(uy)
                ux_y = np.vdot(v[ix, (iy + 1) % n], v[(ix + 1) % n, (iy + 1) % n]); ux_y /= abs(ux_y)
                uy_x = np.vdot(v[(ix + 1) % n, iy], v[(ix + 1) % n, (iy + 1) % n]); uy_x /= abs(uy_x)
                phase_sum += np.angle(ux * uy_x / (ux_y * uy))
        out[s] = phase_sum / (2 * np.pi)
    return out


def determinant_curvature_densities(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    px = spectral_derivative(p, 0)
    py = spectral_derivative(p, 1)
    xcurv = p @ (px @ py - py @ px)
    n = p.shape[0]
    full = np.zeros((n, n), dtype=complex)
    raw = np.zeros((n, n), dtype=complex)
    correction = np.zeros((n, n), dtype=complex)
    for s, w in enumerate(CROSS_WEIGHTS):
        qs = q[s]
        qx = spectral_derivative(qs, 0)
        qy = spectral_derivative(qs, 1)
        full += w * np.trace(qs @ (qx @ qy - qy @ qx), axis1=-2, axis2=-1)
        raw += w * np.trace(qs @ xcurv, axis1=-2, axis2=-1)
        dqx = p @ qx @ p
        dqy = p @ qy @ p
        correction += w * np.trace(qs @ (dqx @ dqy - dqy @ dqx), axis1=-2, axis2=-1)
    fac = 1 / (2j * np.pi)
    return np.real(full * fac), np.real(raw * fac), np.real(correction * fac)


def integrate_density(d: np.ndarray) -> float:
    n = d.shape[0]
    dk = 2 * np.pi / n
    return float(np.sum(d) * dk * dk)


def gauss_codazzi_colormap_and_norm() -> tuple[LinearSegmentedColormap, TwoSlopeNorm]:
    """Return the fixed PRX display map for the Gauss--Codazzi density.

    The asymmetric limits contain the full validated data range, while the
    midpoint is explicitly pure white so that white has the unique meaning
    Xi_AB = 0.  Undefined values, should they occur, are shown in gray.
    """
    base = plt.get_cmap('coolwarm')
    cmap = LinearSegmentedColormap.from_list(
        'gc_blue_white_red',
        [(0.0, base(0.0)), (0.5, '#FFFFFF'), (1.0, base(1.0))],
        N=257,
    )
    cmap.set_bad('#BDBDBD')
    norm = TwoSlopeNorm(vmin=-0.006, vcenter=0.0, vmax=0.035)
    return cmap, norm


def generate_fig2() -> None:
    eps_scan = np.linspace(0, 1.6, 25)
    egaps, rgaps, chis = [], [], []
    sector_cs = []
    for eps in eps_scan:
        p, q, lines, eg, rg = mixed_resolved_grid(float(eps), 23)
        cs = fhs_line_chern(lines)
        egaps.append(eg); rgaps.append(rg); sector_cs.append(cs)
        chis.append(float(np.dot(CROSS_WEIGHTS, cs)))
    sector_cs = np.asarray(sector_cs)
    np.savetxt(
        DATA / 'generic_mixing_gaps_fhs.csv',
        np.c_[eps_scan, egaps, rgaps, sector_cs, chis], delimiter=',',
        header='epsilon,energy_gap,label_resolution_gap,C_pp,C_pm,C_mp,C_mm,chi', comments=''
    )

    eps_decomp = np.linspace(0, 1.6, 13)
    det_i, proj_i, corr_i, residual_i = [], [], [], []
    for eps in eps_decomp:
        p, q, _, _, _ = mixed_resolved_grid(float(eps), 31)
        det, proj, corr = determinant_curvature_densities(p, q)
        det_i.append(integrate_density(det))
        proj_i.append(integrate_density(proj))
        corr_i.append(integrate_density(corr))
        residual_i.append(np.max(np.abs(det - proj - corr)))
    np.savetxt(
        DATA / 'generic_mixing_curvature_integrals.csv',
        np.c_[eps_decomp, det_i, proj_i, corr_i, np.asarray(proj_i)+np.asarray(corr_i), residual_i], delimiter=',',
        header='epsilon,determinant_line,projected,gauss_codazzi,projected_plus_correction,max_pointwise_residual', comments=''
    )

    eps0 = 1.2
    density_n = 151
    p, q, lines, eg0, rg0 = mixed_resolved_grid(eps0, density_n)
    det, proj, corr = determinant_curvature_densities(p, q)
    residual = det - proj - corr
    np.savez_compressed(
        DATA / 'generic_mixing_curvature_density_eps1p2.npz',
        determinant=det, projected=proj, correction=corr, residual=residual,
        epsilon=np.array(eps0), density_grid=np.array(density_n),
        energy_gap=np.array(eg0), label_resolution_gap=np.array(rg0),
        determinant_integral=np.array(integrate_density(det)),
        projected_integral=np.array(integrate_density(proj)),
        gauss_codazzi_integral=np.array(integrate_density(corr)),
        max_residual=np.array(np.max(np.abs(residual))),
        l2_residual=np.array(np.sqrt(np.mean(residual**2))),
    )

    ns = np.array([15, 31, 51, 101, 151])
    conv = []
    for n in ns:
        if int(n) == density_n:
            dd, ppj, cc = det, proj, corr
        else:
            pp, qq, _, _, _ = mixed_resolved_grid(eps0, int(n))
            dd, ppj, cc = determinant_curvature_densities(pp, qq)
        conv.append([
            n,
            integrate_density(dd), integrate_density(ppj), integrate_density(cc),
            np.max(np.abs(dd-ppj-cc)), np.sqrt(np.mean((dd-ppj-cc)**2))
        ])
    conv = np.asarray(conv)
    np.savetxt(
        DATA / 'generic_mixing_curvature_convergence.csv', conv, delimiter=',',
        header='N,determinant_line,projected,gauss_codazzi,max_residual,L2_residual', comments=''
    )

    fig,axs = plt.subplots(2,3,figsize=(7.3,4.65),constrained_layout=True)
    ax=axs[0,0]
    ax.plot(eps_scan, egaps, label='energy gap')
    ax.plot(eps_scan, rgaps, '--', label='label-resolution gap')
    ax.set_xlabel(r'mixing $\epsilon$'); ax.set_ylabel('minimum gap')
    ax.legend(frameon=False, fontsize=6.4, loc='lower left')
    ax.text(.52,.88,r'$\chi=4$',transform=ax.transAxes,fontsize=7.2)
    ax.set_title('(a) Energy and label gaps', loc='left', fontsize=8.5)

    ax=axs[0,1]
    ax.plot(eps_decomp, det_i, label='determinant-line')
    ax.plot(eps_decomp, proj_i, '--', label='projected')
    ax.plot(eps_decomp, np.asarray(proj_i)+np.asarray(corr_i), '-.', label='projected + Gauss-Codazzi')
    ax.plot(eps_decomp, corr_i, ':', label='Gauss-Codazzi')
    ax.axhline(4, lw=0.5)
    ax.set_xlabel(r'$\epsilon$'); ax.set_ylabel('Chern integral')
    ax.legend(frameon=False, fontsize=5.4, ncol=1, loc='center left')
    ax.set_title('(b) Integrated curvatures', loc='left', fontsize=8.5)

    ax=axs[0,2]
    ax.plot(eps_scan, sector_cs[:,0], label=r'$C_{++}=C_{--}$')
    ax.plot(eps_scan, sector_cs[:,1], '--', label=r'$C_{+-}=C_{-+}$')
    ax.set_xlabel(r'$\epsilon$'); ax.set_ylabel('sector Chern number')
    ax.set_ylim(-1.25,1.25); ax.legend(frameon=False, fontsize=6.1, loc='center')
    ax.set_title('(c) Sector Chern numbers', loc='left', fontsize=8.5)

    extent=(-1,1,-1,1)
    ax=axs[1,0]
    im=ax.imshow(
        proj.T, origin='lower', extent=extent, aspect='equal',
        interpolation='bilinear', resample=True
    )
    ax.set_xlabel(r'$k_x/\pi$'); ax.set_ylabel(r'$k_y/\pi$')
    ax.set_title(r'(d) Projected term, $\epsilon=1.2$',loc='left',fontsize=8.5)
    cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.03); cb.ax.tick_params(labelsize=6)

    ax=axs[1,1]
    # The data are asymmetric about zero.  The fixed rounded range contains
    # the complete validated data set, and the custom map assigns pure white
    # uniquely to zero.
    gc_cmap, gc_norm = gauss_codazzi_colormap_and_norm()
    im=ax.imshow(
        corr.T, origin='lower', extent=extent, aspect='equal',
        interpolation='bilinear', resample=True, cmap=gc_cmap, norm=gc_norm
    )
    ax.set_xlabel(r'$k_x/\pi$'); ax.set_ylabel(r'$k_y/\pi$')
    ax.set_title('(e) Gauss-Codazzi term',loc='left',fontsize=8.5)
    cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.03)
    cb.set_ticks([-0.006, 0.0, 0.020, 0.035])
    cb.set_ticklabels(['-0.6', '0', '2.0', '3.5'])
    cb.ax.set_title(r'$\times10^{-2}$', fontsize=6.3, pad=3.0)
    cb.ax.tick_params(labelsize=6)

    ax=axs[1,2]
    ax.semilogy(conv[:,0], conv[:,4], 'o-', label=r'$\|\mathcal{R}\|_\infty$')
    ax.semilogy(conv[:,0], conv[:,5], 's--', label=r'$\|\mathcal{R}\|_2$')
    ax.set_xlabel('spectral grid $N$'); ax.set_ylabel('closure residual')
    ax.legend(frameon=False, fontsize=6.4, loc='upper right')
    ax.text(.04,.06,rf'$\Delta_E={eg0:.2f}$, $\Delta_R={rg0:.2f}$',transform=ax.transAxes,fontsize=6.5)
    ax.set_title('(f) Numerical residual',loc='left',fontsize=8.5)
    fig.savefig(FIG / 'fig2_generic_mixing.pdf', dpi=400, bbox_inches='tight')
    fig.savefig(FIG / 'fig2_generic_mixing.png', dpi=300, bbox_inches='tight')
    plt.close(fig)



# -----------------------------------------------------------------------------
# Projected-label regions and hierarchical label-entanglement gaps
# -----------------------------------------------------------------------------

def _occupied_label_matrices(eps: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ks = 2 * np.pi * np.arange(n) / n - np.pi
    occ_grid = np.zeros((n, n, 8, 4), dtype=complex)
    a_grid = np.zeros((n, n, 4, 4), dtype=complex)
    b_grid = np.zeros((n, n, 4, 4), dtype=complex)
    for ix, kx in enumerate(ks):
        for iy, ky in enumerate(ks):
            h = four_block_hamiltonian(kx, ky) + eps * generic_mixing(kx, ky)
            _, u = np.linalg.eigh(h)
            occ = u[:, :4]
            occ_grid[ix, iy] = occ
            a_grid[ix, iy] = occ.conj().T @ ZA @ occ
            b_grid[ix, iy] = occ.conj().T @ ZB @ occ
    return occ_grid, a_grid, b_grid


def label_gap_from_matrices(a_grid: np.ndarray, b_grid: np.ndarray, alpha: float, beta: float) -> float:
    norm = float(np.hypot(alpha, beta))
    if norm < 1e-12:
        return np.nan
    vals = np.linalg.eigvalsh((alpha * a_grid + beta * b_grid) / norm)
    vals.sort(axis=-1)
    return float(np.min(np.diff(vals, axis=-1)))


def label_gap_map_vectorized(
    a_grid: np.ndarray,
    b_grid: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    angular_points: int = 4096,
    chunk_size: int = 256,
) -> np.ndarray:
    """Evaluate the normalized projected-label gap on a fine angular grid.

    Because the observable uses (alpha,beta)/sqrt(alpha^2+beta^2), the gap is
    exactly homogeneous of degree zero.  We therefore solve the Brillouin-zone
    minimization on a dense periodic angular grid and evaluate the requested
    Cartesian map from that one-dimensional directional data.
    """
    theta = np.linspace(-np.pi, np.pi, int(angular_points), endpoint=False)
    directions = np.column_stack([np.cos(theta), np.sin(theta)])
    flat_a = np.asarray(a_grid).reshape(-1, 4, 4)
    flat_b = np.asarray(b_grid).reshape(-1, 4, 4)
    directional_gap = np.empty(len(theta), dtype=float)
    for start_idx in range(0, len(theta), int(chunk_size)):
        stop = min(start_idx + int(chunk_size), len(theta))
        d = directions[start_idx:stop]
        mats = (
            d[:, 0, None, None, None] * flat_a[None, :, :, :]
            + d[:, 1, None, None, None] * flat_b[None, :, :, :]
        )
        vals = np.linalg.eigvalsh(mats)
        directional_gap[start_idx:stop] = np.min(np.diff(vals, axis=-1), axis=(1, 2))

    aa, bb = np.meshgrid(alphas, betas, indexing='ij')
    rr = np.hypot(aa, bb)
    ang = np.arctan2(bb, aa)
    position = (ang + np.pi) * (len(theta) / (2.0 * np.pi))
    i0 = np.floor(position).astype(int) % len(theta)
    frac = position - np.floor(position)
    i1 = (i0 + 1) % len(theta)
    out = (1.0 - frac) * directional_gap[i0] + frac * directional_gap[i1]
    out[rr < 1e-12] = np.nan
    return out

def label_resolved_lines(alpha: float, beta: float, eps: float, n: int) -> tuple[np.ndarray, float]:
    occ_grid, a_grid, b_grid = _occupied_label_matrices(eps, n)
    lines = np.zeros((4, n, n, 8), dtype=complex)
    min_gap = np.inf
    for ix in range(n):
        for iy in range(n):
            vals, vecs = np.linalg.eigh(alpha * a_grid[ix, iy] + beta * b_grid[ix, iy])
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            min_gap = min(min_gap, float(np.min(np.abs(np.diff(vals)))))
            for j in range(4):
                lines[j, ix, iy] = occ_grid[ix, iy] @ vecs[:, j]
    return fhs_line_chern(lines), min_gap


def hierarchical_label_gaps(eps: float, n: int = 17) -> tuple[float, float, float, float, float, float]:
    _, a_grid, b_grid = _occupied_label_matrices(eps, n)
    gap_a = np.inf
    gap_bp = np.inf
    gap_bm = np.inf
    mid_a = np.inf
    mid_bp = np.inf
    mid_bm = np.inf
    for ix in range(n):
        for iy in range(n):
            vals_a, vecs_a = np.linalg.eigh(a_grid[ix, iy])
            order = np.argsort(vals_a)[::-1]
            vals_a = vals_a[order]
            vecs_a = vecs_a[:, order]
            gap_a = min(gap_a, float(vals_a[1] - vals_a[2]))
            mid_a = min(mid_a, float(np.min(np.abs(vals_a))))
            for sign, inds in ((+1, slice(0, 2)), (-1, slice(2, 4))):
                w = vecs_a[:, inds]
                vals_b = np.linalg.eigvalsh(w.conj().T @ b_grid[ix, iy] @ w)
                gap_b = float(vals_b[1] - vals_b[0])
                mid_b = float(np.min(np.abs(vals_b)))
                if sign > 0:
                    gap_bp = min(gap_bp, gap_b)
                    mid_bp = min(mid_bp, mid_b)
                else:
                    gap_bm = min(gap_bm, gap_b)
                    mid_bm = min(mid_bm, mid_b)
    return gap_a, gap_bp, gap_bm, 0.5 * mid_a, 0.5 * mid_bp, 0.5 * mid_bm




def _panel_heading(ax: plt.Axes, text: str) -> None:
    """Place all panel headings at the same upper-left position outside the axes."""
    ax.text(0.0, 1.045, text, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=8.5, clip_on=False)


def _render_label_resolution_figure(
        alphas: np.ndarray,
        betas: np.ndarray,
        gapmap: np.ndarray,
        path: np.ndarray,
        hierarchy: np.ndarray) -> None:
    """Render the label-resolution figure with square panels and a matched colorbar."""
    fig = plt.figure(figsize=(7.35, 2.60))
    gs = GridSpec(
        1, 3, figure=fig,
        left=0.065, right=0.955, bottom=0.225, top=0.82,
        wspace=0.72,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    for ax in (ax_a, ax_b, ax_c):
        ax.set_box_aspect(1)

    im = ax_a.imshow(
        np.ma.masked_invalid(gapmap.T),
        origin='lower',
        extent=(alphas[0], alphas[-1], betas[0], betas[-1]),
        aspect='equal',
        vmin=0,
        interpolation='bilinear',
    )
    ax_a.set_xlabel(r'$\alpha$')
    ax_a.set_ylabel(r'$\beta$')
    _panel_heading(ax_a, '(a) Projected-label gap')
    # The colorbar is an inset with exactly the same normalized height as panel (a).
    cax = ax_a.inset_axes([1.045, 0.0, 0.045, 1.0])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('minimum separation', fontsize=6.4, labelpad=2)
    cb.ax.tick_params(labelsize=6.8)

    ax_b.plot(path[:, 0], path[:, 2], 'o-', label=r'$C_{++},C_{--}$')
    ax_b.plot(path[:, 0], path[:, 3], 's--', label=r'$C_{+-},C_{-+}$')
    ax_b.plot(path[:, 0], path[:, -1] / 4, 'd-.', label=r'$\chi/4$')
    ax_b.set_xlabel(r'$\beta/\alpha$')
    ax_b.set_ylabel('integer')
    ax_b.set_ylim(-1.25, 1.25)
    ax_b.legend(frameon=False, fontsize=6.2)
    _panel_heading(ax_b, '(b) Sector Chern numbers')

    ax_c.plot(hierarchy[:, 0], hierarchy[:, 1], label=r'$\delta_A$')
    ax_c.plot(hierarchy[:, 0], hierarchy[:, 2], '--', label=r'$\delta_{B|A=+}$')
    ax_c.plot(hierarchy[:, 0], hierarchy[:, 3], ':', label=r'$\delta_{B|A=-}$')
    ax_c_right = ax_c.twinx()
    ax_c_right.plot(hierarchy[:, 0], hierarchy[:, 4], '-.', label='entanglement midgap')
    ax_c.set_xlabel(r'mixing $\epsilon$')
    ax_c.set_ylabel('compressed-label gap')
    ax_c_right.set_ylabel(r'$|\xi-1/2|_{\min}$')
    lines = ax_c.get_lines() + ax_c_right.get_lines()
    ax_c.legend(
        lines, [line.get_label() for line in lines],
        frameon=True, facecolor='white', framealpha=1.0, edgecolor='0.75',
        fontsize=5.1, loc='lower left', borderpad=0.25,
        labelspacing=0.25, handlelength=1.8,
    )
    _panel_heading(ax_c, '(c) Hierarchical gaps')

    fig.savefig(FIG / 'figS1_label_resolution.pdf', bbox_inches='tight', pad_inches=0.08)
    fig.savefig(FIG / 'figS1_label_resolution.png', dpi=320, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)

def plot_figS_label_resolution_from_data() -> None:
    gap_data = np.load(DATA / 'label_resolution_gap_map.npz')
    alphas = np.asarray(gap_data['alpha'], dtype=float)
    betas = np.asarray(gap_data['beta'], dtype=float)
    gapmap = np.asarray(gap_data['gap'], dtype=float)
    path_rec = np.genfromtxt(DATA / 'label_resolution_path.csv', delimiter=',', names=True)
    path = np.column_stack([
        path_rec['beta_over_alpha'], path_rec['min_label_resolution_gap'],
        path_rec['C_pp'], path_rec['C_pm'], path_rec['C_mp'], path_rec['C_mm'], path_rec['chi']
    ])
    hierarchy_rec = np.genfromtxt(DATA / 'hierarchical_label_gaps.csv', delimiter=',', names=True)
    hierarchy = np.column_stack([
        hierarchy_rec['epsilon'], hierarchy_rec['gap_A'], hierarchy_rec['gap_B_given_Aplus'],
        hierarchy_rec['gap_B_given_Aminus'], hierarchy_rec['ent_midgap_A'],
        hierarchy_rec['ent_midgap_B_Aplus'], hierarchy_rec['ent_midgap_B_Aminus']
    ])
    ratios = path[:, 0]

    _render_label_resolution_figure(alphas, betas, gapmap, path, hierarchy)


def generate_figS_label_resolution() -> None:
    eps0 = 1.2
    bz_n = 25
    parameter_n = 301
    _, a_grid, b_grid = _occupied_label_matrices(eps0, bz_n)
    alphas = np.linspace(-2.4, 2.4, parameter_n)
    betas = np.linspace(-2.4, 2.4, parameter_n)
    angular_n = 4096
    gapmap = label_gap_map_vectorized(a_grid, b_grid, alphas, betas, angular_points=angular_n)
    np.savez_compressed(
        DATA / 'label_resolution_gap_map.npz',
        alpha=alphas, beta=betas, gap=gapmap, epsilon=eps0, bz_grid=bz_n,
        parameter_grid=parameter_n, angular_grid=angular_n
    )

    ratios = np.linspace(0.15, 0.85, 8)
    path = []
    for ratio in ratios:
        cs, gap = label_resolved_lines(1.0, float(ratio), eps0, 17)
        path.append([ratio, gap, *cs, float(np.dot(CROSS_WEIGHTS, cs))])
    path = np.asarray(path)
    np.savetxt(
        DATA / 'label_resolution_path.csv', path, delimiter=',',
        header='beta_over_alpha,min_label_resolution_gap,C_pp,C_pm,C_mp,C_mm,chi', comments=''
    )

    epsvals = np.linspace(0, 1.8, 10)
    hierarchy = np.asarray([[eps, *hierarchical_label_gaps(float(eps), 17)] for eps in epsvals])
    np.savetxt(
        DATA / 'hierarchical_label_gaps.csv', hierarchy, delimiter=',',
        header='epsilon,gap_A,gap_B_given_Aplus,gap_B_given_Aminus,ent_midgap_A,ent_midgap_B_Aplus,ent_midgap_B_Aminus', comments=''
    )

    _render_label_resolution_figure(alphas, betas, gapmap, path, hierarchy)



# -----------------------------------------------------------------------------
# Fig. 3: full finite-ribbon spectrum and full finite-cylinder cross pump
# -----------------------------------------------------------------------------

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


TAU_X4 = np.kron(sx, I2)
ZA4 = np.kron(sz, I2)
ZB4 = np.kron(I2, sz)


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
    theta = np.linspace(0.37, 2 * np.pi + 0.37, ntheta)
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
    representative_time: float = 100.0,
    theta0: float = 0.37,
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
    representative_index = int(np.argmin(np.abs(times - representative_time)))
    representative_time = float(times[representative_index])
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
                    rep = states[representative_index]
                    qrep = float(np.real(np.trace(rep.conj().T @ right @ rep))) - q0
                    trajectory_total[j + 1] += qrep
                    trajectory_a[j + 1] += a * qrep
                    trajectory_b[j + 1] += b * qrep

                for it, psi in enumerate(states):
                    q = float(np.real(np.trace(psi.conj().T @ right @ psi))) - q0
                    endpoint_total[it] += q
                    endpoint_a[it] += a * q
                    endpoint_b[it] += b * q
                density_b += b * (x_density(states[representative_index], length) - rho0)

    return (
        flux, trajectory_total, trajectory_a, trajectory_b,
        times, endpoint_total, endpoint_a, endpoint_b,
        density_b, np.asarray([representative_time])
    )


def finite_time_gap_scales(
    length: int = 10,
    circumference: int = 4,
    disorder: float = 0.03,
    edge_potential: float = 0.06,
    theta0: float = 0.37,
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
    theta0 = 0.37
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

def _pump_endpoint(length: int, circumference: int, disorder: float, edge_potential: float) -> tuple[float, float, float]:
    _, total, acharge, bcharge, _, _ = full_cylinder_cross_pump(
        length, circumference, 101, disorder, edge_potential
    )
    return float(total[-1]), float(acharge[-1]), float(bcharge[-1])


def generate_fig3() -> None:
    d0, g0 = full_ribbon_edge_dataset(0.0)
    dg, gg = full_ribbon_edge_dataset(0.25)
    np.savetxt(
        DATA / 'full_ribbon_edge_spectrum_symmetry_preserving.csv', d0, delimiter=',',
        header='ky,energy,edge_weight,ZA_expect,ZB_expect', comments=''
    )
    np.savetxt(
        DATA / 'full_ribbon_edge_spectrum_label_breaking.csv', dg, delimiter=',',
        header='ky,energy,edge_weight,ZA_expect,ZB_expect', comments=''
    )

    # Thermodynamic spectral-flow benchmark retained for comparison.
    flux_pt, total_pt, acharge_pt, bcharge_pt, rho_pt, sectors = full_cylinder_cross_pump()
    np.savetxt(
        DATA / 'finite_cylinder_parallel_transport.csv',
        np.c_[flux_pt, total_pt, acharge_pt, bcharge_pt], delimiter=',',
        header='flux,total_charge_right,A_charge_right,B_charge_right', comments=''
    )
    np.savetxt(
        DATA / 'finite_cylinder_sector_endpoints.csv', sectors, delimiter=',',
        header='A_label,B_label,particle_transfer_right,B_weighted_transfer,min_subspace_singular_value', comments=''
    )
    np.savetxt(
        DATA / 'finite_cylinder_B_density_transfer.csv',
        np.c_[np.arange(len(rho_pt)), rho_pt], delimiter=',',
        header='x,delta_B_density_parallel_transport', comments=''
    )

    ramp_times = np.asarray([0.2, 0.35, 0.6, 1.0, 1.8, 3.0, 5.0, 10.0,
                             20.0, 50.0, 100.0, 200.0, 500.0])
    (
        flux, total, acharge, bcharge, times, endpoint_total, endpoint_a,
        endpoint_b, rho_b, rep_time_arr
    ) = finite_time_cross_pump(ramp_times)
    rep_time = float(rep_time_arr[0])
    np.savetxt(
        DATA / 'finite_time_pump_trajectory.csv',
        np.c_[flux, total, acharge, bcharge], delimiter=',',
        header=f'flux,total_charge_right,A_charge_right,B_charge_right;T={rep_time}', comments=''
    )
    np.savetxt(
        DATA / 'finite_time_pump_endpoints.csv',
        np.c_[times, endpoint_total, endpoint_a, endpoint_b], delimiter=',',
        header='ramp_time,total_transfer,A_transfer,B_transfer', comments=''
    )
    np.savetxt(
        DATA / 'finite_time_B_density_transfer.csv',
        np.c_[np.arange(len(rho_b)), rho_b], delimiter=',',
        header=f'x,delta_B_density;T={rep_time}', comments=''
    )
    edge_gap, bulk_gap = finite_time_gap_scales()
    np.savetxt(
        DATA / 'finite_time_gap_scales.csv',
        np.asarray([[edge_gap, bulk_gap, rep_time]]), delimiter=',',
        header='minimum_opposite_edge_anticrossing,minimum_bulk_like_direct_gap,representative_ramp_time',
        comments=''
    )
    real2d = finite_time_cross_pump_2d_disorder(total_time=rep_time)
    np.savetxt(
        DATA / 'finite_time_full_2d_disorder.csv',
        np.asarray([[8, 4, 0.02, rep_time, *real2d]]), delimiter=',',
        header='Lx,Ly,disorder,ramp_time,total_transfer,A_transfer,B_transfer', comments=''
    )

    step_rows = []
    rep_idx = int(np.argmin(np.abs(times - rep_time)))
    for nstep in (180, 360, 720, 1440):
        if nstep == 720:
            step_rows.append([nstep, endpoint_total[rep_idx], endpoint_a[rep_idx], endpoint_b[rep_idx]])
        else:
            step_out = finite_time_cross_pump(np.asarray([rep_time]), nsteps=nstep)
            step_rows.append([nstep, step_out[5][0], step_out[6][0], step_out[7][0]])
    np.savetxt(
        DATA / 'finite_time_step_convergence.csv', np.asarray(step_rows), delimiter=',',
        header='N_time_steps,total_transfer,A_transfer,B_transfer', comments=''
    )

    fig = plt.figure(figsize=(7.3, 5.05), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    labels = {(1,1):r'$(+,+)$', (1,-1):r'$(+,-)$', (-1,1):r'$(-,+)$', (-1,-1):r'$(-,-)$'}
    markers = {(1,1):'o', (1,-1):'s', (-1,1):'^', (-1,-1):'D'}
    for pair, label in labels.items():
        distance = (d0[:,3] - pair[0])**2 + (d0[:,4] - pair[1])**2
        mask = distance < 0.35
        ax.scatter(
            d0[mask,0] / np.pi, d0[mask,1], s=3 + 8*d0[mask,2],
            alpha=.62, marker=markers[pair], label=label
        )
    ax.set_xlim(-1,1); ax.set_ylim(-.8,.8); ax.axhline(0, lw=.5)
    ax.set_xlabel(r'$k_y/\pi$'); ax.set_ylabel('energy')
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    ax.set_title(r'(a) Symmetric edge spectrum', loc='left', fontsize=8.7)

    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(dg[:,0] / np.pi, dg[:,1], c=dg[:,3], s=3 + 8*dg[:,2], alpha=.72, vmin=-1, vmax=1)
    ax.set_xlim(-1,1); ax.set_ylim(-.8,.8); ax.axhline(0, lw=.5)
    ax.set_xlabel(r'$k_y/\pi$'); ax.set_ylabel('energy')
    cb = fig.colorbar(sc, ax=ax, fraction=.046, pad=.04); cb.set_label(r'$\langle Z_A\rangle$', fontsize=8)
    ax.text(0.58, -0.58, rf'edge gap $\simeq {2*gg:.2f}$', fontsize=7.0, ha='center', bbox=dict(facecolor='white', edgecolor='none', alpha=1.0, pad=1.2))
    ax.set_title('(b) Label-breaking edge gap', loc='left', fontsize=8.7)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(flux, total, label=r'$\Delta Q^R$')
    ax.plot(flux, acharge, '--', label=r'$\Delta Q_A^R$')
    ax.plot(flux, bcharge, '-.', label=r'$\Delta Q_B^R$')
    ax.axhline(4, lw=.45); ax.axhline(0, lw=.45)
    ax.set_xlabel(r'$A$-flux $\theta_A/2\pi$'); ax.set_ylabel('right-half transfer')
    ax.set_ylim(-.35,4.35); ax.legend(frameon=False, fontsize=6.8)
    ax.set_title(rf'(c) Finite-time charge transfer, $T={rep_time:g}$', loc='left', fontsize=8.7)

    ax = fig.add_subplot(gs[1, 1])
    ax.semilogx(times, endpoint_b, 'o-', label=r'$\Delta Q_B^R$')
    ax.semilogx(times, endpoint_total, 's--', ms=3.2, label=r'$\Delta Q^R$')
    ax.semilogx(times, endpoint_a, '^-.', ms=3.2, label=r'$\Delta Q_A^R$')
    ax.axhline(4, lw=.45); ax.axhline(0, lw=.45)
    ax.axvspan(20.0, times[-1], alpha=.08)
    ax.set_xlabel('ramp time $T$'); ax.set_ylabel('one-cycle transfer')
    ax.set_ylim(-.28,4.72); ax.legend(frameon=True, facecolor='white', framealpha=1.0, edgecolor='none', fontsize=5.8, loc='upper center', ncol=3, bbox_to_anchor=(0.50, 0.995))
    ax.text(.03,.74,rf'$\Delta_{{\rm edge}}={edge_gap:.1e}$, $\Delta_{{\rm bulk}}={bulk_gap:.2f}$',
            transform=ax.transAxes, va='top', fontsize=6.6,
            bbox=dict(facecolor='white', edgecolor='none', alpha=.84, pad=1.5))
    ins = ax.inset_axes([.59,.25,.35,.29])
    x = np.arange(len(rho_b))
    ins.set_facecolor('white')
    ins.patch.set_alpha(1.0)
    ins.plot(x, rho_b, 'o-', ms=2.0)
    ins.axhline(0, lw=.4); ins.axvline((len(x)-1)/2, lw=.45, ls=':')
    ins.set_xlabel(''); ins.set_ylabel(r'$\Delta\rho_B$', fontsize=6)
    ins.set_xticks([0, (len(x)-1)//2, len(x)-1])
    ins.tick_params(labelsize=5.3, pad=1.0)
    ax.set_title('(d) Pump window', loc='left', fontsize=8.7)
    fig.savefig(FIG / 'fig3_edge_realspace_pump.pdf', bbox_inches='tight')
    fig.savefig(FIG / 'fig3_edge_realspace_pump.png', dpi=300, bbox_inches='tight')
    plt.close(fig)



def plot_fig3_from_data() -> None:
    """Regenerate Fig. 3 from precomputed finite-system data.

    The direct time evolution and disorder/size ensembles are intentionally
    stored because these are among the most expensive stages.  This function
    keeps the journal-facing one-command build fast while preserving a
    separate deterministic full-regeneration path.
    """
    d0 = np.loadtxt(DATA / 'full_ribbon_edge_spectrum_symmetry_preserving.csv', delimiter=',', skiprows=1)
    dg = np.loadtxt(DATA / 'full_ribbon_edge_spectrum_label_breaking.csv', delimiter=',', skiprows=1)
    edge_states = np.abs(dg[dg[:, 2] > 0.2, 1])
    if edge_states.size == 0:
        raise RuntimeError('label-breaking edge data contain no edge-localized states')
    gg = float(np.min(edge_states))

    trajectory = np.loadtxt(DATA / 'finite_time_pump_trajectory.csv', delimiter=',', skiprows=1)
    flux, total, acharge, bcharge = trajectory.T
    endpoints = np.loadtxt(DATA / 'finite_time_pump_endpoints.csv', delimiter=',', skiprows=1)
    times, endpoint_total, endpoint_a, endpoint_b = endpoints.T
    gap_row = np.loadtxt(DATA / 'finite_time_gap_scales.csv', delimiter=',', skiprows=1).reshape(-1)
    edge_gap, bulk_gap, rep_time = map(float, gap_row[:3])
    rho_b = np.loadtxt(DATA / 'finite_time_B_density_transfer.csv', delimiter=',', skiprows=1)[:, 1]

    fig = plt.figure(figsize=(7.3, 5.05), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    labels = {(1,1):r'$(+,+)$', (1,-1):r'$(+,-)$', (-1,1):r'$(-,+)$', (-1,-1):r'$(-,-)$'}
    markers = {(1,1):'o', (1,-1):'s', (-1,1):'^', (-1,-1):'D'}
    for pair, label in labels.items():
        distance = (d0[:,3] - pair[0])**2 + (d0[:,4] - pair[1])**2
        mask = distance < 0.35
        ax.scatter(
            d0[mask,0] / np.pi, d0[mask,1], s=3 + 8*d0[mask,2],
            alpha=.62, marker=markers[pair], label=label
        )
    ax.set_xlim(-1,1); ax.set_ylim(-.8,.8); ax.axhline(0, lw=.5)
    ax.set_xlabel(r'$k_y/\pi$'); ax.set_ylabel('energy')
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    ax.set_title(r'(a) Symmetric edge spectrum', loc='left', fontsize=8.7)

    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(dg[:,0] / np.pi, dg[:,1], c=dg[:,3], s=3 + 8*dg[:,2], alpha=.72, vmin=-1, vmax=1)
    ax.set_xlim(-1,1); ax.set_ylim(-.8,.8); ax.axhline(0, lw=.5)
    ax.set_xlabel(r'$k_y/\pi$'); ax.set_ylabel('energy')
    cb = fig.colorbar(sc, ax=ax, fraction=.046, pad=.04); cb.set_label(r'$\langle Z_A\rangle$', fontsize=8)
    ax.text(0.58, -0.58, rf'edge gap $\simeq {2*gg:.2f}$', fontsize=7.0, ha='center', bbox=dict(facecolor='white', edgecolor='none', alpha=1.0, pad=1.2))
    ax.set_title('(b) Label-breaking edge gap', loc='left', fontsize=8.7)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(flux, total, label=r'$\Delta Q^R$')
    ax.plot(flux, acharge, '--', label=r'$\Delta Q_A^R$')
    ax.plot(flux, bcharge, '-.', label=r'$\Delta Q_B^R$')
    ax.axhline(4, lw=.45); ax.axhline(0, lw=.45)
    ax.set_xlabel(r'$A$-flux $\theta_A/2\pi$'); ax.set_ylabel('right-half transfer')
    ax.set_ylim(-.35,4.35); ax.legend(frameon=False, fontsize=6.8)
    ax.set_title(rf'(c) Finite-time charge transfer, $T={rep_time:g}$', loc='left', fontsize=8.7)

    ax = fig.add_subplot(gs[1, 1])
    ax.semilogx(times, endpoint_b, 'o-', label=r'$\Delta Q_B^R$')
    ax.semilogx(times, endpoint_total, 's--', ms=3.2, label=r'$\Delta Q^R$')
    ax.semilogx(times, endpoint_a, '^-.', ms=3.2, label=r'$\Delta Q_A^R$')
    ax.axhline(4, lw=.45); ax.axhline(0, lw=.45)
    ax.axvspan(20.0, times[-1], alpha=.08)
    ax.set_xlabel('ramp time $T$'); ax.set_ylabel('one-cycle transfer')
    ax.set_ylim(-.28,4.72); ax.legend(frameon=True, facecolor='white', framealpha=1.0, edgecolor='none', fontsize=5.8, loc='upper center', ncol=3, bbox_to_anchor=(0.50, 0.995))
    ax.text(.03,.74,rf'$\Delta_{{\rm edge}}={edge_gap:.1e}$, $\Delta_{{\rm bulk}}={bulk_gap:.2f}$',
            transform=ax.transAxes, va='top', fontsize=6.6,
            bbox=dict(facecolor='white', edgecolor='none', alpha=.84, pad=1.5))
    ins = ax.inset_axes([.59,.25,.35,.29])
    x = np.arange(len(rho_b))
    ins.set_facecolor('white')
    ins.patch.set_alpha(1.0)
    ins.plot(x, rho_b, 'o-', ms=2.0)
    ins.axhline(0, lw=.4); ins.axvline((len(x)-1)/2, lw=.45, ls=':')
    ins.set_xlabel(''); ins.set_ylabel(r'$\Delta\rho_B$', fontsize=6)
    ins.set_xticks([0, (len(x)-1)//2, len(x)-1])
    ins.tick_params(labelsize=5.3, pad=1.0)
    ax.set_title('(d) Pump window', loc='left', fontsize=8.7)
    fig.savefig(FIG / 'fig3_edge_realspace_pump.pdf', bbox_inches='tight')
    fig.savefig(FIG / 'fig3_edge_realspace_pump.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# -----------------------------------------------------------------------------
# Full factorization diagnostics and reference bundles
# -----------------------------------------------------------------------------

GAMMA = np.array([
    np.kron(sx,sx), np.kron(sx,sy), np.kron(sx,sz), np.kron(sy,I2), np.kron(sz,I2)
])


@lru_cache(maxsize=None)
def _leggauss_cached(order: int):
    return roots_legendre(order)


def c2_shifted_yang(m: float, order: int = 450) -> float:
    x, w = _leggauss_cached(int(order))
    alpha = .5 * (x + 1) * np.pi
    wa = .5 * np.pi * w
    r2 = 1 + m*m + 2*m*np.cos(alpha)
    integrand = .75 * np.sin(alpha)**3 * (1 + m*np.cos(alpha)) / (r2**2.5)
    return float(np.sum(wa * integrand))


def yang_projector(n: np.ndarray) -> np.ndarray:
    return .5 * (np.eye(4) - np.einsum('i,ijk->jk', n, GAMMA))


def operator_schmidt_entropy(u: np.ndarray) -> float:
    t = u.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4)
    singular = np.linalg.svd(t, compute_uv=False)
    prob = singular * singular / np.sum(singular * singular)
    prob = prob[prob > 1e-15]
    return float(-np.sum(prob * np.log2(prob)))


def su2_clutching_and_derivatives(ch: float, th: float, ph: float):
    x = np.array([
        np.sin(ch)*np.sin(th)*np.cos(ph), np.sin(ch)*np.sin(th)*np.sin(ph),
        np.sin(ch)*np.cos(th), np.cos(ch)
    ])
    dc = np.array([
        np.cos(ch)*np.sin(th)*np.cos(ph), np.cos(ch)*np.sin(th)*np.sin(ph),
        np.cos(ch)*np.cos(th), -np.sin(ch)
    ])
    dt = np.array([
        np.sin(ch)*np.cos(th)*np.cos(ph), np.sin(ch)*np.cos(th)*np.sin(ph),
        -np.sin(ch)*np.sin(th), 0
    ])
    dp = np.array([
        -np.sin(ch)*np.sin(th)*np.sin(ph), np.sin(ch)*np.sin(th)*np.cos(ph), 0, 0
    ])
    def mat(v: np.ndarray) -> np.ndarray:
        return v[3]*I2 + 1j*(v[0]*sx + v[1]*sy + v[2]*sz)
    return mat(x), mat(dc), mat(dt), mat(dp)


def clutching_winding_midpoint(n: int, kind: str) -> float:
    # The odd and even embeddings have the same SU(2) Maurer--Cartan
    # form; only the trace multiplicity differs.  Vectorizing the full
    # S^3 midpoint grid keeps this a genuine three-dimensional integral
    # while avoiding slow Python loops over 8x8 matrices.
    dch = np.pi / n
    dth = np.pi / n
    dph = np.pi / n
    ch = ((np.arange(n) + .5) * dch)[:, None, None]
    th = ((np.arange(n) + .5) * dth)[None, :, None]
    ph = ((np.arange(2*n) + .5) * dph)[None, None, :]
    sch, cch = np.sin(ch), np.cos(ch)
    sth, cth = np.sin(th), np.cos(th)
    sph, cph = np.sin(ph), np.cos(ph)
    x = np.stack(np.broadcast_arrays(sch*sth*cph, sch*sth*sph, sch*cth, cch + 0*ph), axis=-1)
    dc = np.stack(np.broadcast_arrays(cch*sth*cph, cch*sth*sph, cch*cth, -sch + 0*ph), axis=-1)
    dt = np.stack(np.broadcast_arrays(sch*cth*cph, sch*cth*sph, -sch*sth, 0*ph), axis=-1)
    dp = np.stack(np.broadcast_arrays(-sch*sth*sph, sch*sth*cph, 0*ph, 0*ph), axis=-1)

    def mats(v: np.ndarray) -> np.ndarray:
        return (
            v[...,3,None,None] * I2
            + 1j * v[...,0,None,None] * sx
            + 1j * v[...,1,None,None] * sy
            + 1j * v[...,2,None,None] * sz
        )
    u, uc, ut, up = mats(x), mats(dc), mats(dt), mats(dp)
    ud = np.swapaxes(u.conj(), -1, -2)
    aa = ud @ uc
    bb = ud @ ut
    cc = ud @ up
    integrand = np.trace(aa @ (bb @ cc - cc @ bb), axis1=-2, axis2=-1)
    multiplicity = 1.0 if kind == 'odd' else 2.0
    total = multiplicity * np.sum(integrand)
    return float(np.real(total * dch * dth * dph / (8*np.pi**2)))


def generic_s4_matrices() -> np.ndarray:
    rng = np.random.default_rng(seed("random_s4_matrices"))
    matrices = []
    for _ in range(6):
        x = rng.normal(size=(8,8)) + 1j*rng.normal(size=(8,8))
        a = (x + x.conj().T) / 2
        a -= np.trace(a) / 8 * np.eye(8)
        a /= np.max(np.abs(np.linalg.eigvalsh(a)))
        matrices.append(a)
    return np.asarray(matrices)


S4_PERTURBATION_SCALE = 2.0
S4_NORM_BOUND = (0.35 + 1.0) / S4_PERTURBATION_SCALE


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


def odd_flat_hamiltonian(n: np.ndarray) -> np.ndarray:
    return block_diag(np.tensordot(n, GAMMA, axes=(0,0)), -I2, I2)


def generic_s4_perturbation(n: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    return (0.35 * matrices[0] + np.tensordot(n, matrices[1:], axes=(0,0)) / np.sqrt(5)) / S4_PERTURBATION_SCALE


def direct_c2_perturbed(epsilon: float, order: int, matrices: np.ndarray) -> tuple[float, float]:
    nodes, weights = _leggauss_cached(int(order))
    angles = .5 * (nodes + 1) * np.pi
    weights = .5 * np.pi * weights
    azimuths = (np.arange(2*order) + .5) * np.pi / order
    waz = np.pi / order
    zero2 = np.zeros((2,2), complex)
    derivatives_n = np.asarray([block_diag(GAMMA[i], zero2, zero2) for i in range(5)])
    derivatives_n += epsilon * matrices[1:] / (np.sqrt(5) * S4_PERTURBATION_SCALE)
    total = 0j
    minimum_gap = np.inf
    for ia, aa in enumerate(angles):
        for ib, bb in enumerate(angles):
            for ic, cc in enumerate(angles):
                weight = weights[ia] * weights[ib] * weights[ic] * waz
                for dd in azimuths:
                    n, dn = s4_coordinates(aa, bb, cc, dd)
                    h = odd_flat_hamiltonian(n) + epsilon * generic_s4_perturbation(n, matrices)
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


def generic_s4_data() -> tuple[np.ndarray, ...]:
    matrices = generic_s4_matrices()
    np.save(DATA / 'generic_s4_perturbation_matrices.npy', matrices)
    with open(DATA / 'generic_s4_seed.txt', 'w') as handle:
        handle.write(f'{seed("random_s4_matrices")}\n')

    rng = np.random.default_rng(seed("random_s4_sampling"))
    nvec = rng.normal(size=(1200,5))
    nvec /= np.linalg.norm(nvec, axis=1)[:,None]
    nvec = np.vstack([nvec, np.eye(5), -np.eye(5)])
    h0 = np.asarray([odd_flat_hamiltonian(n) for n in nvec])
    v = np.asarray([generic_s4_perturbation(n, matrices) for n in nvec])
    eps = np.linspace(0, 1.25, 26)
    sampled_gap = []
    analytic_bound = 2 - 2 * eps * S4_NORM_BOUND
    for value in eps:
        ev = np.linalg.eigvalsh(h0 + value * v)
        sampled_gap.append(float(np.min(ev[:,4]-ev[:,3])))
    sampled_gap = np.asarray(sampled_gap)

    ptriv = block_diag(np.eye(4), np.zeros((4,4)))
    emix = np.linspace(0, 1.25, 21)
    mixing = []
    for value in emix:
        _, u = np.linalg.eigh((h0 + value * v)[:260])
        occ = u[:,:,:4]
        p = np.einsum('nai,nbi->nab', occ, occ.conj())
        comm = p @ ptriv - ptriv @ p
        mixing.append(float(np.mean(np.linalg.norm(comm, axis=(1,2))/2)))
    mixing = np.asarray(mixing)

    eps_c2 = np.linspace(0, 1.25, 6)
    c2_order5 = []
    c2_order6 = []
    c2_gap = []
    for value in eps_c2:
        c5, _ = direct_c2_perturbed(float(value), 5, matrices)
        c6, gap = direct_c2_perturbed(float(value), 6, matrices)
        c2_order5.append(c5); c2_order6.append(c6); c2_gap.append(gap)
    c2_order5 = np.asarray(c2_order5); c2_order6 = np.asarray(c2_order6); c2_gap = np.asarray(c2_gap)
    np.savetxt(
        DATA / 'generic_s4_c2_direct.csv',
        np.c_[eps_c2, c2_order5, c2_order6, c2_gap], delimiter=',',
        header='lambda,C2_order5,C2_order6,min_quadrature_gap', comments=''
    )
    orders_c2 = np.array([3,4,5,6,7,8])
    convergence = np.asarray([[order, *direct_c2_perturbed(1.25, int(order), matrices)] for order in orders_c2])
    np.savetxt(
        DATA / 'generic_s4_c2_convergence.csv', convergence, delimiter=',',
        header='quadrature_order,C2_lambda1p25,min_quadrature_gap', comments=''
    )
    np.savetxt(
        DATA / 'generic_s4_gap_bound.csv', np.c_[eps, sampled_gap, analytic_bound], delimiter=',',
        header='lambda,min_sampled_gap,Weyl_global_lower_bound', comments=''
    )
    np.savetxt(
        DATA / 'generic_s4_block_mixing.csv', np.c_[emix, mixing], delimiter=',',
        header='lambda,mean_block_commutator_norm', comments=''
    )

    t = np.linspace(0,1,81)
    htriv = np.eye(8) - 2 * ptriv
    interpolation_gap = []
    for value in t:
        hh = (1-value)*h0 + value*htriv[None,:,:] + .28*np.sin(np.pi*value)*v
        ev = np.linalg.eigvalsh(hh)
        interpolation_gap.append(float(np.min(ev[:,4]-ev[:,3])))
    interpolation_gap = np.asarray(interpolation_gap)
    np.savetxt(
        DATA / 'generic_s4_odd_to_trivial_interpolation.csv', np.c_[t, interpolation_gap], delimiter=',',
        header='t,min_sampled_gap_illustration', comments=''
    )
    return eps, sampled_gap, analytic_bound, emix, mixing, eps_c2, c2_order5, c2_order6, convergence



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
        hn = odd_flat_hamiltonian(north) + epsilon * generic_s4_perturbation(north, matrices)
        hs = odd_flat_hamiltonian(south) + epsilon * generic_s4_perturbation(south, matrices)
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
        h = odd_flat_hamiltonian(n) + epsilon * generic_s4_perturbation(n, matrices)
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

def main() -> None:
    import argparse
    import subprocess
    import sys
    import time

    jobs = {
        'fig1': ('Fig. 1', generate_fig1),
        'fig2': ('Fig. 2', generate_fig2),
        'figS1': ('Fig. S1 full regeneration', generate_figS_label_resolution),
        'figS1plot': ('Fig. S1 data plot', plot_figS_label_resolution_from_data),
        'fig3': ('Fig. 3 full regeneration', generate_fig3),
        'fig3plot': ('Fig. 3 data plot', plot_fig3_from_data),
    }
    parser = argparse.ArgumentParser(description='Reproduce the numerical data and figures.')
    parser.add_argument('--part', choices=['all', *jobs.keys()], default='all')
    args = parser.parse_args()
    if args.part != 'all':
        label, job = jobs[args.part]
        started = time.time()
        job()
        print(f'{label} complete in {time.time()-started:.2f} s', flush=True)
        return

    # Run numerically distinct panels in fresh processes.  Replacing this
    # process with the shell driver avoids mixed-size LAPACK slowdowns seen in
    # some BLAS builds while preserving a one-command workflow.
    driver = Path(__file__).with_name('run_calculations.sh')
    os.execv('/bin/bash', ['bash', str(driver)])


if __name__ == '__main__':
    main()
