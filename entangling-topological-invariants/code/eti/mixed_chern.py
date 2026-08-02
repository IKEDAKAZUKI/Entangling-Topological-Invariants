from __future__ import annotations

from functools import lru_cache

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Tinos", "Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec

from .common import DATA, FIG, I2, block_diag, kron3, spectral_derivative, sx, sy, sz
from .parameters import LABEL_MIXING_DIRECTION, LABEL_MIXING_ENDPOINT, LABEL_MIXING_SCAN_MAX

ZA = kron3(sz, I2, I2)
ZB = kron3(I2, sz, I2)
PROJECTED_LABEL = 2 * ZA + ZB
CROSS_WEIGHTS = np.array([1, -1, -1, 1], dtype=float)
V1 = kron3(sx, I2, sz)
V2 = kron3(I2, sx, sx)
V3 = kron3(sy, sy, sy)
V4 = kron3(sx, sx, I2)

# Fixed ray used in Fig. 2, Eq. (S24) of the manuscript.
MIXING_DIRECTION = np.asarray(LABEL_MIXING_DIRECTION, dtype=float)

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
    plt.close(fig)

def four_block_hamiltonian(kx: float, ky: float, m: float = 1.0) -> np.ndarray:
    masses = [m, -m, -m, m]
    return block_diag(*(qwz_hamiltonian(kx, ky, mm) for mm in masses))

def label_mixing(kx: float, ky: float, coefficients: np.ndarray = MIXING_DIRECTION) -> np.ndarray:
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
            h = four_block_hamiltonian(kx, ky) + eps * label_mixing(kx, ky)
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
    eps_scan = np.linspace(0, LABEL_MIXING_SCAN_MAX, 25)
    egaps, rgaps, chis = [], [], []
    sector_cs = []
    for eps in eps_scan:
        p, q, lines, eg, rg = mixed_resolved_grid(float(eps), 23)
        cs = fhs_line_chern(lines)
        egaps.append(eg); rgaps.append(rg); sector_cs.append(cs)
        chis.append(float(np.dot(CROSS_WEIGHTS, cs)))
    sector_cs = np.asarray(sector_cs)
    np.savetxt(
        DATA / 'label_mixing_gaps_fhs.csv',
        np.c_[eps_scan, egaps, rgaps, sector_cs, chis], delimiter=',',
        header='epsilon,energy_gap,label_resolution_gap,C_pp,C_pm,C_mp,C_mm,chi', comments=''
    )

    eps_decomp = np.linspace(0, LABEL_MIXING_SCAN_MAX, 13)
    det_i, proj_i, corr_i, residual_i = [], [], [], []
    for eps in eps_decomp:
        p, q, _, _, _ = mixed_resolved_grid(float(eps), 31)
        det, proj, corr = determinant_curvature_densities(p, q)
        det_i.append(integrate_density(det))
        proj_i.append(integrate_density(proj))
        corr_i.append(integrate_density(corr))
        residual_i.append(np.max(np.abs(det - proj - corr)))
    np.savetxt(
        DATA / 'label_mixing_curvature_integrals.csv',
        np.c_[eps_decomp, det_i, proj_i, corr_i, np.asarray(proj_i)+np.asarray(corr_i), residual_i], delimiter=',',
        header='epsilon,determinant_line,projected,gauss_codazzi,projected_plus_correction,max_pointwise_residual', comments=''
    )

    eps0 = LABEL_MIXING_ENDPOINT
    density_n = 151
    p, q, lines, eg0, rg0 = mixed_resolved_grid(eps0, density_n)
    det, proj, corr = determinant_curvature_densities(p, q)
    residual = det - proj - corr
    np.savez_compressed(
        DATA / 'label_mixing_curvature_density_eps6over5.npz',
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
        DATA / 'label_mixing_curvature_convergence.csv', conv, delimiter=',',
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
    ax.set_title(r'(d) Projected term, $\epsilon=6/5$',loc='left',fontsize=8.5)
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
    fig.savefig(FIG / 'fig2_label_mixing.pdf', dpi=400, bbox_inches='tight')
    plt.close(fig)
