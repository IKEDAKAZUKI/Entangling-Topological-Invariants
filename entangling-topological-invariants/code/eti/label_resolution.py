from __future__ import annotations

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
from matplotlib.gridspec import GridSpec

from .common import DATA, FIG
from .parameters import LABEL_MIXING_ENDPOINT
from .mixed_chern import CROSS_WEIGHTS, ZA, ZB, four_block_hamiltonian, label_mixing, fhs_line_chern

def _occupied_label_matrices(eps: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ks = 2 * np.pi * np.arange(n) / n - np.pi
    occ_grid = np.zeros((n, n, 8, 4), dtype=complex)
    a_grid = np.zeros((n, n, 4, 4), dtype=complex)
    b_grid = np.zeros((n, n, 4, 4), dtype=complex)
    for ix, kx in enumerate(ks):
        for iy, ky in enumerate(ks):
            h = four_block_hamiltonian(kx, ky) + eps * label_mixing(kx, ky)
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
    eps0 = LABEL_MIXING_ENDPOINT
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

