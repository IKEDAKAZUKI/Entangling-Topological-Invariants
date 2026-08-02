from __future__ import annotations

import csv

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

from repro_config import seed
from .common import DATA, FIG, I2, kron3, sx, sy, sz
from . import pump
from . import s4_pipeline as pipeline
from .parameters import PUMP_PHASE_OFFSET, S4_COUPLING_ENDPOINT

def _full_disorder_endpoint(seed: int, total_time: float = 100.0, length: int = 8,
                            circumference: int = 4, nsteps: int = 360,
                            disorder: float = .02) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    profile = rng.normal(size=(length, circumference))
    profile -= np.mean(profile)
    profile /= np.max(np.abs(profile))
    theta0 = PUMP_PHASE_OFFSET
    dim = 2 * length * circumference
    right = np.zeros((dim, dim), complex)
    for x in range(length // 2, length):
        for y in range(circumference):
            pos = 2 * (x * circumference + y)
            right[pos:pos + 2, pos:pos + 2] = I2
    total = acharge = bcharge = 0.0
    for a in (1, -1):
        for b in (1, -1):
            h0 = pump.full_realspace_cylinder_hamiltonian(
                theta0, a, b, length, circumference, profile, disorder
            )
            _, u0 = np.linalg.eigh(h0)
            psi = u0[:, :length * circumference]
            q0 = float(np.real(np.trace(psi.conj().T @ right @ psi)))
            for j in range(nsteps):
                theta_mid = theta0 + (j + .5) * 2 * np.pi / nsteps
                h = pump.full_realspace_cylinder_hamiltonian(
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
        out = pump.finite_time_cross_pump(
            np.asarray([100.0]), length=lx, circumference=ly, nsteps=360,
            disorder=.03, edge_potential=.06, snapshot_time=100.0
        )
        size_rows.append([lx, ly, out[5][0], out[6][0], out[7][0]])
    np.savetxt(
        DATA / 'finite_time_size_convergence.csv', np.asarray(size_rows), delimiter=',',
        header='Lx,Ly,total_transfer,A_transfer,B_transfer', comments=''
    )

    offset_rows = []
    for theta0 in (0.15, 0.30, PUMP_PHASE_OFFSET, 0.45, 0.60):
        out = pump.finite_time_cross_pump(
            np.asarray([100.0]), length=10, circumference=4, nsteps=360,
            disorder=.03, edge_potential=.06, snapshot_time=100.0,
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
        ev = np.linalg.eigvalsh(pump.ribbon_hamiltonian(
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
    matrices = pipeline.coupling_matrices()
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

    # Full control graph at the symmetric point n_mu=1/sqrt(5) and lambda=5/4.
    # family_code=0 denotes the flattened Yang/spectator Hamiltonian and
    # family_code=1 denotes the Pauli-string coupling terms.
    p0 = (I2 + sz) / 2
    base_matrices = [
        kron3(p0, sx, sx),
        kron3(p0, sx, sy),
        kron3(p0, sx, sz),
        kron3(p0, sy, I2),
    ]
    base_coefficients = np.full(4, 1 / np.sqrt(5))
    mixing_coefficients = np.asarray([
        S4_COUPLING_ENDPOINT * (7 / 20) / 2,
        *([S4_COUPLING_ENDPOINT / 10] * 5),
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
            'absolute_coupling,phase_radians'
        ), comments=''
    )

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
    ins.semilogy(conv['quadrature_order'], np.abs(conv['C2_endpoint'] - 1), 'o-', ms=2.8)
    ins.set_title(r'convergence at $\lambda=5/4$', fontsize=5.2, pad=1)
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
    plt.close(fig)

