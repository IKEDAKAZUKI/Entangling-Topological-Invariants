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
from .pump import (
    finite_time_cross_pump,
    finite_time_cross_pump_2d_disorder,
    finite_time_gap_scales,
    full_ribbon_edge_dataset,
)

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
        header='minimum_opposite_edge_anticrossing,minimum_bulk_like_direct_gap,density_snapshot_time',
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
    plt.close(fig)

