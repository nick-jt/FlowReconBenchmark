"""All-baselines reconstruction galleries for the PoF 2026 paper.

One figure per 3D dataset, single-sample fields on the SAME snapshot with the
SAME sensors (verified by src/figs_pof/verify_sources.py — run it first; this
script re-asserts the cheap invariants and refuses mismatched panels).

Figure 1  recon_gallery_jhu.{pdf,png}
    JHU 125^3 held-out cube-3 snapshot (val idx 3 = absolute frame 153),
    z-midplane slice.  Rows Ux (observed) / Uy (unobserved); columns DNS
    truth, DMF-Gen sample, latent-FM sample, SiT-point sample, Senseiver,
    IDW k=8, gappy POD (rank 80, basis = train frames 0-149).

Figure 2  recon_gallery_firebench.{pdf,png}
    FireBench 152x126x192 frame 112.  Rows u / theta on the vertical slice
    y=yc (from qual_firebench.npz), rho_f on the fuel-bed plane z=1 (plane of
    max fuel variance, per replot_firebench.py).  Columns LES truth, DMF-Gen
    sample, latent-FM sample, Senseiver, IDW k=8 (+ gappy POD if it fits the
    CPU budget).

Units: panels show PHYSICAL fields; the per-panel number is the relative L2
on the shown slice in STANDARDIZED units (our run's train stats — the paper's
evaluation units, where "predict the train mean" scores 1.0).

CPU-only.  IDW and gappy POD are computed here by importing kd_predict /
GappyPOD / obs_columns from src/baseline_classical_jhu.py (same machinery as
the classical-baseline table).
"""
from paths import DATA_DIR, REPO_ROOT
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

os.environ.setdefault("ALLOW_LOGIN_EVAL", "1")   # import-time safety only; no sensor draw happens here

MAIN = (f'{REPO_ROOT}')
WT = (f'{REPO_ROOT}')
FD = f'{WT}/outputs/pof/field_dumps'
OUT = Path(f'{WT}/outputs/paper/figures')
JHU_H5 = (f'{DATA_DIR}/JHU_4cubes_stride100.h5')
FB_H5 = (f'{DATA_DIR}/FireBench_u10u12_merged.h5')
from methods.classical.anchors_jhu import GappyPOD, kd_predict, obs_columns

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'text.color': '#1f2430', 'axes.edgecolor': '#c9ccd4',
    'axes.linewidth': 0.5, 'figure.dpi': 150, 'savefig.dpi': 300,
    'pdf.fonttype': 42,
})
INK = '#1f2430'


def sensors_by_field(idx, fid, truth_std):
    """{field: (point_idx, standardized value)} from a dump's sensor arrays."""
    out = {}
    for f in np.unique(fid):
        m = fid == f
        pi = idx[m].astype(np.int64)
        out[int(f)] = (pi, truth_std[pi, int(f)].astype(np.float32))
    return out


def rel_l2(pred, true):
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


def annotate(ax, txt):
    ax.text(0.03, 0.965, txt, transform=ax.transAxes, ha='left', va='top',
            fontsize=5.8, color=INK,
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.72))


def style(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)


def load_frames_std(h5, frames, mean, std, n_pts, n_ch):
    X = np.empty((len(frames), n_pts, n_ch), dtype=np.float32)
    with h5py.File(h5, 'r') as f:
        for i, t in enumerate(frames):
            X[i] = (f['fields'][0, int(t), :, 0, 0, :].astype(np.float32) - mean) / std
    return X


# ======================================================================
# Figure 1 — JHU
# ======================================================================

def make_jhu():
    G = 125
    zc = G // 2
    names = ['Ux', 'Uy', 'Uz', 'p']
    qj = np.load(f'{MAIN}/outputs/figures/iclr2027/qual_jhu.npz')
    sf = np.load(f'{MAIN}/outputs/figures/iclr2027/spectra_fields.npz')
    sit = np.load(f'{FD}/jhu_sit.npz')
    sen = np.load(f'{FD}/jhu_senseiver.npz')

    mean = sit['norm_mean'].astype(np.float64)   # == N29 == N23 stats (verified)
    std = sit['norm_std'].astype(np.float64)
    truth_p = sit['truth'].astype(np.float64)    # physical, frame 153
    truth_z = (truth_p - mean) / std

    # --- truth-agreement gates (EXCLUDE a panel rather than mislabel it) ---
    def corr_ok(z_arr):
        return min(float(np.corrcoef(z_arr[:, j], truth_p[:, j])[0, 1])
                   for j in range(4)) > 0.999
    assert np.allclose(sen['truth'], sit['truth']), 'senseiver truth mismatch'
    assert corr_ok(qj['truth']), 'qual_jhu truth mismatch'
    assert np.allclose(sf['truth_3'], qj['truth']), 'spectra truth_3 mismatch'
    assert corr_ok(sf['truth_3']), 'spectra truth mismatch'
    m = json.loads(str(sit['meta']))
    assert int(sit['sensor_indices'].sum()) == m['sensor_idx_sum'], 'idx_sum'
    print('[jhu] truth + sensor gates passed')

    # --- classical baselines on CPU -----------------------------------
    sens = sensors_by_field(sit['sensor_indices'], sit['sensor_field_ids'], truth_z)
    craw = sit['coords_raw'].astype(np.float64)
    lo = craw.min(0)
    dx = (craw.max(0) - lo) / (G - 1)
    coords_box = np.ascontiguousarray((craw - lo) / (G * dx))
    t0 = time.time()
    idw_z = kd_predict(coords_box, sens, 4, 'idw', 8, None)  # non-periodic
    print(f'[jhu] IDW k=8 done in {time.time()-t0:.1f}s')

    t0 = time.time()
    Xtr = load_frames_std(JHU_H5, range(150), mean.astype(np.float32),
                          std.astype(np.float32), G ** 3, 4).reshape(150, -1)
    mu = Xtr.mean(axis=0)
    Xtr -= mu
    pod = GappyPOD(Xtr, mu, rank=80)
    cols, vals = obs_columns(sens, 4)
    pod_z = pod.reconstruct(cols, vals).reshape(G ** 3, 4)
    print(f'[jhu] gappy POD r={pod.r} (train frames 0-149) done in '
          f'{time.time()-t0:.1f}s, cum-energy {pod.energy[pod.r-1]:.4f}')
    del Xtr, pod

    panels = [
        ('DNS truth', truth_z.copy()),
        ('DMF-Gen\n(sample)', qj['sample0'].astype(np.float64)),
        ('latent-FM\n(sample)', sf['latent_fm_3'].astype(np.float64)),
        ('SiT-point\n(sample)', (sit['pred_sample'].astype(np.float64) - mean) / std),
        ('Senseiver\n(determ.)', (sen['pred_mean'].astype(np.float64) - mean) / std),
        ('IDW $k$=8', idw_z.astype(np.float64)),
        ('gappy POD\n($r$=80)', pod_z.astype(np.float64)),
    ]
    rows = [(0, '$U_x$ (observed)'), (1, '$U_y$ (unobserved)')]

    def sl(a, j):
        return a[:, j].reshape(G, G, G)[:, :, zc]

    # sensors of the observed Ux channel that lie in the shown plane k == zc
    pi = sens[0][0]
    in_plane = pi % G == zc
    si, sj = pi[in_plane] // (G * G), (pi[in_plane] // G) % G

    fig = plt.figure(figsize=(7.05, 2.62))
    gs = fig.add_gridspec(2, 8, width_ratios=[1] * 7 + [0.06],
                          left=0.055, right=0.945, top=0.90, bottom=0.065,
                          wspace=0.06, hspace=0.08)
    report = {}
    for r, (j, rlab) in enumerate(rows):
        t_z = sl(truth_z, j)
        t_phys = t_z * std[j] + mean[j]
        q2, q98 = np.percentile(t_phys, [2, 98])
        v = max(abs(q2), abs(q98))
        ims = None
        for c, (lab, arr) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            a_z = sl(arr, j)
            a_phys = a_z * std[j] + mean[j]
            # bicubic raster: the z-midplane is a regular 125x125 grid, and
            # nearest rendering sharpened the chunk-wise speckle of the
            # single generative samples (SiT-point in particular)
            ims = ax.imshow(a_phys.T, origin='lower', cmap='RdBu_r',
                            vmin=-v, vmax=v, interpolation='bicubic',
                            rasterized=True)
            style(ax)
            if c == 0:
                ax.set_ylabel(rlab, fontsize=7.5)
                if r == 0:
                    ax.scatter(si, sj, s=1.1, c='#111111', alpha=0.85,
                               linewidths=0, rasterized=True)
            else:
                e = rel_l2(a_z, t_z)
                annotate(ax, f'{e:.3f}')
                report[f'{names[j]}/{lab.replace(chr(10), " ")}'] = round(e, 4)
            if r == 0:
                ax.set_title(lab, fontsize=7.2, pad=2.5)
        cax = fig.add_subplot(gs[r, 7])
        cb = fig.colorbar(ims, cax=cax)
        cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
        cb.outline.set_visible(False)
    fig.text(0.055, 0.012,
             'JHU forced HIT, held-out cube, $z$-midplane; 1% sensors on $U_x,U_z$ '
             '(dots: $U_x$ sensors in this plane). Number: rel. $L_2$ on the shown '
             'slice, standardized units.', fontsize=6, color='#5a5f6b')
    fig.savefig(OUT / 'recon_gallery_jhu.pdf')
    fig.savefig(OUT / 'recon_gallery_jhu.png')
    plt.close(fig)
    print('[jhu] slice rel-L2:', json.dumps(report, indent=1))
    return report


# ======================================================================
# Figure 2 — FireBench
# ======================================================================

def make_firebench(pod_budget_s=600):
    NX, NY, NZ = 152, 126, 192
    N = NX * NY * NZ
    qf = np.load(f'{MAIN}/outputs/figures/iclr2027/qual_firebench.npz')
    flm = np.load(f'{FD}/firebench_latent_fm.npz')
    fsen = np.load(f'{FD}/firebench_senseiver.npz')
    yc = int(qf['yc'])

    # OUR run's (N18) stats: the qual npz's standardization (verified).
    s18 = torch.load(f'{MAIN}/outputs/runs/firebench/pointcloud_ffm/'
                     'iclr_firebench_v4_DemoN18_20260819_083221/dataset_stats.pt',
                     map_location='cpu', weights_only=False)
    mean = np.asarray(s18['mean'], dtype=np.float64).ravel()
    std = np.asarray(s18['std'], dtype=np.float64).ravel()

    truth_p = flm['truth'].astype(np.float64)     # physical, frame 112
    truth_z = (truth_p - mean) / std

    def corr_ok(z_arr):
        return min(float(np.corrcoef(z_arr[:, j], truth_p[:, j])[0, 1])
                   for j in range(5)) > 0.999
    assert np.allclose(fsen['truth'], flm['truth']), 'senseiver truth mismatch'
    assert corr_ok(qf['truth']), 'qual_firebench truth mismatch'
    assert np.allclose(qf['truth'] * std + mean, truth_p, atol=1e-3), \
        'qual_firebench is not N18-standardized frame 112'
    mf = json.loads(str(flm['meta']))
    assert int(flm['sensor_indices'].sum()) == mf['sensor_idx_sum'], 'idx_sum'
    assert np.array_equal(flm['sensor_indices'], fsen['sensor_indices'])
    print('[fb] truth + sensor gates passed; yc =', yc)

    # fuel-bed plane, per replot_firebench.py: z of max truth fuel variance
    zc = int(np.argmax(truth_z[:, 4].reshape(NX, NY, NZ).var(axis=(0, 1))))
    print('[fb] fuel-bed plane zc =', zc)

    # --- classical baselines (wind sensors only, fields 0/1/2) --------
    sens = sensors_by_field(flm['sensor_indices'], flm['sensor_field_ids'], truth_z)
    craw = flm['coords_raw'].astype(np.float64)
    t0 = time.time()
    idw_z = kd_predict(np.ascontiguousarray(craw), sens, 5, 'idw', 8, None)
    # unobserved channels (theta, rho_f) stay at 0 == the train mean in z units
    print(f'[fb] IDW k=8 done in {time.time()-t0:.1f}s')

    pod_z, pod_note = None, ''
    t0 = time.time()
    try:
        frames = range(80)          # 0.75-split train frames (baseline protocol)
        Xtr = load_frames_std(FB_H5, frames, mean.astype(np.float32),
                              std.astype(np.float32), N, 5).reshape(80, -1)
        mu = Xtr.mean(axis=0)
        Xtr -= mu
        pod = GappyPOD(Xtr, mu, rank=40)
        cols, vals = obs_columns(sens, 5)
        pod_z = pod.reconstruct(cols, vals).reshape(N, 5).astype(np.float64)
        el = time.time() - t0
        print(f'[fb] gappy POD r={pod.r} (train frames 0-79) done in {el:.1f}s, '
              f'cum-energy {pod.energy[pod.r-1]:.4f}')
        del Xtr, pod
        if el > pod_budget_s:
            print('[fb] over budget — dropping POD column')
            pod_z = None
    except MemoryError:
        print('[fb] gappy POD skipped (memory)')
        pod_z = None

    panels = [
        ('LES truth', truth_z.copy()),
        ('DMF-Gen\n(sample)', qf['sample0'].astype(np.float64)),
        ('latent-FM\n(sample)', (flm['pred_sample'].astype(np.float64) - mean) / std),
        ('Senseiver\n(determ.)', (fsen['pred_mean'].astype(np.float64) - mean) / std),
        ('IDW $k$=8', idw_z.astype(np.float64)),
    ]
    if pod_z is not None:
        panels.append(('gappy POD\n($r$=40)', pod_z))
    rows = [(0, 'vert', '$u$ [m/s] (observed)'),
            (3, 'vert', r'$\theta$ [K] (unobserved)'),
            (4, 'horiz', r'$\rho_f$ [kg/m$^3$] (unobs.)')]

    def sl(a, j, plane):
        c = a[:, j].reshape(NX, NY, NZ)
        return c[:, yc, :] if plane == 'vert' else c[:, :, zc]

    # u-sensors lying in the vertical slice j == yc
    pi = sens[0][0]
    in_pl = (pi // NZ) % NY == yc
    si, sk = pi[in_pl] // (NY * NZ), pi[in_pl] % NZ

    nc = len(panels)
    fig = plt.figure(figsize=(7.05, 4.45))
    gs = fig.add_gridspec(3, nc + 1, width_ratios=[1] * nc + [0.06],
                          height_ratios=[1.26, 1.26, 0.86],
                          left=0.06, right=0.94, top=0.925, bottom=0.075,
                          wspace=0.06, hspace=0.10)
    report = {}
    names = ['u', 'v', 'w', 'theta', 'rho_f']
    for r, (j, plane, rlab) in enumerate(rows):
        t_z = sl(truth_z, j, plane)
        t_phys = t_z * std[j] + mean[j]
        vmin, vmax = np.percentile(t_phys, [1, 99])
        if vmax - vmin < 1e-6:                       # rho_f percentile gotcha
            vmin, vmax = t_phys.min(), t_phys.max()
        ims = None
        for c, (lab, arr) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            a_z = sl(arr, j, plane)
            a_phys = a_z * std[j] + mean[j]
            ims = ax.imshow(a_phys.T, origin='lower', cmap='inferno',
                            vmin=vmin, vmax=vmax, aspect='auto',
                            interpolation='nearest', rasterized=True)
            style(ax)
            if c == 0:
                ax.set_ylabel(rlab, fontsize=7)
                if r == 0:
                    ax.scatter(si, sk, s=0.8, c='#00d5ff', alpha=0.8,
                               linewidths=0, rasterized=True)
            else:
                e = rel_l2(a_z, t_z)
                annotate(ax, f'{e:.3f}')
                report[f'{names[j]}/{lab.replace(chr(10), " ")}'] = round(e, 4)
            if r == 0:
                ax.set_title(lab, fontsize=7.2, pad=2.5)
        cax = fig.add_subplot(gs[r, nc])
        cb = fig.colorbar(ims, cax=cax)
        cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
        cb.outline.set_visible(False)
    fig.text(0.06, 0.013,
             f'FireBench wildfire LES, held-out frame; wind sensors only (1% on '
             f'$u,v,w$; dots: $u$ sensors in the slice). Rows 1-2: vertical slice '
             f'$y$={yc}; row 3: fuel-bed plane $z$={zc}. Number: rel. $L_2$ on the '
             'shown slice, standardized units.' + pod_note,
             fontsize=6, color='#5a5f6b')
    fig.savefig(OUT / 'recon_gallery_firebench.pdf')
    fig.savefig(OUT / 'recon_gallery_firebench.png')
    plt.close(fig)
    print('[fb] slice rel-L2:', json.dumps(report, indent=1))
    return report


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    rj = make_jhu()
    rf = make_firebench()
    json.dump({'jhu': rj, 'firebench': rf},
              open(OUT / 'recon_gallery_relL2.json', 'w'), indent=1)
    print('wrote', OUT)
