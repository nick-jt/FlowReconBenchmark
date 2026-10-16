"""Gen4Turb (oommen2026turbulent) with its released weights on its native cylinder-wake case.

The released DM (3_flow_reconstruction/dm/models/best_model.pt of Zenodo record 17088765) is
sampled on the native test frames with Test Setup 1 of the upstream dm/postprocess.ipynb:
velocity observed at one random point set shared by u, v, w (mask fraction = share removed),
pressure unobserved, first 6 test frames (168-173), torch seed 23, 32 Heun steps. Each draw is
scored two ways:
  (a) upstream: one sample per frame, relative L2 over all channels in min-max units (the
      notebook's error_metric) plus per-channel relative L2 in physical and z-scored units, the
      latter also on the observed points alone;
  (b) our JHU wrapper path (methods.gen4turb.eval_jhu, mirrored step by step): points in
      shared z-score units -> physical -> min-max, mask rebuilt from flat sensor indices,
      K members, de-normalized, z-scored with train-frame stats, scored by ensemble_metrics.
The paper reports no numeric error for this task (Fig. 6 and SI Figs. S6-S9 are qualitative),
so (a) is the reference that (b) must reproduce; summary also redraws the KE panel of Fig. S6.

    python -I native_gen4turb.py fetch --dest <empty dir>                      # login node
    python -I native_gen4turb.py run --data <dest> --upstream <code root> --out <json>   # GPU
    python -I native_gen4turb.py summary --data <dest> --run <json> --json outputs/paper/diagnostics/native_gen4turb.json
"""
import argparse
import hashlib
import json
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # python -I drops PYTHONPATH
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ZIP_URL = "https://zenodo.org/api/records/17088765/files/Gen4Turbulence.zip/content"
TASK = "3_flow_reconstruction/"
WANTED = ("data/u.npy", "data/MIN_u.npy", "data/MAX_u.npy", "dm/models/best_model.pt")
FIELDS = ("u", "v", "w", "p")
MASK_FRACTIONS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
N_TRAIN, TEST0, N_FRAMES = 150, 168, 6
PUBLISHED = {
    "source": "Oommen et al., Nat. Commun. 17, 3707 (2026), doi:10.1038/s41467-026-70145-4; "
              "arXiv:2509.08752v1; Zenodo 17088765",
    "numeric_error": None,
    "note": "Task 3 (cylinder wake, Re = 11,000) is reported qualitatively only: Fig. 6 and SI "
            "Figs. S6-S9 (KE slices, spectra, intermittency PDFs). The main text, SI, peer-review "
            "file and arXiv v1 give no reconstruction error; comparison.ipynb computes a KE-slice "
            "nRMSE (get_err) but neither prints nor plots it.",
    "test_setup_1": "velocity observed at a shared random point set (mask fraction = share removed, "
                    "0-0.99), pressure unobserved; DM best_model.pt, 32 Heun steps, S_churn 80",
}
# git blob id of 3_flow_reconstruction/dm/models/best_model.pt in the GitHub release (commit 8af0a7b)
GITHUB_WEIGHTS_BLOB = "fae555243d7cf45173fe16d7129bcdbad60d2903"


class HTTPRange:
    """Read-only seekable view of a remote file through HTTP range requests (enough for zipfile)."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = (off, self.pos + off, self.size + off)[whence]
        return self.pos

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.pos + n, self.size)
        if end <= self.pos:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
        with urllib.request.urlopen(req) as r:
            if r.status != 206:
                raise IOError(f"server ignored the range request (HTTP {r.status})")
            data = r.read()
        self.pos += len(data)
        return data


def fetch(dest: Path):
    """Stream the wanted members of the Zenodo zip into dest, checking size and CRC32."""
    remote = HTTPRange(ZIP_URL)
    zf = zipfile.ZipFile(remote)
    infos = {i.filename: i for i in zf.infolist()}
    root = next(n.split(TASK)[0] for n in infos if TASK in n)
    task = {n: i.file_size for n, i in infos.items() if n.startswith(root + TASK)}
    print(f"zip {remote.size / 1e9:.2f} GB, {len(infos)} members; {TASK} holds {len(task)} "
          f"members, {sum(task.values()) / 1e9:.2f} GB")
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"url": ZIP_URL, "zip_bytes": remote.size, "members": {}}
    for rel in WANTED:
        info = infos[root + TASK + rel]
        if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise SystemExit(f"{rel}: unsupported compression {info.compress_type}")
        z = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
        remote.seek(info.header_offset)
        head = remote.read(30)
        if head[:4] != b"PK\x03\x04":
            raise SystemExit(f"{rel}: bad local header")
        name_len, extra_len = struct.unpack("<HH", head[26:30])
        start = info.header_offset + 30 + name_len + extra_len
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(ZIP_URL, headers={"Range": f"bytes={start}-{start + info.compress_size - 1}"})
        crc, n, t0 = 0, 0, time.time()
        with urllib.request.urlopen(req) as r, open(out, "wb") as f:
            while True:
                raw = r.read(1 << 24)
                chunk = (z.decompress(raw) if raw else z.flush()) if z else raw
                crc, n = zlib.crc32(chunk, crc), n + len(chunk)
                f.write(chunk)
                if not raw:
                    break
        if n != info.file_size or crc != info.CRC:
            raise SystemExit(f"{rel}: size/CRC mismatch ({n} vs {info.file_size})")
        print(f"  {rel}: {info.compress_size / 1e9:.3f} GB fetched, {n / 1e9:.3f} GB written in "
              f"{time.time() - t0:.0f} s, CRC ok", flush=True)
        manifest["members"][rel] = {"bytes": n, "fetched_bytes": info.compress_size, "crc32": f"{crc:08x}"}
    json.dump(manifest, open(dest / "manifest.json", "w"), indent=1)


def rel_l2(pred, true, axis=None):
    return np.sqrt(((pred - true) ** 2).sum(axis) / (true ** 2).sum(axis))


def point_mask(shape, frac, dev):
    """random_point_inpainting_prep of dm/postprocess.ipynb: one randperm mask per frame, shared
    by u, v, w; pressure unobserved. Consumes the CUDA generator in the notebook's order."""
    import torch
    bs, nx, ny, nz = shape
    mask = torch.ones((bs, 1, nx, ny, nz), device=dev)
    for b in range(bs):
        idx = torch.randperm(nx * ny * nz, device=dev)[:int(nx * ny * nz * frac)]
        mask[b, 0, idx // (ny * nz), (idx % (ny * nz)) // nz, idx % nz] = 0
    mask = mask.repeat(1, 4, 1, 1, 1)
    mask[:, 3] = 0
    return mask


def ke_slice(a, z=16):
    """Kinetic energy on the z = 16 plane of frame 0, the panel of Fig. 6 and SI Fig. S6."""
    return 0.5 * (a[0, :3, :, :, z] ** 2).sum(0)


def ke_slice_err(true, pred):
    """get_err of 3_flow_reconstruction/comparison.ipynb: RMSE of that slice, standardized by the
    truth's std."""
    t, p = ke_slice(true), ke_slice(pred)
    return float(np.sqrt((((t - p) / t.std()) ** 2).mean()))


def run(args):
    import torch
    from methods.dmfgen.ensemble_eval import ensemble_metrics
    sys.path.insert(0, str(Path(args.upstream) / "3_flow_reconstruction" / "dm"))
    from utils.architecture import Unet
    from utils.diffusion import ElucidatedDiffusion

    dev = "cuda:0"
    data = Path(args.data)
    u = np.load(data / "data" / "u.npy", mmap_mode="r", allow_pickle=False)   # [4, nt, nx, ny, nz]
    lo = np.load(data / "data" / "MIN_u.npy", allow_pickle=False).astype(np.float32).reshape(-1)
    span = np.load(data / "data" / "MAX_u.npy", allow_pickle=False).astype(np.float32).reshape(-1) - lo
    nx, ny, nz = 256, 256, u.shape[-1]
    side, cb = (nx, ny, nz), (slice(None), None, None, None)                  # cb: per-channel broadcast
    train = np.asarray(u[:, :N_TRAIN, :nx, :ny], dtype=np.float32)            # [4, 150, nx, ny, nz]
    sigma_data = float(np.std((train - lo[cb + (None,)]) / span[cb + (None,)]))
    mean, std = train.mean(axis=(1, 2, 3, 4)), train.std(axis=(1, 2, 3, 4))   # z-score stats
    del train
    phys = np.moveaxis(np.asarray(u[:, TEST0:TEST0 + N_FRAMES, :nx, :ny], dtype=np.float32), 0, 1)
    zt = (phys - mean[cb]) / std[cb]                                          # [6, 4, nx, ny, nz]

    net = Unet(dim=16, dim_mults=(1, 2, 4, 8), channels=4, self_condition=True, flash_attn=True)
    model = ElucidatedDiffusion(net, channels=4, image_size_h=nx, image_size_w=ny, image_size_d=nz,
                                sigma_data=sigma_data).to(dev)
    model.load_state_dict(torch.load(data / "dm" / "models" / "best_model.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()

    # (a) upstream Test Setup 1: the notebook's mask-fraction loop, one sample per frame
    rows, masks, ke = [], {}, {"dns": ke_slice(phys)}
    y = torch.from_numpy((phys - lo[cb]) / span[cb]).to(dev)
    torch.manual_seed(23)
    t0 = time.time()
    for frac in MASK_FRACTIONS:
        mask = point_mask((N_FRAMES, *side), frac, dev)
        with torch.no_grad():
            pred = model.sample(torch.cat([y * mask, mask], dim=1), num_sample_steps=32)
        pa = pred.cpu().numpy() * span[cb] + lo[cb]
        za = (pa - mean[cb]) / std[cb]
        masks[frac] = mask.cpu().numpy().astype(bool)
        ke[f"dm_{frac}"] = ke_slice(pa)
        obs = masks[frac][:, :3]                                              # u, v, w observed
        err_obs = np.sqrt(((za - zt)[:, :3] ** 2 * obs).sum((2, 3, 4)) / (zt[:, :3] ** 2 * obs).sum((2, 3, 4)))
        rows.append({"mask_fraction": frac, "observed_points_per_frame": int(masks[frac][0, 0].sum()),
                     "upstream": {
                         "rel_l2_all_minmax": float(torch.norm(y - pred) / torch.norm(y)),
                         "ke_slice_nrmse_frame0": ke_slice_err(phys, pa),
                         "rel_l2_phys": dict(zip(FIELDS, rel_l2(pa, phys, axis=(2, 3, 4)).mean(0).tolist())),
                         "rel_l2_z": dict(zip(FIELDS, rel_l2(za, zt, axis=(2, 3, 4)).mean(0).tolist())),
                         "rel_l2_z_observed_points": dict(zip(FIELDS, err_obs.mean(0).tolist()))}})
    np.savez(Path(args.out).with_suffix(".npz"), **ke)
    sec_a = (time.time() - t0) / (len(MASK_FRACTIONS) * N_FRAMES)

    # (b) the eval_jhu.py path on the same draws: flat sensor indices per observed field,
    # min-max from the shared z-score, K seeded members, de-normalized, ensemble_metrics
    t0 = time.time()
    for row in rows:
        mask_a, per_frame = masks[row["mask_fraction"]], []
        for i in range(N_FRAMES):
            truth = np.moveaxis(zt[i], 0, -1).reshape(-1, 4)                  # [N, C] shared z-score
            sens = {f: np.flatnonzero(mask_a[i, f]) for f in (0, 1, 2)}
            x_norm = np.moveaxis((truth.reshape(*side, 4) * std + mean - lo) / span, -1, 0)
            m = np.zeros((4, *side), dtype=np.float32)
            for f, idx in sens.items():
                ijk = np.stack(np.unravel_index(idx, side))
                m[f, ijk[0], ijk[1], ijk[2]] = 1.0
            if not np.array_equal(m.astype(bool), mask_a[i]):
                raise SystemExit("[native] wrapper mask differs from the upstream mask")
            c = torch.from_numpy(np.concatenate([x_norm * m, m])[None]).float().to(dev)
            ens = []
            with torch.no_grad():
                for k in range(args.K):
                    torch.manual_seed(9000 + 97 * i + k)
                    ens.append(model.sample(c)[0].float().cpu().numpy())
            ens = np.moveaxis(np.stack(ens), 1, -1) * span + lo                # [K, nx, ny, nz, C]
            ens_z = ((ens - mean) / std).reshape(args.K, -1, 4)
            per_frame.append(ensemble_metrics(ens_z, truth, list(FIELDS))["per_field"])
        row["wrapper"] = {key: {f: float(np.mean([pf[f][key] for pf in per_frame])) for f in FIELDS}
                          for key in ("rel_l2_single", "rel_l2_mean", "crps", "spread_error_ratio")}
        a, w = row["upstream"]["rel_l2_z"], row["wrapper"]["rel_l2_single"]
        print(f"[native] frac {row['mask_fraction']}: relL2(all, minmax) "
              f"{row['upstream']['rel_l2_all_minmax']:.4f}; z-scored upstream/wrapper "
              + " ".join(f"{f} {a[f]:.3f}/{w[f]:.3f}" for f in FIELDS), flush=True)
    sec_b = (time.time() - t0) / (len(rows) * N_FRAMES * args.K)

    out = {"rows": rows,
           "setup": {"frames": f"test frames {TEST0}-{TEST0 + N_FRAMES - 1} of {u.shape[1]}",
                     "grid": list(side), "sigma_data": sigma_data, "K_wrapper": args.K,
                     "seconds_per_sample": {"upstream_batch6": sec_a, "wrapper_single": sec_b},
                     "train_mean": mean.tolist(), "train_std": std.tolist(),
                     "minmax_lo": lo.tolist(), "minmax_span": span.tolist()}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


def summary(args):
    run_ = json.load(open(args.run))
    w = (Path(args.data) / "dm" / "models" / "best_model.pt").read_bytes()
    blob = hashlib.sha1(b"blob %d\0" % len(w) + w).hexdigest()
    s = run_["setup"]
    mean, std, span = (np.array(s[k]) for k in ("train_mean", "train_std", "minmax_span"))
    u = np.load(Path(args.data) / "data" / "u.npy", mmap_mode="r", allow_pickle=False)
    test = np.asarray(u[:, TEST0:TEST0 + N_FRAMES, :256, :256], dtype=np.float32)
    rms_z = np.sqrt((((test - mean[:, None, None, None, None]) / std[:, None, None, None, None]) ** 2)
                    .mean(axis=(1, 2, 3, 4)))
    rows = []
    for r in run_["rows"]:
        a, b = r["upstream"]["rel_l2_z"], r["wrapper"]["rel_l2_single"]
        # the model's own units: per-channel RMSE in min-max [0, 1] units (frame-averaged ratios)
        mm = {f: a[f] * rms_z[c] * std[c] / span[c] for c, f in enumerate(FIELDS)}
        rows.append({**r, "wrapper_minus_upstream_z": {f: b[f] - a[f] for f in FIELDS},
                     "upstream_rmse_minmax": mm})
    out = {"published": PUBLISHED,
           "download": json.load(open(Path(args.data) / "manifest.json")),
           "weights_match_github_release": blob == GITHUB_WEIGHTS_BLOB,
           "setup": {**s, "test_rms_z": rms_z.tolist(), "span_over_std": (span / std).tolist()},
           "rows": rows,
           "gpu": {"job": args.job, "gpu_hours": args.gpu_hours, "au": round(256 * args.gpu_hours, 1)}}
    Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("weights_match_github_release", "gpu")}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ke = np.load(Path(args.run).with_suffix(".npz"))
    fracs = (0.0, 0.5, 0.99)
    fig, ax = plt.subplots(1, 1 + len(fracs), figsize=(4 * (1 + len(fracs)), 4))
    vmin, vmax = ke["dns"].min(), ke["dns"].max()
    for a, key, title in zip(ax, ["dns"] + [f"dm_{f}" for f in fracs],
                             ["KE (DNS)"] + [f"KE (DM), mask fraction {f}" for f in fracs]):
        a.imshow(ke[key], cmap="pink", vmin=vmin, vmax=vmax)
        a.set_title(title)
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(Path(args.run).with_name("ke_slices_fig_s6.png"), dpi=120)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--dest", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True, help="fetch --dest directory")
    r.add_argument("--upstream", required=True, help="Gen4Turbulence code root (utils/ of dm/)")
    r.add_argument("--K", type=int, default=8)
    r.add_argument("--out", required=True)
    m = sub.add_parser("summary")
    m.add_argument("--data", required=True)
    m.add_argument("--run", required=True, help="run --out JSON")
    m.add_argument("--job", default="", help="Slurm job id of the run")
    m.add_argument("--gpu-hours", type=float, required=True, help="elapsed H100 hours of the run")
    m.add_argument("--json", required=True)
    args = p.parse_args()
    {"fetch": lambda: fetch(Path(args.dest)), "run": lambda: run(args), "summary": lambda: summary(args)}[args.cmd]()


if __name__ == "__main__":
    main()
