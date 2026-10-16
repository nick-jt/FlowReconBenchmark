"""CoNFiLD (du2024confild) with its released weights on its native sensor-reconstruction task.

Case 4 of the paper (3D wall-bounded turbulence over cubic roughness, Re_h = 3200): one test
trajectory of N_T = 384 snapshots on the 58,483-point CNF grid, velocity (u, v, w) observed at
1, 10 or 100 random grid points at every snapshot, 10 conditional samples each. Published error
(SI Fig. S23a; Fig. 9j and 9l for 100 sensors): the space-time mean of
    Delta(x, t) = (|u|_gen - |u|_gt)^2 / (sigma_gen sigma_gt),
    sigma = RMS of |u| over the unobserved points, the snapshots and (for gen) the samples,
reported as the mean over the 10 samples with a max-min bar. Inputs come from Zenodo records
14058363 (cond_input.zip: weights, normalizers, sensor coords and measurements) and 14037782
(data.zip: data/Case4; latents.zip: Case4/valid_data.npy). The released measurements are the
DNS velocities of frames 1-384 of data/Case4 at the sensor points (found in `prep`).

Every draw is scored against two references: the DNS frames, and the "CNF label", i.e. the
latents the released CNF checkpoint fitted to those frames, decoded by the released CNF. Both
Delta and the same mean without the 1/(sigma_gen sigma_gt) factor (mse_abs) are reported.

Each draw is produced two ways with the same weights, data and sensors:
  (a) upstream: ConditionalDiffusionGeneration/inference_scripts/Case4/random_sensor notebook
      verbatim (Case4Operator in physical units, PS guidance scale 1.0, 1000-step DDPM, seed 42);
  (b) port: methods.confild.eval_unified (load_stage1/load_stage2 on converted
      checkpoints, WindowSensorOperator, measurements in the decoder's [-1, 1] field units,
      per-member seeds), i.e. the JHU window-joint arm with every row conditioned on its sensors.
Downloads are read with python -I; every checkpoint is loaded with weights_only=True (see safe_load).

    python -I native_confild.py fetch --dest <empty dir>                       # login node
    python -I native_confild.py prep --data <dest> --ckpt <dir> --out <dir>    # login node
    python -I native_confild.py run --path upstream|port|uncond|cnf --sensors 100 ...   # GPU
    python -I native_confild.py rescore --data <dest> --out <dir>                # GPU, saved latents
    python -I native_confild.py summary --data <dest> --out <dir> --json outputs/paper/diagnostics/native_confild.json
"""
import argparse
import json
import os
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from functools import partial
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # python -I drops PYTHONPATH
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SOURCES = {
    "https://zenodo.org/records/14037782/files/data.zip?download=1":
        ("data/Case4/coords.npy", "data/Case4/data.npy"),
    "https://zenodo.org/records/14037782/files/latents.zip?download=1": ("Case4/valid_data.npy",),
}
COND_URL = "https://zenodo.org/records/14058363/files/cond_input.zip?download=1"
SENSORS = (1, 10, 100)
N_T, LATENT = 384, 384
UNET = dict(image_size=384, num_channels=128, num_res_blocks=2, channel_mult="1, 1, 2, 2, 4, 4",
            num_heads=4, num_head_channels=64, attention_resolutions="32,16,8")
SAMPLER = dict(sampler="ddpm", steps=1000, noise_schedule="cosine", model_mean_type="epsilon",
               model_var_type="fixed_large", dynamic_threshold=False, clip_denoised=True,
               rescale_timesteps=False, timestep_respacing="")
# Read off the raster figures by pixel position (bar tops; time mean of the Fig. 9j curves).
# One pixel is 8e-5 in S23a and Fig. 9l and 4e-5 in Fig. 9j.
PUBLISHED = {
    "source": "du2024confild, Nat. Commun. 15:10416; SI Fig. S23a (Case 4, 10 samples per "
              "sensor count); main Fig. 9j and 9l (Case 4, 100 sensors, 10 conditional and 10 "
              "unconditional samples)",
    "metric": "<Delta_|u|>^(s,t), Eqs. 5-8",
    "sensors_1_figS23a": 0.0114, "sensors_10_figS23a": 0.0081, "sensors_100_figS23a": 0.00065,
    "sensors_100_figS23a_errorbar_top": 0.0019, "sensors_100_fig9l": 0.00061,
    "sensors_100_fig9j_time_mean": 0.00064, "unconditional_fig9l": 0.0115,
    "unconditional_fig9j_time_mean": 0.0114,
}


# ----------------------------------------------------------------------------- fetch
class HTTPRange:
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
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        data = get_range(self.url, self.pos, self.pos + n - 1)
        self.pos += len(data)
        return data


def get_range(url, a, b, tries=6):
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == b - a + 1:
                return data
        except OSError as exc:
            print("retry", a, exc, flush=True)
    raise RuntimeError(f"range {a}-{b} of {url} failed")


def fetch(dest: Path):
    """Pull only the Case-4 members of the data record, CRC-checked, plus cond_input.zip."""
    from concurrent.futures import ThreadPoolExecutor
    chunk = 16 << 20
    for url, members in SOURCES.items():
        z = zipfile.ZipFile(HTTPRange(url))
        sub = dest / url.split("/")[-1].split("?")[0].replace(".", "_")
        for name in members:
            info = z.getinfo(name)
            n, m = struct.unpack("<HH", get_range(url, info.header_offset + 26, info.header_offset + 29))
            start = info.header_offset + 30 + n + m
            spans = [(a, min(a + chunk, start + info.compress_size) - 1)
                     for a in range(start, start + info.compress_size, chunk)]
            out = sub / name
            out.parent.mkdir(parents=True, exist_ok=True)
            dec, crc, size = zlib.decompressobj(-15), 0, 0
            with open(out, "wb") as f, ThreadPoolExecutor(8) as pool:
                for blob in pool.map(lambda s: get_range(url, *s), spans):
                    piece = dec.decompress(blob)
                    crc, size = zlib.crc32(piece, crc), size + len(piece)
                    f.write(piece)
                piece = dec.flush()
                crc, size = zlib.crc32(piece, crc), size + len(piece)
                f.write(piece)
            if (crc, size) != (info.CRC, info.file_size):
                raise RuntimeError(f"{name}: CRC/size mismatch")
            print(f"OK {name} {size} B", flush=True)
    urllib.request.urlretrieve(COND_URL, dest / "cond_input.zip")
    with zipfile.ZipFile(dest / "cond_input.zip") as z:
        z.extractall(dest / "cond_input")


# ----------------------------------------------------------------------------- inputs
def paths(data: Path):
    inp = data / "cond_input" / "input"
    return dict(
        grid=data / "data_zip" / "data" / "Case4" / "coords.npy",
        fields=data / "data_zip" / "data" / "Case4" / "data.npy",
        valid_latents=data / "latents_zip" / "Case4" / "valid_data.npy",
        coords=inp / "cnf_model" / "coords.npy",
        infos=inp / "cnf_model" / "infos.npz",
        cnf=inp / "cnf_model" / "checkpoint_20000.pt",
        normalizer=inp / "cnf_model" / "Hirachical2-11-11_normalizer_params.pt",
        ema=inp / "diff_model" / "ema_0.9999_390000.pt",
        lat_max=inp / "data_scale" / "data_max.npy",
        lat_min=inp / "data_scale" / "data_min.npy",
        sensors=lambda n: (inp / "random_sensor" / str(n) / "coords.npy",
                           inp / "random_sensor" / str(n) / "measures.npy"),
    )


def sensor_points(grid, sensor_coords):
    """Grid index of each sensor; the sensors sit exactly on CNF grid points."""
    d = np.abs(grid[None, :, :] - sensor_coords[:, None, :]).max(-1)
    idx = d.argmin(1)
    if d[np.arange(len(idx)), idx].max() > 1e-9:
        raise RuntimeError("sensor coordinates are not grid points")
    return idx


def safe_load(load):
    """torch.load that never fully unpickles: upstream calls it bare, the port with weights_only=False.

    The released CNF checkpoint stores its training-log arrays (eval_losses, psnrs) as numpy
    arrays, so the restricted unpickler is given exactly the numpy names those need."""
    import torch

    def _reconstruct(*a):
        return np._core.multiarray._reconstruct(*a)
    _reconstruct.__module__ = "numpy.core.multiarray"                 # name stored in the pickle
    allow = [_reconstruct, np.ndarray, np.dtype, np.dtypes.Float32DType]

    def wrapped(*a, **k):
        k["weights_only"] = True
        with torch.serialization.safe_globals(allow):
            return load(*a, **k)
    return wrapped


def prep(args):
    """Find the test window the released measurements come from; convert weights for the port."""
    import torch
    P = paths(args.data)
    grid = np.load(P["grid"])
    if not np.array_equal(grid, np.load(P["coords"])):
        raise RuntimeError("data.zip and cond_input grids differ")
    fields = np.load(P["fields"], mmap_mode="r")                       # [T, N, 3]
    report = {"fields_shape": list(fields.shape), "fields_dtype": str(fields.dtype)}
    idx100 = sensor_points(grid, np.load(P["sensors"](100)[0]))
    meas = np.load(P["sensors"](100)[1])                               # [384, 100, 3]
    probe = np.asarray(fields[:, idx100, :], dtype=np.float64)         # [T, 100, 3]
    err = np.array([np.abs(probe[t:t + N_T] - meas).max() for t in range(len(probe) - N_T + 1)])
    t0 = int(err.argmin())
    report.update(t0=t0, t0_max_abs_mismatch=float(err[t0]),
                  next_best_mismatch=float(np.partition(err, 1)[1]), meas_scale=float(np.abs(meas).max()))
    window = np.asarray(fields[t0:t0 + N_T])
    for n in SENSORS + (1000,):
        c, m = (np.load(p) for p in P["sensors"](n))
        idx = sensor_points(grid, c)
        report[f"sensors_{n}_mismatch"] = float(np.abs(window[:, idx] - m).max())
        report[f"sensors_{n}_index"] = idx.tolist()
    print(json.dumps({k: v for k, v in report.items() if "index" not in k}, indent=1), flush=True)

    # Port-format checkpoints (what methods.confild.eval_unified.load_stage1/2 read).
    load = safe_load(torch.load)
    sd = lambda p: load(p, map_location="cpu")  # noqa: E731
    cnf, norm = sd(P["cnf"]), sd(P["normalizer"])
    ub, lb = norm["y_normalizer0u_params"][0][..., :3], norm["y_normalizer0l_params"][1][..., :3]
    report["y_normalizer_shape"] = list(ub.shape)
    report["x_normalizer"] = [v.tolist() for v in norm["x_normalizer_params"]]
    stats = {"minimum": lb.reshape(-1, 3).float(), "maximum": ub.reshape(-1, 3).float(),
             "benchmark_mean": torch.zeros(3), "benchmark_std": torch.ones(3)}
    args.ckpt.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 2, "training_stage": 1, "field_statistics": stats,
                "architecture": {"latent_dim": LATENT, "layers": 15, "hidden_features": 384},
                "decoder": cnf["model_state_dict"]}, args.ckpt / "stage1.pt")
    lat = [torch.tensor(np.load(P[k]), dtype=torch.float32).reshape(()) for k in ("lat_min", "lat_max")]
    arch = dict(UNET)
    arch.update(model_image_size=arch.pop("image_size"), channel_mult="1,1,2,2,4,4", window_length=N_T)
    torch.save({"format_version": 2, "training_stage": 2, "architecture": arch, "ema": sd(P["ema"]),
                "latent_min": lat[0], "latent_max": lat[1]}, args.ckpt / "stage2.pt")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "prep.json").write_text(json.dumps(report) + "\n")


# ----------------------------------------------------------------------------- metric
def delta_metric(gen, truth, observed):
    """<Delta_|u|>^(s,t) per sample (Eqs. 5-8), the same mean squared |u| error without the
    1/E normalization (velocity in the data's bulk units), and per-sample velocity relative L2.

    gen [S, T, N, 3] and truth [T, N, 3] in physical units; observed are sensor grid indices,
    excluded from the unobserved volume."""
    keep = np.setdiff1d(np.arange(truth.shape[1]), observed)
    g = np.linalg.norm(gen[:, :, keep], axis=-1).astype(np.float64)
    t = np.linalg.norm(truth[:, keep], axis=-1).astype(np.float64)
    e = np.sqrt((g ** 2).mean()) * np.sqrt((t ** 2).mean())
    mse = ((g - t[None]) ** 2).mean(axis=(1, 2))
    d = (gen[:, :, keep] - truth[None, :, keep]).astype(np.float64)
    rel = np.sqrt((d ** 2).sum(axis=(1, 2, 3)) / (truth[:, keep].astype(np.float64) ** 2).sum())
    mean_rel = float(np.linalg.norm(gen[:, :, keep].mean(0) - truth[:, keep]) / np.linalg.norm(truth[:, keep]))
    out = {"E": float(e), "rel_l2_per_sample": rel.tolist(), "rel_l2_mean": float(rel.mean()),
           "rel_l2_ensemble_mean": mean_rel}
    for name, v in (("delta", mse / e), ("mse_abs", mse)):
        out.update({f"{name}_per_sample": v.tolist(), f"{name}_mean": float(v.mean()),
                    f"{name}_median": float(np.median(v)), f"{name}_min": float(v.min()),
                    f"{name}_max": float(v.max())})
    return out


# ----------------------------------------------------------------------------- GPU runs
def upstream_operator(P, sensor_coords, dev):
    from ConditionalDiffusionGeneration.src.guided_diffusion.measurements import get_operator
    return get_operator(device=dev, name="case4", coords_path=str(sensor_coords), max_val_path=str(P["lat_max"]),
                        min_val_path=str(P["lat_min"]), normalizer_params_path=str(P["normalizer"]),
                        ckpt_path=str(P["cnf"]), batch_size=384)


def upstream_draw(P, n, samples, scale, dev):
    """The released Case-4 random-sensor notebook, cell by cell; returns latents [S, T, D]."""
    import torch
    from ConditionalDiffusionGeneration.src.guided_diffusion.unet import create_model
    from ConditionalDiffusionGeneration.src.guided_diffusion.condition_methods import get_conditioning_method
    from ConditionalDiffusionGeneration.src.guided_diffusion.measurements import get_noise
    from ConditionalDiffusionGeneration.src.guided_diffusion.gaussian_diffusion import create_sampler
    torch.manual_seed(42)
    np.random.seed(42)
    sc, sm = P["sensors"](max(n, 1))
    meas = torch.from_numpy(np.load(sm)).to(dev)
    # The notebook names ema_0.9999_400000.pt, which the release does not contain, and upstream
    # create_model silently falls back to random weights on a failed load: load strictly here.
    unet = create_model(**UNET, model_path=str(P["ema"]))
    unet.load_state_dict(torch.load(P["ema"], map_location="cpu"))
    unet.to(dev).eval()
    op = upstream_operator(P, sc, dev)
    noiser = get_noise(sigma=0.0, name="gaussian")
    cond = get_conditioning_method(operator=op, noiser=noiser, name="ps" if n else "vanilla", scale=scale)
    sampler = create_sampler(**SAMPLER)
    sample_fn = partial(sampler.p_sample_loop, model=unet, measurement_cond_fn=partial(cond.conditioning))
    x_start = torch.randn(samples, 1, N_T, LATENT, device=dev)
    mask = torch.ones_like(meas)
    out = [sample_fn(x_start=x_start[i:i + 1], measurement=mask * meas, record=False, save_root=None)
           for i in range(samples)]
    return op._unnorm(torch.cat(out))[:, 0].detach(), op


def upstream_decode(P, op, latents, dev):
    import torch
    from einops import rearrange
    from ConditionalNeuralField.cnf.inference_function import decoder
    coords = torch.tensor(np.load(P["coords"]), device=dev, dtype=torch.float32)
    f = decoder(coords, rearrange(latents, "s t l -> (s t) l"), op.model, op.x_normalizer,
                op.y_normalizer, batch_size=16, device=dev)
    return rearrange(f, "(s t) n c -> s t n c", t=N_T).numpy()


def port_draw(P, args, n, samples, scale, dev):
    """methods.confild.eval_unified window-joint arm on the native window; returns physical fields."""
    import torch
    from methods.confild.eval_unified import (WindowSensorOperator, create_sampler,
                                                    get_conditioning_method, get_noise,
                                                    load_stage1, load_stage2)
    decoder, stats, _ = load_stage1(str(args.ckpt / "stage1.pt"), dev)
    denoiser, lat_min, lat_max, window = load_stage2(str(args.ckpt / "stage2.pt"), dev, LATENT, N_T)
    norm = torch.load(P["normalizer"], map_location="cpu", weights_only=True)
    x_ub, x_lb = (v.to(dev).float() for v in norm["x_normalizer_params"])
    grid = torch.tensor(np.load(P["coords"]), dtype=torch.float32, device=dev)
    coords_full = (grid - x_lb) / (x_ub - x_lb) * 2.0 - 1.0          # upstream x-normalizer, [-1, 1]
    sc, sm = P["sensors"](n)
    idx = torch.from_numpy(sensor_points(np.load(P["coords"]), np.load(sc))).to(dev)
    meas = torch.from_numpy(np.load(sm)).float().to(dev)              # [T, n, 3] physical
    # Each sensor carries u, v, w: three (point, field) observations, as cond_fields does on JHU.
    ids = idx.repeat_interleave(3)
    fld = torch.arange(3, device=dev).repeat(n)
    raw = meas.repeat_interleave(3, dim=1)                             # [T, 3n, 3]
    y = torch.stack([stats.normalize(raw[t], ids).gather(1, fld[:, None])[:, 0]
                     for t in range(window)]).reshape(1, -1)           # eval_unified2.measurement_rows
    op = WindowSensorOperator(decoder, coords_full[ids].expand(window, -1, -1),
                              fld.expand(window, -1), lat_min, lat_max, row_chunk=window)
    cond = get_conditioning_method(name="ps", operator=op, noiser=get_noise(sigma=0.0, name="gaussian"),
                                   scale=scale)
    sampler = create_sampler(**SAMPLER)
    sample_fn = partial(sampler.p_sample_loop, model=denoiser, measurement_cond_fn=partial(cond.conditioning),
                        record=False, save_root=None)
    gen = np.zeros((samples, window, grid.shape[0], 3), dtype=np.float32)
    lat = []
    for k in range(samples):
        torch.manual_seed(args.op_seed + 100003 * k)                   # eval_unified2 member seed, start 0
        z_n = sample_fn(x_start=torch.randn(1, 1, window, LATENT, device=dev), measurement=y)
        latents = op.unnorm(z_n.detach()).reshape(window, LATENT)
        lat.append(latents)
        with torch.no_grad():                                          # eval_unified2.decode_latent
            for r in range(window):
                pm1 = decoder(coords_full[None], latents[r].view(1, 1, -1))[0]
                gen[k, r] = stats.benchmark_normalize(stats.denormalize(pm1)).cpu().numpy()
        print(f"[port] member {k + 1}/{samples}", flush=True)
    return gen, torch.stack(lat)


def references(args, P, n, dev):
    """DNS truth window, CNF label (released fitted latents decoded by the released CNF),
    observed sensor indices and the upstream operator."""
    import torch
    prep_rep = json.loads((args.out / "prep.json").read_text())
    t0 = prep_rep["t0"]
    truth = np.asarray(np.load(P["fields"], mmap_mode="r")[t0:t0 + N_T], dtype=np.float32)
    observed = np.asarray(prep_rep[f"sensors_{max(n, 1)}_index"]) if n else np.array([], dtype=int)
    op = upstream_operator(P, P["sensors"](max(n, 1))[0], dev)
    fitted = torch.load(P["cnf"], map_location="cpu")["hidden_states_model"]["latents"][None, t0:t0 + N_T]
    label = upstream_decode(P, op, fitted.to(dev), dev)[0]
    return t0, truth, label, observed, op, fitted


def gpu_setup(args):
    import torch
    torch.load = safe_load(torch.load)
    sys.path.insert(0, str(args.upstream))
    return paths(args.data), torch.device("cuda")


def run(args):
    import torch
    P, dev = gpu_setup(args)
    n = 0 if args.path == "uncond" else args.sensors
    t0, truth, label, observed, op, fitted = references(args, P, n, dev)
    tic = time.time()
    if args.path == "port":
        gen, lat = port_draw(P, args, n, args.samples, args.scale, dev)
    elif args.path == "cnf":
        args.samples, gen, lat = 1, label[None], fitted
    else:
        lat, op = upstream_draw(P, n, args.samples, args.scale, dev)
        gen = upstream_decode(P, op, lat, dev)
    wall = time.time() - tic
    res = {"path": args.path, "sensors": n, "samples": args.samples,
           "scale": args.scale if n and args.path != "cnf" else 0.0,
           "steps": SAMPLER["steps"], "t0": t0, "wall_s": wall, "gpu": torch.cuda.get_device_name(),
           "vs_dns": delta_metric(gen, truth, observed), "vs_cnf_label": delta_metric(gen, label, observed)}
    if args.path == "port":
        res["op_seed"] = args.op_seed
    print(json.dumps({k: ({a: b for a, b in v.items() if "per_sample" not in a} if isinstance(v, dict) else v)
                      for k, v in res.items()}), flush=True)
    tag = f"{args.path}_s{n}"
    np.save(args.out / f"latents_{tag}.npy", lat.float().cpu().numpy())
    (args.out / f"{tag}.json").write_text(json.dumps(res, indent=1) + "\n")


def rescore(args):
    """Re-score saved draws: decode latents_<arm>.npy with the released CNF (as upstream does)."""
    import torch
    P, dev = gpu_setup(args)
    for f in sorted(args.out.glob("*_s*.json")):
        res = json.loads(f.read_text())
        _, truth, label, observed, op, _ = references(args, P, res["sensors"], dev)
        lat = torch.from_numpy(np.load(args.out / f"latents_{f.stem}.npy")).to(dev)
        gen = upstream_decode(P, op, lat, dev)
        res.update(vs_dns=delta_metric(gen, truth, observed), vs_cnf_label=delta_metric(gen, label, observed))
        f.write_text(json.dumps(res, indent=1) + "\n")
        print(f.stem, {r: round(res[r]["mse_abs_mean"], 6) for r in ("vs_dns", "vs_cnf_label")}, flush=True)


def split_check(P, t0):
    """Is the truth window one of the released diffusion validation windows (stride-1 windows)?"""
    import torch
    fitted = safe_load(torch.load)(P["cnf"], map_location="cpu")["hidden_states_model"]["latents"].numpy()
    first = {fitted[j].tobytes(): j for j in range(len(fitted) - N_T + 1)}
    starts = sorted(first[w[0].tobytes()] for w in np.load(P["valid_latents"]))
    return {"stride1_windows": len(fitted) - N_T + 1, "validation_windows": len(starts),
            "truth_window_is_validation_window": t0 in starts,
            "validation_window_frames_shared_with_truth_max": int(max(N_T - abs(j - t0) for j in starts))}


PORT_DIFFERENCES = (
    "DPS residual in the decoder's [-1, 1] field units (global per-channel min-max of the released "
    "normalizer) instead of physical units, so at the same nominal scale the guidance step is "
    "larger by 2/(max-min) per channel (guidance_unit_factor)",
    "each sensor enters as three (point, channel) observations through WindowSensorOperator, "
    "decoded with gradient checkpointing; same values as upstream's (point, u/v/w) measurement",
    "UNet built by UnconditionalDiffusionTraining script_util.create_model (same UNetModel class) "
    "and loaded strictly; upstream's create_model falls back to random weights on a failed load",
    "per-member seeds (op_seed + 100003 k) instead of one seeded batch of initial noise",
    "measurements in float32 instead of float64",
)
TRAIN_SCAN = ("Case4/train_data.npy (653 windows) streamed from latents.zip: every window is a "
              "stride-1 window of the CNF-fitted latents; one starts at frame 1, the truth window")


def summary(args):
    import torch
    runs = {p.stem: json.loads(p.read_text()) for p in sorted(args.out.glob("*_s*.json"))}
    prep_rep = json.loads((args.out / "prep.json").read_text())
    P = paths(args.data)
    norm = safe_load(torch.load)(P["normalizer"], map_location="cpu")
    span = norm["y_normalizer0u_params"][0][:3] - norm["y_normalizer0l_params"][1][:3]
    keep = [f"{m}_{s}" for m in ("delta", "mse_abs") for s in ("mean", "median", "min", "max")]
    keep += ["E", "rel_l2_mean", "rel_l2_ensemble_mean"]
    out = {"published": PUBLISHED,
           "truth_window": {**{k: prep_rep[k] for k in ("t0", "t0_max_abs_mismatch", "fields_shape")},
                            **split_check(P, prep_rep["t0"]), "train_windows": TRAIN_SCAN},
           "port_differences": PORT_DIFFERENCES,
           "guidance_unit_factor_uvw": (2.0 / span).tolist(),
           "sampling_gpu_hours": sum(v["wall_s"] for v in runs.values()) / 3600,
           "runs": {k: {**{f: v[f] for f in ("sensors", "samples", "scale", "steps", "wall_s", "gpu")},
                        # draws that land on the CNF-fitted truth window (velocity rel. L2 < 5% to it)
                        "samples_at_cnf_label": sum(r < 0.05 for r in v["vs_cnf_label"]["rel_l2_per_sample"]),
                        **{ref: {f: v[ref][f] for f in keep} for ref in ("vs_dns", "vs_cnf_label")}}
                    for k, v in runs.items()}}
    args.json.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--dest", type=Path, required=True)
    for name in ("prep", "run", "rescore", "summary"):
        s = sub.add_parser(name)
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--data", type=Path, required=True)
        if name in ("prep", "run"):
            s.add_argument("--ckpt", type=Path, required=True, help="port-format checkpoint dir")
        if name in ("run", "rescore"):
            s.add_argument("--upstream", type=Path, default=Path(os.environ.get(
                "CONFILD_ROOT", str(Path(__file__).resolve().parents[3] / "third_party" / "CoNFiLD"))))
    r = sub.choices["run"]
    r.add_argument("--path", choices=("upstream", "port", "uncond", "cnf"), required=True)
    r.add_argument("--sensors", type=int, default=100, choices=SENSORS)
    r.add_argument("--samples", type=int, default=10)
    r.add_argument("--scale", type=float, default=1.0)
    r.add_argument("--op-seed", type=int, default=1000)
    sub.choices["summary"].add_argument("--json", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "fetch":
        fetch(args.dest)
    else:
        {"prep": prep, "run": run, "rescore": rescore, "summary": summary}[args.cmd](args)


if __name__ == "__main__":
    main()
