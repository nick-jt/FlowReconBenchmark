"""Collect the pod_diag_2d / pod_diag_jhu / pod_dense_limit outputs into
outputs/paper/diagnostics/pod_table.json (table rows + full detail).

    python -m analysis.diagnostics.pod_build
"""
import json

from paths import REPO_ROOT

H = REPO_ROOT / "outputs/paper/diagnostics"
d2 = json.load(open(H / "pod_2d.json")); dj = json.load(open(H / "pod_jhu.json"))["jhu"]; dd = json.load(open(H / "pod_dense.json"))

def pick(case_d, r, n):
    for s in case_d["settings"]:
        if s["r"] == r and s.get("n_obs_per_observed_field") == n:
            return s
    raise KeyError((case_d["case"], r, n))

def row(name, case_d, r, n, sensing, dense=None, notes=""):
    s = pick(case_d, r, n)
    out = {"setting": name, "sensing": sensing, "r": r, "n_obs_per_observed_field": n,
           "n_rows_HPhi": s["n_rows_HPhi"],
           "energy_fraction": round(s["energy_fraction_train"], 4),
           "projection_floor_relL2": round(s["projection_floor_relL2"], 4),
           "projection_floor_relL2_per_field": s["projection_floor_relL2_per_field"],
           "cond_HPhi": round(s["cond_HPhi_median"], 3 if s["cond_HPhi_median"] < 100 else 0),
           "cond_HPhi_p10_p90": [round(x, 2) for x in s["cond_HPhi_p10_p90"]],
           "sigma_min": s["sigma_min_median"], "sigma_max": s["sigma_max_median"],
           "sigma_min_normalized": round(s["sigma_min_normalized_median"], 4),
           "gappy_relL2_measured": s.get("gappy_relL2_measured_canonical"),
           "gappy_relL2_measured_per_field": s.get("gappy_relL2_measured_canonical_per_field"),
           "gappy_relL2_cpu_redraw": round(s["gappy_relL2_cpu_draws"], 4),
           "total_norm_split": {"gappy": round(s["gappy_total_norm_cpu"], 4), "coef": round(s["coef_error_total_norm_cpu"], 4),
                                "floor": round(s["projection_floor_total_norm"], 4)},
           "notes": notes}
    if dense:
        out["dense_sensor_limit_relL2"] = round(dense["dense_limit_relL2"], 4)
        out["dense_sensor_limit_per_field"] = dense["dense_limit_relL2_per_field"]
    for k in ("projection_floor_relL2_Re80_Re250", "gappy_relL2_cpu_Re80_Re250"):
        if k in s: out[k] = [round(x, 4) for x in s[k]]
    return out

cyl, srf, kol = d2["cylinder"], d2["cylinder_surface"], d2["kolmogorov"]
table = [
    row("cylinder volume 1%", cyl, 20, 238, "u_x,u_y at 238 random mesh cells each; p unobserved", dd["cylinder_r20"],
        "anchor rank (holdout scan); gappy within 0.009 of the floor, 0.006 of the dense-sensor limit"),
    row("cylinder surface 128 taps", srf, 5, 128, "u_x,u_y,p at 128 of the 360 wall cells each", None,
        "anchor rank r=5 (holdout scan: every rank > 1 on the Re-200 holdout, r=5 least bad)"),
    row("cylinder surface 128 taps, r=20", srf, 20, 128, "same taps, compressible rank of the volume task", None,
        "not an anchor row: same basis as the volume task; cpu redraw only"),
    row("cylinder surface 360 taps (full ring)", srf, 5, 360, "u_x,u_y,p at all 360 wall cells", None, "anchor rank r=5"),
    row("cylinder surface 360 taps, r=20", srf, 20, 360, "full ring, rank 20", None, "not an anchor row"),
    row("Kolmogorov 1%", kol, 160, 655, "vorticity at 655 random grid points", None,
        "anchor rank 160; single channel, so the dense-sensor limit equals the projection floor"),
    row("JHTDB 1%", dj, 80, 19531, "U_x,U_z at 19,531 points each; U_y,p unobserved", dd["jhu_r80"],
        "anchor rank 80; energy fraction is in-sample over 150 correlated training frames (3 cubes x 50)"),
    row("JHTDB 0.1%", dj, 80, 1953, "U_x,U_z at 1,953 points each", None, "gappy error flat across 10x density"),
]
full = {"cylinder": cyl, "cylinder_surface": srf, "kolmogorov": kol, "jhu": dj, "dense_limit": dd}
json.dump({"definitions": {
    "energy_fraction": "cumulative POD energy of the first r modes over the TRAINING frames (anchor's own basis: z-scored with train stats, centred, channels interleaved)",
    "projection_floor_relL2": "relative L2 (per field ||.||/||y|| in z-units, mean over fields, mean over the TEST frames the anchor scored) of the orthogonal projection of each TEST frame onto mu + span(Phi_r): the lowest error any coefficient fit with this basis can reach",
    "cond_HPhi": "median over the TEST-frame sensor draws of sigma_max/sigma_min of H Phi_r (rows of the orthonormal modes at the sensed (point, channel) entries); draws use the canonical seed rule on CPU (statistically equivalent, not bit-identical, to the canonical CUDA draws)",
    "sigma_min_normalized": "sigma_min(H Phi_r) * sqrt(N*C/M); about 1 when M random entries of all channels observe every mode as well as full sampling would",
    "gappy_relL2_measured": "the reported anchor number from the classical JSON (canonical CUDA sensor draws)",
    "total_norm_split": "all-channel relative errors (||.||/||y|| over the stacked vector), which split exactly: gappy^2 = coef^2 + floor^2",
    "dense_sensor_limit_relL2": "gappy POD with every point of every observed channel sensed (normal equations); isolates cross-channel transfer from sampling"},
    "table_rows": table, "full": full}, open(H / "pod_table.json", "w"), indent=1)
for t in table:
    print(f"{t['setting']:40s} r={t['r']:3d} E={t['energy_fraction']:.4f} floor={t['projection_floor_relL2']:.4f} "
          f"cond={t['cond_HPhi']} gappy={t['gappy_relL2_measured']} cpu={t['gappy_relL2_cpu_redraw']} dense={t.get('dense_sensor_limit_relL2')} split={t['total_norm_split']}")
