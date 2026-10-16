"""Collect the idw_diag / idw_diag_local outputs into outputs/paper/diagnostics/idw_table.json.

    python -m analysis.diagnostics.idw_build
"""
import json

from paths import REPO_ROOT

H = REPO_ROOT / "outputs/paper/diagnostics"
K = json.load(open(H / "idw_kolmogorov.json"))
J = json.load(open(H / "idw_jhu.json"))
C_iso = json.load(open(H / "idw_cylinder.json"))
L = json.load(open(H / "idw_local.json"))
ell_w = L["cylinder_wake"]["ell_wake_D"]
JHU_DX = 2 * 3.141592653589793 / 1024
KOLM_DX = 2 * 3.141592653589793 / 256
r3 = lambda x: None if x is None else round(float(x), 4)

settings = []
for r in K["rows"]:
    settings.append({"setting": r["setting"], "regime": "kolmogorov", "field": r["field"], "n_sensors": r["n_sensors"],
                     "sensor_pool": r["pool"], "nn_search": "periodic" if "periodic" in r["setting"] else "non-periodic",
                     "mean_gap_cells": r3(r["mean_gap_cells"]), "mean_gap_phys": r3(r["mean_gap_phys"]), "phys_unit": "box length (domain [0,2pi))",
                     "ell_cells": r3(r["ell_cells"]), "ell_phys": r3(r["ell_phys"]), "gap_over_ell": r3(r["gap_over_ell"]),
                     "eps_pred": r3(r["eps_pred_train"]), "eps_nn_measured": r3(r["eps_nn_measured"]),
                     "eps_idw_measured": r3(r["eps_idw_measured"]),
                     "pred_over_measured_nn": r3(r["eps_pred_train"] / r["eps_nn_measured"]),
                     "measured_source": r["measured_source"]})
for r in J["rows"]:
    settings.append({"setting": r["setting"], "regime": "jhtdb", "field": r["field"], "n_sensors": r["n_sensors"],
                     "sensor_pool": r["pool"], "nn_search": "non-periodic",
                     "mean_gap_cells": r3(r["mean_gap_cells"]), "mean_gap_phys": r3(r["mean_gap_phys"]), "phys_unit": "JHTDB length (dx = 2pi/1024)",
                     "ell_cells": "> 62 (S2 stays below <u^2> out to half the 125^3 cutout)", "ell_phys": "> 0.38",
                     "gap_over_ell": f"< {r['mean_gap_cells']/62:.3f}",
                     "eps_pred": r3(r["eps_pred_train"]), "eps_nn_measured": r3(r["eps_nn_measured"]),
                     "eps_idw_measured": r3(r["eps_idw_measured"]),
                     "pred_over_measured_nn": r3(r["eps_pred_train"] / r["eps_nn_measured"]),
                     "measured_source": r["measured_source"]})
iso = {(r["setting"], r["field"]): r for r in C_iso["rows"]}
for r in L["cylinder_local"]:
    e = ell_w[r["field"]]
    surf = "surface" in r["setting"]
    settings.append({"setting": r["setting"], "regime": "cylinder " + ("surface" if surf else "volume"), "field": r["field"],
                     "n_sensors": r["n_sensors"],
                     "sensor_pool": "360 wall-adjacent cells" if surf else "all 23800 mesh cells",
                     "nn_search": "non-periodic",
                     "mean_gap_cells": None, "mean_gap_phys": r3(r["mean_gap_D"]), "phys_unit": "cylinder diameters D",
                     "mean_gap_phys_wake": r3(r["mean_gap_D_wake"]), "mean_gap_phys_near_body": r3(r["mean_gap_D_near_body"]),
                     "ell_cells": None, "ell_phys": r3(e), "ell_note": "wake box x>1D, |y|<4D (domain-wide isotropic ell is set by the boundary layer and is not meaningful on the graded mesh)",
                     "gap_over_ell": r3(r["mean_gap_D_wake"] / e), "gap_over_ell_note": "wake mean gap / wake ell",
                     "eps_pred": r3(r["eps_pred_local"]), "eps_pred_isotropic_S2": r3(iso[(r["setting"], r["field"])]["eps_pred_train"]),
                     "eps_nn_measured": r3(r["eps_nn_measured"]), "eps_idw_measured": r3(r["eps_idw_measured"]),
                     "pred_over_measured_nn": r3(r["eps_pred_local"] / r["eps_nn_measured"]),
                     "frac_sq_error_from_wake_box": r3(r["frac_sq_error_from_wake"]),
                     "measured_source": ("outputs/runs/cylinder2d_surface/baseline_classical/classical_baselines_surface_2d.json" if surf
                                         else "outputs/runs/cylinder2d/baseline_classical/classical_baselines_main_1pct_2d.json")})

def pick(regime, n, fields, nn="non-periodic"):
    rows = [s for s in settings if s["regime"] == regime and s["n_sensors"] == n and s["field"] in fields and s["nn_search"] == nn]
    return {s["field"]: {k: s[k] for k in ("gap_over_ell", "eps_pred", "eps_nn_measured", "eps_idw_measured")} for s in rows}

table_rows = {
    "Cylinder, volume, 1% (238/field)": {"per_field": pick("cylinder volume", 238, ["Ux", "Uy"]),
                                          "mean_gap": "0.67 D (wake 0.68 D)", "ell": "wake: Ux 0.52 D, Uy 0.72 D"},
    "Cylinder, surface, 128 taps/field": {"per_field": pick("cylinder surface", 128, ["Ux", "Uy", "p"]),
                                           "mean_gap": "6.8 D (wake 9.6 D)", "ell": "wake: Ux 0.52 D, Uy 0.72 D, p 0.76 D"},
    "Kolmogorov, 1% (655)": {"per_field": pick("kolmogorov", 655, ["vorticity"], "periodic"),
                              "per_field_nonperiodic": pick("kolmogorov", 655, ["vorticity"]),
                              "mean_gap": "5.0 cells (0.12)", "ell": "12.6 cells (0.31)"},
    "JHTDB, 1% (19531/channel), observed channels": {"per_field": pick("jhtdb", 19531, ["Ux", "Uz"]),
                                                      "mean_gap": "2.6 cells (0.016)", "ell": "> 62 cells (> half the cutout)"},
}
out = {
    "definitions": {
        "units": "fields z-scored per channel with TRAIN-split mean/std (the table units; the training-mean predictor has relative L2 = 1)",
        "S2": "second-order structure function S2(D) = <[u(x+D) - u(x)]^2> (the geostatistical variogram 2*gamma) from TRAINING frames only: Kolmogorov 320 frames (periodic FFT), JHTDB 30 frames of cubes 0-2 (zero-padded FFT, exact overlap normalization), cylinder 80 frames (20 per training Re)",
        "eps_pred": "sqrt( E_x[S2(D(x))] / E_train[u^2] ), D(x) = displacement from query point x to its nearest sensor; uniform random sensor draws of the evaluated count from the same pool (numpy; canonical draws use torch.randperm on CUDA and cannot be reproduced on CPU, the distance distribution is the same). Kolmogorov/JHTDB: S2 at the exact displacement (homogeneous). Cylinder: location-resolved, eps^2 = mean_x sum_k w_k E_train[(u(x)-u(y_k))^2] / E_train[u^2] with w_k = C(P-k, n-1)/C(P, n) the probability that the k-th nearest pool point is the nearest of n sensors drawn from a pool of P (exact under uniform sampling without replacement); the isotropic-S2 version fails on the graded, inhomogeneous mesh (kept as eps_pred_isotropic_S2)",
        "ell": "decorrelation length: smallest separation at which the radially averaged S2 reaches the mean square of the field (R = 1/2 for a zero-mean field); beyond it, copying the nearest reading no longer beats the training-mean predictor. Cylinder: wake box x > 1 D, |y| < 4 D, threshold = wake mean square",
        "mean_gap": "mean nearest-sensor distance over all query points (cylinder: expected distance under the same weights; wake and near-body (r < 1.5 D) means also given)",
        "eps_nn_measured / eps_idw_measured": "per-field relative L2 of the KD-tree nearest-neighbour and IDW (k=8, 1/d) anchors on the scored TEST-half frames, read from the classical JSONs",
    },
    "headline": {
        "kolmogorov_sweep": "eps_pred within 1% of measured NN at every density 65-6554 (ratios 0.990-1.009)",
        "cylinder": "location-resolved eps_pred within 7% of measured NN on the volume task (Ux 0.491 vs 0.526, Uy 0.611 vs 0.612) and within 6% on the surface task at every tap count",
        "jhtdb": "eps_pred from training cubes under-predicts measured NN by 8-9% (Uz) and 14-16% (Ux) at every density; with S2 measured on the scored cube the prediction is within 4% (Ux 0.215 vs 0.211, Uz 0.229 vs 0.221 at 1%): the held-out cube has ~30% larger mean-square increments at the sensor gap than the training-cube average",
        "gap_over_ell_at_1pct": "JHTDB velocity < 0.05; Kolmogorov vorticity 0.40; cylinder wake 0.95 (Uy) - 1.33 (Ux); cylinder surface (128 taps) 13-19 in the wake",
    },
    "table_rows_for_tab_diagnostics": table_rows,
    "jhu_testcube_check": L["jhu_testcube_check"],
    "settings": settings,
    "scripts": ["analysis.diagnostics.idw_diag (Kolmogorov, JHTDB, isotropic cylinder)", "analysis.diagnostics.idw_diag_local (location-resolved cylinder, JHTDB scored-cube check)", "analysis.diagnostics.idw_build"],
}
json.dump(out, open(H / "idw_table.json", "w"), indent=1)
for k, v in table_rows.items():
    print(k, json.dumps(v))
