#!/usr/bin/env python3
"""Emit the LaTeX row body of the surface-to-field table from the eval JSONs.

Source: Save_TrainedModel/cylinder2d_surface/*/*/Evaluation/cyl_surface_*.json
(learned, n=50 Re-stratified held-out frames) and baseline_classical/*.json
(all 600 frames). Columns are tap budgets per field on the 360-cell wall ring.

Grid-locked models (geofno, sit, latent_fm, s3gm) draw from the 62 unique grid
cells the ring rasterizes onto, so their 128 and 360 entries are identical to
64 BY CONSTRUCTION; where the eval was not re-run at those budgets we print the
64-tap value with a dagger rather than leaving a hole or implying new work.
"""
from paths import REPO_ROOT
import glob, json, re
from pathlib import Path

ROOT = REPO_ROOT
STM = ROOT / "outputs/runs" / "cylinder2d_surface"
OUT = ROOT / "outputs" / "paper"
TAPS = [32, 64, 128, 360]
GRID = {"geofno", "sit", "latent_fm", "s3gm"}          # pool caps at 62 cells
NICE = {"senseiver": "Senseiver", "geofno": "Geo-FNO", "dmfgen": r"\dmfgen{}",
        "mlp_rbf": "MLP-RBF", "sit": "SiT-point"}
ORDER = ["sit", "dmfgen", "senseiver", "geofno", "mlp_rbf"]   # the mesh-native fleet; grid-locked families have no cylinder row


def learned():
    """Rows of each model's reported family only (fleet_select.REPORTED_FAMILY,
    cylinder entry); rows under any other family are labelled variants."""
    from eval.fleet_select import REPORTED_FAMILY
    out, fam_of = {}, {}
    for p in sorted(glob.glob(str(STM / "*" / "*" / "Evaluation" / "cyl_surface_*.json"))):
        d = json.load(open(p))
        # the combined sweep payload (cyl_surface_<m>_sweep.json) carries
        # rel_l2_by_n instead of a single n_obs; the per-density files are the
        # ones with a summary, so skip anything without both keys
        if "n_obs" not in d or "summary" not in d:
            continue
        m, n = d["model"], int(d["n_obs"][0])
        fam = Path(p).parents[2].name
        want = REPORTED_FAMILY[m.replace("_", "")]
        if isinstance(want, dict):
            want = want["cylinder2d"]
        if fam != want:
            print(f"[surface] {m}: ignoring variant row {Path(p).name} ({fam} != {want})")
            continue
        fam_of[m] = fam
        a = d["summary"]["aggregate"]
        out.setdefault(m, {})[n] = (a["rel_l2_mean"], a["crps"], a.get("spread_error_ratio"))
    for m, fam in fam_of.items():
        print(f"[surface] {m}: rows from {fam}")
    return out


def classical():
    out, ranks = {}, set()
    for f in glob.glob(str(STM / "baseline_classical" / "*.json")):
        tag = Path(f).stem
        for k, v in json.load(open(f)).get("results", {}).items():
            if "aggregate" not in v:
                continue
            if "_n" not in k:          # e.g. constant_train_mean: no tap budget
                continue
            base, n = k.rsplit("_n", 1)
            if not n.isdigit():
                continue
            lbl = {"idw": r"IDW ($k{=}8$)", "kdtree": "Nearest neighbor", "gappy_pod": "Gappy POD"}.get(base)
            if lbl is None:
                continue
            if base == "gappy_pod":
                ranks.add(int(v["rank"]))
            out.setdefault(lbl, {})[int(n)] = (v["aggregate"]["rel_l2_mean"], v["aggregate"]["crps"])
    if "Gappy POD" in out:          # rank chosen per tap budget on the training holdout
        lbl = (rf"Gappy POD ($r{{=}}{ranks.pop()}$)" if len(ranks) == 1
               else "Gappy POD (rank per budget)")
        out[lbl] = out.pop("Gappy POD")
    # Ridge gappy POD and LSE live in their own directory so their IDW rows (scanned
    # exponent) cannot replace the canonical ones read above.
    ridge = STM / "baseline_classical_ridge_lse" / "classical_baselines_surface_ridge_lse_2d.json"
    if ridge.exists():
        res = json.load(open(ridge))["results"]
        for base, lbl in (("gappy_pod_ridge", "Ridge gappy POD"), ("lse", "LSE")):
            rows = {int(k.rsplit("_n", 1)[1]): v for k, v in res.items() if k.rsplit("_n", 1)[0] == base}
            out[lbl] = {n: (v["aggregate"]["rel_l2_mean"], v["aggregate"]["crps"]) for n, v in rows.items()}
    return out


def fmt(vals, m=None):
    cells = []
    for t in TAPS:
        if t in vals:
            cells.append(f"{vals[t][0]:.3f}")
        elif m in GRID and 64 in vals and t > 64:
            cells.append(rf"{vals[64][0]:.3f}$^\dagger$")   # pool-capped, identical by construction
        else:
            cells.append("---")
    return cells


if __name__ == "__main__":
    L, C = learned(), classical()
    best = min(v[max(v)][0] for v in L.values() if v)
    lines = []
    for m in ORDER:
        if m not in L:
            continue
        v = L[m]
        c = fmt(v, m)
        top = v[max(v)]
        if abs(top[0] - best) < 1e-9:
            c[-1] = rf"\textbf{{{c[-1]}}}"          # best full-ring entry
        spr = "---" if top[2] is None else f"{top[2]:.2f}"
        lines.append(f"{NICE[m]} & " + " & ".join(c) + f" & {top[1]:.3f} & {spr} \\\\")
    lines.append(r"\hline")
    for lbl in sorted([k for k in C if k.startswith(("Gappy POD", "Ridge gappy POD", "LSE"))], key=lambda k: C[k][128][0]) \
            + [r"IDW ($k{=}8$)", "Nearest neighbor"]:
        if lbl not in C:
            continue
        v = C[lbl]
        lines.append(f"{lbl} & " + " & ".join(fmt(v)) + f" & {v[max(v)][1]:.3f} & --- \\\\")
    (OUT / "tab_surface_body.tex").write_text(
        "% GENERATED by src/figs_pof/make_surface_table.py -- do not hand-edit\n"
        + "\n".join(lines) + "\n")
    print((OUT / "tab_surface_body.tex").read_text())
