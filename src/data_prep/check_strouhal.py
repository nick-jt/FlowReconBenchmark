"""Strouhal-number V&V for the generated 2D cylinder cases.

Reads postProcessing/forceCoeffs1/*/forceCoeffs.dat (OpenFOAM 9 layout;
falls back to coefficient.dat) and prints St = f D / U (D = U = 1) plus mean Cd
over the sampling window (t >= 120). The shedding frequency is the inverse of the
mean period between upward zero crossings of the lift coefficient over t >= 210,
where every case has reached its limit cycle. An FFT peak over the 150-unit window
would resolve St only to 1/150 = 0.0067, about 4% of St.

Literature anchors (Williamson 1996; Norberg 2003): St(Re=60) ~ 0.135,
St(100) ~ 0.164, St(150) ~ 0.183, St(200) ~ 0.196. Mean Cd(100) ~ 1.33.
"""
from paths import DATA_DIR

import argparse
import glob
import os

import numpy as np


def load_coeffs(case_dir: str) -> np.ndarray:
    pats = [
        os.path.join(case_dir, "postProcessing", "forceCoeffs1", "*", "forceCoeffs.dat"),
        os.path.join(case_dir, "postProcessing", "forceCoeffs1", "*", "coefficient.dat"),
    ]
    files = sorted(f for p in pats for f in glob.glob(p))
    if not files:
        raise FileNotFoundError(f"no forceCoeffs output under {case_dir}")
    rows = []
    for fp in files:
        with open(fp) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    rows.append([float(parts[0]), float(parts[2]), float(parts[3])])
    arr = np.array(rows)  # time, Cd, Cl (OF9 columns: Time Cm Cd Cl Cl(f) Cl(r))
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    _, keep = np.unique(arr[:, 0], return_index=True)
    return arr[keep]


def strouhal(case_dir: str, t_min: float = 120.0, t_fit: float = 210.0):
    arr = load_coeffs(case_dir)
    m = arr[:, 0] >= t_min
    if m.sum() < 100:
        raise RuntimeError(f"{case_dir}: only {m.sum()} samples past t={t_min}")
    t, cd, cl = arr[m, 0], arr[m, 1], arr[m, 2]
    f = t >= t_fit
    tf, c = t[f], cl[f] - cl[f].mean()
    up = np.nonzero((c[:-1] < 0) & (c[1:] >= 0))[0]
    if len(up) < 3:
        raise RuntimeError(f"{case_dir}: fewer than 3 lift zero crossings past t={t_fit}")
    tz = tf[up] - c[up] * (tf[up + 1] - tf[up]) / (c[up + 1] - c[up])  # linear interpolation
    st = 1.0 / np.mean(np.diff(tz))
    return st, float(cd.mean()), float(np.abs(cl - cl.mean()).max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=f"{DATA_DIR}/cylinder2d/runs")
    ap.add_argument("--t-min", type=float, default=120.0)
    ap.add_argument("--t-fit", type=float, default=210.0, help="start of the zero-crossing fit")
    args = ap.parse_args()
    for case in sorted(glob.glob(os.path.join(args.runs, "Re*"))):
        try:
            st, cd, cl_amp = strouhal(case, args.t_min, args.t_fit)
            print(f"{os.path.basename(case):8s}  St = {st:.4f}   mean Cd = {cd:.3f}   |Cl| amp = {cl_amp:.3f}")
        except Exception as exc:
            print(f"{os.path.basename(case):8s}  FAILED: {exc}")


if __name__ == "__main__":
    main()
