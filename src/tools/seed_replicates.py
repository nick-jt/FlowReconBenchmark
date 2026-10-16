"""Regenerate the training-seed replicate configs of the 2D headline rows.

    python -m tools.seed_replicates            # writes configs/<ds>_seed{7,1337}/

Every replicate is the frozen base config with only the seed, the run identifiers
(demo number +100 / +200, as the existing replicates) and the output roots changed,
so a seed statement never mixes protocols. The seed directories are rebuilt from
scratch; stale replicates of superseded configurations are removed.
"""
from __future__ import annotations

import re
from pathlib import Path

from paths import CONFIGS_DIR

SEEDS = {7: 100, 1337: 200}          # seed -> demo-number offset
HEADLINE = {
    "kolmogorov2d": ["config_bench_kolm_ffm.yaml", "config_baseline_Det_kolm.yaml",
                     "config_baseline_MLPRBF_kolm_sig002.yaml", "config_baseline_GeoFNO_kolm.yaml",
                     "config_baseline_SiT_kolm_point.yaml", "config_baseline_Gen_kolm.yaml",
                     "config_baseline_S3GM_kolm.yaml"],
    "cylinder2d": ["config_bench_cyl_ffm.yaml", "config_baseline_Det_cyl.yaml",
                   "config_baseline_MLPRBF_cyl_sig002.yaml", "config_baseline_GeoFNO_cyl_mesh.yaml",
                   "config_baseline_SiT_cyl_point.yaml"],
}


def replicate(text: str, ds: str, seed: int, offset: int) -> str:
    out, n = re.subn(r"^(\s*seed\s*:\s*)42\b", rf"\g<1>{seed}", text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit(f"{ds}: no `seed: 42` line")
    def bump(m):
        return f"{m.group(1)}{int(m.group(2)) + offset}"
    out, n = re.subn(r"^(\s*(?:demo_num|Demo_Num)\s*:\s*)(\d+)", bump, out, count=1, flags=re.M)
    if n != 1:
        raise SystemExit(f"{ds}: no demo number line")
    out = out.replace(f"outputs/runs/{ds}/", f"outputs/runs/{ds}_seed{seed}/")
    out = out.replace(f"configs/{ds}/", f"configs/{ds}_seed{seed}/")
    return out


def main() -> None:
    for ds, names in HEADLINE.items():
        for seed, offset in SEEDS.items():
            d = CONFIGS_DIR / f"{ds}_seed{seed}"
            d.mkdir(exist_ok=True)
            for stale in d.glob("*.yaml"):
                stale.unlink()
            for name in names:
                src = CONFIGS_DIR / ds / name
                dst = d / name.replace(".yaml", f"_s{seed}.yaml")
                body = replicate(src.read_text(), ds, seed, offset)
                dst.write_text(f"# SEED REPLICATE (seed {seed}) of configs/{ds}/{name} -- identical otherwise.\n" + body)
                print("wrote", dst.relative_to(CONFIGS_DIR.parent))


if __name__ == "__main__":
    main()
