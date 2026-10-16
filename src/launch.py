"""Launcher for every benchmark job: renders a SLURM script from a case manifest and
the site settings, then submits (or runs) it.

    python -m launch list [group]                  # what cases exist
    python -m launch show <case>                   # rendered script, not submitted
    python -m launch submit [sbatch opts] <case> [--variant NAME] [-- args...]
    python -m launch run <case> [args...]          # execute inline (driver scripts)

Cases live in launchers/cases/<group>.yaml; a group may sit in a subdirectory (2d/train).
Each case is the body of one job (flags and chained submissions preserved) with:
  * the site header (account, partition, GPU gres, Python environment) taken from
    configs/site.yaml (FRB_PROFILE=<name> selects a profile),
  * `$FRB_ROOT` / `$FRB_LOGS` / `$FRB_SRC` exported and the working directory set to
    the repository root, so every path in a case is repository-relative,
  * `frb_submit` / `frb_run` shell functions for chained cases (they call this module),
  * outputs/logs/<case>_<jobid>.log as the SLURM output file.
Cases with `variants` (e.g. one DMF-Gen trainer per config) take --variant NAME, which
sets ${CONFIG} / ${DEMO_NUM} in the body.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from paths import REPO_ROOT, site

CASES_DIR = REPO_ROOT / "launchers" / "cases"
LOGS_DIR = REPO_ROOT / "outputs" / "logs"


def load_cases() -> dict[str, dict]:
    cases = {}
    for f in sorted(CASES_DIR.rglob("*.yaml")):
        group = f.relative_to(CASES_DIR).with_suffix("").as_posix()
        for name, case in (yaml.safe_load(open(f)) or {}).items():
            if name in cases:
                raise SystemExit(f"duplicate case name {name!r} in {group} and {cases[name]['group']}")
            case = dict(case); case["group"] = group
            cases[name] = case
    return cases


def site_settings() -> dict:
    s = dict(site())
    prof = os.environ.get("FRB_PROFILE")
    if prof:
        s.update(s.get("profiles", {}).get(prof) or {})
    return s


def render(name: str, case: dict, variant: str | None = None) -> str:
    st = site_settings()
    sb = case.get("sbatch", {})
    body = case["body"]
    if case.get("variants"):
        if variant is None:
            variant = next(iter(case["variants"]))
        if variant not in case["variants"]:
            raise SystemExit(f"{name}: unknown variant {variant!r}; choose from {list(case['variants'])}")
        for k, v in case["variants"][variant].items():
            if k == "sbatch":
                sb = {**sb, **v}
            else:
                body = body.replace("${" + ("CONFIG" if k == "config" else k) + "}", str(v))
    lines = ["#!/bin/bash", f"#SBATCH --job-name={variant or name}", "#SBATCH --nodes=1", "#SBATCH --ntasks-per-node=1",
             f"#SBATCH --time={sb.get('time', '04:00:00')}", f"#SBATCH --mem={sb.get('mem', '64G')}",
             f"#SBATCH --cpus-per-task={sb.get('cpus', 8)}", f"#SBATCH --partition={st['partition']}",
             f"#SBATCH --account={st['account']}", f"#SBATCH --output={LOGS_DIR}/{variant or name}_%j.log"]
    if sb.get("gpus", 0):
        gres = st.get("gpu_gres", "gpu:1")
        if int(sb["gpus"]) > 1:
            gres = gres.rsplit(":", 1)[0] + f":{sb['gpus']}"
        lines.append(f"#SBATCH --gres={gres}")
    if sb.get("array"):
        lines.append(f"#SBATCH --array={sb['array']}")
    if st.get("mail_user"):
        lines += [f"#SBATCH --mail-user={st['mail_user']}", "#SBATCH --mail-type=END,FAIL"]
    lines += [
        "",
        f'export FRB_ROOT="{REPO_ROOT}"',
        f'export FRB_CASE="{variant or name}"',
        'export FRB_SRC="$FRB_ROOT/src"',
        'export FRB_LOGS="$FRB_ROOT/outputs/logs"',
        'export PYTHONPATH="$FRB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"',
        'mkdir -p "$FRB_LOGS"',
        st.get("env_activate", ""),
        'cd "$FRB_ROOT"',
        'frb_submit() { python -m launch submit "$@"; }',
        'frb_run() { python -m launch run "$@"; }',
        'export -f frb_submit frb_run',
        "",
        f"# ---- case {name} ({case['group']}) ----",
        body.rstrip("\n"),
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m launch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("group", nargs="?")
    p = sub.add_parser("show"); p.add_argument("case"); p.add_argument("--variant")
    p = sub.add_parser("submit", add_help=False)   # sbatch options are passed through untouched
    p = sub.add_parser("run"); p.add_argument("case"); p.add_argument("args", nargs=argparse.REMAINDER)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "submit":
        return submit(argv[1:])
    a = ap.parse_args(argv)
    cases = load_cases()
    if a.cmd == "list":
        for n, c in cases.items():
            if a.group and c["group"] != a.group: continue
            v = f"  variants: {', '.join(c['variants'])}" if c.get("variants") else ""
            print(f"{c['group']:22s} {n:40s} {c.get('purpose', '')[:90]}{v}")
    elif a.cmd == "show":
        print(render(a.case, cases[a.case], a.variant))
    elif a.cmd == "run":
        script = render(a.case, cases[a.case])
        with tempfile.NamedTemporaryFile("w", suffix=f"_{a.case}.sh", delete=False) as fh:
            fh.write(script); path = fh.name
        rc = subprocess.call(["bash", path] + a.args)
        os.unlink(path); sys.exit(rc)


def submit(argv: list[str]) -> None:
    """submit [sbatch opts] <case> [--variant NAME] [-- script args]"""
    cases = load_cases()
    sb_opts, rest, case, variant = [], [], None, None
    i = 0
    while i < len(argv):
        t = argv[i]
        if case is None and t.startswith("-"):
            if t == "--variant": variant = argv[i + 1]; i += 2; continue
            sb_opts.append(t)
            if t in ("-J", "-t", "-p", "-A", "-d", "-o", "-e", "-N", "-n", "-c") and i + 1 < len(argv): sb_opts.append(argv[i + 1]); i += 1
        elif case is None:
            case = t
        elif t == "--variant":
            variant = argv[i + 1]; i += 1
        elif t == "--":
            pass
        else:
            rest.append(t)
        i += 1
    if case is None or case not in cases:
        raise SystemExit(f"usage: python -m launch submit [sbatch opts] <case> [--variant NAME] [args]\nunknown case {case!r}")
    script = render(case, cases[case], variant)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=f"_{variant or case}.sbatch", delete=False, dir=str(LOGS_DIR)) as fh:
        fh.write(script); path = fh.name
    out = subprocess.run(["sbatch"] + sb_opts + [path] + rest, capture_output=True, text=True)
    sys.stdout.write(out.stdout); sys.stderr.write(out.stderr)
    if out.returncode != 0:
        sys.exit(out.returncode)


if __name__ == "__main__":
    main()
