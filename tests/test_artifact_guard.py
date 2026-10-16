"""Result-identity guard (eval.artifact_guard): replays the 2026-09-10 incidents in
which an evaluation JSON was silently overwritten by a run with a different identity,
checks the legitimate re-writes still go through, and (slow) checks that every real
artifact in the run tree parses and never conflicts with itself."""
import glob
import json
import tempfile
from pathlib import Path

from eval import artifact_guard as G
from paths import RUNS_DIR


def P(**kw):
    base = {"protocol": "kolm2d_matched_v1", "model": "s3gm", "n_obs": [655],
            "K": 8, "nfe": 200, "cond_source": "points", "cond_fields": [0],
            "summary": {"aggregate": {"rel_l2_mean": 0.5}}}
    base.update(kw)
    return base


def quiet(*a, **k):
    return None


def outcome(existing, new):
    """(conflicted, existing_kept, output_exists) for writing `new` over `existing`."""
    G.CONFLICTS.clear()
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "target.json"
        if existing is not None:
            f.write_text(existing if isinstance(existing, str) else json.dumps(existing))
        before = f.read_text() if f.exists() else None
        out = G.safe_write_json(f, new, log=quiet)
        conflicted = bool(G.CONFLICTS)
        kept = (f.read_text() == before) if conflicted else True
        return conflicted, kept, out.exists()


CONFLICTS = [
    ("#1 NOBS override (65) over canonical 655", P(), P(n_obs=[65])),
    ("#3 operator aggregate over clean aggregate", P(), P(protocol="kolm2d_matched_v1_op_occl0.25")),
    ("#4 dmfgen occl sweep over clean sweep", P(model="dmfgen", K=8, nfe=4),
     P(model="dmfgen", K=8, nfe=4, protocol="kolm2d_matched_v1_op_occl0.25")),
    ("#4b clean roll-up (no n_obs) vs operator roll-up",
     {"protocol": "kolm2d_matched_v1", "model": "dmfgen", "K": 8, "nfe": 4},
     {"protocol": "kolm2d_matched_v1_op_noise0.1", "model": "dmfgen", "K": 8, "nfe": 4}),
    ("K collision: K=64 run over K=8 sweep", P(model="dmfgen", nfe=4), P(model="dmfgen", nfe=4, K=64)),
    ("pre-stamp clean cache vs operator record", {"n_obs": 655, "K": 8, "nfe": 200},
     {"n_obs": 655, "K": 8, "nfe": 200, "protocol": "kolm2d_matched_v1_op_noise0.3"}),
    ("surface 64 taps vs 128 taps", P(n_obs=[64, 64, 64], cond_source="surface"),
     P(n_obs=[128, 128, 128], cond_source="surface")),
    ("3-frame smoke over 50-frame canonical (n_frames)", P(n_frames=50), P(n_frames=3)),
    ("best vs last checkpoint (ckpt)", P(ckpt="best"), P(ckpt="last")),
    ("different evaluation seed (sensor draws)", P(seed=0), P(seed=1)),
    ("val vs test split", P(split="val"), P(split="test")),
]
LEGITIMATE = [
    ("pre-n_frames artifact vs 3-frame run (no key: skip)",
     {"protocol": "kolm2d_matched_v1", "model": "s3gm", "n_obs": [655], "K": 8}, P(n_frames=3)),
    ("same measurement re-run, new numbers", P(), P(summary={"aggregate": {"rel_l2_mean": 0.49}})),
    ("pre-stamp clean cache vs clean record", {"n_obs": 655, "K": 8, "nfe": 200},
     {"n_obs": 655, "K": 8, "nfe": 200, "protocol": "kolm2d_matched_v1"}),
    ("cache n_obs int vs payload n_obs list", {"n_obs": 655, "K": 1}, P(K=1)),
    ("roll-up re-written with a different density list",
     {"protocol": "kolm2d_matched_v1", "model": "dmfgen", "K": 8, "nfe": 4, "rel_l2_by_n": {"655": 1}},
     {"protocol": "kolm2d_matched_v1", "model": "dmfgen", "K": 8, "nfe": 4, "rel_l2_by_n": {"65": 1}}),
    ("fresh file", None, P()),
    ("corrupt existing file", "{not json", P()),
    ("empty existing file", "", P()),
]


def test_identity_conflicts_are_refused_and_existing_kept():
    for name, existing, new in CONFLICTS:
        conflicted, kept, exists = outcome(existing, new)
        assert conflicted and kept and exists, name


def test_legitimate_rewrites_go_through():
    for name, existing, new in LEGITIMATE:
        conflicted, kept, exists = outcome(existing, new)
        assert not conflicted and exists, name


def test_sidecar_named_by_identity_and_exit_code():
    G.CONFLICTS.clear()
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.json"
        f.write_text(json.dumps(P()))
        side = G.safe_write_json(f, P(K=64), log=quiet)
        assert side.name.startswith("t.CONFLICT_")
        assert json.loads(side.read_text())["K"] == 64
        try:
            G.exit_if_conflicts(log=quiet)
        except SystemExit as e:
            assert e.code == 3
        else:
            raise AssertionError("exit_if_conflicts did not fail the job")


def test_every_real_artifact_is_self_consistent():
    """Slow: scans every eval JSON under outputs/runs for the 2D fleets."""
    n = bad = 0
    for pat in ("kolmogorov2d", "cylinder2d", "cylinder2d_surface"):
        for f in glob.glob(str(RUNS_DIR / pat / "*" / "*" / "Evaluation" / "*.json")):
            try:
                d = json.loads(Path(f).read_text())
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            n += 1
            G.CONFLICTS.clear()
            with tempfile.TemporaryDirectory() as td:
                t = Path(td) / "x.json"
                t.write_text(json.dumps(d))
                G.safe_write_json(t, d, log=quiet)
                bad += bool(G.CONFLICTS)
    assert n > 0, "no artifacts found under outputs/runs (is the run tree linked?)"
    assert bad == 0, f"{bad} of {n} artifacts conflict with themselves"
