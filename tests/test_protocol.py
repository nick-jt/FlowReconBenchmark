"""Protocol rules every trainer and evaluator share (protocol, eval.fleet_select)."""
import json

import numpy as np
import pytest
import torch
import yaml

from eval import fleet_select
from protocol import (PROTOCOL_2D, count_parameters, fixed_validation_draw, held_out_subset,
                          resume_best_val, within_budget)


def test_tune_and_test_halves_partition_the_held_out_block():
    held = np.arange(2560, 3200)
    tune, test = held_out_subset(held, "tune"), held_out_subset(held, "test")
    assert len(tune) == len(test) == 320
    assert not set(tune) & set(test)
    assert sorted(set(tune) | set(test)) == list(held)
    assert test[0] == 2560 and tune[0] == 2561          # even positions = TEST, odd = TUNE
    assert held_out_subset(held, "val") is held


def test_fixed_validation_draw_is_repeatable_and_leaves_training_rng_alone():
    cond = {"cond_fields": [0, 1], "n_obs_min_list": [24], "n_obs_max_list": [476], "vis_n_obs_list": [238]}
    torch.manual_seed(123)
    draws = []
    for _ in range(2):
        with fixed_validation_draw(cond) as n_obs:
            assert n_obs == [238, 238]
            assert cond["n_obs_min_list"] == cond["n_obs_max_list"] == [238, 238]
            draws.append(torch.randperm(1000)[:5].tolist())
        assert cond["n_obs_min_list"] == [24] and cond["n_obs_max_list"] == [476]
    assert draws[0] == draws[1]
    torch.manual_seed(123)
    expected = torch.randperm(1000)[:5].tolist()
    torch.manual_seed(123)
    with fixed_validation_draw(cond):
        torch.randperm(1000)
    assert torch.randperm(1000)[:5].tolist() == expected


def test_count_parameters_counts_complex_elements_once():
    m = torch.nn.Module()
    m.w = torch.nn.Parameter(torch.zeros(3, 4, dtype=torch.cfloat))
    m.b = torch.nn.Parameter(torch.zeros(5))
    c = count_parameters(m)
    assert c["total"] == 17 and c["complex_elements"] == 12
    assert within_budget(6_500_000 * 1.09) and not within_budget(6_500_000 * 1.11)


def test_resume_best_val_never_uses_last_pt_val_loss(tmp_path):
    assert resume_best_val({"val_loss": 0.1, "best_val": 0.05}, tmp_path) == 0.05
    torch.save({"val_loss": 0.07}, tmp_path / "best.pt")
    assert resume_best_val({"val_loss": 0.1}, tmp_path) == 0.07
    assert resume_best_val({"val_loss": 0.1}, tmp_path / "nowhere") == float("inf")


def _payload(**kw):
    d = {"protocol": PROTOCOL_2D, "split": "test", "clamp_hard": False, "n_obs": [655],
         "model": "senseiver", "K": 1, "n_frames": 50, "snapshots": [{}] * 50, "summary": {}}
    d.update(kw)
    return d


def test_canonical_row_requires_the_full_protocol_stamp():
    assert fleet_select.why_not_canonical(_payload(), "kolmogorov2d") is None
    assert "split" in fleet_select.why_not_canonical(_payload(split="val"), "kolmogorov2d")
    assert "clamp" in fleet_select.why_not_canonical(_payload(clamp_hard=True), "kolmogorov2d")
    assert "protocol" in fleet_select.why_not_canonical(_payload(protocol="kolm2d_matched_v1"), "kolmogorov2d")
    assert "snapshot records" in fleet_select.why_not_canonical(_payload(snapshots=[{}] * 49), "kolmogorov2d")
    assert "K=" in fleet_select.why_not_canonical(_payload(model="dmfgen", K=1), "kolmogorov2d")


def test_missing_reported_family_is_an_error_not_a_fallback(tmp_path):
    ev = tmp_path / "kolmogorov2d" / "baseline_sit" / "run" / "Evaluation"
    ev.mkdir(parents=True)
    json.dump(_payload(model="sit", K=8), open(ev / "kolm_fleet_sit_K8_nfe50.json", "w"))
    with pytest.raises(SystemExit, match="not the reported family baseline_sit_point"):
        fleet_select.load_canonical_fleet("kolmogorov2d", tmp_path)


def test_verify_run_config_rejects_configs_outside_the_repo_tree(tmp_path):
    run = tmp_path / "run"; run.mkdir()
    json.dump({"config_path": str(tmp_path / "elsewhere.yaml")}, open(run / "run_metadata.json", "w"))
    with pytest.raises(SystemExit, match="not a file under"):
        fleet_select.verify_run_config(run)
    run2 = tmp_path / "run2"; run2.mkdir()
    cfg = fleet_select.CONFIGS_DIR / "kolmogorov2d" / "config_baseline_Det_kolm.yaml"
    json.dump({"config_path": str(cfg)}, open(run2 / "run_metadata.json", "w"))
    from methods.common.config import validate_and_normalize_config
    normalised = validate_and_normalize_config(yaml.safe_load(open(cfg)))
    yaml.safe_dump(normalised, open(run2 / "run_config.yaml", "w"))
    assert fleet_select.verify_run_config(run2) == cfg
    normalised["shared"]["conditioning"]["n_obs_max_list"] = [655]
    yaml.safe_dump(normalised, open(run2 / "run_config.yaml", "w"))
    with pytest.raises(SystemExit, match="differs from"):
        fleet_select.verify_run_config(run2)
